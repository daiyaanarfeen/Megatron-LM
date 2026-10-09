"""Validate GPU-only traces and compare the same four recorded iterations."""

import argparse
import json
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from analyze_exact_uniform_nep_slowdown import intersection_duration, union_duration
from analyze_nsys_mbs_pair import ROOT, labels


def analyze(path, reference, experts, first=10, last=14):
    data = json.loads(path.read_text())
    assert data['dropped_records'] == data['invalid_records'] == 0, path
    events = data['traceEvents']
    names = sorted({e['name'] for e in events if e['cat'] == 'kernel'})
    demangled = subprocess.run(['c++filt'], input='\n'.join(names)+'\n', text=True,
                               capture_output=True, check=True).stdout.splitlines()
    mapping = dict(zip(names, demangled))
    for e in events:
        e.update(a=e['ts']/1000, b=(e['ts']+e['dur'])/1000)
        if e['cat'] == 'kernel':
            e['name'] = mapping[e['name']]
    ranges = sorted((e for e in events if e['name'] in
                     [f'iteration_{i}' for i in range(first,last)]), key=lambda e:e['a'])
    assert len(ranges) == last-first, path
    lo,hi = ranges[0]['a'],ranges[-1]['b']
    gpu = [e for e in events if e['cat'] != 'capture_range' and lo <= e['a'] < hi]
    nccl = defaultdict(list)
    for e in gpu:
        if 'nccl' in e['name'].lower():
            nccl[str(e['args']['stream'])].append(e)
    assert set(nccl) == set(reference['groups']), (path, list(nccl), list(reference['groups']))
    for stream, ops in nccl.items():
        ops.sort(key=lambda e:e['a'])
        original = reference['groups'][stream]
        assert original == original[:len(original)//4]*4
        expected = original[:len(original)//4]*(last-first)
        assert [e['name'].split('(')[0] for e in ops] == [n for n,k in expected], (path,stream,'NCCL mismatch')
        for e,(_,kind) in zip(ops,expected):
            e['label'] = kind
    rows=[]
    for window in ranges:
        start,end=window['a'],window['b']
        intervals=defaultdict(list)
        gemms=[]
        comm_ops=[]
        gpu_events=[]
        for e in gpu:
            if e['a'] >= end or e['b'] <= start:continue
            span=(max(start,e['a']),min(end,e['b']))
            kind=e.get('label', 'graph_span' if e['cat']=='graph_span' else 'memory' if e['cat'] != 'kernel' else
                       'hybrid_ep' if 'hybrid_ep::' in e['name'] or 'ag_nvl_kernel' in e['name'] else 'other_kernels')
            intervals[kind].append(span)
            gpu_events.append((span,kind,e['name'],e['args']['stream']))
            if 'label' in e:comm_ops.append(dict(a=span[0],b=span[1],category=kind,stream=e['args']['stream'],name=e['name']))
            if e['cat']=='kernel' and e['args']['stream'] in reference['experts']:
                assert 'nvjet' in e['name'], (path,e['name'])
                gemms.append(span)
        assert len(gemms)==15*3*experts*6,(path,window['name'],len(gemms),experts)
        all_gpu=[s for vs in intervals.values() for s in vs]
        known_gpu=[s for k,vs in intervals.items() if k!='graph_span' for s in vs]
        categories={}
        for kind,spans in intervals.items():
            others=[s for k,vs in intervals.items() if k!=kind for s in vs]
            residency=union_duration(spans)
            known_others=[s for k,vs in intervals.items() if k not in (kind,'graph_span') for s in vs]
            categories[kind]={'union_ms':residency,'exposed_ms':residency-intersection_duration(spans,others),
                              'exposed_upper_ms':residency-intersection_duration(spans,known_others)}
        scopes=defaultdict(list)
        for e in events:
            if e['cat']=='capture_range' and e['name'].startswith('nep.') and e['a']<end and e['b']>start:
                scopes[e['name']].append((max(e['a'],start),min(e['b'],end)))
        scopes['all_staging']=[s for vs in scopes.values() for s in vs]
        staging={k:{'cpu_ms':union_duration(v),'gpu_idle_ms':union_duration(v)-intersection_duration(v,all_gpu),
                    'gpu_idle_upper_ms':union_duration(v)-intersection_duration(v,known_gpu),'calls':len(v)} for k,v in scopes.items()}
        rows.append(dict(iteration=int(window['name'].split('_')[1]),range_ms=end-start,
                         window_start_ms=start,window_end_ms=end,
                         gpu_union_ms=union_duration(all_gpu),gpu_idle_ms=end-start-union_duration(all_gpu),
                         gpu_idle_upper_ms=end-start-union_duration(known_gpu),
                         expert_gemm_ms=union_duration(gemms),expert_kernels=len(gemms),
                         first_expert_ms=min(a for a,b in gemms)-start,
                         last_expert_ms=max(b for a,b in gemms)-start,
                         tail_ms=end-max(b for a,b in gemms),categories=categories,staging=staging,
                         comm_ops=comm_ops,staging_intervals=dict(scopes)))
    return dict(path=str(path),memory_activity=data.get('memory_activity',True),
                origin_ns=data.get('origin_ns'),zero_duration_graphs=data.get('zero_duration_graphs',0),
                coarse_graph_spans=any(e['cat']=='graph_span' for e in events),iterations=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--start',type=int,default=10)
    parser.add_argument('--end',type=int,default=14)
    parser.add_argument('--fresh-controls',action='store_true')
    args=parser.parse_args()
    root=ROOT/'slurm_runs/soft_clique_baseline'/args.job
    output=root/'cupti_mbs_analysis'
    output.mkdir(exist_ok=True)
    result={}
    for mbs in [1,2]:
        result[mbs]={'timing':{},'ranks':{}}
        for case,size in [('healthy',16),('nep',14)]:
            run=root/f'mbs{mbs}/cupti/ep8/scaling_ladder__a770m_4b_moe_215b'/case/args.job
            vals={int(i):float(v) for i,v in re.findall(r'iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)',(run/'driver.log').read_text())}
            result[mbs]['timing'][case]={}
            for label,ix in [('pre',range(max(6,args.start-11),args.start-1)),('recorded',range(args.start,args.end)),('post',range(args.end+1,args.end+11))]:
                values=[vals[i] for i in ix]
                result[mbs]['timing'][case][label]=dict(ms=values,mean_ms=statistics.mean(values),std_ms=statistics.stdev(values))
            if args.fresh_controls:
                control=root/f'control_mbs{mbs}/cupti/ep8/scaling_ladder__a770m_4b_moe_215b'/case/args.job/'driver.log'
                content=control.read_text()
                assert 'mode=control_no_cupti' in content and 'GPU_TRACE_CUPTI_VERSION' not in content, control
                cv={int(i):float(v) for i,v in re.findall(r'iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)',content)}
                values=[cv[i] for i in range(args.start,args.end)]
                result[mbs]['timing'][case]['fresh_control']=dict(ms=values,mean_ms=statistics.mean(values),std_ms=statistics.stdev(values))
            for rank in range(size):
                cache=output/f'mbs{mbs}_{case}_{rank}.json'
                if cache.exists():row=json.loads(cache.read_text())
                else:
                    experts=16 if case=='healthy' or rank<8 else 22 if rank<10 else 21
                    row=analyze(run/f'cupti_profile/rank-{rank}.json',labels(case,rank),experts,args.start,args.end)
                    cache.write_text(json.dumps(row,indent=2)+'\n')
                result[mbs]['ranks'][f'{case}_{rank}']=row
                print(mbs,case,rank,'validated',flush=True)
        windows=['pre','recorded','post']+(['fresh_control'] if args.fresh_controls else [])
        result[mbs]['parity']={w:result[mbs]['timing']['healthy'][w]['mean_ms']/result[mbs]['timing']['nep'][w]['mean_ms'] for w in windows}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m:{'timing':v['timing'],'parity':v['parity']} for m,v in result.items()},indent=2))


if __name__=='__main__':main()
