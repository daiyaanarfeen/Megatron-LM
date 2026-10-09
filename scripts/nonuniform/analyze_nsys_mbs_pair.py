"""Same-iteration MBS comparison; coarse graph spans are never called pure compute."""

import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_exact_uniform_nep_slowdown import intersection_duration, union_duration
from analyze_gpu_only_mbs_pair import template, trace_path

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen')


def labels(case, rank):
    cache = ROOT / 'analysis/nsys_reference_labels' / f'{case}_{rank}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    experts, groups = template(trace_path('2997621', 'cpu_cuda', case, rank))
    result = {'experts':sorted(experts), 'groups':{str(s):[(n.split('(')[0],k) for n,k in v] for s,v in groups.items()}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result)+'\n')
    return result


def analyze(path, reference, experts):
    c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    names = dict(c.execute('SELECT id,value FROM StringIds'))
    diagnostics = [r[0] for r in c.execute('SELECT text FROM DIAGNOSTIC_EVENT')]
    assert any('Hardware tracing used' in d or 'software instrumented trace was collected' in d for d in diagnostics)
    ranges = [(a,b,text or names.get(tid,'')) for a,b,text,tid in c.execute('SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL')]
    ranges = sorted((a,b,int(text.split('_')[1])) for a,b,text in ranges if text in [f'iteration_{i}' for i in range(10,14)])
    assert [i for _,_,i in ranges] == [10,11,12,13], path
    lo,hi=ranges[0][0],ranges[-1][1]
    kernels = [dict(a=a/1e6,b=b/1e6,stream=s,name=names[n]) for a,b,s,n in c.execute('SELECT start,end,streamId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND start<? ORDER BY start',(lo,hi))]
    nccl=defaultdict(list)
    for e in kernels:
        if 'nccl' in e['name'].lower():nccl[str(e['stream'])].append(e)
    assert set(nccl)==set(reference['groups']), (path,set(nccl),set(reference['groups']))
    for s,events in nccl.items():
        expected=reference['groups'][s]
        assert [e['name'].split('(')[0] for e in events]==[n for n,k in expected], (path,s,'NCCL sequence mismatch')
        for e,(_,k) in zip(events,expected):e['label']=k
    tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    extra=[]
    for table,category in [('CUPTI_ACTIVITY_KIND_GRAPH_TRACE','graph_span'),('CUPTI_ACTIVITY_KIND_MEMCPY','memory'),('CUPTI_ACTIVITY_KIND_MEMSET','memory')]:
        if table in tables:extra.extend((a/1e6,b/1e6,category) for a,b in c.execute(f'SELECT start,end FROM {table} WHERE end>? AND start<?',(lo,hi)))
    rows=[]
    for start,end,iteration in ranges:
        start,end=start/1e6,end/1e6
        intervals=defaultdict(list)
        gemms=[]
        comm_ops=[]
        for e in kernels:
            if e['a']>=end or e['b']<=start:continue
            span=(max(start,e['a']),min(end,e['b']))
            category=e.get('label','hybrid_ep' if 'hybrid_ep::' in e['name'] or 'ag_nvl_kernel' in e['name'] else 'other_kernels')
            intervals[category].append(span)
            if 'label' in e:comm_ops.append(dict(e,category=category))
            if e['stream'] in reference['experts']:
                assert 'nvjet' in e['name'],(path,e['name'])
                gemms.append(span)
        assert len(gemms)==15*3*experts*6,(path,iteration,len(gemms),experts)
        for a,b,category in extra:
            if b>start and a<end:intervals[category].append((max(a,start),min(b,end)))
        categories={}
        for category,spans in intervals.items():
            others=[v for k,vs in intervals.items() if k!=category for v in vs]
            known_others=[v for k,vs in intervals.items() if k not in (category,'graph_span') for v in vs]
            residency=union_duration(spans)
            categories[category]={'union_ms':residency,'exposed_lower_ms':residency-intersection_duration(spans,others),'exposed_upper_ms':residency-intersection_duration(spans,known_others)}
        busy_upper=union_duration([s for spans in intervals.values() for s in spans])
        last_expert=max(b for a,b in gemms)
        rows.append({'iteration':iteration,'nvtx_ms':end-start,'activity_span_union_ms':busy_upper,'outside_activity_spans_ms':end-start-busy_upper,'expert_gemm_ms':union_duration(gemms),'expert_kernel_count':len(gemms),'last_expert_from_start_ms':last_expert-start,'after_last_expert_ms':end-last_expert,'categories':categories,'comm_ops':comm_ops})
    c.close()
    return {'path':str(path),'diagnostics':diagnostics,'iterations':rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job')
    args=parser.parse_args()
    root=ROOT/'slurm_runs/soft_clique_baseline'/args.job
    output=root/'nsys_mbs_analysis'
    output.mkdir(exist_ok=True)
    result={}
    for mbs in [1,2]:
        result[mbs]={'timing':{},'ranks':{}}
        for case,size in [('healthy',16),('nep',14)]:
            run=root/f'mbs{mbs}/nsys/ep8/scaling_ladder__a770m_4b_moe_215b'/case/args.job
            text=(run/'driver.log').read_text()
            vals={int(i):float(v) for i,v in re.findall(r'iteration\s+(\d+)/\s*24.*?elapsed time per iteration \(ms\):\s*([\d.]+)',text)}
            result[mbs]['timing'][case]={}
            for label,ix in [('pre',range(6,9)),('recorded',range(10,14)),('post',range(15,25))]:
                values=[vals[i] for i in ix]
                result[mbs]['timing'][case][label]={'ms':values,'mean_ms':statistics.mean(values),'std_ms':statistics.stdev(values)}
            for rank in range(size):
                cache=output/f'mbs{mbs}_{case}_{rank}.json'
                if cache.exists():row=json.loads(cache.read_text())
                else:
                    experts=16 if case=='healthy' or rank<8 else 22 if rank<10 else 21
                    row=analyze(run/f'nsys_profile/rank-{rank}.sqlite',labels(case,rank),experts)
                    cache.write_text(json.dumps(row,indent=2)+'\n')
                result[mbs]['ranks'][f'{case}_{rank}']=row
                print(mbs,case,rank,'validated',flush=True)
        result[mbs]['parity']={w:result[mbs]['timing']['healthy'][w]['mean_ms']/result[mbs]['timing']['nep'][w]['mean_ms'] for w in ['pre','recorded','post']}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m:{'timing':v['timing'],'parity':v['parity']} for m,v in result.items()},indent=2))


if __name__=='__main__':main()
