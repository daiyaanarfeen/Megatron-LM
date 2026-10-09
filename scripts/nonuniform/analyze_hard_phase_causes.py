"""Existing-profile phase and kernel accounting; no training changes."""

import argparse
import bisect
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_a8b_current_overheads import category


def span(e):
    return e['ts'] / 1000, (e['ts'] + e['dur']) / 1000


def partition(a, b, events):
    edges = [(a, 0, ''), (b, 0, '')]
    for x, y, name in events:
        x, y = max(x, a), min(y, b)
        if y > x:
            edges.extend(((x, 1, name), (y, -1, name)))
    counts, result, prev = Counter(), defaultdict(float), a
    for t, delta, name in sorted(edges):
        label = '+'.join(sorted(k for k, v in counts.items() if v)) or 'idle'
        result[label] += t - prev
        counts[name] += delta
        prev = t
    assert abs(sum(result.values()) - (b-a)) < 1e-5
    return dict(result)


def run(path, rank, kernels=False, steps=range(25, 35)):
    trace = json.loads((path / 'cupti_profile' / f'rank-{rank}.json').read_text())
    by_step = defaultdict(list)
    its = {}
    for e in trace['traceEvents']:
        if e['cat'] == 'iteration':
            its[int(e['name'].split('_')[1])] = e
        else:
            by_step[e['args']['iteration']].append(e)
    sql_data = None
    if kernels:
        con = sqlite3.connect(f'file:{path}/nsys_profile/rank-{rank}.sqlite?mode=ro', uri=True)
        con.row_factory = sqlite3.Row
        strings = dict(con.execute('select id,value from StringIds'))
        nvtx = [dict(r) for r in con.execute('select * from NVTX_EVENTS where end is not null order by start')]
        for n in nvtx:
            n['name'] = n['text'] or strings.get(n['textId'], '')
        cpu = sorted((e for e in trace['traceEvents'] if e['cat']=='cpu_operation_range' and e['name']=='microbatch.forward'), key=lambda e:e['ts'])
        nv = [n for n in nvtx if n['name']=='microbatch.forward']
        assert len(cpu)==len(nv)
        lo = max(n['start']/1e6-e['ts']/1000 for n,e in zip(nv,cpu))
        hi = min(n['end']/1e6-(e['ts']+e['dur'])/1000 for n,e in zip(nv,cpu))
        assert hi >= lo, (lo,hi)
        offset = (lo+hi)/2
        runtime = {r['correlationId']:dict(r) for r in con.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
        collectives = [n for n in nvtx if n['name'].startswith('collective.')]
        commstarts = [n['start'] for n in collectives]
        def label(k):
            name = strings[k['shortName']]
            if 'nccl' in name.lower():
                rt = runtime.get(k['correlationId'])
                if rt:
                    i = bisect.bisect_right(commstarts, rt['start'])-1
                    for n in reversed(collectives[max(0,i-4):i+1]):
                        if n['start'] <= rt['start'] <= n['end']:
                            return n['name']
                return 'nccl.graph_or_other'
            if name in ('device_sync_kernel','ag_nvl_kernel','dispatch_kernel','combine_kernel'):
                return 'hybrid'
            return 'compute'
        ks = []
        for r in con.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
            k = dict(r)
            ks.append(dict(start=k['start']/1e6-offset, end=k['end']/1e6-offset,
                           label=label(k), name=strings[k['shortName']], corr=k['correlationId'],
                           launch_ms=runtime.get(k['correlationId'],{}).get('start',0)/1e6-offset))
        for table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
            for k in con.execute(f'select start,end from {table}'):
                ks.append(dict(start=k['start']/1e6-offset,end=k['end']/1e6-offset,label='memory',name=table,corr=-1,launch_ms=0))
        ks.sort(key=lambda k:k['start'])
        sql_data = dict(offset_ms=offset, alignment_interval_ms=[lo,hi], kernels=ks)
        con.close()
    rows = []
    for step in steps:
        events = by_step[step]
        gpu = [e for e in events if e['cat']=='gpu_operation_range']
        cpu = [e for e in events if e['cat']=='cpu_operation_range']
        assert [e['name'] for e in gpu]==[e['name'] for e in cpu]
        pairs = list(zip(gpu,cpu))
        named = lambda n: sorted([(g,c) for g,c in pairs if g['name']==n], key=lambda p:p[0]['ts'])
        fs,bs = named('microbatch.forward'),named('microbatch.backward')
        op=named('optimizer.step')[0]
        phases=[('firstF',*fs[0]),('otherF',*fs[1]),('otherB',*bs[0]),('finalB',*bs[1]),('optimizer',*op)]
        ranges={n:span(g) for n,g,c in phases}
        ranges['startup']=(span(its[step])[0],span(fs[0][0])[0])
        ranges['B_to_opt']=(span(bs[-1][0])[1],span(op[0])[0])
        ranges['after_opt']=(span(op[0])[1],span(its[step])[1])
        detail={}
        for n,g,c in phases:
            gs=[x for x,y in pairs if x['name'].startswith('graph.') and c['ts']<=y['ts']<c['ts']+c['dur']]
            sums=defaultdict(float)
            for x in gs:
                sums[x['name']]+=x['dur']/1000
            detail[n]=dict(graph_sums=dict(sums), outside_graph_ms=g['dur']/1000-sum(sums.values()),
                           graphs=[dict(name=x['name'],start=span(x)[0],end=span(x)[1]) for x in gs])
        scopes={}
        for g,c in pairs:
            if g['name'].startswith(('optimizer.','ddp.','nep.')):
                v=scopes.setdefault(g['name'],dict(gpu_ms=0,cpu_ms=0,thread_ms=0,count=0))
                v['gpu_ms']+=g['dur']/1000;v['cpu_ms']+=c['dur']/1000
                v['thread_ms']+=c['args']['thread_cpu_us']/1000;v['count']+=1
        comm=defaultdict(lambda:dict(elapsed_ms=0,count=0,input_bytes=0,output_bytes=0))
        for e in gpu:
            if 'collective' not in e['args']:continue
            v=comm[category(e)];v['elapsed_ms']+=e['dur']/1000;v['count']+=1
            for operation in e['args']['collective']['operations']:
                for k in ('input_bytes','output_bytes'):v[k]+=operation[k]
        row=dict(iteration=step,ranges=ranges,details=detail,scopes=scopes,comm=dict(comm))
        if sql_data:
            a,b=span(its[step]);ks=[k for k in sql_data['kernels'] if k['end']>a and k['start']<b]
            row['kernel_partitions']={n:partition(x,y,[(k['start'],k['end'],k['label']) for k in ks]) for n,(x,y) in ranges.items()}
            row['kernel_sums']={}
            for n,(x,y) in ranges.items():
                sums=defaultdict(float)
                for k in ks:
                    dt=max(0,min(y,k['end'])-max(x,k['start']))
                    if dt:sums[k['name']]+=dt
                row['kernel_sums'][n]=dict(sums)
            row['graph_kernel_parts']={}
            for n,v in detail.items():
                parts=[]
                for g in v['graphs']:
                    parts.append(dict(name=g['name'],start=g['start'],end=g['end'],
                        parts=partition(g['start'],g['end'],[(k['start'],k['end'],k['label']) for k in ks])))
                row['graph_kernel_parts'][n]=parts
        rows.append(row)
    return dict(origin_ns=trace['origin_ns'],rows=rows,
                alignment={k:v for k,v in (sql_data or {}).items() if k!='kernels'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--kernels',action='store_true');p.add_argument('--ranks',default='0,4,8,16,20,24');a=p.parse_args()
    root=Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3121678')
    data=json.loads((root/'hard_graph_phase_analysis.json').read_text())
    out=root/'phase_causes';out.mkdir(exist_ok=True)
    for case,c in data['cases'].items():
        for rank in map(int,a.ranks.split(',')):
            result=run(Path(c['run']),rank,a.kernels)
            target=out/f'{case}-rank-{rank}{"-kernels" if a.kernels else ""}.json'
            target.write_text(json.dumps(result)+'\n')
            print(target,flush=True)


if __name__=='__main__':main()
