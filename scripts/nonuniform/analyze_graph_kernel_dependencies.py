"""Match graph launches to kernels and quantify internal gaps without double counting."""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def union(intervals):
    total, last = 0, -1
    for a,b in sorted(intervals):
        total += max(0,b-max(a,last))
        last = max(last,b)
    return total


def partition(kernels):
    edges = []
    for k in kernels:
        name = k['name']
        kind = ('nccl' if 'nccl' in name else 'hybrid' if name in
                ('device_sync_kernel','ag_nvl_kernel','dispatch_kernel','combine_kernel') else 'compute')
        edges.extend([(k['start'],1,kind),(k['end'],-1,kind)])
    counts = defaultdict(int)
    result = defaultdict(float)
    prev = min(k['start'] for k in kernels)
    for t,d,kind in sorted(edges):
        label = '+'.join(sorted(k for k,v in counts.items() if v)) or 'gap'
        result[label] += (t-prev)/1e6
        counts[kind] += d
        prev = t
    return dict(result)


def analyze(path):
    con = sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    con.row_factory = sqlite3.Row
    strings = dict(con.execute('select id,value from StringIds'))
    nvtx = [dict(x) for x in con.execute('select * from NVTX_EVENTS where end is not null order by start')]
    for x in nvtx:
        x['name'] = x['text'] or strings.get(x['textId'],'')
    runtime = [dict(x) for x in con.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME order by start')]
    kernels = [dict(x) for x in con.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
    by_corr = defaultdict(list)
    for k in kernels:
        k['name'] = strings[k['shortName']]
        by_corr[k['correlationId']].append(k)
    rows = []
    for step in range(25,28):
        it, = [x for x in nvtx if x['name']==f'iteration_{step}']
        micros = {name:[x for x in nvtx if x['name']==f'microbatch.{name}' and it['start']<=x['start']<it['end']]
                  for name in ['forward','backward']}
        for phase,p in [('first_forward',micros['forward'][0]),('final_backward',micros['backward'][-1])]:
            graphs = [r for r in runtime if 'cudaGraphLaunch' in strings[r['nameId']] and p['start']<=r['start']<p['end']]
            assert len(graphs)==32
            for ordinal,r in enumerate(graphs):
                ks = by_corr[r['correlationId']]
                assert ks
                start,end = min(k['start'] for k in ks),max(k['end'] for k in ks)
                gaps = []; cursor = start
                for k in ks:
                    if k['start']-cursor>100000:
                        other = [o for o in kernels if o['correlationId']!=r['correlationId'] and o['end']>cursor and o['start']<k['start']]
                        last_comm = [o for o in kernels if o['correlationId']!=r['correlationId'] and 'nccl' in o['name'] and o['start']<=k['start']]
                        last = max(last_comm,key=lambda o:o['start'],default=None)
                        gaps.append(dict(start_ns=cursor,end_ns=k['start'],ms=(k['start']-cursor)/1e6,
                                         before=k['name'], other_kernel_names=sorted({o['name'] for o in other}),
                                         last_comm_start_ns=last['start'] if last else None,
                                         last_comm_name=last['name'] if last else None,
                                         last_comm_to_next_kernel_us=(k['start']-last['start'])/1e3 if last else None))
                    cursor = max(cursor,k['end'])
                names = defaultdict(float)
                for k in ks:names[k['name']] += (k['end']-k['start'])/1e6
                rows.append(dict(iteration=step,phase=phase,ordinal=ordinal,
                                 start_ns=start,end_ns=end,span_ms=(end-start)/1e6,
                                 busy_union_ms=union([(k['start'],k['end']) for k in ks])/1e6,
                                 gaps=gaps,kernel_sums_ms=dict(names),host_launch_ns=r['start'],
                                 kernel_partition_ms=partition(ks)))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite',type=Path)
    args = parser.parse_args()
    result = analyze(args.sqlite)
    out = args.sqlite.with_suffix('.graph_analysis.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    for phase in ['first_forward','final_backward']:
        rows = [r for r in result if r['phase']==phase and r['iteration']==25]
        print(phase, 'span',sum(r['span_ms'] for r in rows),'gaps',sum(r['span_ms']-r['busy_union_ms'] for r in rows))
        for r in rows:
            big = [g for g in r['gaps'] if g['ms']>.5]
            if big:print(r['ordinal'],big)
    print(out)
