"""Measure captured RMSNorm launch gaps, retaining exact graph-node matches."""

import argparse
import bisect
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(run, rank, steps):
    trace = json.loads((run / 'cupti_profile' / f'rank-{rank}.json').read_text())
    con = sqlite3.connect(f'file:{run}/nsys_profile/rank-{rank}.sqlite?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    strings = dict(con.execute('select id,value from StringIds'))
    nvtx = [dict(r) for r in con.execute('select * from NVTX_EVENTS where end is not null order by start')]
    for n in nvtx:
        n['name'] = n['text'] or strings.get(n['textId'], '')
    cpu = [e for e in trace['traceEvents'] if e['cat'] == 'cpu_operation_range' and e['name'] == 'microbatch.forward']
    nv = [n for n in nvtx if n['name'] == 'microbatch.forward']
    assert len(cpu) == len(nv)
    lo = max(n['start']/1e6-e['ts']/1000 for n, e in zip(nv, cpu))
    hi = min(n['end']/1e6-(e['ts']+e['dur'])/1000 for n, e in zip(nv, cpu))
    assert hi >= lo
    offset = (lo+hi)/2
    kernels = [dict(r) for r in con.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
    memsets = defaultdict(list)
    for r in con.execute('select * from CUPTI_ACTIVITY_KIND_MEMSET order by start'):
        memsets[r['graphNodeId']].append(dict(r))
    runtime = {r['correlationId']: dict(r) for r in con.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
    collectives = [n for n in nvtx if n['name'].startswith('collective.')]
    starts = [n['start'] for n in collectives]
    comm = []
    for k in kernels:
        name = strings[k['shortName']]
        if 'nccl' not in name.lower():
            continue
        label = 'nccl.unclassified'
        api = runtime.get(k['correlationId'])
        if api:
            i = bisect.bisect_right(starts, api['start'])
            for n in reversed(collectives[max(0, i-8):i]):
                if n['start'] <= api['start'] <= n['end']:
                    label = n['name']
                    break
        comm.append((k, label))
    result = []
    for step in steps:
        back = [e for e in trace['traceEvents'] if e['cat'] == 'gpu_operation_range' and e['name'] == 'microbatch.backward' and e['args']['iteration'] == step]
        if not back:
            continue
        b = max(back, key=lambda e: e['ts'])
        begin, end = b['ts']/1000+offset, (b['ts']+b['dur'])/1000+offset
        for k in kernels:
            if strings[k['shortName']] != 'rmsnorm_bwd_general_kernel' or k['graphNodeId'] is None:
                continue
            if not begin <= k['start']/1e6 < end:
                continue
            predecessors = [m for m in memsets[k['graphNodeId']-1] if begin <= m['end']/1e6 <= k['start']/1e6]
            if not predecessors:
                continue
            m = max(predecessors, key=lambda m: m['end'])
            concurrent = [dict(name=label, start_ms=(c['start']-m['end'])/1e6, end_ms=(c['end']-m['end'])/1e6)
                          for c, label in comm if c['start'] < k['start'] and c['end'] > m['end']]
            result.append(dict(iteration=step, rank=rank, graph_node=k['graphNodeId'],
                               memset_node=m['graphNodeId'], grid=k['gridX'],
                               gap_ms=(k['start']-m['end'])/1e6,
                               kernel_ms=(k['end']-k['start'])/1e6,
                               memset_end_sql_ms=m['end']/1e6,
                               norm_start_sql_ms=k['start']/1e6,
                               concurrent_nccl=concurrent))
    con.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job')
    p.add_argument('--ranks', type=int, nargs='+', default=[0, 2, 8, 11, 16])
    args = p.parse_args()
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline') / args.job
    result = {}
    for run in sorted(root.glob('cleanup_ab_*/nsys/ep16/a8b_120b_latentmoe_1t/*/' + args.job)):
        if not (run / 'baseline_uniform.json').is_file():
            continue
        label = run.parents[4].name
        rows = [row for rank in args.ranks for row in analyze(run, rank, range(20, 28))]
        result[label] = rows
        totals = defaultdict(float)
        for row in rows:
            totals[(row['rank'], row['iteration'])] += row['gap_ms']
        print(label, 'matched_nodes', len(rows), 'mean gap sum per rank-step', statistics.mean(totals.values()),
              'largest gap', max(rows, key=lambda r: r['gap_ms']))
    (root / 'norm_admission_gaps.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
