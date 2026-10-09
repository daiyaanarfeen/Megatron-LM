"""Offline, non-additive residency and additive phase accounting for job3122515."""

import bisect
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from analyze_hard_phase_causes import partition


ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3122515')
OUT = ROOT / 'first_forward_attribution'


def read_rank(run, rank, per_iteration_alignment=False, steps=range(20, 28), phase='first_forward'):
    assert phase in ('first_forward', 'final_backward_tail')
    data = json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())
    es = data['traceEvents']
    con = sqlite3.connect(f'file:{run}/nsys_profile/rank-{rank}.sqlite?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    strings = dict(con.execute('select id,value from StringIds'))
    nv = [dict(r) for r in con.execute('select * from NVTX_EVENTS where end is not null order by start')]
    for n in nv:
        n['name'] = n['text'] or strings.get(n['textId'], '')
    cf = [e for e in es if e['cat'] == 'cpu_operation_range' and e['name'] == 'microbatch.forward']
    nf = [e for e in nv if e['name'] == 'microbatch.forward']
    assert len(cf) == len(nf)
    lo = max(n['start']/1e6-e['ts']/1000 for n, e in zip(nf, cf))
    hi = min(n['end']/1e6-(e['ts']+e['dur'])/1000 for n, e in zip(nf, cf))
    if not per_iteration_alignment:
        assert hi >= lo
    offset = (hi+lo)/2
    graph_launch_ranges = {}
    if per_iteration_alignment:
        cg = [e for e in es if e['cat'] == 'cpu_operation_range' and e['name'].startswith('graph.')]
        ng = [n for n in nv if n['name'].startswith('graph.')]
        assert ['graph.' + e['name'].rsplit('.', 1)[-1] for e in cg] == [n['name'] for n in ng]
        graph_launch_ranges = {id(e): (n['start']/1e6, n['end']/1e6) for e, n in zip(cg, ng)}
    calls = [n for n in nv if n['name'].startswith('collective.')]
    starts = [n['start'] for n in calls]
    staging = [n for n in nv if n['name'] in ('nep.redistribute_params', 'nep.stage_grads')]
    st = [n['start'] for n in staging]
    rt = {r['correlationId']: dict(r) for r in con.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
    kernels = []
    for row in con.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
        k = dict(row)
        name = strings[k['shortName']]
        api = rt.get(k['correlationId'], {})
        label = 'model_compute'
        graph = bool(k['graphNodeId'])
        if 'nccl' in name.lower():
            label = 'model_TP' if graph else 'other_comm'
            i = bisect.bisect_right(starts, api.get('start', -1))
            for n in reversed(calls[max(0, i-8):i]):
                if n['start'] <= api['start'] <= n['end']:
                    desc = n['name'].lower()
                    is_tp = desc.endswith(':tp') or 'tensor_model_parallel' in desc
                    if desc.startswith('collective.all_gather'):
                        label = 'expert_AG' if ('ep_dp' in desc or 'expert_data' in desc) else 'model_TP' if is_tp else 'dense_AG'
                    elif desc.startswith('collective.reduce_scatter'):
                        label = 'expert_RS' if ('ep_dp' in desc or 'expert_data' in desc) else 'model_TP' if is_tp else 'dense_RS'
                    elif 'nep_owner_gather' in desc:
                        label = 'NEP_gather'
                    elif 'nep_owner_transfer' in desc:
                        label = 'NEP_broadcast'
                    else:
                        label = 'model_TP' if is_tp else 'other_comm'
                    break
        elif name in ('device_sync_kernel', 'ag_nvl_kernel', 'dispatch_kernel', 'combine_kernel'):
            label = 'model_EP'
        elif not graph:
            i = bisect.bisect_right(st, api.get('start', -1)) - 1
            if i >= 0 and staging[i]['start'] <= api['start'] <= staging[i]['end']:
                label = 'NEP_staging'
        kernels.append(dict(start=k['start']/1e6-offset, end=k['end']/1e6-offset,
                            name=name, label=label, graph=graph, stream=k['streamId'],
                            launch=api.get('start', 0)/1e6-offset,
                            launch_end=api.get('end', 0)/1e6-offset))
    for table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
        for r in con.execute(f'select * from {table}'):
            api = rt.get(r['correlationId'], {})
            memory_label = 'memory'
            i = bisect.bisect_right(st, api.get('start', -1)) - 1
            if not r['graphNodeId'] and i >= 0 and staging[i]['start'] <= api['start'] <= staging[i]['end']:
                memory_label = 'NEP_staging'
            kernels.append(dict(start=r['start']/1e6-offset, end=r['end']/1e6-offset,
                                name=table, label=memory_label, graph=bool(r['graphNodeId']),
                                stream=r['streamId'], launch=api.get('start', 0)/1e6-offset,
                                launch_end=api.get('end', 0)/1e6-offset))
    con.close()
    result = []
    alignments = {}
    for step in steps:
        gpu = [e for e in es if e['cat'] == 'gpu_operation_range' and e['args']['iteration'] == step]
        cpu = [e for e in es if e['cat'] == 'cpu_operation_range' and e['args']['iteration'] == step]
        assert [e['name'] for e in gpu] == [e['name'] for e in cpu]
        pairs = list(zip(gpu, cpu))
        fg, fc = next((g, c) for g, c in pairs if g['name'] == 'microbatch.forward')
        if phase == 'final_backward_tail':
            fg, fc = [(g, c) for g, c in pairs if g['name'] == 'microbatch.backward'][-1]
        a, b = fg['ts']/1000, (fg['ts']+fg['dur'])/1000
        if phase == 'final_backward_tail':
            b = next(g['ts']/1000 for g, _ in pairs if g['name'] == 'optimizer.step')
        delta = 0.0
        if per_iteration_alignment:
            matched = [(n, e) for n, e in zip(nf, cf) if e['args']['iteration'] == step]
            lower = max(n['start']/1e6-e['ts']/1000 for n, e in matched)
            upper = min(n['end']/1e6-(e['ts']+e['dur'])/1000 for n, e in matched)
            assert upper >= lower, (rank, step, lower, upper)
            alignments[step] = [lower, upper]
            delta = offset - (lower + upper) / 2
        ks = [dict(k, **{key: None if k[key] is None else k[key] + delta
                         for key in ('start', 'end', 'launch', 'launch_end')})
              for k in kernels if k['start'] + delta < b and k['end'] + delta > a]
        gs = [(g, c) for g, c in pairs if g['name'].startswith('graph.') and fc['ts'] <= c['ts'] < fc['ts']+fc['dur']]
        gs.sort(key=lambda p: p[0]['ts'])
        graphs, gaps, previous = [], [], a
        for i, (g, c) in enumerate(gs):
            x, y = g['ts']/1000, (g['ts']+g['dur'])/1000
            assert x >= previous-0.001
            gaps.append((previous, x))
            native = [k for k in ks if k['graph'] and k['start'] < y and k['end'] > x]
            if per_iteration_alignment:
                launch_start, launch_end = graph_launch_ranges[id(c)]
                native = [k for k in ks if k['graph']
                          and ((k['launch'] is None and k['start'] < y and k['end'] > x)
                               or (k['launch'] is not None
                                   and launch_start - offset + delta <= k['launch'] <= launch_end - offset + delta))]
            graphs.append(dict(index=i, name=g['name'], start=x, end=y,
                               parts=partition(x, y, [(k['start'], k['end'], k['label']) for k in native]),
                               model_ep=[k for k in native if k['label']=='model_EP']))
            previous = y
        gaps.append((previous, b))
        gap_parts = defaultdict(float)
        for x, y in gaps:
            if y <= x:
                continue
            for label, dt in partition(x, y, [(k['start'], k['end'], k['label']) for k in ks]).items():
                gap_parts[label] += dt
        assert abs(sum(g['end']-g['start'] for g in graphs)+sum(gap_parts.values())-(b-a)) < 1e-5
        ops = []
        for g, c in pairs:
            if g['name'] == 'nep.redistribute_params' or g['name']=='collective.all_gather':
                if not fc['ts'] <= c['ts'] < fc['ts']+fc['dur']:
                    continue
                ops.append(dict(name=g['name'], start=g['ts']/1000, end=(g['ts']+g['dur'])/1000,
                                cpu_start=c['ts']/1000, cpu_end=(c['ts']+c['dur'])/1000,
                                meta=g['args'].get('collective')))
        result.append(dict(iteration=step, start=a, end=b, graphs=graphs, gaps=gaps,
                           gap_parts=dict(gap_parts), ops=ops, kernels=ks))
    return dict(schema_version=4 if per_iteration_alignment else 2,
                memory_staging_labeled=True,
                rank=rank, origin_ns=data['origin_ns'], clock_error_ns=data['clock_error_bound_ns'],
                alignment_interval_ms=[lo, hi], alignment_by_iteration_ms=alignments, rows=result)


def main():
    OUT.mkdir(exist_ok=True)
    cases = json.loads((ROOT/'cleanup_phase_analysis.json').read_text())
    for label, size in [('healthy_margin', 32), ('cleanup_margin', 28)]:
        run = Path(cases['cleanup_ab_'+label]['run'])
        for rank in range(size):
            target = OUT/f'delivery-{label}-rank-{rank}.json'
            if not target.exists() or json.loads(target.read_text()).get('schema_version') != 2:
                target.write_text(json.dumps(read_rank(run, rank))+'\n')
            print(label, rank, flush=True)


if __name__ == '__main__':
    main()
