"""All-rank operation-range audit for the original a8b MBS experiment.

Expert anchors partition elapsed time, not pure compute. Recompute is backward.
Collective ranges include readiness/queue waits, not just NCCL kernel service.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_exact_uniform_nep_slowdown import intersection_duration, union_duration


def category(e):
    meta = e.get('args', {}).get('collective')
    if meta is None:
        return e['name']
    desc = meta['group_desc'].lower()
    desc = {'expert_data_parallel_group': 'expert', 'ep_dp': 'expert',
            'tensor_model_parallel_group': 'tp', 'pipeline_model_parallel_group': 'pp',
            'data_parallel_group_with_cp': 'dense', 'dp_cp': 'dense',
            'nep_owner_transfer': 'nep_param', 'nep_owner_gather': 'nep_grad'}.get(desc, desc)
    return desc + '.' + e['name'].removeprefix('collective.')


def span(e):
    return e['ts'] / 1000, (e['ts'] + e['dur']) / 1000


def stats(values):
    return dict(mean=statistics.mean(values), sd=statistics.pstdev(values),
                min=min(values), max=max(values), samples=values)


def blocks(events, moe_layers=30):
    forward = sorted([e for e in events if e['name'] == 'expert.forward'], key=lambda e: e['ts'])
    backward = sorted([e for e in events if e['name'] == 'expert.backward'], key=lambda e: e['ts'])
    ops = 2 * moe_layers
    assert len(forward) == 16 * ops and len(backward) == 8 * ops, (len(forward), len(backward))
    out = []
    for mb in range(8):
        f = forward[mb*2*ops:mb*2*ops+ops]
        r = forward[mb*2*ops+ops:(mb+1)*2*ops]
        b = backward[mb*ops:(mb+1)*ops]
        assert f[-1]['ts'] < r[0]['ts'] < b[0]['ts']
        assert max(span(e)[1] for e in r) <= span(b[-1])[1] + 0.01
        out.append((f, r, b))
    return out


def analyze_rank(path, moe_layers=30):
    data = json.loads(path.read_text())
    grouped = defaultdict(list)
    iterations = {}
    for e in data['traceEvents']:
        if e['cat'] == 'iteration':
            iterations[int(e['name'].split('_')[1])] = e
        else:
            grouped[e['args']['iteration']].append(e)
    result = {}
    for step in range(25, 35):
        gpu = sorted([e for e in grouped[step] if e['cat'] == 'gpu_operation_range'], key=lambda e: e['ts'])
        cpu = sorted([e for e in grouped[step] if e['cat'] == 'cpu_operation_range'], key=lambda e: e['ts'])
        gb, cb = blocks(gpu, moe_layers), blocks(cpu, moe_layers)
        opt = next(e for e in gpu if e['name'] == 'optimizer.step')
        start, end = span(iterations[step])
        phases = {'startup': [(start, span(gb[0][0][0])[0])],
                  'first_forward': [], 'later_forward': [], 'forward_backward_transition': [],
                  'earlier_backward': [], 'last_backward': [], 'between_microbatches': [],
                  'backward_to_optimizer': [(span(gb[-1][2][-1])[1], span(opt)[0])],
                  'optimizer': [span(opt)], 'after_optimizer': [(span(opt)[1], end)]}
        micros = []
        for mb, ((f, r, b), (cf, cr, cbb)) in enumerate(zip(gb, cb)):
            phases['first_forward' if mb == 0 else 'later_forward'].append((span(f[0])[0], span(f[-1])[1]))
            phases['forward_backward_transition'].append((span(f[-1])[1], span(r[0])[0]))
            phases['last_backward' if mb == 7 else 'earlier_backward'].append((span(r[0])[0], span(b[-1])[1]))
            if mb:
                phases['between_microbatches'].append((span(gb[mb-1][2][-1])[1], span(f[0])[0]))
            micro = dict(forward_ms=span(f[-1])[1]-span(f[0])[0],
                         backward_ms=span(b[-1])[1]-span(r[0])[0],
                         f_start_ms=span(f[0])[0], f_end_ms=span(f[-1])[1],
                         b_start_ms=span(r[0])[0], b_end_ms=span(b[-1])[1])
            for label, ge, ce in [('f', f, cf), ('recompute', r, cr), ('b', b, cbb)]:
                micro[label+'_gpu_range_sum_ms'] = sum(e['dur'] for e in ge)/1000
                micro[label+'_cpu_sum_ms'] = sum(e['dur'] for e in ce)/1000
                micro[label+'_cpu_thread_ms'] = sum(e['args']['thread_cpu_us'] for e in ce)/1000
                leads = [span(g)[0]-span(c)[1] for g, c in zip(ge, ce)]
                micro[label+'_host_ahead_fraction'] = statistics.mean(v >= 0 for v in leads)
                micro[label+'_host_lead_ms'] = statistics.mean(leads)
            micro['layers'] = {
                label: [dict(gpu_span_ms=span(ge[2*i+1])[1]-span(ge[2*i])[0],
                             gpu_range_sum_ms=(ge[2*i]['dur']+ge[2*i+1]['dur'])/1000,
                             cpu_span_ms=span(ce[2*i+1])[1]-span(ce[2*i])[0],
                             cpu_sum_ms=(ce[2*i]['dur']+ce[2*i+1]['dur'])/1000,
                             start_ms=span(ge[2*i])[0], end_ms=span(ge[2*i+1])[1])
                        for i in range(moe_layers)]
                for label, ge, ce in [('f', f, cf), ('recompute', r, cr), ('b', b, cbb)]}
            micros.append(micro)
        phase_ms = {k: sum(b-a for a,b in ranges) for k,ranges in phases.items()}
        assert all(v >= -0.5 for v in phase_ms.values()), phase_ms
        assert abs(sum(phase_ms.values())-(end-start)) < 1e-5
        comm = defaultdict(list)
        for e in gpu:
            if e['name'].startswith(('collective.', 'nep.')):
                comm[category(e)].append(e)
        comm_stats = {}
        for name, evs in comm.items():
            ranges = [span(e) for e in evs]
            host = [e for e in cpu if category(e) == name]
            comm_stats[name] = dict(count=len(evs), sum_ms=sum(e['dur'] for e in evs)/1000,
                                   union_ms=union_duration(ranges), host_ms=sum(e['dur'] for e in host)/1000,
                                   host_thread_ms=sum(e['args']['thread_cpu_us'] for e in host)/1000,
                                   observer_host_ms=sum(e['args'].get('observer_enqueue_cpu_us',0) for e in host)/1000,
                                   input_bytes=sum(op['input_bytes'] for e in evs for op in e['args'].get('collective',{}).get('operations',[])),
                                   output_bytes=sum(op['output_bytes'] for e in evs for op in e['args'].get('collective',{}).get('operations',[])),
                                   phases={k: intersection_duration(ranges,v) for k,v in phases.items()})
        result[step] = dict(iteration_ms=end-start, phases=phase_ms, micros=micros,
                            comm=comm_stats, events=gpu, cpu=cpu, start_ms=start, end_ms=end)
    return dict(origin_ns=data['origin_ns'], clock_error_bound_ns=data['clock_error_bound_ns'], steps=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', default='3078207')
    parser.add_argument('--tag', default='mbs2')
    parser.add_argument('--healthy-tag')
    parser.add_argument('--healthy-job')
    parser.add_argument('--moe-layers', type=int, default=30)
    args = parser.parse_args()
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline')/args.job
    output = root/f'{args.tag}_overhead_analysis'
    output.mkdir(exist_ok=True)
    summary = {}
    for case, size in [('healthy',32),('nep',28)]:
        all_ranks = {}
        for rank in range(size):
            cache = output/f'{case}-rank-{rank}.json'
            if cache.exists():
                item = json.loads(cache.read_text())
                item['steps'] = {int(k):v for k,v in item['steps'].items()}
            else:
                tag = args.healthy_tag if case == 'healthy' and args.healthy_tag else args.tag
                job = args.healthy_job if case == 'healthy' and args.healthy_job else args.job
                path = root.parent/job/tag/'cupti/ep16/a8b_120b_latentmoe_1t'/case/job/'cupti_profile'/f'rank-{rank}.json'
                item = analyze_rank(path, args.moe_layers)
                cache.write_text(json.dumps(item)+'\n')
            all_ranks[rank] = item
        groups = {'owners':range(12), 'followers':range(12,16), 'second':range(16,28)}
        summary[case] = {}
        for group,ranks in groups.items():
            def average(f):
                return stats([statistics.mean(f(all_ranks[r]['steps'][s]) for r in ranks) for s in range(25,35)])
            sample = all_ranks[0]['steps'][25]
            phases = {k:average(lambda row:row['phases'][k]) for k in sample['phases']}
            micros = [{k:average(lambda row:row['micros'][mb][k]) for k in sample['micros'][mb] if k != 'layers'} for mb in range(8)]
            names = set().union(*(set(all_ranks[r]['steps'][s]['comm']) for r in ranks for s in range(25,35)))
            comm = {}
            for name in names:
                exemplar = next(all_ranks[r]['steps'][25]['comm'][name] for r in ranks if name in all_ranks[r]['steps'][25]['comm'])
                comm[name] = {k:average(lambda row:row['comm'].get(name,{}).get(k,0)) for k in exemplar if k != 'phases'}
                comm[name]['phases'] = {k:average(lambda row:row['comm'].get(name,{}).get('phases',{}).get(k,0)) for k in sample['phases']}
            summary[case][group] = dict(phases=phases, micros=micros, comm=comm)
            print(case,group,'phases', {k:round(v['mean'],3) for k,v in phases.items()},flush=True)
    summary['delta'] = {role:{k:summary['nep'][role]['phases'][k]['mean']-summary['healthy'][role]['phases'][k]['mean'] for k in summary['healthy'][role]['phases']} for role in ['owners','followers','second']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('DELTAS',json.dumps(summary['delta'],indent=2))
    print(output)


if __name__ == '__main__':
    main()
