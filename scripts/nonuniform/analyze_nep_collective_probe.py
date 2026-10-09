"""Audit asynchronous AG/RS completion ranges, not kernel-only residency."""

import argparse
import json
import statistics
from collections import defaultdict

from analyze_exact_uniform_nep_slowdown import intersection_duration, union_duration
from analyze_nep_sync_workload import ROOT, SLUG, compare, read_case


def interval(event):
    return (event['ts'] / 1000, (event['ts'] + event['dur']) / 1000)


def category(event):
    metadata = event['args'].get('collective')
    if metadata is None:
        return event['name']
    desc = metadata['group_desc'].lower()
    kind = event['name'].removeprefix('collective.')
    if desc == 'nep_owner_gather':
        assert kind == 'all_to_all'
        return 'nep_grad_gather'
    if desc == 'nep_owner_transfer':
        assert kind == 'broadcast'
        return 'nep_param_broadcast'
    if desc == 'ep_dp' or 'expert_data_parallel' in desc:
        return 'expert_' + kind
    return desc + ':' + kind


def stats(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values), samples=values)


def inspect_rank(path):
    data = json.loads(path.read_text())
    assert data['collective_schema_version'] == 1
    events = data['traceEvents']
    rows, groups = [], {}
    for step in range(40, 50):
        selected = [e for e in events if e.get('args', {}).get('iteration') == step]
        gpu = [e for e in selected if e['cat'] == 'gpu_operation_range']
        cpu = [e for e in selected if e['cat'] == 'cpu_operation_range']
        forward = sorted([e for e in gpu if e['name'] == 'expert.forward'], key=lambda e: e['ts'])
        backward = sorted([e for e in gpu if e['name'] == 'expert.backward'], key=lambda e: e['ts'])
        assert len(forward) == len(backward) == 90
        optimizer = next(e for e in gpu if e['name'] == 'optimizer.step')
        first_f = [(interval(forward[0])[0], interval(forward[29])[1])]
        last_b = [(interval(backward[60])[0], interval(backward[89])[1])]
        tail = [(last_b[0][1], interval(optimizer)[0])]
        expert_work = [interval(e) for e in forward + backward]
        buckets = defaultdict(list)
        for event in gpu:
            if not event['name'].startswith('collective.') and not event['name'].startswith('nep.'):
                continue
            buckets[category(event)].append(event)
            if 'collective' in event['args']:
                meta = event['args']['collective']
                assert meta['interval'] == 'input_ready_to_completion'
                key = meta['group_id']
                groups[key] = {k: meta[k] for k in ('group_id', 'group_desc', 'ranks')}
        row = dict(iteration=step, categories={})
        for name, records in buckets.items():
            ranges = [interval(e) for e in records]
            host = [e for e in cpu if category(e) == name]
            assert len(records) == len(host)
            row['categories'][name] = dict(
                count=len(records),
                sum_ms=sum(e['dur'] for e in records) / 1000,
                union_ms=union_duration(ranges),
                first_forward_window_ms=intersection_duration(ranges, first_f),
                final_backward_window_ms=intersection_duration(ranges, last_b),
                after_last_expert_before_optimizer_ms=intersection_duration(ranges, tail),
                overlap_recorded_expert_ranges_ms=intersection_duration(ranges, expert_work),
                host_wall_ms=sum(e['dur'] for e in host) / 1000,
                observer_enqueue_cpu_ms=sum(e['args'].get('observer_enqueue_cpu_us', 0) for e in host) / 1000,
                input_bytes=sum(op['input_bytes'] for e in records for op in e['args'].get('collective', {}).get('operations', [])),
                output_bytes=sum(op['output_bytes'] for e in records for op in e['args'].get('collective', {}).get('operations', [])),
            )
        row['final_backward_events'] = [
            dict(category=category(e), start_relative_first_expert_ms=interval(e)[0] - last_b[0][0],
                 end_relative_first_expert_ms=interval(e)[1] - last_b[0][0],
                 duration_ms=e['dur'] / 1000, collective=e['args'].get('collective'))
            for e in gpu
            if (e['name'].startswith('collective.') or e['name'] == 'nep.stage_grads')
            and interval(e)[1] >= last_b[0][0]
        ]
        row['final_backward_span_ms'] = last_b[0][1] - last_b[0][0]
        rows.append(row)
    return dict(groups=list(groups.values()), iterations=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--tag', default='e512_seq2048')
    args = parser.parse_args()
    root = ROOT / args.job
    pair = {case: read_case(root, args.tag, case) for case in ('healthy', 'nep')}
    result = dict(job=args.job, comparison=compare(pair), phases=pair, cases={}, notes=[
        'Input-ready-to-completion intervals include host, queue, and peer waiting; not pure NCCL kernel or wire time.',
        'Overlap is only with recorded grouped-expert operation ranges, which themselves can include gaps.',
        'Attention/Mamba/HybridEP kernels are not captured individually; non-overlap is not proof of exposure or GPU idle.',
        'Distinct categories can overlap. Sums are not additive causes of iteration slowdown.',
        'Pre/post windows diagnose drift/profiling distortion; no subtraction corrects measured timing.',
    ])
    for case, size in (('healthy', 16), ('nep', 14)):
        path = root / args.tag / 'cupti/ep8' / SLUG / case / args.job
        ranks = {r: inspect_rank(path / f'cupti_profile/rank-{r}.json') for r in range(size)}
        cohorts = dict(full_owners=range(6), second_owners=range(8, 14), followers=range(6, 8))
        summaries = {}
        for cohort, members in cohorts.items():
            names = set().union(*(set(step['categories']) for r in members for step in ranks[r]['iterations']))
            summaries[cohort] = {}
            for name in sorted(names):
                exemplar = next(step['categories'][name] for r in members for step in ranks[r]['iterations'] if name in step['categories'])
                summaries[cohort][name] = {
                    key: stats([statistics.mean(ranks[r]['iterations'][i]['categories'].get(name, {}).get(key, 0) for r in members) for i in range(10)])
                    for key in exemplar
                }
            for required in ('expert_all_gather', 'expert_reduce_scatter'):
                if case == 'healthy' or cohort != 'followers':
                    assert required in summaries[cohort], (case, cohort, names)
        result['cases'][case] = dict(ranks=ranks, cohorts=summaries)
    output = root / 'collective_analysis'
    output.mkdir(exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['comparison'], indent=2))
    for case, data in result['cases'].items():
        for cohort in ('full_owners', 'second_owners'):
            print(case, cohort)
            for name, metrics in data['cohorts'][cohort].items():
                print(name, {key: round(metrics[key]['mean'], 3) for key in ('count', 'sum_ms', 'union_ms', 'final_backward_window_ms', 'after_last_expert_before_optimizer_ms', 'observer_enqueue_cpu_ms')})
    print(output / 'summary.json')


if __name__ == '__main__':
    main()
