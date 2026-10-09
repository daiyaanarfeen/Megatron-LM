"""Estimate parameter-release advances, not a full counterfactual runtime replay."""

import json
import statistics
from pathlib import Path

from analyze_nep_ag_red_overlap import duration
from analyze_nep_delivery_factorial import intersection_duration
from analyze_nep_first_forward_delivery import read_rank

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3215521')
OUT = ROOT / 'zero_param_delivery_20261006'


def stats(values: list[float]) -> dict:
    """Report descriptive statistics without treating ranks as repeats."""
    return dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values))


def main() -> None:
    """Inspect both compact ascending repeats using the original measured steps."""
    OUT.mkdir(exist_ok=True)
    result = dict(repeats=[], note='Frozen AG/compute schedules; not a causal runtime measurement.')
    for repeat in range(2):
        run = ROOT / f'controlled_actual_A_r{repeat}_nep_moe'
        traces = {
            rank: json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())
            for rank in range(28)
        }
        origin = min(trace['origin_ns'] for trace in traces.values())
        rows, frontiers = [], []
        for step in range(25, 31):
            ranks = {}
            for rank, trace in traces.items():
                events = [
                    e
                    for e in trace['traceEvents']
                    if e.get('args', {}).get('iteration') == step
                    and e['cat'] == 'gpu_operation_range'
                ]
                shift = (trace['origin_ns'] - origin) / 1e6
                absolute = lambda e: (shift + e['ts'] / 1000, shift + (e['ts'] + e['dur']) / 1000)
                forward = next(e for e in events if e['name'] == 'microbatch.forward')
                a, b = absolute(forward)
                graphs = [
                    e for e in events if e['name'].startswith('graph.') and a <= absolute(e)[0] < b
                ]
                moes = [e for e in graphs if e['name'] == 'graph.MoETransformerLayer.forward']
                assert len(moes) == 15, (rank, step, len(moes))
                red = [e for e in events if e['name'] == 'nep.redistribute_params']
                ag = [
                    e
                    for e in events
                    if e['name'] == 'collective.all_gather'
                    and e['args']['collective']['group_desc'] == 'ep_dp'
                ]
                assert len(red) == 3 and len(ag) == (0 if 12 <= rank < 16 else 3)
                groups = [
                    e['args']['collective']['ranks']
                    for e in events
                    if e['name'] == 'collective.all_to_all'
                    and e['args']['collective']['group_desc'] == 'nep_owner_transfer'
                ]
                sources = {r for group in groups for r in group if r < 12}
                if rank < 12 or rank >= 16:
                    sources.add(rank)
                assert sources
                consumers = []
                for index in (0, 5, 10):
                    consumer = moes[index]
                    previous = graphs[graphs.index(consumer) - 1]
                    consumers.append((absolute(previous)[1], absolute(consumer)[0]))
                ranks[rank] = dict(
                    ag=[absolute(e) for e in ag],
                    red=[absolute(e) for e in red],
                    consumers=consumers,
                    sources=sorted(sources),
                )
            for rank, data in ranks.items():
                for bucket in range(3):
                    source_ready = max(ranks[source]['ag'][bucket][1] for source in data['sources'])
                    previous, consumer = data['consumers'][bucket]
                    red_start, red_end = data['red'][bucket]
                    rows.append(
                        dict(
                            rank=rank,
                            iteration=step,
                            bucket=bucket,
                            red_span_ms=red_end - red_start,
                            required_source_ag_end_ms=source_ready,
                            red_end_ms=red_end,
                            consumer_start_ms=consumer,
                            previous_graph_end_ms=previous,
                            ready_margin_ms=consumer - red_end,
                            after_source_ag_ms=max(0, red_end - source_ready),
                            direct_release_advance_ms=max(
                                0, min(consumer, red_end) - max(previous, source_ready)
                            ),
                        )
                    )
            # The first expert's model-EP operation needs the whole replica.
            # Keep observed upstream readiness fixed; do not sum rank-local waits.
            for role, ids in [('full', range(16)), ('reduced', range(16, 28))]:
                bucket0 = [
                    row
                    for row in rows
                    if row['iteration'] == step and row['bucket'] == 0 and row['rank'] in ids
                ]
                actual = max(
                    max(row['red_end_ms'], row['previous_graph_end_ms']) for row in bucket0
                )
                ideal = max(
                    max(row['required_source_ag_end_ms'], row['previous_graph_end_ms'])
                    for row in bucket0
                )
                frontiers.append(dict(iteration=step, role=role, release_advance_ms=actual - ideal))
        kernels = []
        for rank in range(12):
            cache = OUT / f'r{repeat}-rank-{rank}-kernels.json'
            if cache.exists():
                kernel_rows = json.loads(cache.read_text())
            else:
                parsed = read_rank(run, rank, True, range(25, 31))
                kernel_rows = []
                for row in parsed['rows']:
                    red, other = [], []
                    for kernel in row['kernels']:
                        intervals = (
                            red if kernel['label'] in ('NEP_broadcast', 'NEP_staging') else other
                        )
                        intervals.append(
                            (max(row['start'], kernel['start']), min(row['end'], kernel['end']))
                        )
                    kernel_rows.append(
                        dict(
                            rank=rank,
                            iteration=row['iteration'],
                            union_ms=duration(red),
                            exclusive_ms=duration(red) - intersection_duration(red, other),
                        )
                    )
                cache.write_text(json.dumps(kernel_rows) + '\n')
            kernels.extend(kernel_rows)
            print('KERNELS', repeat, rank, flush=True)
        summary = dict(repeat=repeat, rows=rows, frontiers=frontiers, kernels=kernels)
        summary['first_bucket_full_release_advance_ms'] = stats(
            [r['release_advance_ms'] for r in frontiers if r['role'] == 'full']
        )
        summary['first_bucket_reduced_release_advance_ms'] = stats(
            [r['release_advance_ms'] for r in frontiers if r['role'] == 'reduced']
        )
        summary['later_delivery_min_margin_ms'] = min(
            r['ready_margin_ms'] for r in rows if r['bucket']
        )
        summary['owner_later_direct_release_advance_ms'] = stats(
            [r['direct_release_advance_ms'] for r in rows if r['bucket'] and r['rank'] < 12]
        )
        summary['owner_redistribution_union_ms'] = stats([r['union_ms'] for r in kernels])
        summary['owner_redistribution_exclusive_ms'] = stats([r['exclusive_ms'] for r in kernels])
        result['repeats'].append(summary)
        (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(
            json.dumps(
                {k: v for k, v in summary.items() if k not in ('rows', 'frontiers', 'kernels')}
            ),
            flush=True,
        )


if __name__ == '__main__':
    main()
