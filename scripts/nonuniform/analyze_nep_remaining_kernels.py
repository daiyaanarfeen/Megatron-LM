"""Kernel accounting for final backward and the sync-to-scaling interval."""

import json
import statistics
from collections import defaultdict

from analyze_nep_ag_red_overlap import duration
from analyze_nep_delivery_factorial import intersection_duration
from analyze_nep_first_forward_delivery import read_rank
from analyze_nep_remaining_overheads import OUT, ROOT, span


def partition_window(a: float, b: float, kernels: list[dict]) -> dict:
    """Return additive compute/communication/idle accounting, not causal costs."""
    compute, comm, other = [], [], []
    groups = defaultdict(list)
    for kernel in kernels:
        interval = max(a, kernel['start']), min(b, kernel['end'])
        if interval[1] <= interval[0]:
            continue
        label = kernel['label']
        groups[label].append(interval)
        if label in ('model_compute', 'memory'):
            compute.append(interval)
        elif label == 'NEP_staging':
            other.append(interval)
        else:
            comm.append(interval)
    busy = duration(compute + comm + other)
    overlap = intersection_duration(compute, comm)
    ordinary = duration(compute + comm)
    result = dict(
        compute_only_ms=duration(compute) - overlap,
        comm_only_ms=duration(comm) - overlap,
        compute_comm_overlap_ms=overlap,
        staging_without_compute_comm_ms=busy - ordinary,
        idle_ms=b - a - busy,
    )
    assert abs(sum(result.values()) - (b - a)) < 1e-5
    result['duration_ms'] = b - a
    for label, intervals in groups.items():
        result[label + '_union_ms'] = duration(intervals)
    result['expert_rs_model_ep_overlap_ms'] = intersection_duration(
        groups['expert_RS'], groups['model_EP']
    )
    return result


def main() -> None:
    """Analyze the full owner cohort, preserving separate repeats."""
    OUT.mkdir(exist_ok=True)
    result = {}
    for repeat in range(2):
        for cell, topology in [('H', 'healthy'), ('A', 'nep')]:
            run = ROOT / f'controlled_actual_{cell}_r{repeat}_{topology}_moe'
            rows = []
            for rank in range(12):
                cache = OUT / f'{topology}-r{repeat}-rank-{rank}-kernel_summary.json'
                if cache.exists():
                    rank_rows = json.loads(cache.read_text())
                else:
                    parsed = read_rank(run, rank, True, range(25, 31), phase='final_backward_tail')
                    events = json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())[
                        'traceEvents'
                    ]
                    rank_rows = []
                    for row in parsed['rows']:
                        step = row['iteration']
                        gpu = [
                            e
                            for e in events
                            if e.get('cat') == 'gpu_operation_range'
                            and e.get('args', {}).get('iteration') == step
                        ]
                        backward = [e for e in gpu if e['name'] == 'microbatch.backward'][-1]
                        sync = next(e for e in gpu if e['name'] == 'ddp.finish_grad_sync')
                        scale = next(
                            e
                            for e in gpu
                            if e['name'] in ('nep.scale_gradients', 'ddp.scale_gradients')
                        )
                        a, b = span(backward)
                        graphs = defaultdict(float)
                        for graph in row['graphs']:
                            for key, value in partition_window(
                                graph['start'], graph['end'], row['kernels']
                            ).items():
                                graphs[key] += value
                        rank_rows.append(
                            dict(
                                rank=rank,
                                iteration=step,
                                final_backward=partition_window(a, b, row['kernels']),
                                graphs=dict(graphs),
                                sync_to_scale=partition_window(
                                    span(sync)[1], span(scale)[0], row['kernels']
                                ),
                            )
                        )
                    cache.write_text(json.dumps(rank_rows) + '\n')
                rows.extend(rank_rows)
                print('KERNELS', topology, repeat, rank, flush=True)
            means = {}
            for phase in ('final_backward', 'graphs', 'sync_to_scale'):
                keys = set().union(*(row[phase] for row in rows))
                means[phase] = {
                    key: statistics.mean(row[phase].get(key, 0) for row in rows)
                    for key in sorted(keys)
                }
            result[f'{topology}_r{repeat}'] = dict(means=means, rows=rows)
            (OUT / 'kernels.json').write_text(json.dumps(result, indent=2) + '\n')
            print(topology, repeat, json.dumps(means), flush=True)


if __name__ == '__main__':
    main()
