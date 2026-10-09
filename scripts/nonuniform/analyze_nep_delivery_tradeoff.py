"""Account for consumer-wait savings that do not advance first-forward completion."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from analyze_nep_delivery_factorial import intersection_duration


def analyze(root, cell, repeat):
    name = f'controlled_actual_{cell}_r{repeat}_nep_moe'
    data = {r: json.loads((root / f'{name}-rank-{r}.json').read_text()) for r in range(16)}
    deps = json.loads((root / f'{name}-dependencies.json').read_text())
    steps = [row['iteration'] for row in data[0]['rows']]
    divisor = 12 * len(steps)
    totals, graph_parts, graph_times = defaultdict(float), defaultdict(float), defaultdict(float)
    families, latest_ranks = defaultdict(float), Counter()
    for rank in range(12):
        for row in data[rank]['rows']:
            waits = [
                (d['demand_ms'] + row['start'], d['ready_ms'] + row['start'])
                for d in deps
                if d['rank'] == rank and d['iteration'] == row['iteration']
            ]
            graphs = [(g['start'], g['end']) for g in row['graphs']]
            overlap = intersection_duration(waits, graphs)
            assert overlap < 0.05, (name, rank, row['iteration'], overlap)
            span = row['end'] - row['start']
            wait = sum(b - a for a, b in waits)
            graph = sum(b - a for a, b in graphs)
            totals['first_forward_ms'] += span / divisor
            totals['parameter_wait_ms'] += wait / divisor
            totals['native_graph_elapsed_ms'] += graph / divisor
            totals['other_first_forward_ms'] += (span - wait - graph) / divisor
            for g in row['graphs']:
                graph_times[f"{g['index']}:{g['name']}"] += (g['end'] - g['start']) / divisor
                for label, duration in g['parts'].items():
                    graph_parts[label] += duration / divisor
            for k in row['kernels']:
                if k['graph'] and k['label'] == 'model_compute':
                    families[k['name']] += (k['end'] - k['start']) / divisor
    for index, step in enumerate(steps):
        kernels = {
            r: [k for k in data[r]['rows'][index]['kernels'] if k['label'] == 'model_EP']
            for r in range(16)
        }
        assert all(
            [k['name'] for k in ks] == [k['name'] for k in kernels[0]] for ks in kernels.values()
        )
        for call in range(len(kernels[0])):
            starts = {
                r: ks[call]['start'] + data[r]['origin_ns'] / 1e6 for r, ks in kernels.items()
            }
            last = max(starts, key=starts.get)
            latest_ranks[last] += 1
            for r in range(12):
                k = kernels[r][call]
                duration = k['end'] - k['start']
                totals['model_EP_kernel_sum_ms'] += duration / divisor
                totals['model_EP_before_latest_peer_start_ms'] += (
                    min(duration, max(0, starts[last] - starts[r])) / divisor
                )
    return dict(
        totals=dict(totals),
        native_graph_partition_ms=dict(graph_parts),
        graph_elapsed_ms=dict(graph_times),
        compute_kernel_residency_ms=dict(families),
        latest_model_EP_kernel_rank_counts=dict(latest_ranks),
        note='Graph idle means no native graph kernel, not global GPU idle. '
        'Peer-start diagnostic is not pure wait or wire service time. '
        'Cross-host clock offset is not independently measured.',
    )


def main(job):
    root = (
        Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab')
        / job
        / 'delivery_factorial'
    )
    result = {}
    for repeat in (0, 1):
        cells = {cell: analyze(root, cell, repeat) for cell in ('Q10', 'Q11')}
        differences = {}
        for section in (
            'totals',
            'native_graph_partition_ms',
            'graph_elapsed_ms',
            'compute_kernel_residency_ms',
        ):
            before, after = cells['Q10'][section], cells['Q11'][section]
            differences[section] = {
                k: after.get(k, 0) - before.get(k, 0) for k in before.keys() | after.keys()
            }
        result[str(repeat)] = dict(cells=cells, early_minus_consumer=differences)
        print('REPEAT', repeat, differences['totals'], flush=True)
        print('GRAPH_PARTITIONS', differences['native_graph_partition_ms'], flush=True)
    (root / 'early-release-tradeoff.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
