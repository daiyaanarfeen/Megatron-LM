"""Validate and summarize matched, all-rank placement-ablation event profiles."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_a8b_graph_attribution import analyze
from training_scripts_nep_matrix import STATUS_PATTERN, TIMING_PATTERN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('--full', type=int, default=8)
    parser.add_argument('--reduced', type=int, default=6)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--hard', action='store_true')
    args = parser.parse_args()
    if args.hard:
        args.full, args.reduced = 16, 12
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs') / ('soft_clique_baseline' if args.hard else 'nep_cleanup_ab') / args.job
    result = {}
    steps = list(range(20, 28) if args.hard else range(15, 21))
    total = 35 if args.hard else 25
    runs = sorted(root.glob('cleanup_ab_ablation_*/nsys/ep16/a8b_120b_latentmoe_1t/*/' + args.job)) if args.hard else sorted(p for p in root.glob('ablation_*') if p.is_dir())
    for run in runs:
        label = run.parents[4].name.removeprefix('cleanup_ab_') + '_' + run.parent.name + '_moe' if args.hard else run.name
        if args.cases and label not in args.cases:
            continue
        healthy = '_healthy_' in label
        world = 2 * args.full if healthy else args.full + args.reduced
        paths = sorted((run / 'cupti_profile').glob('rank-*.json'))
        assert len(paths) == world, (run, len(paths), world)
        log = (run / 'driver.log').read_text()
        status = [tuple(map(int, m.groups())) for m in STATUS_PATTERN.finditer(log)][-total:]
        assert status == [(i, total, 0, 0) for i in range(1, total + 1)], (run, status)
        ranks, collectives = {}, {}
        broadcast_bytes = {step: dict(source_payload_bytes=0, aggregate_receive_bytes=0, groups=0, gather_input_bytes=0, gather_output_bytes=0) for step in steps}
        for path in paths:
            rank = int(path.stem.split('-')[1])
            raw = json.loads(path.read_text())
            counts = {}
            for step in steps:
                events = [e for e in raw['traceEvents'] if e.get('cat') == 'gpu_operation_range' and e.get('args', {}).get('iteration') == step]
                names = {e['name'] for e in events}
                assert {'collective.all_gather', 'collective.reduce_scatter'} <= names, (run, rank, step, names)
                if not healthy:
                    assert 'nep.redistribute_params' in names, (run, rank, step)
                counts[step] = {}
                for event in events:
                    if not event['name'].startswith('collective.'):
                        continue
                    meta = event['args']['collective']
                    assert meta['ranks'] and meta['operations']
                    assert 'completion_stream' in event['args']
                    key = event['name'] + ':' + str(meta['group_desc'])
                    row = counts[step].setdefault(key, dict(count=0, elapsed_ms=0))
                    row['count'] += 1
                    row['elapsed_ms'] += event['dur'] / 1000
                    if rank < args.full and meta['group_desc'] == 'nep_owner_transfer' and event['name'] == 'collective.broadcast':
                        operation = meta['operations'][0]
                        if operation['source_rank'] == rank:
                            payload = operation['input_bytes']
                            broadcast_bytes[step]['source_payload_bytes'] += payload
                            broadcast_bytes[step]['aggregate_receive_bytes'] += payload * (len(meta['ranks']) - 1)
                            broadcast_bytes[step]['groups'] += 1
                    if rank < args.full and meta['group_desc'] == 'nep_owner_gather' and event['name'] == 'collective.all_to_all':
                        broadcast_bytes[step]['gather_input_bytes'] += sum(op['input_bytes'] for op in meta['operations'])
                        broadcast_bytes[step]['gather_output_bytes'] += sum(op['output_bytes'] for op in meta['operations'])
            assert (run / 'nsys_profile' / f'rank-{rank}.sqlite').is_file()
            ranks[rank] = analyze(path, steps, num_microbatches=2)
            collectives[rank] = counts
        cohorts = {}
        for name, members in [('owners', range(args.reduced)), ('followers', range(args.reduced, args.full)), ('second', range(args.full, world))]:
            rows = [{key: statistics.mean(ranks[r][i][key] for r in members) for key, value in ranks[0][i].items() if isinstance(value, (int, float))} for i in range(len(steps))]
            summary = {key: dict(mean=statistics.mean(row[key] for row in rows), sd=statistics.stdev(row[key] for row in rows)) for key in rows[0]}
            cohorts[name] = dict(steps=rows, summary=summary)
        timings = {int(i): float(ms) for i, ms in TIMING_PATTERN.findall(log)}
        log_windows = {name: dict(mean=statistics.mean(timings[i] for i in ids), sd=statistics.stdev(timings[i] for i in ids)) for name, ids in [('measured', steps), ('post', range(steps[-1] + 2, total + 1))]}
        result[label] = dict(run=str(run), ranks=ranks, cohorts=cohorts, collective_elapsed_diagnostics=collectives, log_windows=log_windows, broadcast_bytes=broadcast_bytes)
        print(label, {k: round(v['mean'], 3) for k, v in cohorts['owners']['summary'].items() if k in ('iteration_ms', 'first_forward_ms', 'other_forward_ms', 'nonfinal_backward_ms', 'final_backward_ms', 'outside_fb_ms')}, flush=True)
    assert result, root
    (root / 'placement_analysis.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
