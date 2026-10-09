"""Matched measured-iteration accounting for isolated NEP cleanup experiments."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_a8b_graph_attribution import analyze
from training_scripts_nep_matrix import STATUS_PATTERN, TIMING_PATTERN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('--hard', action='store_true')
    args = parser.parse_args()
    base = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs')
    root = base / ('soft_clique_baseline' if args.hard else 'nep_cleanup_ab') / args.job
    runs = sorted(root.glob('cleanup_ab_*/nsys/ep16/a8b_120b_latentmoe_1t/*/' + args.job)) if args.hard else sorted(root.glob('*_moe'))
    steps = range(20, 28) if args.hard else range(15, 21)
    result = {}
    for run in runs:
        label = run.parents[4].name if args.hard else run.name
        profiles = sorted((run / 'cupti_profile').glob('rank-*.json'))
        world = (32 if run.parent.name == 'healthy' else 28) if args.hard else 14
        if len(profiles) != world:
            print('INCOMPLETE', label, len(profiles), world)
            continue
        log = (run / 'driver.log').read_text()
        total = 35 if args.hard else 25
        statuses = [tuple(map(int, m.groups())) for m in STATUS_PATTERN.finditer(log)][-total:]
        assert statuses == [(i, total, 0, 0) for i in range(1, total + 1)], (label, statuses)
        ranks = {}
        for path in profiles:
            rank = int(path.stem.split('-')[1])
            data = json.loads(path.read_text())
            for step in steps:
                events = [e for e in data['traceEvents'] if e.get('args', {}).get('iteration') == step and e['cat'] == 'gpu_operation_range']
                names = {e['name'] for e in events}
                assert {'collective.all_gather', 'collective.reduce_scatter'} <= names, (label, rank, step)
                if 'healthy' not in label:
                    assert 'nep.redistribute_params' in names, (label, rank, step)
                for e in events:
                    if e['name'].startswith('collective.'):
                        assert e['args']['collective']['ranks'] and e['args']['collective']['operations']
                        assert 'completion_stream' in e['args']
            assert (run / 'nsys_profile' / f'rank-{rank}.sqlite').is_file()
            ranks[rank] = analyze(path, steps, num_microbatches=2)
        full, reduced = (16, 12) if args.hard else (8, 6)
        cohorts = {}
        for name, members in [('owners', range(reduced)), ('followers', range(reduced, full)), ('second', range(full, world))]:
            rows = [{k: statistics.mean(ranks[r][i][k] for r in members)
                     for k, v in ranks[0][i].items() if isinstance(v, (float, int))}
                    for i in range(len(steps))]
            summary = {k: dict(mean=statistics.mean(row[k] for row in rows), sd=statistics.stdev(row[k] for row in rows)) for k in rows[0]}
            scopes = {k: statistics.mean(ranks[r][i]['extra_scopes_ms'][k] for r in members for i in range(len(steps)))
                      for k in ranks[0][0]['extra_scopes_ms']}
            cohorts[name] = dict(steps=rows, summary=summary, extra_scopes_ms=scopes)
        timings = {int(i): float(ms) for i, ms in TIMING_PATTERN.findall(log)}
        windows = {'measured': list(steps), 'post': list(range(steps.stop + 1, total + 1))}
        log_means = {k: statistics.mean(timings[i] for i in ids) for k, ids in windows.items()}
        result[label] = dict(run=str(run), ranks=ranks, cohorts=cohorts, log_means=log_means)
        print(label, {k: round(v['mean'], 3) for k, v in cohorts['owners']['summary'].items() if k in ('iteration_ms', 'first_forward_ms', 'other_forward_ms', 'nonfinal_backward_ms', 'final_backward_ms', 'outside_fb_ms')}, log_means)
    (root / 'cleanup_phase_analysis.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
