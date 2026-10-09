"""Validate all-rank profiles and report complete schedule-phase parity."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_a8b_graph_attribution import analyze

PHASES = (
    'first_forward_ms',
    'other_forward_ms',
    'nonfinal_backward_ms',
    'final_backward_ms',
    'outside_fb_ms',
    'iteration_ms',
)


def summarize(rows: list[dict]) -> dict:
    """Use per-iteration rank means as samples, not ranks as independent samples."""
    return {
        key: dict(
            mean=statistics.mean(row[key] for row in rows),
            sd=statistics.stdev(row[key] for row in rows),
            minimum=min(row[key] for row in rows),
            maximum=max(row[key] for row in rows),
        )
        for key in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    args = parser.parse_args()
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline') / args.job
    result = dict(job=args.job, measured_steps=list(range(25, 35)), cases={})
    for case, world in [('healthy', 32), ('nep', 28)]:
        run = root / f'hard_mbs1_ga2_{case}' / 'nsys/ep16/a8b_120b_latentmoe_1t' / case / args.job
        if not (run / 'baseline_uniform.json').is_file():
            print('Not yet validated:', case)
            continue
        ranks = {}
        for rank in range(world):
            path = run / 'cupti_profile' / f'rank-{rank}.json'
            trace = json.loads(path.read_text())
            for step in range(25, 35):
                events = [e for e in trace['traceEvents'] if e.get('args', {}).get('iteration') == step]
                gpu = [e for e in events if e['cat'] == 'gpu_operation_range']
                cpu = [e for e in events if e['cat'] == 'cpu_operation_range']
                assert [e['name'] for e in gpu] == [e['name'] for e in cpu], (case, rank, step)
                names = {e['name'] for e in gpu}
                assert {'collective.all_gather', 'collective.reduce_scatter'} <= names
                if case == 'nep':
                    assert 'nep.redistribute_params' in names
                    if rank < 12 or rank >= 16:
                        assert 'nep.stage_grads' in names
                for event in gpu:
                    if event['name'].startswith('collective.'):
                        meta = event['args']['collective']
                        assert meta['ranks'] and meta['operations']
                        assert 'completion_stream' in event['args']
                for direction in ('forward', 'backward'):
                    count = sum(e['name'] == f'graph.MoETransformerLayer.{direction}' for e in gpu)
                    assert count == 30, (case, rank, step, direction, count)
            assert (run / 'nsys_profile' / f'rank-{rank}.sqlite').is_file()
            ranks[str(rank)] = analyze(path, range(25, 35), num_microbatches=2)
        cohorts = {}
        for role, members in [('owners', range(12)), ('followers', range(12, 16)), ('second', range(16, world))]:
            steps = []
            for index, step in enumerate(range(25, 35)):
                rows = [ranks[str(rank)][index] for rank in members]
                steps.append({
                    key: statistics.mean(row[key] for row in rows)
                    for key, value in rows[0].items() if isinstance(value, (float, int))
                })
                assert steps[-1]['iteration'] == step
            cohorts[role] = dict(steps=steps, summary=summarize(steps))
        logged = json.loads((run / 'baseline_uniform.json').read_text())['timings']
        windows = {name: dict(mean=statistics.mean(float(logged[str(i)]) for i in ids),
                             sd=statistics.stdev(float(logged[str(i)]) for i in ids))
                   for name, ids in [('pre', range(15, 24)), ('measured', range(25, 35)), ('post', range(36, 46))]}
        result['cases'][case] = dict(run=str(run), ranks=ranks, cohorts=cohorts, log_windows=windows)
        print(case, json.dumps(cohorts['owners']['summary'], indent=2))
    if len(result['cases']) == 2:
        comparisons = {}
        for role in ('owners', 'followers', 'second'):
            h = result['cases']['healthy']['cohorts'][role]['summary']
            n = result['cases']['nep']['cohorts'][role]['summary']
            comparisons[role] = {
                key: dict(healthy_ms=h[key]['mean'], nep_ms=n[key]['mean'],
                          extra_ms=n[key]['mean'] - h[key]['mean'],
                          parity_pct=100 * h[key]['mean'] / n[key]['mean'],
                          slowdown_pct=100 * (n[key]['mean'] / h[key]['mean'] - 1))
                for key in PHASES
            }
            delta = comparisons[role]
            assert abs(sum(delta[k]['extra_ms'] for k in PHASES[:-1]) - delta['iteration_ms']['extra_ms']) < .001
        result['comparisons'] = comparisons
        print('OWNER_COMPARISON', json.dumps(comparisons['owners'], indent=2))
    (root / 'hard_graph_phase_analysis.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
