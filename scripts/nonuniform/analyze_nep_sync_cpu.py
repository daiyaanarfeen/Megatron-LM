"""Summarize matched measured-window NEP sync-copy validation cases."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_cuda_event_mbs_pair import analyze, timing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--tags', nargs='+', required=True)
    args = parser.parse_args()
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline') / args.job
    result = {}
    for tag in args.tags:
        runs = list((root / tag / 'cupti/ep8/scaling_ladder__a770m_4b_moe_215b').glob('*/*'))
        runs = [run for run in runs if (run / 'driver.log').exists()]
        assert len(runs) == 1, (tag, runs)
        run = runs[0]
        case = run.parent.name
        size = 16 if case == 'healthy' else 14
        ranks = {rank: analyze(run / f'cupti_profile/rank-{rank}.json', 40, 50, case)
                 for rank in range(size)}
        roles = {'full_owner': range(6), 'reduced_owner': range(8, 14)}
        if case == 'healthy':
            roles = {'matched_full': range(6), 'matched_second': range(8, 14)}
        row = {'windows': {window: timing(run / 'driver.log', begin, end)
                           for window, begin, end in [('pre', 29, 39), ('recorded', 40, 50), ('post', 51, 61)]},
               'roles': {}}
        for role, selected in roles.items():
            samples = [sample for rank in selected for sample in ranks[rank]['iterations']]
            # SD across iterations of the rank-mean, not a misleading pool of
            # strongly correlated ranks as independent observations.
            iteration_means = [statistics.mean(ranks[rank]['iterations'][i]['range_ms'] for rank in selected)
                               for i in range(10)]
            operations = {}
            for name in sorted(samples[0]['operations']):
                operations[name] = {key: statistics.mean(sample['operations'][name][key] for sample in samples)
                                    for key in samples[0]['operations'][name]}
            spans = {key: statistics.mean(sample['temporal_spans_ms'][key] for sample in samples)
                     for key in samples[0]['temporal_spans_ms']}
            row['roles'][role] = dict(mean_ms=statistics.mean(iteration_means),
                                      std_ms=statistics.stdev(iteration_means),
                                      iterations_ms=iteration_means, operations=operations,
                                      temporal_spans_ms=spans)
        result[tag] = row
    destination = root / 'sync_cpu_analysis.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(destination)
    for tag, row in result.items():
        print(tag, json.dumps(row['windows']))
        for role, values in row['roles'].items():
            print(role, 'iteration', values['mean_ms'], '+/-', values['std_ms'])
            for name, measurements in values['operations'].items():
                if name.startswith('nep.'):
                    print(name, json.dumps(measurements))


if __name__ == '__main__':
    main()
