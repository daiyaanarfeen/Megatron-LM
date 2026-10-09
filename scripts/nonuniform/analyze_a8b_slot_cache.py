"""Audit slot-cache A/B using matched operation profiles, not kernel service times."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_a8b_current_overheads import stats


def summarize(root):
    result = {}
    for case, size in [('healthy', 32), ('nep', 28)]:
        ranks = {
            rank: json.loads((root / f'{case}-rank-{rank}.json').read_text())
            for rank in range(size)
        }
        origin = min(data['origin_ns'] for data in ranks.values())
        result[case] = {}
        for role, members in [('owners', range(12)), ('second', range(16, size))]:
            steps = []
            for step in range(25, 35):
                rows = [ranks[r]['steps'][str(step)] for r in members]

                def mean(fn):
                    return statistics.mean(fn(row) for row in rows)

                phases = {key: mean(lambda row: row['phases'][key]) for key in rows[0]['phases']}
                micro = []
                for mb in range(8):
                    times = {}
                    for field in ['f_start_ms', 'f_end_ms', 'b_start_ms', 'b_end_ms']:
                        times[field] = statistics.mean(
                            (ranks[r]['origin_ns'] - origin) / 1e6
                            + ranks[r]['steps'][str(step)]['micros'][mb][field]
                            for r in members
                        )
                    times['expert_backward_recompute_cpu_thread_ms'] = mean(
                        lambda row: row['micros'][mb]['b_cpu_thread_ms']
                        + row['micros'][mb]['recompute_cpu_thread_ms']
                    )
                    micro.append(times)
                leads = []
                for row in rows:
                    dispatch = sorted(
                        (e for e in row['events'] if e['name'] == 'hybridep.dispatch.forward'),
                        key=lambda e: e['ts'],
                    )
                    experts = sorted(
                        (e for e in row['cpu'] if e['name'] == 'moe.experts.forward'),
                        key=lambda e: e['ts'],
                    )
                    assert len(dispatch) == len(experts) == 240
                    leads.extend(
                        (e['ts'] - d['ts'] - d['dur']) / 1000 for d, e in zip(dispatch, experts)
                    )
                steps.append(
                    dict(
                        iteration=step,
                        elapsed_ms=mean(lambda row: row['iteration_ms']),
                        phases=phases,
                        microbatches=micro,
                        postprocess_cpu_ms=mean(
                            lambda row: sum(
                                e['dur'] / 1000
                                for e in row['cpu']
                                if e['name'] == 'moe.dispatch_postprocess'
                            )
                        ),
                        postprocess_gpu_range_ms=mean(
                            lambda row: sum(
                                e['dur'] / 1000
                                for e in row['events']
                                if e['name'] == 'moe.dispatch_postprocess'
                            )
                        ),
                        expert_cpu_start_minus_dispatch_gpu_end_ms=statistics.mean(leads),
                        early_expert_submission_fraction=statistics.mean(x < 0 for x in leads),
                    )
                )
            result[case][role] = steps
        skews = []
        for owner, second in zip(result[case]['owners'], result[case]['second']):
            o, s = owner['microbatches'][-1], second['microbatches'][-1]
            skews.append(
                dict(
                    iteration=owner['iteration'],
                    final_b_start_skew_ms=s['b_start_ms'] - o['b_start_ms'],
                    final_b_end_skew_ms=s['b_end_ms'] - o['b_end_ms'],
                    second_final_b_ms=s['b_end_ms'] - s['b_start_ms'],
                )
            )
        result[case]['skews'] = skews
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache_dir', type=Path)
    args = parser.parse_args()
    assert str(args.cache_dir.resolve()).startswith('/lustre/')
    result = summarize(args.cache_dir)
    output = args.cache_dir / 'slot_cache_attribution.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    for case in result:
        for role in ['owners', 'second']:
            rows = result[case][role]
            scalar = {
                k: stats([row[k] for row in rows])['mean']
                for k in rows[0]
                if k not in ['iteration', 'phases', 'microbatches']
            }
            print(case, role, scalar)
        print(
            case,
            'skews',
            {
                k: statistics.mean(row[k] for row in result[case]['skews'])
                for k in result[case]['skews'][0]
                if k != 'iteration'
            },
        )
    print(output)


if __name__ == '__main__':
    main()
