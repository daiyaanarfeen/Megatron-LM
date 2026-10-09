"""Compare sync-adjacent temporal envelopes without claiming kernel attribution."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path

from analyze_cuda_event_mbs_pair import analyze, timing

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline')
SLUG = 'scaling_ladder__a770m_4b_moe_215b'


def summarize(values: list[float]) -> dict:
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values), samples=values)


def phases(sample: dict) -> dict:
    spans = sample['temporal_spans_ms']
    fwd = [spans[f'mb{i}_first_forward -> mb{i}_last_forward'] for i in range(3)]
    bwd = [spans[f'mb{i}_first_backward -> mb{i}_last_backward'] for i in range(3)]
    additive = dict(
        before_first_expert=spans['iteration_start -> mb0_first_forward'],
        first_forward=fwd[0],
        later_forwards=sum(fwd[1:]),
        forward_to_backward=sum(
            spans[f'mb{i}_last_forward -> mb{i}_first_backward'] for i in range(3)
        ),
        earlier_backwards=sum(bwd[:2]),
        final_backward=bwd[2],
        between_microbatches=sum(
            spans[f'mb{i}_last_backward -> mb{i+1}_first_forward'] for i in range(2)
        ),
        last_backward_to_optimizer=spans['mb2_last_backward -> optimizer_start'],
        optimizer=spans['optimizer_start -> optimizer_end'],
        after_optimizer=spans['optimizer_end -> iteration_end'],
    )
    assert math.isclose(sum(additive.values()), sample['range_ms'], abs_tol=1e-6)
    return dict(
        iteration=sample['range_ms'],
        additive=additive,
        comparisons=dict(
            first_forward=fwd[0],
            later_forward_mean=statistics.mean(fwd[1:]),
            final_backward=bwd[2],
            earlier_backward_mean=statistics.mean(bwd[:2]),
            first_forward_minus_later=fwd[0] - statistics.mean(fwd[1:]),
            final_backward_minus_earlier=bwd[2] - statistics.mean(bwd[:2]),
        ),
    )


def read_case(root: Path, tag: str, case: str) -> dict:
    run = root / tag / 'cupti/ep8' / SLUG / case / root.name
    log = (run / 'driver.log').read_text()
    statuses = re.findall(
        r'iteration\s+(\d+)/\s*60.*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)',
        log,
    )
    assert len(statuses) == 60 and all(s == n == '0' for _, s, n in statuses), (run, statuses)
    assert {int(i) for i, _, _ in statuses} == set(range(1, 61))
    size = 16 if case == 'healthy' else 14
    assert log.count('GPU_EVENT_TRACE_RESULT') == size
    records = [
        analyze(run / f'cupti_profile/rank-{rank}.json', 40, 50, case)['iterations']
        for rank in range(size)
    ]
    windows = {
        name: timing(run / 'driver.log', a, b)
        for name, a, b in (('pre', 29, 39), ('recorded', 40, 50), ('post', 51, 61))
    }
    groups = {'full_owners': range(6), 'second_replica_owners': range(8, 14)}
    if case == 'nep':
        groups['followers'] = range(6, 8)
    result = dict(path=str(run), windows=windows, cohorts={})
    for group, ranks in groups.items():
        per_rank = {rank: [phases(sample) for sample in records[rank]] for rank in ranks}
        entry = dict(
            iteration=summarize(
                [statistics.mean(per_rank[r][i]['iteration'] for r in ranks) for i in range(10)]
            )
        )
        for category in ('additive', 'comparisons'):
            keys = per_rank[next(iter(ranks))][0][category]
            entry[category] = {
                key: summarize(
                    [
                        statistics.mean(per_rank[r][i][category][key] for r in ranks)
                        for i in range(10)
                    ]
                )
                for key in keys
            }
        names = set().union(*(sample['operations'] for r in ranks for sample in records[r]))
        entry['operations'] = {}
        for name in sorted(names):
            measures = next(
                sample['operations'][name]
                for r in ranks
                for sample in records[r]
                if name in sample['operations']
            )
            entry['operations'][name] = {
                key: summarize(
                    [
                        statistics.mean(
                            records[r][i]['operations'].get(name, {}).get(key, 0) for r in ranks
                        )
                        for i in range(10)
                    ]
                )
                for key in measures
            }
        result['cohorts'][group] = entry
    manifests = sorted((run / 'launch_manifest').glob('rank-*.json'))
    assert len(manifests) == size, (run, len(manifests))
    for path in manifests:
        manifest = json.loads(path.read_text())
        env = manifest['environment']
        rank = int(env['RANK'])
        assert env['REPLICA_NUM_MICROBATCHES'] == '3 3'
        assert env['NCCL_MNNVL_CLIQUE_ID'] == str(1000 + int(rank >= 8))
        assert env['CUPTI_EVENTS_ONLY'] == '1' and env['CUPTI_COLLECTION_DISABLED'] == '0'
    result['manifest_rank0'] = json.loads((run / 'launch_manifest/rank-0.json').read_text())
    # Architecture changes can legitimately change native bucket thresholds.
    # Retain the actual sizes instead of silently assuming equal buckets.
    result['bucket_log'] = [
        line
        for line in log.splitlines()
        if 'Number of buckets for gradient all-reduce / reduce-scatter:' in line
        or line.startswith('Params for bucket ')
    ]
    assert result['bucket_log']
    return result


def compare(pair: dict) -> dict:
    healthy = pair['healthy']['cohorts']['full_owners']
    nep = pair['nep']['cohorts']['full_owners']
    delta = {
        key: nep['additive'][key]['mean'] - healthy['additive'][key]['mean']
        for key in healthy['additive']
    }
    assert math.isclose(
        sum(delta.values()), nep['iteration']['mean'] - healthy['iteration']['mean'], abs_tol=1e-6
    )
    result = dict(
        owner_parity=healthy['iteration']['mean'] / nep['iteration']['mean'],
        delta_iteration_ms=nep['iteration']['mean'] - healthy['iteration']['mean'],
        delta_temporal_envelopes_ms=delta,
        delta_comparisons_ms={
            key: nep['comparisons'][key]['mean'] - healthy['comparisons'][key]['mean']
            for key in healthy['comparisons']
        },
        logged_window_parity={
            window: pair['healthy']['windows'][window]['mean_ms']
            / pair['nep']['windows'][window]['mean_ms']
            for window in ('pre', 'recorded', 'post')
        },
    )
    result['profile_excess_over_post_ms'] = {
        case: pair[case]['windows']['recorded']['mean_ms']
        - pair[case]['windows']['post']['mean_ms']
        for case in ('healthy', 'nep')
    }
    # This diagnoses possible distortion/drift, not a profiler correction.
    result['differential_recorded_post_ms'] = (
        result['profile_excess_over_post_ms']['nep']
        - result['profile_excess_over_post_ms']['healthy']
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--tags', nargs='+', required=True)
    args = parser.parse_args()
    root = ROOT / args.job
    output = root / 'sync_workload_analysis'
    output.mkdir(exist_ok=True)
    result = dict(
        job=args.job,
        notes=[
            'GPU event envelopes include dependencies and host spacing; not kernel busy time or exposed communication.',
            'Temporal categories are additive but not causal attribution; ordinary compute and native collectives occur inside.',
            'SD is across iteration means over ranks, not pooled ranks as independent samples.',
            'First-minus-later and final-minus-earlier are descriptive controls, not exact causal sync cost.',
            'Recorded and post-profile windows are never combined or corrected by subtraction.',
        ],
        cases={},
    )
    for tag in args.tags:
        pair = {case: read_case(root, tag, case) for case in ('healthy', 'nep')}
        pair['comparison'] = compare(pair)
        result['cases'][tag] = pair
        print(tag, json.dumps(pair['comparison']), flush=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(output / 'summary.json')


if __name__ == '__main__':
    main()
