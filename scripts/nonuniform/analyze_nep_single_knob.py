"""Audit same-window owner parity and operation ranges for single-knob runs."""

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from analyze_cuda_event_mbs_pair import analyze, timing


ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline')
TAGS = ('anchor_start', 'seq4096', 'topk2', 'anchor_end')
SLUG = 'scaling_ladder__a770m_4b_moe_215b'


def summary(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values), samples=values)


def phase_metrics(path):
    data = json.loads(path.read_text())
    events = data['traceEvents']
    result = []
    for step in range(40, 50):
        row = {}
        for direction in ('forward', 'backward'):
            name = f'expert.{direction}'
            groups = {}
            for kind in ('cpu', 'gpu'):
                es = sorted((e for e in events if e['cat'] == f'{kind}_operation_range'
                             and e['name'] == name and e['args']['iteration'] == step),
                            key=lambda e: e['ts'])
                assert len(es) == 90, (path, step, direction, kind, len(es))
                groups[kind] = [es[i:i + 30] for i in (0, 30, 60)]
            for index in range(3):
                c, g = groups['cpu'][index], groups['gpu'][index]
                prefix = f'{direction}{index}'
                row[prefix + '_span_ms'] = (g[-1]['ts'] + g[-1]['dur'] - g[0]['ts']) / 1000
                row[prefix + '_gpu_range_sum_ms'] = sum(e['dur'] for e in g) / 1000
                row[prefix + '_cpu_wall_ms'] = sum(e['dur'] for e in c) / 1000
                row[prefix + '_cpu_thread_ms'] = sum(e['args']['thread_cpu_us'] for e in c) / 1000
                row[prefix + '_launch_ahead_fraction'] = statistics.mean(
                    ge['ts'] > ce['ts'] + ce['dur'] + 3 * data['clock_error_bound_ns'] / 1000
                    for ce, ge in zip(c, g))
                row[prefix + '_mean_start_lag_ms'] = statistics.mean(
                    (ge['ts'] - ce['ts']) / 1000 for ce, ge in zip(c, g))
        result.append(row)
    return result


def bucket_signature(log):
    lines = [line for line in log.splitlines() if
             'Number of buckets for gradient all-reduce / reduce-scatter:' in line
             or line.startswith('Params for bucket ')]
    # Ignore logger prefixes; retained entries encode actual padded/unpadded sizes.
    lines = [line[line.index('Number of buckets'):] if 'Number of buckets' in line else line
             for line in lines]
    return dict(lines=lines, sha256=hashlib.sha256('\n'.join(lines).encode()).hexdigest())


def normalized_options(tokens, variant):
    tokens = tokens.copy()
    if variant == 'seq4096':
        i = tokens.index('--seq-length') + 1
        assert tokens[i] == '4096'
        tokens[i] = '8192'
    elif variant == 'topk2':
        i = tokens.index('--moe-router-topk') + 1
        assert tokens[i] == '2'
        tokens[i] = '6'
    return ['ARTIFACT_PATH' if token.startswith('/lustre/') else token for token in tokens]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--tags', nargs='+', default=TAGS)
    args = parser.parse_args()
    root = ROOT / args.job
    out = root / 'sensitivity_analysis'
    out.mkdir(exist_ok=True)
    result = {}
    reference_options, reference_buckets = {}, {}
    for tag in args.tags:
        pair = {}
        for case, size in [('healthy', 16), ('nep', 14)]:
            run = root / tag / 'cupti/ep8' / SLUG / case / args.job
            paths = [run / f'cupti_profile/rank-{rank}.json' for rank in range(size)]
            log_path = run / 'driver.log'
            if not log_path.exists() or not all(p.exists() for p in paths):
                if args.partial:
                    break
                raise RuntimeError(f'Incomplete case: {run}')
            log = log_path.read_text()
            if not re.search(r'iteration\s+60/\s*60', log):
                if args.partial:
                    break
                raise RuntimeError(f'Case did not finish60 iterations: {run}')
            assert log.count('GPU_EVENT_TRACE_RESULT') == size, (run, 'profile count')
            assert 'backend=cuda_events' in log and 'GPU_TRACE_CUPTI_VERSION' not in log
            statuses = re.findall(r'iteration\s+(\d+)/\s*60.*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)', log)
            assert len(statuses) == 60 and all(s == n == '0' for _, s, n in statuses), (run, statuses)
            losses = re.findall(r'lm loss:\s*([\d.eE+-]+)', log)
            assert len(losses) == 60 and all(math.isfinite(float(x)) for x in losses)
            buckets = bucket_signature(log)
            assert buckets['lines']
            if case in reference_buckets:
                assert reference_buckets[case] == buckets, (tag, case, 'rebucketing')
            else:
                reference_buckets[case] = buckets
            rank_data, phases = [], []
            for rank, path in enumerate(paths):
                manifest = json.loads((run / f'launch_manifest/rank-{rank}.json').read_text())
                expected_variant = tag if tag in ('mbs1', 'seq4096', 'topk2') else 'anchor'
                assert manifest['variant'] == expected_variant
                opts = normalized_options(manifest['options'], expected_variant)
                key = case, rank
                if key in reference_options:
                    assert reference_options[key] == opts, (tag, key, 'unrelated option changed')
                else:
                    reference_options[key] = opts
                env = manifest['environment']
                assert env['NCCL_MNNVL_CLIQUE_ID'] == str(1000 + (rank >= 8))
                expected_mbs = '1 1' if tag == 'mbs1' else '2 2'
                assert env['REPLICA_MICRO_BATCH_SIZES'] == expected_mbs
                assert env['REPLICA_NUM_MICROBATCHES'] == '3 3'
                assert env['CUPTI_EVENTS_ONLY'] == '1' and env['CUPTI_COLLECTION_DISABLED'] == '0'
                rank_data.append(analyze(path, 40, 50, case))
                phases.append(phase_metrics(path))
            cohorts = {'owners': range(6), 'other_replica': range(8, 14)}
            if case == 'nep':
                cohorts['followers'] = range(6, 8)
            cohort_data = {}
            for label, ranks in cohorts.items():
                per_iteration = [statistics.mean(rank_data[r]['iterations'][i]['range_ms'] for r in ranks)
                                 for i in range(10)]
                phase_stats = {key: summary([statistics.mean(phases[r][i][key] for r in ranks)
                                             for i in range(10)])
                               for key in phases[0][0]}
                names = set().union(*(rank_data[r]['iterations'][0]['operations'].keys() for r in ranks))
                ops = {}
                for name in names:
                    measures = next(rank_data[r]['iterations'][0]['operations'][name]
                                    for r in ranks if name in rank_data[r]['iterations'][0]['operations'])
                    ops[name] = {key: summary([statistics.mean(
                        rank_data[r]['iterations'][i]['operations'].get(name, {}).get(key, 0)
                        for r in ranks) for i in range(10)]) for key in measures}
                cohort_data[label] = dict(iteration_ms=summary(per_iteration), phases=phase_stats,
                                          operations=ops)
            windows = {name: timing(log_path, first, last) for name, first, last in
                       [('pre', 29, 39), ('recorded', 40, 50), ('post', 51, 61)]}
            pair[case] = dict(log_windows=windows, cohorts=cohort_data, buckets=buckets,
                              loss_first=float(losses[0]), loss_last=float(losses[-1]))
            (out / f'{tag}_{case}_ranks.json').write_text(json.dumps(dict(ranks=rank_data, phases=phases)) + '\n')
        if len(pair) != 2:
            continue
        pair['parity'] = {window: pair['healthy']['log_windows'][window]['mean_ms'] /
                          pair['nep']['log_windows'][window]['mean_ms']
                          for window in ('pre', 'recorded', 'post')}
        pair['owner_parity'] = pair['healthy']['cohorts']['owners']['iteration_ms']['mean'] / pair['nep']['cohorts']['owners']['iteration_ms']['mean']
        result[tag] = pair
        print(tag, 'logged_parity', pair['parity'], 'owner_parity', pair['owner_parity'], flush=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Completed comparisons:', list(result))


if __name__ == '__main__':
    main()
