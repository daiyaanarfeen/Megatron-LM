"""Actual first-forward kernel overlap for placement/order ablations.

Kernel residency includes participant waiting. Exclusive union time is direct
rank-local exposure, not the complete causal iteration cost of redistribution.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_nep_ag_red_overlap import duration
from analyze_nep_first_forward_delivery import read_rank
from validate_nep_parameter_waves import inspect


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
    out = root / 'delivery_analysis'
    out.mkdir(exist_ok=True)
    steps = list(range(20, 28) if args.hard else range(15, 21))
    result = {}
    signatures = {}
    runs = sorted(root.glob('cleanup_ab_ablation_*/nsys/ep16/a8b_120b_latentmoe_1t/*/' + args.job)) if args.hard else sorted(p for p in root.glob('ablation_*') if p.is_dir())
    for run in runs:
        label = run.parents[4].name.removeprefix('cleanup_ab_') + '_' + run.parent.name + '_moe' if args.hard else run.name
        if args.cases and label not in args.cases:
            continue
        healthy = '_healthy_' in label
        samples, ready, signatures_by_rank = [], defaultdict(list), None
        if not healthy:
            signatures_by_rank, waves = inspect(run, steps, '_current_ascending_' not in label and '_spread_ascending_' not in label)
            signatures[label] = signatures_by_rank
        for rank in range(args.full):
            cache = out / f'{label}-rank-{rank}.json'
            if not cache.exists() or not json.loads(cache.read_text()).get('memory_staging_labeled'):
                cache.write_text(json.dumps(read_rank(run, rank, True, steps=steps)) + '\n')
            data = json.loads(cache.read_text())
            for row in data['rows']:
                a, b = row['start'], row['end']
                intervals = defaultdict(list)
                for kernel in row['kernels']:
                    intervals[kernel['label']].append((max(a, kernel['start']), min(b, kernel['end'])))
                shuffle = intervals['NEP_broadcast'] + intervals['NEP_staging']
                other = [interval for name, values in intervals.items() if name not in ('NEP_broadcast', 'NEP_staging') for interval in values]
                overlapped = [(max(x, u), min(y, v)) for x, y in shuffle for u, v in other]
                samples.append(dict(rank=rank, step=row['iteration'], first_forward_ms=b-a,
                                    redistribution_union_ms=duration(shuffle),
                                    redistribution_overlapped_ms=duration(overlapped),
                                    redistribution_exclusive_ms=duration(shuffle)-duration(overlapped),
                                    expert_ag_union_ms=duration(intervals['expert_AG']),
                                    model_ep_union_ms=duration(intervals['model_EP'])))
                if shuffle:
                    ready[row['iteration']].append(dict(rank=rank, completion_ms=max(y for x, y in shuffle), origin_ms=data['origin_ns']/1e6, clock_error_us=data['clock_error_ns']/1000))
        cohorts = {}
        for name, ids in [('owners', range(args.reduced)), ('followers', range(args.reduced, args.full))]:
            rows = [row for row in samples if row['rank'] in ids]
            cohorts[name] = {key: statistics.mean(row[key] for row in rows) for key in rows[0] if key not in ('rank', 'step')}
        result[label] = dict(cohorts=cohorts, samples=samples, readiness=ready,
                               note='Non-additive GPU kernel unions; includes waiting; not pure transport time.')
        print(label, cohorts, flush=True)
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for placement in ('current', 'spread'):
        for scope in ('moe', 'dropless'):
            a = signatures.get(f'ablation_{placement}_ascending_nep_{scope}')
            b = signatures.get(f'ablation_{placement}_waves_nep_{scope}')
            if a is not None and b is not None:
                assert a == b, f'Changed collective calls/payloads for {placement} order ablation'
                print('ORDER_PAYLOADS_MATCH', placement, scope, flush=True)


if __name__ == '__main__':
    main()
