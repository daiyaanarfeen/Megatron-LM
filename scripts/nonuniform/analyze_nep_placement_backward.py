"""Final-backward/tail kernel exposure using the existing all-rank captures."""

import argparse
import json
import statistics
from pathlib import Path

from analyze_nep_ag_red_overlap import duration
from analyze_nep_first_forward_delivery import read_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs') / ('soft_clique_baseline' if args.hard else 'nep_cleanup_ab') / args.job
    full, reduced = (16, 12) if args.hard else (8, 6)
    steps = list(range(20, 28) if args.hard else range(15, 21))
    runs = sorted(root.glob('cleanup_ab_ablation_*/nsys/ep16/a8b_120b_latentmoe_1t/*/' + args.job)) if args.hard else sorted(p for p in root.glob('ablation_*') if p.is_dir())
    out = root / 'backward_analysis'
    out.mkdir(exist_ok=True)
    result = {}
    for run in runs:
        label = run.parents[4].name.removeprefix('cleanup_ab_') + '_' + run.parent.name + '_moe' if args.hard else run.name
        if args.cases and label not in args.cases:
            continue
        world = full * 2 if '_healthy_' in label else full + reduced
        samples = []
        for rank in range(world):
            cache = out / f'{label}-rank-{rank}.json'
            if not cache.exists() or not json.loads(cache.read_text()).get('memory_staging_labeled'):
                cache.write_text(json.dumps(read_rank(run, rank, True, steps, phase='final_backward_tail')) + '\n')
            data = json.loads(cache.read_text())
            for row in data['rows']:
                a, b = row['start'], row['end']
                kernels = [(max(a, k['start']), min(b, k['end']), k['label']) for k in row['kernels']]
                sample = dict(rank=rank, step=row['iteration'], final_backward_and_tail_ms=b-a)
                for name, categories in [('gather', {'NEP_gather'}), ('owner_staging', {'NEP_staging'}), ('expert_rs', {'expert_RS'}), ('model_ep', {'model_EP'})]:
                    target = [(x, y) for x, y, cat in kernels if cat in categories]
                    other = [(x, y) for x, y, cat in kernels if cat not in categories]
                    sample[name + '_union_ms'] = duration(target)
                    # |target \ other| = |target union other| - |other|.
                    sample[name + '_exclusive_ms'] = duration(target + other) - duration(other)
                samples.append(sample)
        cohorts = {}
        for name, ids in [('owners', range(reduced)), ('followers', range(reduced, full)), ('second', range(full, world))]:
            rows = [r for r in samples if r['rank'] in ids]
            cohorts[name] = {key: statistics.mean(r[key] for r in rows) for key in rows[0] if key not in ('rank', 'step')}
        result[label] = dict(cohorts=cohorts, samples=samples, note='Kernel unions include participant waits. Explicitly tagged owner staging excludes untagged Gather packing. Not additive causal costs.')
        print(label, cohorts['owners'], flush=True)
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
