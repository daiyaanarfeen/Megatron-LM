"""Matched sequence sensitivity: keep timing modes and exposure metrics separate."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analyze_nep_actual_ablation import stats, summarize
from analyze_nep_ag_red_overlap import duration
from analyze_nep_delivery_factorial import (
    aggregate,
    event_dependencies,
    intersection_duration,
    kernel_dependencies,
)

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab')
PHASES = (
    'first_forward_ms',
    'other_forward_ms',
    'nonfinal_backward_ms',
    'final_backward_ms',
    'outside_fb_ms',
)


def cached_owner_kernels(out: Path) -> dict:
    """Summarize first-repeat owner kernels; do not extrapolate to companions."""
    results = {}
    for cell, topology in (
        ('H8K', 'healthy'),
        ('N8K', 'nep'),
        ('H4K', 'healthy'),
        ('N4K', 'nep'),
        ('H2K', 'healthy'),
        ('N2K', 'nep'),
    ):
        name = f'controlled_actual_{cell}_r0_{topology}_moe'
        files = [out / f'{name}-rank-{rank}.json' for rank in range(12)]
        if not all(path.exists() for path in files):
            continue
        values = defaultdict(list)
        for path in files:
            for row in json.loads(path.read_text())['rows']:
                groups = defaultdict(list)
                for kernel in row['kernels']:
                    groups[kernel['label']].append(
                        (max(row['start'], kernel['start']), min(row['end'], kernel['end']))
                    )
                red = groups['NEP_broadcast'] + groups['NEP_staging']
                compute, ag = groups['model_compute'], groups['expert_AG']
                other = [
                    span
                    for label, spans in groups.items()
                    if label not in ('NEP_broadcast', 'NEP_staging')
                    for span in spans
                ]
                metrics = dict(
                    first_forward_ms=row['end'] - row['start'],
                    redistribution_union_ms=duration(red),
                    redistribution_exclusive_ms=duration(red) - intersection_duration(red, other),
                    redistribution_compute_overlap_ms=intersection_duration(red, compute),
                    expert_ag_union_ms=duration(ag),
                    ag_compute_overlap_ms=intersection_duration(ag, compute),
                    compute_union_ms=duration(compute),
                )
                for key, value in metrics.items():
                    values[key].append(value)
        results[cell] = {key: stats(samples) for key, samples in values.items()}
    return results


def main(job: str, partial: bool = False, kernels: bool = False) -> None:
    """Report paired run-level effects without treating ranks as repetitions."""
    root = ROOT / job
    summarize(root)
    data = json.loads((root / 'actual-summary.json').read_text())
    out = root / 'sequence_hiding'
    out.mkdir(exist_ok=True)
    if not partial:
        assert len(data['cases']) == 36, len(data['cases'])
    lookup = {}
    for name, row in data['cases'].items():
        spec = row['spec']
        repeat = int(name.split('_')[3][1:])
        healthy = spec['cell'].startswith('H')
        lookup[spec['seqlen'], repeat, row['profiled'], healthy] = row
        if not row['profiled']:
            continue
        cache = out / f'{name}-dependencies.json'
        if not cache.exists():
            cache.write_text(json.dumps(event_dependencies(root / name, spec)) + '\n')
        deps = json.loads(cache.read_text())
        row['bucket_dependencies'] = aggregate(deps, ('role', 'position'))
        samples = defaultdict(lambda: defaultdict(float))
        for dep in deps:
            sample = samples[dep['role'], dep['rank'], dep['iteration']]
            sample['parameter_wait_ms'] += dep['wait_ms']
            part = 'initial' if dep['position'] == 0 else 'later'
            sample[part + '_parameter_wait_ms'] += dep['wait_ms']
            for key in ('wait_before_ag_completion_ms', 'wait_after_ag_completion_ms'):
                if key in dep:
                    sample[key] += dep[key]
        row['delivery_waits'] = aggregate(
            [dict(role=k[0], rank=k[1], iteration=k[2], **v) for k, v in samples.items()], ('role',)
        )
        if kernels:
            row['kernel_unions'] = aggregate(kernel_dependencies(root / name, spec, out), ('role',))
    comparisons = []
    for (seq, repeat, profiled, healthy), row in sorted(lookup.items()):
        if healthy or (seq, repeat, profiled, True) not in lookup:
            continue
        h = lookup[seq, repeat, profiled, True]
        hm, nm = h['timing']['mean'], row['timing']['mean']
        result = dict(
            sequence=seq,
            repeat=repeat,
            profiled=profiled,
            healthy_ms=hm,
            nep_ms=nm,
            extra_ms=nm - hm,
            parity=100 * hm / nm,
        )
        if profiled:
            result['phase_excess_ms'] = {
                phase: row['cohorts']['owners'][phase]['mean']
                - h['cohorts']['owners'][phase]['mean']
                for phase in PHASES
            }
            for label, case in (('healthy', h), ('nep', row)):
                waits = next(x for x in case['delivery_waits'] if x['role'] == 'owners')
                result[label + '_delivery_wait_ms'] = {
                    k: v['mean'] for k, v in waits.items() if isinstance(v, dict)
                }
        comparisons.append(result)
        print('PAIRED', json.dumps(result), flush=True)
    effects = []
    for seq in (4096, 2048):
        for profiled in (False, True):
            matched = []
            for repeat in range(3):
                baseline = next(
                    (
                        x
                        for x in comparisons
                        if (x['sequence'], x['repeat'], x['profiled']) == (8192, repeat, profiled)
                    ),
                    None,
                )
                variant = next(
                    (
                        x
                        for x in comparisons
                        if (x['sequence'], x['repeat'], x['profiled']) == (seq, repeat, profiled)
                    ),
                    None,
                )
                if baseline is not None and variant is not None:
                    matched.append(baseline['parity'] - variant['parity'])
            if matched:
                effects.append(
                    dict(
                        sequence=seq,
                        profiled=profiled,
                        parity_drop_pp=stats(matched),
                        all_three_exceed_two_pp=len(matched) == 3 and min(matched) > 2,
                        all_three_exceed_three_pp=len(matched) == 3 and min(matched) > 3,
                    )
                )
    result = dict(
        cases=data['cases'],
        comparisons=comparisons,
        effects=effects,
        complete=len(data['cases']) == 36,
        first_repeat_owner_kernels=cached_owner_kernels(out),
        note='Sequence changes token traffic and compute shapes, not parameter bytes. '
        'Consumer waits are dependency spans, not pure transfer time. Timing '
        'companions and profiles are separate runs with matched iteration indices. '
        'Do not attribute companion latency using profiled phase arithmetic.',
    )
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('EFFECTS', json.dumps(effects, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--kernels', action='store_true')
    args = parser.parse_args()
    main(args.job, args.partial, args.kernels)
