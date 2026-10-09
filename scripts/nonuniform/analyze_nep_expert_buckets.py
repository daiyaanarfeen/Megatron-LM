"""Require the complete 3/5/6-bucket study before reporting paired results."""

import json
import statistics
import sys
from pathlib import Path

from analyze_nep_actual_ablation import summarize
from analyze_nep_remaining_overheads import analyze


def main(root):
    expected = {
        f'controlled_actual_{cell}_r{rep}{suffix}_{topology}_moe'
        for cell, topology in [('H8K', 'healthy'), ('B3', 'nep'), ('B5', 'nep'), ('B6', 'nep')]
        for rep in range(2)
        for suffix in ('', '_off')
    }
    assert all((root / case / 'actual-validation.json').exists() for case in expected)
    assert len({(root / case / 'runtime.sha256').read_text() for case in expected}) == 1
    specs = [json.loads((root / case / 'actual-spec.json').read_text()) for case in expected]
    nep_specs = [
        {k: v for k, v in spec.items() if k not in ('cell', 'expert_bucket_groups')}
        for spec in specs
        if not spec['cell'].startswith('H')
    ]
    assert all(spec == nep_specs[0] for spec in nep_specs)
    summarize(root, allow_expert_rebucketing=True)
    source = json.loads((root / 'actual-summary.json').read_text())
    rows = {}
    phase_keys = (
        'first_forward_ms',
        'other_forward_ms',
        'nonfinal_backward_ms',
        'final_backward_ms',
        'outside_fb_ms',
    )
    for cell, topology in [('H8K', 'healthy'), ('B3', 'nep'), ('B5', 'nep'), ('B6', 'nep')]:
        row = {}
        for suffix, mode in [('', 'profiled'), ('_off', 'companion')]:
            paired = []
            for rep in range(2):
                h = source['cases'][f'controlled_actual_H8K_r{rep}{suffix}_healthy_moe']
                n = source['cases'][f'controlled_actual_{cell}_r{rep}{suffix}_{topology}_moe']
                hm, nm = h['timing']['mean'], n['timing']['mean']
                paired.append(
                    dict(
                        repeat=rep,
                        healthy_ms=hm,
                        case_ms=nm,
                        delta_ms=nm - hm,
                        parity=100 * hm / nm,
                        step_sd_ms=n['timing']['sd'],
                    )
                )
            row[mode] = dict(
                repeats=paired,
                case_ms=statistics.mean(x['case_ms'] for x in paired),
                delta_ms=statistics.mean(x['delta_ms'] for x in paired),
                parity=100
                * sum(x['healthy_ms'] for x in paired)
                / sum(x['case_ms'] for x in paired),
            )
        row['differential_instrumentation_ms'] = (
            row['profiled']['delta_ms'] - row['companion']['delta_ms']
        )
        raw = [
            record
            for rep in range(2)
            for rank in range(12)
            for record in analyze(root / f'controlled_actual_{cell}_r{rep}_{topology}_moe', rank)
        ]
        row['operation_range_means'] = {
            key: statistics.mean(x[key] for x in raw)
            for key in raw[0]
            if key not in ('rank', 'iteration')
        }
        row['profiled_phase_ms'] = {
            key: statistics.mean(
                source['cases'][f'controlled_actual_{cell}_r{rep}_{topology}_moe']['cohorts'][
                    'owners'
                ][key]['mean']
                for rep in range(2)
            )
            for key in phase_keys
        }
        rows[cell] = row
    for cell, row in rows.items():
        row['profiled_phase_delta_ms'] = {
            key: row['profiled_phase_ms'][key] - rows['H8K']['profiled_phase_ms'][key]
            for key in phase_keys
        }
        print(
            'BUCKET_RESULT',
            cell,
            json.dumps(
                {key: row[key] for key in ('profiled', 'companion', 'profiled_phase_delta_ms')}
            ),
            flush=True,
        )
    (root / 'expert-bucket-summary.json').write_text(
        json.dumps(
            dict(
                cases=rows,
                notes=[
                    'Owners0-11; steps25-30; two reversed-order repeats; no post-profile substitution.',
                    'Operation ranges include waits/queueing; overlapping ranges are not additive costs.',
                    'Healthy retains five native expert buckets; NEP count is the only workload change.',
                ],
            ),
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main(
        Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab') / sys.argv[1]
    )
