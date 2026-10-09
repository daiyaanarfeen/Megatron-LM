"""Publish withdrawal notices, never figures, for invalid broadcast ablations."""

import argparse
import html
import json
from pathlib import Path


ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs')
JOBS = {
    '3181167': 'nep_cleanup_ab',
    '3181303': 'nep_cleanup_ab',
    '3181315': 'nep_cleanup_ab',
    '3181288': 'soft_clique_baseline',
    '3181404': 'nep_cleanup_ab',
    '3181439': 'soft_clique_baseline',
}
DATE = '2026-09-29'
REASON = (
    'Parameter redistribution broadcast every offloaded expert of an owner to '
    'every participating follower, including experts that follower did not need. '
    'Both compact and spread placement used this incorrect transfer policy.'
)
NOTICE = (
    '# WITHDRAWN: NEP placement and transfer-scheduling ablations\n\n'
    f'Withdrawn {DATE} at the user request.\n\n{REASON}\n\n'
    'All performance figures, tables, and conclusions from this ablation set '
    'are withdrawn as paper evidence. This includes scheduling comparisons '
    'performed on the same broadcast implementation. The previous recommendation '
    'of compact over spread placement is retracted.\n\n'
    'Raw logs, traces, and analysis summaries are retained only for provenance '
    'and debugging. Successful execution and parameter-value tests did not '
    'validate that only necessary data crossed each GPU pair.\n\n'
    'New measurements require destination-specific payloads, validation of '
    'per-pair expert identities and bytes, and fresh placement/scheduling '
    'comparisons. Do not reuse these timings for that implementation.\n'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mark-source', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = dict(status='withdrawn', date=DATE, reason=REASON, usable_for_paper=False)
    for job, family in JOBS.items():
        # Replace old URLs so detached downloads cannot appear to be valid results.
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="240" '
            'viewBox="0 0 1500 240"><rect width="100%" height="100%" fill="white"/>'
            '<g font-family="Arial,sans-serif" fill="#202020">'
            f'<text x="40" y="65" font-size="30">WITHDRAWN: job {job}</text>'
            '<text x="40" y="115" font-size="21">Incorrect broadcast payloads: '
            'followers received experts they did not need.</text>'
            '<text x="40" y="155" font-size="21">This figure is removed from '
            'the paper evidence. Measurements must be repeated after correction.</text>'
            f'<text x="40" y="200" font-size="18">{DATE}</text></g></svg>\n'
        )
        (args.output / f'{job}.svg').write_text(svg)
        for path in args.output.glob(f'{job}-*.json'):
            path.write_text(json.dumps(dict(status, job=job), indent=2) + '\n')
        if args.mark_source:
            root = ROOT / family / job
            assert root.is_dir(), root
            (root / 'WITHDRAWN.md').write_text(NOTICE)
            (root / 'result_status.json').write_text(
                json.dumps(dict(status, job=job), indent=2) + '\n'
            )
    for name in ('report.md', 'experiment_record.md', 'WITHDRAWN.md'):
        (args.output / name).write_text(NOTICE)
    (args.output / 'result_status.json').write_text(
        json.dumps(dict(status, jobs=list(JOBS)), indent=2) + '\n'
    )
    (args.output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>WITHDRAWN: NEP ablations</title>'
        '<main style="font:18px Arial;max-width:1000px;margin:40px auto">'
        '<h1>WITHDRAWN: NEP placement and scheduling results</h1>'
        f'<p>{html.escape(REASON)}</p>'
        '<p>All figures, tables and paper conclusions from this set have been removed. '
        'The compact-versus-spread recommendation is retracted.</p>'
        '<p>Raw traces remain only for debugging. New results require corrected '
        'destination-specific transfers and new measurements.</p>'
        f'<p>Withdrawn {DATE}. <a href="report.md">Withdrawal record</a></p></main>\n'
    )
    print(f'Published withdrawal notices: {args.output}')


if __name__ == '__main__':
    main()
