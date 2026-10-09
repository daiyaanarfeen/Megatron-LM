"""Plot parity separately from absolute overhead and profiled delivery waiting."""

import csv
import json
import statistics as st
import sys
from pathlib import Path

from report_nep_delivery_factorial import Figure


def main(job: str) -> None:
    """Publish only the complete matched three-repeat sequence comparison."""
    base = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen')
    root = base / 'slurm_runs/nep_cleanup_ab' / job
    data = json.loads((root / 'sequence_hiding/summary.json').read_text())
    assert data['complete'] and len(data['comparisons']) == 18
    out = base / 'plots' / f'nep_sequence_hiding_{job}'
    out.mkdir(exist_ok=True)
    rows = []
    sequences = (8192, 4096, 2048)
    for seq in sequences:
        for profiled in (False, True):
            matches = [
                r for r in data['comparisons'] if r['sequence'] == seq and r['profiled'] == profiled
            ]
            assert len(matches) == 3
            row = dict(sequence=seq, profiled=profiled)
            for key in ('healthy_ms', 'nep_ms', 'extra_ms', 'parity'):
                values = [r[key] for r in matches]
                row.update(
                    {key: st.mean(values), key + '_min': min(values), key + '_max': max(values)}
                )
            if profiled:
                for topology in ('healthy', 'nep'):
                    row[topology + '_later_wait_ms'] = st.mean(
                        r[topology + '_delivery_wait_ms']['later_parameter_wait_ms']
                        for r in matches
                    )
            rows.append(row)
    fig = Figure(1440, 1030)
    fig.text(40, 36, 'Sequence length: parity, absolute overhead, and actual delivery stalls', 25)
    fig.text(
        40, 66, 'a8b hybrid/MTP: 15 MoE layers | H4608 / latent1152 / expert F3072 / E512 / top-k6'
    )
    fig.text(
        40,
        92,
        'GB300 | healthy EP16/16, NEP EP16/12 | TP2 / CP1 | MBS2 / GA8 | DistOpt + HybridEP + graphs',
    )
    fig.text(
        40,
        118,
        'Local BS128/128 healthy,128/96 NEP; GBS256/224 | dense buckets3/3, expert5/3 | prefetch1 / early delivery',
    )
    fig.text(
        40,
        144,
        'Same allocation; three counterbalanced repeats; matched iterations25-30 after24 warmup. Whiskers: repeat range.',
    )
    colors = {False: '#287b82', True: '#ba643d'}
    panels = (
        ('Owner iteration parity (%)', 'parity', 180, 87, 100.5),
        (
            'Absolute NEP-minus-healthy iteration time (ms)',
            'extra_ms',
            445,
            0,
            max(r['extra_ms_max'] for r in rows) * 1.2,
        ),
    )
    for title, key, top, lower, upper in panels:
        fig.text(40, top + 15, title, 21)
        y0, height = top + 210, 165
        for tick in range(5):
            value = lower + (upper - lower) * tick / 4
            y = y0 - height * tick / 4
            fig.node('line', x1=100, x2=1360, y1=y, y2=y, stroke='#dbe1e5')
            fig.text(85, y + 4, f'{value:.1f}', 12, 'end')
        for index, seq in enumerate(sequences):
            center = 300 + index * 420
            for profiled in (False, True):
                row = next(r for r in rows if r['sequence'] == seq and r['profiled'] == profiled)
                x = center + (55 if profiled else -55)
                y = y0 - (row[key] - lower) / (upper - lower) * height
                fig.node('rect', x=x - 42, y=y, width=84, height=y0 - y, fill=colors[profiled])
                lo = y0 - (row[key + '_min'] - lower) / (upper - lower) * height
                hi = y0 - (row[key + '_max'] - lower) / (upper - lower) * height
                fig.node('line', x1=x, x2=x, y1=hi, y2=lo, stroke='#20272b', stroke_width=2)
                fig.text(x, hi - 8, f'{row[key]:.2f}', 15, 'middle')
            fig.text(center, y0 + 25, f'S{seq}', 16, 'middle')
    fig.text(40, 745, 'Later-bucket parameter-ready waiting in first forward (profiled, ms)', 21)
    fig.text(
        40,
        773,
        'Excludes the initial bucket. These are consumer dependency stalls, not pure transfer service times.',
        15,
    )
    for index, seq in enumerate(sequences):
        row = next(r for r in rows if r['sequence'] == seq and r['profiled'])
        x = 300 + index * 420
        fig.text(x, 817, f'S{seq}', 19, 'middle')
        fig.text(x, 850, f"Healthy: {row['healthy_later_wait_ms']:.2f} ms", 19, 'middle')
        fig.text(x, 882, f"NEP: {row['nep_later_wait_ms']:.2f} ms", 19, 'middle')
    for x, profiled, label in ((45, False, 'Timing companion'), (285, True, 'Profiled run')):
        fig.node('rect', x=x, y=917, width=17, height=17, fill=colors[profiled])
        fig.text(x + 26, 931, label)
    fig.text(
        40,
        968,
        'Keep measurement modes separate: profiler intrusion changes the measured NEP gap. No post-profile substitution.',
        15,
    )
    fig.text(
        40,
        994,
        'Sequence changes compute shapes and token traffic; parameter bytes and scheduling stay fixed. Parity alone does not prove hiding.',
        15,
    )
    fig.save(out / 'sequence_hiding.svg')
    (out / 'results.json').write_text(
        json.dumps(dict(rows=rows, effects=data['effects']), indent=2) + '\n'
    )
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (out / 'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out / 'index.html').write_text(
        '<!doctype html><html><head><title>NEP sequence sensitivity</title></head><body>'
        '<h1>NEP sequence sensitivity</h1><p><a href="results.csv">CSV</a> | '
        '<a href="results.json">JSON</a> | <a href="README.md">Report</a></p>'
        '<img style="width:100%;max-width:1440px" src="sequence_hiding.svg"></body></html>'
    )
    print(out)


if __name__ == '__main__':
    main(sys.argv[1])
