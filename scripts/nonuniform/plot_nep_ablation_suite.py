"""Consolidate accepted a8b ablations without pooling jobs or timing modes."""

import csv
import hashlib
import json
import statistics as st
from pathlib import Path

from report_nep_delivery_factorial import Figure

BASE = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen')
ROOT = BASE / 'slurm_runs/nep_cleanup_ab'
OUT = BASE / 'plots/nep_ablation_suite_20261006'
JOBS = {3215521: 28, 3215822: 16, 3238625: 16, 3238626: 32, 3240633: 24, 3241762: 36}
PANELS = [
    (
        'sequence',
        'Sequence length',
        3241762,
        'GB300 | MBS2 / GA8',
        [('8K', 'N8K', 'H8K'), ('4K', 'N4K', 'H4K'), ('2K', 'N2K', 'H2K')],
        'Parameter bytes fixed; compute shapes and token traffic change.',
    ),
    (
        'batching',
        'Fixed-work MBS / GA',
        3238626,
        'GB300 | S8192',
        [('1 / 16', 'M1', 'H1'), ('2 / 8', 'N8K', 'H8K'), ('4 / 4', 'M4', 'H4')],
        'MBS x GA = 16; healthy is matched to each MBS / GA.',
    ),
    (
        'allocation',
        'Reduced-replica GA',
        3215822,
        'GB300 | S8192 | MBS2; full GA8',
        [('8: proportional', 'B', 'H'), ('6: underload', 'R', 'H'), ('10: overload', 'O', 'H')],
        'Reduced local BS96 / 72 / 120; NEP GBS224 / 200 / 248.',
    ),
    (
        'delivery4k',
        'Delivery scheduling at 4K',
        3238625,
        'GB300 | S4096 | MBS2 / GA8',
        [('Demand / d1', 'D4K', 'H4K'), ('Early / d1', 'N4K', 'H4K'), ('Early / d2', 'P4K', 'H4K')],
        'd = AG prefetch depth; same healthy depth1 reference.',
    ),
    (
        'delivery8k',
        'Delivery scheduling at 8K',
        3238626,
        'GB300 | S8192 | MBS2 / GA8',
        [('Demand / d1', 'D8K', 'H8K'), ('Early / d1', 'N8K', 'H8K'), ('Early / d2', 'P8K', 'H8K')],
        'Small effects retained, including repeat-order reversals.',
    ),
    (
        'prefetch',
        'AG prefetch x release policy',
        3240633,
        'GB200 | S4096 | MBS2 / GA8',
        [
            ('d0 / demand', 'Q00', 'H0'),
            ('d0 / early', 'Q01', 'H0'),
            ('d1 / demand', 'Q10', 'H1P'),
            ('d1 / early', 'Q11', 'H1P'),
        ],
        'Healthy AG depth matches NEP; no fixed-prefetched denominator.',
    ),
    (
        'placement',
        'Placement / transfer order / streams',
        3215521,
        'GB300 | S8192 | MBS2 / GA8',
        [
            ('Compact / asc / 1', 'A', 'H'),
            ('Compact / disjoint / 1', 'B', 'H'),
            ('Compact / asc / 2', 'C', 'H'),
            ('Compact / disjoint / 2', 'D', 'H'),
            ('Spread / asc / 1', 'E', 'H'),
            ('Spread / asc / 2', 'F', 'H'),
        ],
        'Destination-exact payloads; both buffer lanes preallocated in every case.',
    ),
]
CONTRASTS = [
    ('Sequence: 8K -> 4K', 'sequence', 'N8K', 'N4K'),
    ('Sequence: 8K -> 2K', 'sequence', 'N8K', 'N2K'),
    ('MBS/GA: 2/8 -> 1/16', 'batching', 'N8K', 'M1'),
    ('MBS/GA: 2/8 -> 4/4', 'batching', 'N8K', 'M4'),
    ('Reduced GA: 8 -> 6', 'allocation', 'B', 'R'),
    ('Reduced GA: 8 -> 10', 'allocation', 'B', 'O'),
    ('4K: demand -> early', 'delivery4k', 'D4K', 'N4K'),
    ('4K: prefetch depth 1 -> 2', 'delivery4k', 'N4K', 'P4K'),
    ('8K: demand -> early', 'delivery8k', 'D8K', 'N8K'),
    ('8K: prefetch depth 1 -> 2', 'delivery8k', 'N8K', 'P8K'),
    ('AG depth 0 -> 1; both topologies', 'prefetch', 'Q00', 'Q10'),
    ('Early release; no AG prefetch', 'prefetch', 'Q00', 'Q01'),
    ('Early release; AG prefetch enabled', 'prefetch', 'Q10', 'Q11'),
    ('Compact: ascending -> disjoint; 1 stream', 'placement', 'A', 'B'),
    ('Compact: ascending -> disjoint; 2 streams', 'placement', 'C', 'D'),
    ('Compact: 1 -> 2 streams; ascending', 'placement', 'A', 'C'),
    ('Compact: 1 -> 2 streams; disjoint', 'placement', 'B', 'D'),
    ('Spread: 1 -> 2 streams', 'placement', 'E', 'F'),
    ('Compact -> spread; ascending / 1 stream', 'placement', 'A', 'E'),
    ('Compact -> spread; ascending / 2 streams', 'placement', 'C', 'F'),
]


def load_points() -> tuple[list, list]:
    """Validate source cases and pair native owner timings within each study."""
    sources, lookup = [], {}
    for job, count in JOBS.items():
        root = ROOT / str(job)
        source = root / 'actual-summary.json'
        data = json.loads(source.read_text())
        assert len(data['cases']) == count
        assert json.loads((root / 'work-controls-audit.json').read_text())['status'] == 'PASS'
        for name, case in data['cases'].items():
            assert (
                json.loads((root / name / 'actual-validation.json').read_text())['status'] == 'PASS'
            )
            spec = case['spec']
            assert (spec['profile_start'], spec['profile_end'], spec['train_iters']) == (25, 31, 35)
            assert spec['native_rank_timing']
            raw = [
                json.loads((root / name / f'native-timing-{r}.json').read_text()) for r in range(12)
            ]
            assert all(len(x) == 35 for x in raw)
            owner_mean = st.mean(st.mean(x[24:30]) for x in raw)
            assert abs(owner_mean - case['timing']['mean']) < 1e-6
            repeat = int(name.split('_')[3][1:])
            lookup[job, spec['cell'], repeat, case['profiled']] = (owner_mean, name, spec)
        sources.append(
            dict(
                job=job,
                path=str(source),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                cases=count,
            )
        )
    points = []
    for panel, title, job, metadata, cells, note in PANELS:
        for profiled in (False, True):
            for label, cell, healthy in cells:
                repeats = sorted(
                    k[2] for k in lookup if k[0] == job and k[1] == cell and k[3] == profiled
                )
                assert len(repeats) == (3 if job == 3241762 else 2)
                samples = []
                for repeat in repeats:
                    n, nname, nspec = lookup[job, cell, repeat, profiled]
                    h, hname, hspec = lookup[job, healthy, repeat, profiled]
                    assert nspec['seqlen'] == hspec['seqlen']
                    assert nspec['mbs'] == hspec['mbs'] and nspec['ga'] == hspec['ga']
                    samples.append(
                        dict(
                            repeat=repeat,
                            parity=100 * h / n,
                            healthy_ms=h,
                            nep_ms=n,
                            extra_ms=n - h,
                            healthy_case=hname,
                            nep_case=nname,
                        )
                    )
                points.append(
                    dict(
                        panel=panel,
                        title=title,
                        job=job,
                        metadata=metadata,
                        note=note,
                        cell=cell,
                        healthy_cell=healthy,
                        label=label,
                        profiled=profiled,
                        mean=st.mean(x['parity'] for x in samples),
                        low=min(x['parity'] for x in samples),
                        high=max(x['parity'] for x in samples),
                        samples=samples,
                    )
                )
    return points, sources


def heading(fig: Figure, title: str, profiled: bool) -> None:
    """Use common architectural and measurement labels on every figure."""
    fig.text(36, 35, title, 26)
    fig.text(
        36,
        64,
        'a8b hybrid/MTP, 15 MoE layers | H4608 / latent1152 / expert F3072 / E512 / top-k6',
        17,
    )
    fig.text(
        36,
        90,
        'Healthy EP16/16; NEP EP16/12 | TP2 / CP1 | DistOpt + Flex/HybridEP + CUDA graphs',
        17,
    )
    fig.text(
        36,
        116,
        'Default: MBS2 / GA8; local BS128/128 vs128/96; GBS256 vs224 | actual dense buckets3/3, expert5/3',
        16,
    )
    mode = (
        'PROFILED runs (instrumentation can change parity)'
        if profiled
        else 'TIMING COMPANIONS (separate runs, same measured steps)'
    )
    fig.text(36, 146, mode + ' | Owners0-11; steps25-30 after24 warmup', 16)


def panel(fig: Figure, points: list, x: int, y: int, width: int, low: float, high: float) -> None:
    """Horizontal bars share limits within each figure; dots are run repeats."""
    p = points[0]
    fig.text(x, y, p['title'], 22)
    fig.text(x, y + 26, p['metadata'], 15)
    fig.text(x, y + 49, f"Job {p['job']} | {len(p['samples'])} counterbalanced repeats", 14)
    left, right = x + 215, x + width - 85
    fig.text(right, y + 49, 'Owner iteration parity (%)', 14, 'end')
    top = y + 88

    def scale(value):
        return left + (value - low) / (high - low) * (right - left)

    for tick in range(6):
        value = low + (high - low) * tick / 5
        xx = scale(value)
        fig.node(
            'line',
            x1=xx,
            x2=xx,
            y1=top - 18,
            y2=top + 58 * (len(points) - 1) + 22,
            stroke='#dce2e6',
        )
        fig.text(xx, top - 28, f'{value:.0f}', 12, 'middle')
    for i, row in enumerate(points):
        yy = top + i * 58
        assert low <= row['low'] <= row['high'] <= high
        fig.text(x, yy + 5, row['label'], 15)
        fig.node(
            'rect', x=left, y=yy - 12, width=scale(row['mean']) - left, height=24, fill='#287b82'
        )
        fig.node(
            'line',
            x1=scale(row['low']),
            x2=scale(row['high']),
            y1=yy,
            y2=yy,
            stroke='#192f39',
            stroke_width=2,
        )
        for j, sample in enumerate(row['samples']):
            fig.node(
                'circle',
                cx=scale(sample['parity']),
                cy=yy + (j - (len(row['samples']) - 1) / 2) * 5,
                r=3,
                fill='#192f39',
            )
        fig.text(right + 10, yy + 5, f"{row['mean']:.2f}%", 16)
    fig.text(x, top + 58 * len(points) + 16, p['note'], 14)


def effects(points: list) -> list:
    """Compute paired parity differences, not differences of pooled timings."""
    result = []
    for profiled in (False, True):
        index = {(p['panel'], p['cell']): p for p in points if p['profiled'] == profiled}
        for label, name, first, last in CONTRASTS:
            a, b = index[name, first], index[name, last]
            assert a['job'] == b['job']
            av = {s['repeat']: s['parity'] for s in a['samples']}
            bv = {s['repeat']: s['parity'] for s in b['samples']}
            assert av.keys() == bv.keys()
            values = [bv[r] - av[r] for r in sorted(av)]
            result.append(
                dict(
                    label=label,
                    panel=name,
                    job=a['job'],
                    profiled=profiled,
                    before=first,
                    after=last,
                    mean=st.mean(values),
                    low=min(values),
                    high=max(values),
                    samples=values,
                )
            )
    return result


def effect_plot(rows: list, profiled: bool, zoom: bool) -> str:
    """Show large and small effects without hiding negative or null outcomes."""
    rows = [r for r in rows if r['profiled'] == profiled]
    if zoom:
        rows = [r for r in rows if r['low'] > -2 and r['high'] < 2]
    lo, hi = (-2, 2) if zoom else (-20, 4)
    height = 245 + 39 * len(rows) + 85
    fig = Figure(1500, height)
    heading(
        fig,
        (
            'Small / null parity effects (zoom)'
            if zoom
            else 'Ablation effects on owner iteration parity'
        ),
        profiled,
    )
    fig.text(
        36,
        179,
        'Positive = higher parity. Dots = paired repeats; line = repeat range, not a confidence interval.',
        15,
    )
    left, right, top = 625, 1285, 238
    scale = lambda value: left + (value - lo) / (hi - lo) * (right - left)
    ticks = (-2, -1, 0, 1, 2) if zoom else (-20, -15, -10, -5, -3, -2, 0, 2, 3, 4)
    for value in ticks:
        xx = scale(value)
        fig.node(
            'line',
            x1=xx,
            x2=xx,
            y1=top - 12,
            y2=top + 39 * len(rows) - 17,
            stroke='#6c777d' if value == 0 else '#dce2e6',
            stroke_dasharray='4 4' if abs(value) in (2, 3) else 'none',
        )
        fig.text(xx, top - 23, str(value), 12, 'middle')
    for i, row in enumerate(rows):
        yy = top + 39 * i
        fig.text(36, yy + 5, row['label'], 16)
        fig.text(575, yy + 5, row['job'], 12, 'end')
        color = '#287b82' if row['mean'] >= 0 else '#ba643d'
        fig.node(
            'line',
            x1=scale(row['low']),
            x2=scale(row['high']),
            y1=yy,
            y2=yy,
            stroke=color,
            stroke_width=3,
        )
        for j, value in enumerate(row['samples']):
            fig.node(
                'circle',
                cx=scale(value),
                cy=yy + (j - (len(row['samples']) - 1) / 2) * 5,
                r=4,
                fill=color,
            )
        fig.text(1320, yy + 5, f"{row['mean']:+.3f} pp", 16)
    fig.text(
        36,
        height - 55,
        'Every contrast is within one study/allocation. No cross-job subtraction. 2-3 pp is the requested practical threshold.',
        15,
    )
    fig.text(
        36,
        height - 29,
        'Parity changes are not automatically hiding improvements; workload changes can alter work and amortization.',
        15,
    )
    name = ('small_effects' if zoom else 'effects') + ('_profiled' if profiled else '_companion')
    fig.save(OUT / f'{name}.svg')
    return name


def main() -> None:
    """Generate reproducible figures and download tables from accepted sources."""
    OUT.mkdir(exist_ok=True)
    points, sources = load_points()
    contrasts = effects(points)
    names = {False: [], True: []}
    for profiled in (False, True):
        by_panel = {
            name: [p for p in points if p['panel'] == name and p['profiled'] == profiled]
            for name, *_ in PANELS
        }
        suffix = 'profiled' if profiled else 'companion'
        for group, title, selected, low in (
            (
                'workloads',
                'Workload ablations, including small effects',
                ('sequence', 'batching', 'allocation'),
                75,
            ),
            (
                'delivery',
                'Delivery scheduling and native prefetch ablations',
                ('delivery4k', 'delivery8k', 'prefetch'),
                90,
            ),
        ):
            height = 1230 if group == 'delivery' else 1175
            fig = Figure(1500, height)
            heading(fig, title, profiled)
            for i, name in enumerate(selected):
                panel(fig, by_panel[name], 36, 205 + i * 305, 1410, low, 100)
            fig.text(
                36,
                height - 31,
                'Whiskers/dots show repeat variation. Compare only within panels; allocation, hardware and runtime versions differ between studies.',
                15,
            )
            name = f'{group}_{suffix}'
            fig.save(OUT / f'{name}.svg')
            names[profiled].append(name)
        fig = Figure(1500, 740)
        heading(fig, 'Expert placement, owner order, and transfer streams', profiled)
        panel(fig, by_panel['placement'], 36, 208, 1410, 90, 100)
        fig.text(
            36,
            702,
            'Accepted exact-payload implementation only. Spread has larger profiling distortion; see the separate timing-companion chart.',
            15,
        )
        name = f'placement_{suffix}'
        fig.save(OUT / f'{name}.svg')
        names[profiled].append(name)
        names[profiled].extend(effect_plot(contrasts, profiled, zoom) for zoom in (False, True))
    (OUT / 'results.json').write_text(
        json.dumps(dict(points=points, effects=contrasts, sources=sources), indent=2) + '\n'
    )
    for name, data in (('parity', points), ('effects', contrasts)):
        with (OUT / f'{name}.csv').open('w') as stream:
            keys = [k for k in data[0] if k != 'samples'] + ['samples']
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({**r, 'samples': json.dumps(r['samples'])} for r in data)
    sections = []
    for mode, title in ((False, 'Timing companions'), (True, 'Profiled runs')):
        sections.append(f'<h2 id="{str(mode).lower()}">{title}</h2>')
        for name in names[mode]:
            sections.append(
                f'<p><a href="{name}.png">PNG</a> | <a href="{name}.svg">SVG</a></p>'
                f'<img loading="lazy" style="width:100%;max-width:1500px" src="{name}.svg">'
            )
    (OUT / 'index.html').write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>NEP ablation suite</title></head><body>'
        '<h1>NEP ablations: large, small, and null effects</h1>'
        '<p>Accepted actual a8b workload studies. Every ratio uses matched healthy and NEP owner measurements. '
        'No new jobs; no post-profile substitutions. Different jobs are not pooled.</p>'
        '<p>Raw tables and reproducible sources are retained on the cluster under '
        '/lustre/fsw/coreai_comparch_sysarch/darfeen/plots/nep_ablation_suite_20261006/.</p>'
        '<p>'
        '<a href="#true">Profiled-run figures</a></p>' + ''.join(sections) + '</body></html>'
    )
    report = [
        '# Consolidated NEP Ablation Plots',
        '',
        'Generated 2026-10-06 from existing accepted actual-a8b studies. No new GPU jobs.',
        'Includes small, null, and negative effects, not just favorable results.',
        '',
        '## Measurement',
        '',
        '- Parity is healthy owner iteration time / NEP full-replica owner iteration time.',
        '- Owners0-11, iterations25-30 after24 warmup; native per-rank timers.',
        '- Each ratio pairs the same study, repeat, workload and profiling mode.',
        '- Points/whiskers are two or three repeat values/ranges, not confidence intervals.',
        '- Companion and profiled results have separate figures. No post-profile substitution.',
        '- Every plotted source timing was checked against the raw native owner timers.',
        '- Source case and cross-case audits must pass before plotting.',
        '',
        '## Scope And Limits',
        '',
        'The model is the memory-limited 15-MoE a8b hybrid/MTP workload, not full-depth120B.',
        'H4608, latent1152, expertF3072,512experts,topk6;EP16/12 versus healthy16/16;',
        'TP2,CP1,DistOpt,Flex/HybridEP,native graphs. Actual dense3/3 and expert5/3 buckets',
        'are fixed per topology, not iso-bucket between healthy and NEP.',
        '',
        'Each panel is a separate study. Hardware, allocation and implementation revisions',
        'differ between panels. Never compare absolute heights across jobs as a causal effect.',
        'Every effect-chart subtraction is within one job, not across studies.',
        'Older studies use a different profiler configuration; profiling distortion can be',
        'policy-dependent, particularly for spread placement. Keep the two modes separate.',
        '',
        'The factorial AG-prefetch plot matches healthy depth0/depth1 to NEP. This deliberately',
        'avoids the earlier fixed-H1P denominator that made native prefetch look NEP-specific.',
        'Other delivery-depth comparisons retain their native healthy depth1 reference.',
        '',
        'Sequence changes token traffic and compute shapes. Work allocation changes total',
        'samples. Neither is a pure hiding-only intervention. Parity is not token throughput.',
        'As established in the sequence report, a large parity effect is not by itself proof',
        'of an equally large change in hidden communication.',
        '',
        'Parity bar axes are zoomed to75-100% or90-100%, with limits explicitly labeled.',
        'Use the effect figures for signed percentage-point differences and the small-effect',
        'zoom to inspect null results. The requested practical threshold is2-3pp, not a',
        'population-level statistical significance claim from two or three repeats.',
        '',
        'Excluded: failed/incomplete cases (including3215526), obsolete broadcast-based',
        'placement experiments, and four-layer toy studies. The latter have different',
        'workloads and timing protocols and are not silently mixed with actual-a8b results.',
        'No standalone transport-throttling or isolated interference replay has been completed;',
        'these figures do not invent results for those proposals.',
        '',
        '## Sources',
        '',
        '| Job | Accepted cases | Hardware |',
        '| --- | ---: | --- |',
    ]
    report.extend(
        f"| {s['job']} | {s['cases']} | {'GB200' if s['job'] == 3240633 else 'GB300'} |"
        for s in sources
    )
    report.extend(
        [
            '',
            '## Timing-Companion Effects',
            '',
            '| Intervention | Mean change (pp) | Paired-repeat range (pp) |',
            '| --- | ---: | ---: |',
        ]
    )
    report.extend(
        f"| {r['label']} | {r['mean']:+.4f} | {r['low']:+.4f} to {r['high']:+.4f} |"
        for r in contrasts
        if not r['profiled']
    )
    report.extend(
        [
            '',
            '## Reproduction',
            '',
            'Run `python3 scripts/nonuniform/plot_nep_ablation_suite.py` from the repository.',
            'The source archive includes that script and its SVG helper. Outputs are on Lustre.',
            '`results.json` retains individual paired timings, source case names, job IDs,',
            'source hashes and every plotted effect. CSV files expose the same numeric results.',
            'PNG and SVG downloads are provided for all figures.',
        ]
    )
    (OUT / 'README.md').write_text('\n'.join(report) + '\n')
    print(OUT)
    for row in contrasts:
        if not row['profiled']:
            print(f"{row['label']}: {row['mean']:+.4f} pp [{row['low']:+.4f}, {row['high']:+.4f}]")


if __name__ == '__main__':
    main()
