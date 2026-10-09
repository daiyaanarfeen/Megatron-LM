"""Publish scheduling effects without adding overlapping communication spans."""

import csv
import json
import statistics as st
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CELLS = ('H0', 'H1P', 'Q00', 'Q01', 'Q10', 'Q11')
LABELS = {
    'H0': 'Healthy / no prefetch',
    'H1P': 'Healthy / prefetch',
    'Q00': 'No prefetch / demand release',
    'Q01': 'No prefetch / early release',
    'Q10': 'Prefetch / demand release',
    'Q11': 'Prefetch / early release',
}


class Figure:
    def __init__(self, width=1440, height=940):
        self.svg = ET.Element(
            'svg',
            xmlns='http://www.w3.org/2000/svg',
            width=str(width),
            height=str(height),
            viewBox=f'0 0 {width} {height}',
        )
        self.node('rect', width=width, height=height, fill='white')

    def node(self, tag, **attrs):
        return ET.SubElement(self.svg, tag, {k.replace('_', '-'): str(v) for k, v in attrs.items()})

    def text(self, x, y, value, size=16, anchor='start'):
        self.node(
            'text',
            x=x,
            y=y,
            font_family='Arial, sans-serif',
            font_size=size,
            text_anchor=anchor,
            fill='#252b30',
        ).text = str(value)

    def save(self, path):
        ET.ElementTree(self.svg).write(path, encoding='utf-8', xml_declaration=True)
        ET.parse(path)


def main(job):
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab') / job
    data = json.loads((root / 'delivery_factorial/summary.json').read_text())
    assert json.loads((root / 'work-controls-audit.json').read_text())['status'] == 'PASS'
    assert len(data['cases']) == 24, 'Do not publish an incomplete factorial study'
    out = (
        Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/plots') / f'nep_delivery_factorial_{job}'
    )
    out.mkdir(parents=True, exist_ok=True)
    lookup = {
        (v['spec']['cell'], int(k.split('_')[3][1:]), v['profiled']): (k, v)
        for k, v in data['cases'].items()
    }
    rows = []
    for cell in CELLS:
        profiles = [lookup[cell, r, True][1] for r in (0, 1)]
        companions = [lookup[cell, r, False][1] for r in (0, 1)]
        before, after = [], []
        for profile in profiles:
            deps = [x for x in profile['bucket_dependencies'] if x['role'] == 'owners']
            before.append(sum(x['wait_before_ag_completion_ms']['mean'] for x in deps))
            after.append(sum(x['wait_after_ag_completion_ms']['mean'] for x in deps))
        first = [p['cohorts']['owners']['first_forward_ms']['mean'] for p in profiles]
        hc = cell if cell.startswith('H') else 'H0' if cell[1] == '0' else 'H1P'
        times = [p['timing']['mean'] for p in companions]
        profile_times = [p['timing']['mean'] for p in profiles]
        parity = [100 * lookup[hc, r, False][1]['timing']['mean'] / times[r] for r in (0, 1)]
        fixed_parity = [
            100 * lookup['H1P', r, False][1]['timing']['mean'] / times[r] for r in (0, 1)
        ]
        rows.append(
            dict(
                cell=cell,
                label=LABELS[cell],
                companion_ms=st.mean(times),
                companion_repeat_ms=times,
                profile_ms=st.mean(profile_times),
                profile_repeat_ms=profile_times,
                matched_policy_parity=st.mean(parity),
                fixed_healthy_parity=st.mean(fixed_parity),
                first_forward_ms=st.mean(first),
                first_forward_repeat_ms=first,
                wait_before_ag_ms=st.mean(before),
                wait_after_ag_ms=st.mean(after),
                remaining_first_forward_ms=st.mean(first) - st.mean(before) - st.mean(after),
            )
        )
    f = Figure()
    f.text(40, 38, 'What parameter-delivery scheduling hides', 27)
    f.text(
        40,
        69,
        'a8b hybrid/MTP, 15 MoE layers | H4608 / latent1152 / expert F3072 / E512 / top-k6',
        17,
    )
    f.text(
        40,
        96,
        'S4096 | MBS2 / GA8 | TP2 / CP1 / ETP1 | EP16/16 healthy, EP16/12 NEP | DistOpt + HybridEP + graphs',
        16,
    )
    f.text(
        40,
        123,
        'Local batches128/128 vs128/96; GBS256 vs224 | actual dense buckets3/3, expert5/3 | owner ranks0-11',
        16,
    )
    f.text(
        40,
        150,
        'Steps25-30 after24 warmup; two reversed-order repeats. Points are repeat means. Same resources within study.',
        15,
    )
    colors = ('#497da4', '#bd6c42', '#bac6cc')
    legend = (
        'Consumer wait while own AG unfinished',
        'Remaining parameter-ready wait',
        'Rest of first forward (not pure compute)',
    )
    top, bottom, left, width = 230, 590, 95, 1250
    limit = max(r['first_forward_ms'] for r in rows) * 1.25
    y = lambda value: bottom - value / limit * (bottom - top)
    f.text(40, 190, 'Profiled first forward: disjoint compute-stream intervals (ms)', 21)
    for tick in range(6):
        value = limit * tick / 5
        f.node('line', x1=left, x2=left + width, y1=y(value), y2=y(value), stroke='#dce1e5')
        f.text(left - 12, y(value) + 4, f'{value:.0f}', 13, 'end')
    for i, row in enumerate(rows):
        x = left + (i + 0.5) * width / len(rows)
        base = 0
        for key, color in zip(
            ('wait_before_ag_ms', 'wait_after_ag_ms', 'remaining_first_forward_ms'), colors
        ):
            value = row[key]
            assert value >= 0
            f.node(
                'rect',
                x=x - 35,
                y=y(base + value),
                width=70,
                height=y(base) - y(base + value),
                fill=color,
            )
            base += value
        for dx, value in zip((-5, 5), row['first_forward_repeat_ms']):
            f.node('circle', cx=x + dx, cy=y(value), r=3, fill='#222222')
        f.text(x, y(base) - 12, f'{base:.1f}', 15, 'middle')
        f.text(x, 617, row['cell'], 17, 'middle')
        f.text(x, 645, f"Iter {row['companion_ms']:.1f} ms", 14, 'middle')
        f.text(
            x,
            666,
            (
                'Healthy control'
                if row['cell'].startswith('H')
                else f"Parity {row['fixed_healthy_parity']:.2f}%"
            ),
            14,
            'middle',
        )
    for i, (color, label) in enumerate(zip(colors, legend)):
        f.node('rect', x=45, y=696 + i * 25, width=15, height=15, fill=color)
        f.text(68, 709 + i * 25, label, 15)
    for i, row in enumerate(rows):
        f.text(45 + (i % 2) * 700, 797 + (i // 2) * 27, f"{row['cell']}: {row['label']}", 15)
    f.text(
        40,
        900,
        'Iteration/parity annotations use separate timing companions; all NEP parity uses the same prefetched healthy reference H1P.',
        15,
    )
    f.text(
        40,
        924,
        'The waiting bars are dependency spans, not transfer service time; they are not added to companion iteration savings.',
        14,
    )
    f.save(out / 'delivery_effects.svg')

    timeline = Figure(height=1000)
    timeline.text(
        40, 37, 'Parameter readiness: same rank and iteration across all four schedules', 25
    )
    timeline.text(
        40,
        65,
        'Rank0, iteration27, repeat0 | time relative to first-forward GPU start | elapsed ranges include queueing/waits',
        16,
    )
    traces = {}
    for cell in ('Q00', 'Q01', 'Q10', 'Q11'):
        name = lookup[cell, 0, True][0]
        deps = json.loads((root / 'delivery_factorial' / f'{name}-dependencies.json').read_text())
        traces[cell] = [r for r in deps if r['rank'] == 0 and r['iteration'] == 27]
    limit = max(r['first_forward_ms'] for ds in traces.values() for r in ds) * 1.04
    x = lambda value: 240 + value / limit * 1120
    for i, (cell, ds) in enumerate(traces.items()):
        top = 118 + i * 205
        timeline.text(40, top, f'{cell}: {LABELS[cell]}', 18)
        for j, d in enumerate(ds):
            y0 = top + 27 + j * 48
            timeline.text(55, y0 + 10, f'Expert bucket {d["bucket"]}', 15)
            for prefix, yy, color in [('ag', y0, '#497da4'), ('red', y0 + 15, '#bd6c42')]:
                timeline.node(
                    'rect',
                    x=x(d[prefix + '_start_ms']),
                    y=yy,
                    width=max(0.5, x(d[prefix + '_end_ms']) - x(d[prefix + '_start_ms'])),
                    height=11,
                    fill=color,
                )
            for key, color in [('demand_ms', '#252b30'), ('ready_ms', '#2c885c')]:
                timeline.node(
                    'line',
                    x1=x(d[key]),
                    x2=x(d[key]),
                    y1=y0 - 4,
                    y2=y0 + 29,
                    stroke=color,
                    stroke_width=2,
                )
        timeline.node(
            'line',
            x1=x(ds[0]['first_forward_ms']),
            x2=x(ds[0]['first_forward_ms']),
            y1=top + 15,
            y2=top + 175,
            stroke='#a5394d',
            stroke_dasharray='5 4',
        )
    for i in range(7):
        value = limit * i / 6
        timeline.text(x(value), 945, f'{value:.0f} ms', 14, 'middle')
    timeline.text(
        40,
        982,
        'Blue: AG input-ready/completion | orange: redistribution | black: demand | green: consumer released | dashed: forward end',
        15,
    )
    timeline.save(out / 'delivery_timeline.svg')
    tradeoff = json.loads((root / 'delivery_factorial/early-release-tradeoff.json').read_text())
    change = {
        key: st.mean(tradeoff[str(r)]['early_minus_consumer']['totals'][key] for r in (0, 1))
        for key in (
            'parameter_wait_ms',
            'native_graph_elapsed_ms',
            'other_first_forward_ms',
            'first_forward_ms',
        )
    }
    bridge = Figure(height=660)
    bridge.text(40, 40, 'Where the early-redistribution first-forward savings go', 25)
    bridge.text(
        40,
        72,
        'Q10 -> Q11: same AG prefetch, change only redistribution release | owners0-11, both repeats, profiled steps25-30',
        16,
    )
    labels = [
        ('parameter_wait_ms', 'Parameter-ready wait'),
        ('native_graph_elapsed_ms', 'Native graph execution'),
        ('other_first_forward_ms', 'Other first-forward time'),
        ('first_forward_ms', 'Net first-forward change'),
    ]
    zero, scale = 945, 53
    bridge.node('line', x1=zero, x2=zero, y1=110, y2=430, stroke='#42484d')
    for i, (key, label) in enumerate(labels):
        value = change[key]
        yy = 130 + i * 75
        bridge.text(40, yy + 27, label, 19)
        bridge.node(
            'rect',
            x=min(zero, zero + scale * value),
            y=yy,
            width=abs(value) * scale,
            height=38,
            fill='#357c75' if value < 0 else '#b66b43',
        )
        bridge.text(
            zero + scale * value + (-10 if value < 0 else 10),
            yy + 26,
            f'{value:+.2f} ms',
            18,
            'end' if value < 0 else 'start',
        )
    graph_changes = [
        tradeoff[str(r)]['early_minus_consumer']['totals']['native_graph_elapsed_ms']
        for r in (0, 1)
    ]
    graph_parts = {
        label: st.mean(
            tradeoff[str(r)]['early_minus_consumer']['native_graph_partition_ms'][label]
            for r in (0, 1)
        )
        for label in ('model_compute', 'model_EP')
    }
    bridge.text(
        40,
        473,
        f'Native graph change by repeat: {graph_changes[0]:+.2f} / {graph_changes[1]:+.2f} ms.',
        18,
    )
    bridge.text(
        40,
        507,
        f"Its partition shifts mainly into native compute ({graph_parts['model_compute']:+.2f} ms) and model-EP ({graph_parts['model_EP']:+.2f} ms).",
        17,
    )
    bridge.text(
        40,
        540,
        'Graph gaps and remaining categories partly offset those increases. No extra model kernels are assumed.',
        16,
    )
    bridge.text(
        40,
        583,
        'Disjoint phase accounting: parameter waits do not overlap native graph ranges (checked in every sample).',
        15,
    )
    bridge.text(
        40,
        612,
        'This explains the measured first-forward effect, not unprofiled kernel service or a hardware-bandwidth decomposition.',
        15,
    )
    bridge.save(out / 'early_release_tradeoff.svg')
    (out / 'results.json').write_text(
        json.dumps(
            dict(rows=rows, effects=data['factorial_comparisons'], tradeoff=tradeoff), indent=2
        )
        + '\n'
    )
    with (out / 'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'index.html').write_text(
        '<!doctype html><html><head><title>NEP delivery scheduling</title></head><body>'
        '<h1>NEP delivery scheduling attribution</h1><p><a href="results.csv">CSV</a> | '
        '<a href="results.json">JSON</a> | <a href="README.md">Full report and limitations</a></p>'
        '<img style="width:100%;max-width:1440px" src="delivery_effects.svg">'
        '<img style="width:100%;max-width:1440px" src="delivery_timeline.svg">'
        '<img style="width:100%;max-width:1440px" src="early_release_tradeoff.svg"></body></html>'
    )
    print(out)


if __name__ == '__main__':
    main(sys.argv[1])
