"""Additive replica-progress accounting; not a causal critical-path attribution."""

import csv
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw

from analyze_cuda_event_mbs_pair import ROOT, analyze
from plot_nep_failure_categories import load_font
from plot_nep_strong_microbatch_phases import SLUG, SWEEPS

OUT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/plots/nep_accumulated_slowdown_20260915')
SOURCE = OUT.parent / 'nep_strong_microbatch_phases_20260915/plotted_values.json'
COMPONENTS = ('first_forward', 'other_forwards', 'earlier_backwards', 'remaining')
LABELS = ('First forward', 'All other forwards', 'All backwards except last',
          'Remaining: last backward + tail / gaps')
COLORS = ('#487fa8', '#e2a33e', '#579974', '#a2769b')
METHOD = """Accumulation-aware temporal accounting, NOT causal collective attribution.

All inputs are the existing targeted CPU/CUDA-event profiles, iterations 40-49.
No new jobs, corrected timings, or profiler-window substitutions are used.
Both healthy and NEP use cohorts ranks 0-5 and 8-13. These are cohort means,
not the last individual collective participant; full follower ranks 6-7 and
healthy second-replica ranks 14-15 are not in the cohort statistic.

At each expert-bounded phase endpoint, compute elapsed time from that rank's
own iteration start, average within each cohort, then take the maximum of the
two cohort means. This is a replica progress envelope, not a GPU busy timeline.
Local-start alignment avoids assuming calibrated cross-host epoch offsets;
it excludes actual iteration-start skew. A changing envelope leader is allowed.
This construction accounts for accumulated progress instead of adding reduced
duration penalties on top of a full-owner synchronization wait.

For each iteration, first-forward contribution is the NEP-minus-healthy
envelope advance from first expert forward start to last expert forward end
in MB0. Other forwards sum the corresponding advances in MB1 and MB2.
Earlier backwards sum the corresponding advances in MB0 and MB1.
Remaining = full-owner iteration difference minus those three contributions.
It includes last backward, post-backward optimizer/synchronization tail, and
all gaps outside the expert-bounded spans. It is NOT labeled pure last-backward
work or proven synchronization cost. A detailed residual breakdown is exported.

These four signed components add exactly to the measured full-owner iteration
gap. Every component is divided by the SAME mean healthy full-owner iteration
time. Slowdown = 100 * (NEP / healthy - 1), not 100 minus parity.
Negative components are retained below zero; black dots mark net slowdown.
The first three colors indicate where the progress envelope advances more,
not counterfactual causal overheads. Uninstrumented waits cannot be uniquely
assigned to their originating operation using these endpoint profiles.
"""


def points(record: dict) -> dict:
    """Recover the ordered local-elapsed endpoints from the audited partition."""
    result = {'iteration_start': 0.0}
    elapsed = 0.0
    for span, duration in record['temporal_spans_ms'].items():
        start, end = span.split(' -> ')
        assert abs(result[start] - elapsed) < 1e-6
        elapsed += duration
        result[end] = elapsed
    assert math.isclose(elapsed, record['range_ms'], abs_tol=1e-6)
    return result


def split(envelope: dict, owner_iteration: float) -> dict:
    """Partition the envelope, closing the tail at the full-owner iteration end."""
    durations = {}
    for mb in range(3):
        for direction in ('forward', 'backward'):
            value = (envelope[f'mb{mb}_last_{direction}']
                     - envelope[f'mb{mb}_first_{direction}'])
            assert value >= 0
            durations[f'{direction}_{mb}'] = value
    result = {
        'first_forward': durations['forward_0'],
        'other_forwards': durations['forward_1'] + durations['forward_2'],
        'earlier_backwards': durations['backward_0'] + durations['backward_1'],
        'last_backward': durations['backward_2'],
        'post_backward_tail': owner_iteration - envelope['mb2_last_backward'],
        'gaps': envelope['mb2_last_backward'] - sum(durations.values()),
    }
    assert abs(sum(result.values()) - owner_iteration) < 1e-6
    result['remaining'] = sum(result[k] for k in ('last_backward', 'post_backward_tail', 'gaps'))
    return result


def collect(source: dict) -> dict:
    """Use identical iterations and matched cohorts in both configurations."""
    job, tag = source['job'], source['tag']
    modes = {}
    for mode in ('healthy', 'nep'):
        path = ROOT / job / tag / 'cupti/ep8' / SLUG / mode / job / 'cupti_profile'
        groups = []
        for ranks in (range(6), range(8, 14)):
            records = [analyze(path / f'rank-{rank}.json', 40, 50, mode)['iterations'] for rank in ranks]
            grouped = []
            for i in range(10):
                ps = [points(rank[i]) for rank in records]
                grouped.append({k: statistics.mean(p[k] for p in ps) for k in ps[0]})
            groups.append(grouped)
        samples = []
        for i in range(10):
            full, other = groups[0][i], groups[1][i]
            envelope = {k: max(full[k], other[k]) for k in full}
            assert all(b >= a for a, b in zip(envelope.values(), list(envelope.values())[1:]))
            samples.append(dict(owner_iteration_ms=full['iteration_end'],
                                components_ms=split(envelope, full['iteration_end']),
                                full_ms=full, other_ms=other, envelope_ms=envelope))
        modes[mode] = samples
    healthy = statistics.mean(s['owner_iteration_ms'] for s in modes['healthy'])
    nep = statistics.mean(s['owner_iteration_ms'] for s in modes['nep'])
    assert math.isclose(healthy / nep, source['owner_parity'], abs_tol=1e-10)
    deltas = []
    for h, n in zip(modes['healthy'], modes['nep']):
        d = {k: n['components_ms'][k] - h['components_ms'][k] for k in h['components_ms']}
        d['total'] = n['owner_iteration_ms'] - h['owner_iteration_ms']
        assert abs(sum(d[k] for k in COMPONENTS) - d['total']) < 1e-6
        deltas.append(d)
    means = {k: statistics.mean(d[k] for d in deltas) for k in deltas[0]}
    percents = {k: 100 * v / healthy for k, v in means.items()}
    assert abs(sum(percents[k] for k in COMPONENTS) - percents['total']) < 1e-8
    print(job, tag, {k: round(percents[k], 2) for k in COMPONENTS + ('total',)}, flush=True)
    return dict(job=job, tag=tag, title=source['title'], config=source['config'],
                healthy_ms=healthy, nep_ms=nep, parity=healthy / nep,
                delta_ms=means, percent=percents, samples=deltas, modes=modes)


def render(cases: list[dict], title: str, output: Path, groups: list[tuple[str, int]] | None = None) -> None:
    """Draw signed stacks; total markers use net rather than positive height."""
    width, height = max(2000, (270 if groups else 370) * len(cases) + 220), 1280
    im = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(im)

    def text(x, y, value, size=23, bold=False, anchor=None):
        font = load_font(size, bold)
        bbox = draw.textbbox((x, y), value, font=font, anchor=anchor)
        assert min(bbox[:2]) >= 0 and bbox[2] <= width and bbox[3] <= height, value
        draw.text((x, y), value, font=font, fill='#243139', anchor=anchor)

    text(50, 25, f'NEP accumulated slowdown | {title}', 35, True)
    text(50, 83, 'a770-family: 15 Mamba + 15 MoE + 3 GQA | shared F1536, top-k6 | EP8/8 healthy vs EP8/6 NEP', 25)
    text(50, 126, 'MBS2, GA3 | GBS96/84; replica BS48/48 vs48/36 | TP1 CP1 PP1 ETP1 | Flex/HybridEP + DistOpt', 25)
    text(50, 169, 'GB200; replicas NVL-contained, cross-replica RDMA | Same measured iterations40-49 | No new jobs', 24)
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        x, y = 60 + (i % 2) * (width // 2), 230 + (i // 2) * 43
        draw.rectangle((x, y + 4, x + 24, y + 28), fill=color)
        text(x + 38, y, label, 24, True)
    text(60, 326, 'Slowdown (% of healthy iteration); black dot = full-owner total. Negative segments are speedups.', 24)
    low = min(sum(min(0, c['percent'][k]) for k in COMPONENTS) for c in cases)
    high = max(sum(max(0, c['percent'][k]) for k in COMPONENTS) for c in cases)
    low = min(-5, math.floor(low / 5) * 5 - 2)
    high = math.ceil(high / 5) * 5 + 5
    top, bottom, left, right = 440 if groups else 395, 930, 120, width - 60
    cell = (right - left) / len(cases)
    if groups:
        assert sum(count for _, count in groups) == len(cases)
        offset = 0
        for label, count in groups:
            start, end = left + offset * cell, left + (offset + count) * cell
            text((start + end) / 2, 380, label, 27, True, 'mt')
            draw.line((start + 18, 423, end - 18, 423), fill='#7b8b96', width=2)
            if offset:
                draw.line((start, 442, start, 1070), fill='#a7b4bd', width=2)
            offset += count

    def y(value):
        return bottom - (value - low) / (high - low) * (bottom - top)

    for tick in range(math.ceil(low / 5) * 5, math.floor(high / 5) * 5 + 1, 5):
        draw.line((left, y(tick), right, y(tick)), fill='#ccd5dc', width=3 if tick == 0 else 1)
        text(left - 15, y(tick), f'{tick}%', 21, anchor='rm')
    for i, case in enumerate(cases):
        x = left + cell * (i + .5)
        positive = negative = 0.0
        for key, color in zip(COMPONENTS, COLORS):
            value = case['percent'][key]
            base = positive if value >= 0 else negative
            end = base + value
            draw.rectangle((x - 66, min(y(base), y(end)), x + 66, max(y(base), y(end))), fill=color)
            if abs(y(base) - y(end)) > 27:
                text(x, (y(base) + y(end)) / 2, f'{value:+.1f}', 20, True, 'mm')
            if value >= 0:
                positive = end
            else:
                negative = end
        total = case['percent']['total']
        draw.line((x - 76, y(total), x + 76, y(total)), fill='#17232b', width=3)
        draw.ellipse((x - 5, y(total) - 5, x + 5, y(total) + 5), fill='#17232b')
        text(x, y(positive) - 15, f'{total:.1f}% slower', 24, True, 'mb')
        config = case['config']
        label = case['tag'].replace('anchor_start', 'Original before').replace('anchor_end', 'Original after')
        label = label.replace('expert_f', 'F').replace('_seq', ' / S').replace('h320_e512', 'H320 / E512')
        if groups and case['tag'] == 'anchor_start':
            label = 'Original unchanged'
        text(x, 960, label, 23, True, 'mt')
        text(x, 998, f'H{config["H"]} F{config["F"]} E{config["E"]}', 20, anchor='mt')
        text(x, 1030, f'S{config["seq"]}; parity {100 * case["parity"]:.1f}%', 20, anchor='mt')
    text(50, 1090, 'Accounting: at each boundary use the later of full/reduced cohort elapsed times; apply identically to healthy.', 23)
    text(50, 1130, 'This counts accumulated delay once. It is a progress-envelope decomposition, NOT a proven causal wait attribution.', 23, True)
    text(50, 1170, 'First 3 colors: expert-bounded spans. Remaining includes last backward, optimizer/sync tail and intervening gaps.', 23)
    text(50, 1210, 'Each stack sums exactly to owner slowdown, 100 x (NEP / healthy - 1). Signed details and methodology accompany plot.', 23)
    im.save(output.with_suffix('.png'))
    im.save(output.with_suffix('.pdf'), resolution=160)


def render_grouped(cases: list[dict]) -> None:
    """Put each sweep's own initial unchanged workload before its variants."""
    lookup = {(c['job'], c['tag']): c for c in cases}
    groups = []
    selected = []
    for _, title, job, _, tags in SWEEPS:
        initial_and_variants = tags[:-1]
        assert initial_and_variants[0] == 'anchor_start' and tags[-1] == 'anchor_end'
        selected.extend(lookup[job, tag] for tag in initial_and_variants)
        groups.append((title.replace(' sweep', ''), len(initial_and_variants)))
    render(selected, 'Grouped workloads with initial baselines', OUT / 'overview_grouped', groups)


def main() -> None:
    """Generate plots and auditable samples from existing profiles only."""
    OUT.mkdir(parents=True, exist_ok=True)
    sources = json.loads(SOURCE.read_text())
    cases = [collect(source) for source in sources]
    lookup = {(c['job'], c['tag']): c for c in cases}
    for name, title, job, _, tags in SWEEPS:
        render([lookup[job, tag] for tag in tags], title, OUT / name)
    selected = [('3063815', 'expert_f3072'), ('3063815', 'expert_f3072_seq2048'),
                ('3064000', 'e512'), ('3064000', 'e512_seq2048'),
                ('3064218', 'h320_e512'), ('3063804', 'seq2048'), ('3063804', 'seq1024')]
    render([lookup[key] for key in selected], 'All changed low-parity cases', OUT / 'overview')
    render_grouped(cases)
    (OUT / 'methodology.txt').write_text(METHOD)
    (OUT / 'plotted_values.json').write_text(json.dumps(cases, indent=2) + '\n')
    with (OUT / 'plotted_values.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['job', 'case', 'healthy_ms', 'nep_ms', 'component', 'delta_ms', 'percent_of_healthy'])
        for case in cases:
            for key, value in case['delta_ms'].items():
                writer.writerow([case['job'], case['tag'], case['healthy_ms'], case['nep_ms'],
                                 key, value, case['percent'][key]])
    print(f'Validated {len(cases)} cases; parity reproduced; all stacks reconcile exactly.')


if __name__ == '__main__':
    main()
