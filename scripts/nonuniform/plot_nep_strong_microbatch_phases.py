"""Plot literal first/last microbatch envelopes from existing all-rank profiles."""

import csv
import json
import math
import re
import statistics
from pathlib import Path

from PIL import Image, ImageDraw

from analyze_cuda_event_mbs_pair import ROOT, analyze
from plot_nep_failure_categories import load_font

OUT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/plots/nep_strong_microbatch_phases_20260915')
SLUG = 'scaling_ladder__a770m_4b_moe_215b'
ROLES = ('Healthy', 'NEP full owners', 'NEP reduced owners')
COLORS = ('#48755a', '#2677b8', '#bf722d')
PHASES = {
    'first_forward': 'mb0_first_forward -> mb0_last_forward',
    'last_forward': 'mb2_first_forward -> mb2_last_forward',
    'first_backward': 'mb0_first_backward -> mb0_last_backward',
    'last_backward': 'mb2_first_backward -> mb2_last_backward',
}
SWEEPS = (
    ('expert_width', 'Expert-width sweep', '3063815', 'sync_workload_analysis',
     ('anchor_start', 'expert_f3072', 'expert_f3072_seq2048', 'anchor_end')),
    ('expert_count', 'Expert-count sweep', '3064000', 'ordinary_experts_analysis',
     ('anchor_start', 'e512', 'e512_seq2048', 'anchor_end')),
    ('hidden_width', 'Hidden-width / expert-count sweep', '3064218', 'ordinary_hidden_analysis',
     ('anchor_start', 'h320', 'h320_e512', 'anchor_end')),
    ('sequence_length', 'Sequence-length sweep', '3063804', 'ordinary_strong_analysis',
     ('anchor_start', 'seq2048', 'seq1024', 'anchor_end')),
)
TITLES = {
    'anchor_start': 'Original: before sweep', 'anchor_end': 'Original: after sweep',
    'expert_f3072': 'Expert F: 768 -> 3072',
    'expert_f3072_seq2048': 'F3072 + sequence2048',
    'e512': 'Experts: 128 -> 512', 'e512_seq2048': '512 experts + sequence2048',
    'h320': 'Hidden H: 1280 -> 320', 'h320_e512': 'H320 + 512 experts',
    'seq2048': 'Sequence: 8192 -> 2048', 'seq1024': 'Sequence: 8192 -> 1024',
}


def collect(job: str, tag: str, source_pair: dict) -> dict:
    """Average ranks within each iteration, then summarize the ten iterations."""
    run = ROOT / job / tag / 'cupti/ep8' / SLUG
    series = {}
    for role, case, ranks in zip(ROLES, ('healthy', 'nep', 'nep'), (range(6), range(6), range(8, 14))):
        records = [analyze(run / case / job / f'cupti_profile/rank-{rank}.json', 40, 50, case)
                   for rank in ranks]
        if case == 'nep':
            assert all(sample['operations']['nep.stage_grads']['count'] == 8
                       for record in records for sample in record['iterations'])
        metrics = {}
        for key, span in dict(PHASES, iteration=None).items():
            samples = [statistics.mean(
                record['iterations'][i]['temporal_spans_ms'][span] if span else record['iterations'][i]['range_ms']
                for record in records) for i in range(10)]
            assert all(math.isfinite(v) and v > 0 for v in samples)
            metrics[key] = dict(mean=statistics.mean(samples), sd=statistics.stdev(samples), samples=samples)
        series[role] = metrics
    parity = series[ROLES[0]]['iteration']['mean'] / series[ROLES[1]]['iteration']['mean']
    recorded = source_pair.get('owner_parity', source_pair.get('comparison', {}).get('owner_parity'))
    assert math.isclose(parity, recorded, abs_tol=1e-10), (job, tag, parity, recorded)
    manifest = json.loads((run / 'healthy' / job / 'launch_manifest/rank-0.json').read_text())
    options = manifest['options']
    config = {name: int(options[options.index(flag) + 1]) for name, flag in (
        ('H', '--hidden-size'), ('F', '--ffn-hidden-size'), ('E', '--num-experts'), ('seq', '--seq-length'))}
    log = (run / 'healthy' / job / 'driver.log').read_text()
    buckets = [int(v) for v in re.findall(r'Number of buckets for gradient all-reduce / reduce-scatter: (\d+)', log)]
    assert len(buckets) == 2
    nep_log = (run / 'nep' / job / 'driver.log').read_text()
    nep_buckets = [int(v) for v in re.findall(r'Number of buckets for gradient all-reduce / reduce-scatter: (\d+)', nep_log)]
    assert nep_buckets[:2] == buckets and nep_buckets[2:] == [1] * 8
    return dict(job=job, tag=tag, title=TITLES[tag], config=config, series=series,
                owner_parity=parity, healthy_buckets=buckets, nep_expert_groups=8)


def render(cases: list[dict], title: str, output: Path) -> None:
    """Render paired first/last bars with common scales within each direction."""
    width = 100 + 600 * len(cases)
    im = Image.new('RGB', (width, 1460), 'white')
    draw = ImageDraw.Draw(im)

    def text(x, y, value, size=23, bold=False, anchor=None):
        font = load_font(size, bold)
        box = draw.textbbox((x, y), value, font=font, anchor=anchor)
        assert 0 <= box[0] and box[2] <= width and 0 <= box[1] and box[3] < im.height, value
        draw.text((x, y), value, fill='#202b32', font=font, anchor=anchor)

    text(45, 20, f'NEP first / last microbatch phases | {title}', 37, True)
    text(45, 78, 'a770-family: 15 Mamba + 15 MoE + 3 GQA | shared F1536, top-k6 | H / expert F / expert count / sequence below', 25)
    text(45, 117, 'MBS2, GA3 | EP8/8 healthy vs EP8/6 NEP | GBS96/84; replica BS48/48 vs48/36 | TP1 CP1 PP1 ETP1', 25)
    text(45, 156, 'GB200, segment4; cross-replica RDMA | Flex/HybridEP + DistOpt | Same measured iterations40-49; mean +/- SD', 25)
    for r, (role, color) in enumerate(zip(ROLES, COLORS)):
        x = 360 + r * 620
        draw.rectangle((x, 212, x + 26, 238), fill=color)
        text(x + 39, 207, role, 26, True)
    cell = (width - 100) / len(cases)
    for col, case in enumerate(cases):
        center = 50 + cell * (col + .5)
        c = case['config']
        text(center, 268, case['title'], 26, True, 'mt')
        text(center, 308, f'H{c["H"]} / F{c["F"]} / E{c["E"]} / seq{c["seq"]}', 23, anchor='mt')
        text(center, 344, f'Owner parity {100 * case["owner_parity"]:.2f}% | job {case["job"]}', 23, True, 'mt')
        dense, expert = case['healthy_buckets']
        text(center, 380, f'Dense H/N {dense}/{dense}; expert H/N {expert}/8', 21, anchor='mt')

    for row, (keys, labels, heading) in enumerate((
        (('first_forward', 'last_forward'), ('First forward', 'Last forward'), 'Forward GPU elapsed span (ms)'),
        (('first_backward', 'last_backward'), ('First backward', 'Last backward'), 'Backward GPU elapsed span (ms)'),
    )):
        top, bottom = 490 + row * 425, 765 + row * 425
        text(45, top - 61, heading, 28, True)
        maximum = max(c['series'][r][k]['mean'] + c['series'][r][k]['sd']
                      for c in cases for r in ROLES for k in keys)
        upper = math.ceil(maximum * 1.18 / 20) * 20
        for col, case in enumerate(cases):
            left, right = 50 + cell * col + 45, 50 + cell * (col + 1) - 30

            def y(value):
                return bottom - value / upper * (bottom - top)

            for tick in range(5):
                value = upper * tick / 4
                draw.line((left, y(value), right, y(value)), fill='#dce2e6', width=2)
                text(left - 8, y(value), f'{value:.0f}', 18, anchor='rm')
            for group, (key, label) in enumerate(zip(keys, labels)):
                center = left + (right - left) * (.26 if group == 0 else .76)
                for r, (role, color) in enumerate(zip(ROLES, COLORS)):
                    m = case['series'][role][key]
                    mean, sd = m['mean'], m['sd']
                    x = center + (r - 1) * 64
                    draw.rectangle((x - 20, y(mean), x + 20, bottom), fill=color)
                    draw.line((x, y(max(0, mean - sd)), x, y(mean + sd)), fill='#202b32', width=2)
                    for value in (max(0, mean - sd), mean + sd):
                        draw.line((x - 6, y(value), x + 6, y(value)), fill='#202b32', width=2)
                    text(x, y(mean + sd) - 8 - (19 if r == 1 else 0), f'{mean:.1f}', 17, True, 'ms')
                text(center, bottom + 15, label, 22, True, 'mt')
    text(45, 1270, 'First / last = microbatch1 / microbatch3, not layers; middle microbatch omitted. Each bar is ONE microbatch.', 24)
    text(45, 1310, 'Spans: first grouped-expert GPU operation to last in that phase, including intervening work and waits; not summed GEMM time.', 23)
    text(45, 1350, 'Post-backward optimizer/tail excluded. Temporal spans do not isolate transfer cost or explain cross-replica readiness by themselves.', 23)
    text(45, 1390, 'Rank means: healthy0-5, NEP full0-5, NEP reduced8-13. Whiskers: SD across10 iteration means. Expert H/N = actual sync groups.', 22)
    im.save(output.with_suffix('.png'))
    im.save(output.with_suffix('.pdf'), resolution=160)
    print(output.with_suffix('.png'), flush=True)


def main() -> None:
    """Validate source parity and publish plots plus their exact numerical data."""
    OUT.mkdir(parents=True, exist_ok=True)
    all_cases = []
    for filename, title, job, analysis, tags in SWEEPS:
        source = json.loads((ROOT / job / analysis / 'summary.json').read_text())
        pairs = source.get('cases', source)
        cases = [collect(job, tag, pairs[tag]) for tag in tags]
        render(cases, title, OUT / filename)
        all_cases.extend(cases)
    lookup = {(c['job'], c['tag']): c for c in all_cases}
    selection = [('3063815', 'expert_f3072'), ('3063815', 'expert_f3072_seq2048'),
                 ('3064000', 'e512'), ('3064000', 'e512_seq2048'), ('3064218', 'h320_e512')]
    render([lookup[key] for key in selection], 'Five reported low-parity configurations', OUT / 'overview')
    (OUT / 'plotted_values.json').write_text(json.dumps(all_cases, indent=2) + '\n')
    with (OUT / 'plotted_values.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['job', 'case', 'role', 'phase', 'mean_ms', 'sd_ms'] + [f'iteration{i}_ms' for i in range(40, 50)])
        for c in all_cases:
            for role, phases in c['series'].items():
                for phase, values in phases.items():
                    writer.writerow([c['job'], c['tag'], role, phase, values['mean'], values['sd']] + values['samples'])
    print(f'Validated {len(all_cases)} cases; literal first/last microbatches, all source parities match.')


if __name__ == '__main__':
    main()
