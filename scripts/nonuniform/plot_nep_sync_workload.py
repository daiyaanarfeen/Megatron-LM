"""Render measured microbatch envelopes, not inferred exposed collective time."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from plot_nep_failure_categories import load_font

COLORS = ('#48755a', '#2677b8', '#bf722d')
ROLES = ('Healthy full', 'NEP full owners', 'NEP reduced owners')


def render(source: Path, output: Path) -> None:
    data = json.loads(source.read_text())
    cases = list(data['cases'].items())
    width = max(2400, 590 * len(cases) + 100)
    image = Image.new('RGB', (width, 1540), 'white')
    draw = ImageDraw.Draw(image)

    def text(x, y, value, size=24, bold=False, anchor=None):
        font = load_font(size, bold)
        box = draw.textbbox((x, y), value, font=font, anchor=anchor)
        assert box[0] >= 0 and box[2] <= width and box[1] >= 0 and box[3] < image.height, value
        draw.text((x, y), value, fill='#202b32', font=font, anchor=anchor)

    text(45, 22, 'NEP first-forward / final-backward workload ablation', 40, True)
    text(45, 82, 'Anchor a770: 15 Mamba + 15 MoE + 3 GQA | H1280, expert F768, shared F1536, E128, top-k6 | TP1 CP1', 27)
    text(45, 125, 'Anchor MBS2, seq8192; GA3 fixed | GB200, EP8/8 vs EP8/6; local BS48/48 vs48/36 | DistOpt + Flex; requested DDP8', 26)
    text(45, 165, f'Job {data["job"]}; measured iterations40-49 | CUDA-event expert envelopes; not kernel busy time or exposed NCCL', 25)
    for i, (role, color) in enumerate(zip(ROLES, COLORS)):
        x = 200 + i * 610
        draw.rectangle((x, 215, x + 24, 239), fill=color)
        text(x + 38, 211, role, 25, True)
    series = []
    for tag, pair in cases:
        series.append([
            pair['healthy']['cohorts']['full_owners'],
            pair['nep']['cohorts']['full_owners'],
            pair['nep']['cohorts']['second_replica_owners'],
        ])
    cell = (width - 100) / len(cases)
    titles = {'anchor_start': 'Anchor start', 'anchor_end': 'Anchor repeat',
              'expert_f1536': 'Expert F768 ->1536', 'shared_f3072': 'Shared F1536 ->3072',
              'mbs1': 'MBS2 ->1 [nonstationary]', 'seq4096': 'Sequence8192 ->4096', 'topk2': 'Top-k6 ->2'}
    for col, (tag, pair) in enumerate(cases):
        center = 50 + cell * (col + .5)
        text(center, 268, titles.get(tag, tag), 28, True, 'mt')
        h_gbs = pair['healthy']['manifest_rank0']['environment']['TRUE_GLOBAL_BATCH_SIZE']
        n_gbs = pair['nep']['manifest_rank0']['environment']['TRUE_GLOBAL_BATCH_SIZE']
        text(center, 310, f'Parity {100 * pair["comparison"]["owner_parity"]:.2f}% | GBS {h_gbs}/{n_gbs}', 25, True, 'mt')
        counts = [int(line.rsplit(':', 1)[-1]) for line in pair['healthy']['bucket_log']
                  if 'Number of buckets' in line]
        owner_groups = int(pair['nep']['cohorts']['full_owners']['operations']['nep.stage_grads']['count']['mean'])
        text(center, 346, f'Delta +{pair["comparison"]["delta_iteration_ms"]:.2f} ms | D{counts[0]} E{counts[1]}/{owner_groups}', 23, False, 'mt')

    for row, (keys, labels, title) in enumerate((
        (('first_forward', 'later_forward_mean'), ('First', 'Later mean'), 'Forward: expert-bounded GPU elapsed span (ms)'),
        (('final_backward', 'earlier_backward_mean'), ('Final', 'Earlier mean'), 'Backward: expert-bounded GPU elapsed span (ms)'),
    )):
        top = 455 + 425 * row
        bottom = top + 260
        text(45, top - 57, title, 29, True)
        maximum = max(role['comparisons'][key]['mean'] + role['comparisons'][key]['sd']
                      for column in series for role in column for key in keys)
        upper = math.ceil(maximum * 1.16 / 10) * 10
        for col, column in enumerate(series):
            left, right = 50 + cell * col + 45, 50 + cell * (col + 1) - 30
            y = lambda v: bottom - (bottom - top) * v / upper
            for tick in range(5):
                value = upper * tick / 4
                draw.line((left, y(value), right, y(value)), fill='#dce2e6', width=2)
                text(left - 8, y(value), f'{value:.0f}', 18, anchor='rm')
            for index, (key, label) in enumerate(zip(keys, labels)):
                center = left + (right - left) * (.27 if index == 0 else .77)
                for r, color in enumerate(COLORS):
                    value = column[r]['comparisons'][key]
                    mean, sd = value['mean'], value['sd']
                    x = center + (r - 1) * 52
                    draw.rectangle((x - 18, y(mean), x + 18, bottom), fill=color)
                    draw.line((x, y(max(0, mean - sd)), x, y(mean + sd)), fill='#202b32', width=2)
                    draw.line((x - 6, y(mean + sd), x + 6, y(mean + sd)), fill='#202b32', width=2)
                    text(x, y(mean + sd) - 7 - (18 if r == 1 else 0), f'{mean:.1f}', 17, True, 'ms')
                text(center, bottom + 15, label, 23, True, 'mt')

    text(45, 1230, 'Additional full-owner temporal deltas (NEP minus healthy; ms)', 28, True)
    for col, (_, pair) in enumerate(cases):
        center = 50 + cell * (col + .5)
        delta = pair['comparison']['delta_temporal_envelopes_ms']
        for i, (key, label) in enumerate((
            ('before_first_expert', 'Iteration prefix'),
            ('last_backward_to_optimizer', 'Last expert backward -> optimizer'),
            ('optimizer', 'Optimizer'),
        )):
            text(center, 1280 + i * 37, f'{label}: {delta[key]:+.2f}', 21, anchor='mt')
    text(45, 1420, 'Mean +/- SD over10 iterations; full ranks0-5, reduced8-13. D = dense buckets; E = healthy expert / NEP owner groups.', 23)
    note = ('MBS1 measured timing is nonstationary; its apparent parity improvement is not a reliable general conclusion.'
            if 'mbs1' in data['cases'] else
            'Repeated anchors quantify drift. GPU envelopes are not causal attribution; wider shared experts also increase dense traffic.')
    text(45, 1460, note, 22)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.source, args.output)
