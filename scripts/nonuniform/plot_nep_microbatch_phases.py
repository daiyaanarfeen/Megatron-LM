"""Plot first-forward and final-backward effects without mixing timing windows."""

import json
import math

from PIL import Image, ImageDraw

from analyze_nep_microbatch_phases import OUT
from plot_nep_failure_categories import load_font

COLORS = {'Healthy': '#567561', 'NEP full owner': '#2277AA', 'NEP reduced owner': '#BC7134'}


def render():
    rows = json.loads((OUT/'summary.json').read_text())
    im = Image.new('RGB', (2700, 1890), 'white')
    d = ImageDraw.Draw(im)

    def text(x, y, value, size=25, bold=False, anchor=None, color='#202c34'):
        f = load_font(size, bold)
        bounds = d.textbbox((x, y), value, font=f, anchor=anchor)
        assert 0 <= bounds[0] and bounds[2] <= im.width and 0 <= bounds[1] and bounds[3] < im.height, value
        d.text((x, y), value, font=f, anchor=anchor, fill=color)

    text(65, 25, 'Where NEP costs appear across gradient-accumulation microbatches', 43, True)
    text(65, 90, 'Same jobs as small_model_parity_final_ga | Existing PyTorch profiles, iterations 10-13 only | No CPU-cleanup changes', 27)
    text(65, 135, 'GPU elapsed envelopes, not summed kernel time. Bars: mean of 4 iterations; whiskers: sample SD. Rest = mean per microbatch.', 26)
    text(65, 180, 'Representative ranks: healthy rank 0; NEP full owner rank 0; NEP reduced owner rank 8 (small models) or 16 (a8b).', 26)
    for x, (role, color) in zip([700, 1190, 1720], COLORS.items()):
        d.rectangle((x, 240, x+28, 268), fill=color)
        text(x+42, 236, role, 27, True)
    text(65, 295, 'Small: GB200, EP8+8 / EP8+6, TP1 CP1, seq8192 GA3, 128 experts top-k6 | a8b: GB300, EP16+16 / EP16+12, TP2 CP1, seq8192 GA8', 24)
    text(65, 335, 'All: DistOpt + Flex/HybridEP; intra-replica NVLink, cross-replica RDMA. a500 H1024/F640; a770 H1280/F768; a8b H4608/latent1152.', 24)

    def chart(col, top, bottom, keys, labels):
        row = rows[col]
        left, right = 100+col*530, 535+col*530
        maximum = max(row['series'][role][k]['mean']+row['series'][role][k]['std'] for k in keys for role in COLORS)
        upper = max(5, math.ceil(maximum*1.25/5)*5)
        def y(v):
            return bottom-(bottom-top)*v/upper
        for j in range(5):
            v = upper*j/4
            d.line((left, y(v), right, y(v)), fill='#dfe5e7', width=2)
            text(left-8, y(v), f'{v:.0f}', 19, anchor='rm')
        centers = [(left+right)/2] if len(keys)==1 else [left+115, right-105]
        for key, label, center in zip(keys, labels, centers):
            for offset, (role, color) in zip([-56, 0, 56], COLORS.items()):
                value = row['series'][role][key]
                m, sd = value['mean'], value['std']
                x = center+offset
                d.rectangle((x-22, y(m), x+22, bottom), fill=color)
                d.line((x, y(max(0, m-sd)), x, y(m+sd)), fill='#26333b', width=2)
                d.line((x-7, y(m+sd), x+7, y(m+sd)), fill='#26333b', width=2)
                text(x, y(m+sd)-7, f'{m:.1f}', 17, True, anchor='ms')
            text(center, bottom+15, label, 22, True, anchor='mt')

    for col, row in enumerate(rows):
        center = 317+col*530
        if col:
            d.line((65+col*530, 393, 65+col*530, 1610), fill='#dfe5e7', width=2)
        text(center, 395, row['title'], 30, True, anchor='mt')
        h = (48 if row['ep']==8 else 128)*row['mbs']
        n = (42 if row['ep']==8 else 112)*row['mbs']
        text(center, 442, f'GA{row["ga"]}; GBS H/N {h}/{n}', 22, anchor='mt')
        text(center, 478, f'Job {row["job"]}', 21, anchor='mt', color='#59646a')
    text(65, 527, 'Forward + preparation: elapsed GPU span (ms)', 28, True)
    for col in range(5):
        chart(col, 595, 835, ['F1_span_ms', 'Frest_span_ms'], ['First', 'Later (mean)'])
    text(65, 905, 'Backward execution: elapsed GPU span, excluding gradient-sync/staging operations (ms)', 28, True)
    for col in range(5):
        chart(col, 980, 1220, ['Brest_span_ms', 'Blast_span_ms'], ['Earlier (mean)', 'Final'])
    text(65, 1290, 'After final backward: gradient-finalization tail before first optimizer GPU activity (ms)', 28, True)
    for col in range(5):
        chart(col, 1360, 1545, ['tail_ms'], ['Finalization tail'])
    d.line((65, 1630, 2640, 1630), fill='#c7d0d5', width=2)
    text(65, 1660, 'Forward includes input/preparation and parameter synchronization launched between backward calls; no explicit model-forward trace scope exists.', 25)
    text(65, 1700, 'Backward includes native token communication and internal stalls; excludes identified gradient collectives and NEP staging, whose interference can still delay it.', 25)
    text(65, 1740, 'Tail includes remaining synchronization, checks and host launch gaps, not just NEP communication. Axes are independently scaled; phase spans are not additive.', 25)
    text(65, 1780, 'Full profiling can distort healthy/NEP differently. These are measured-window observations, not post-profile timing estimates or all-rank averages.', 25)
    text(65, 1820, 'NEP parameter redistribution appears in first forward, with no calls in later forward microbatches. Individual microbatch data and compute sums are retained in JSON.', 25)
    im.save(OUT/'microbatch_phase_comparison.png')
    im.save(OUT/'microbatch_phase_comparison.pdf', resolution=160)


if __name__ == '__main__':
    render()
