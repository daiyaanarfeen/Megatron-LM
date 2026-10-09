"""Plot topology-controlled small-model parity without mixing timing windows."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from plot_nep_failure_categories import load_font

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen')
OUTPUT = ROOT / 'analysis/nep_small_model_parity_20260911'
COLORS = {'Before': '#B84C48', 'After': '#167B77', 'Original only': '#78828C'}


def load_rows():
    source = ROOT / 'analysis/profile_window_parity_20260910.json'
    old = {r['job']: r for r in json.loads(source.read_text())}
    event_path = ROOT / 'slurm_runs/soft_clique_baseline/3007029/cuda_event_analysis/summary.json'
    event = json.loads(event_path.read_text())

    def historical(job):
        paths = list((ROOT / 'slurm_runs/soft_clique_baseline' / job).glob(f'**/comparison_{job}.json'))
        assert len(paths) == 1
        post = json.loads(paths[0].read_text())
        return {'job': job, 'profiled': old[job]['parity_percent'],
                'post': post['comparison']['owner_iteration_time_parity_percent']}

    rows = [
        {'model': 'a500m / 2b', 'change': 'MBS1 -> MBS3; GA3 unchanged',
         'gbs': 'GBS H/N: 48/42 -> 144/126', 'dimensions': 'H1024, F640',
         'sampling': 'Full PyTorch profile: iterations 10-13',
         'post_sampling': 'Post-profile: iterations 15-24',
         'bars': [('Before', historical('2994105')), ('After', historical('2994279'))]},
        {'model': 'a770m / 4b', 'change': 'MBS1 -> MBS2; GA3 unchanged',
         'gbs': 'GBS H/N: 48/42 -> 96/84', 'dimensions': 'H1280, F768',
         'sampling': 'Low-overhead CUDA events: iterations 40-49',
         'post_sampling': 'Post-profile: iterations 51-60',
         'bars': [(label, {'job': '3007029', 'profiled': event[str(mbs)]['parity']['recorded'] * 100,
                          'post': event[str(mbs)]['parity']['post'] * 100})
                  for label, mbs in [('Before', 1), ('After', 2)]]},
    ]
    expected = [(73.82, 91.13), (88.36, 95.76)]
    for row, values in zip(rows, expected):
        assert tuple(round(b['profiled'], 2) for _, b in row['bars']) == values
    return rows


def render(rows, window):
    canvas = Image.new('RGB', (1800, 1160), 'white')
    draw = ImageDraw.Draw(canvas)

    def text(x, y, value, size=23, bold=False, anchor=None, color='#242B31', background=None):
        font = load_font(size, bold)
        bounds = draw.textbbox((x, y), value, font=font, anchor=anchor)
        assert bounds[0] >= 0 and bounds[2] <= 1800 and bounds[1] >= 0 and bounds[3] < 1160, value
        if background:
            draw.rectangle((bounds[0]-3, bounds[1]-2, bounds[2]+3, bounds[3]+2), fill=background)
        draw.text((x, y), value, font=font, fill=color, anchor=anchor)

    text(70, 30, 'NEP parity before and after increasing microbatch size', 38, True)
    title = 'Measured / profiled iterations' if window == 'profiled' else 'Post-profile iterations (separate timing window)'
    text(70, 88, title, 29, True)
    text(70, 138, 'Hybrid Mamba / attention / MoE | 128 experts, top-k 6 | sequence 8192 | GA3', 23)
    text(70, 176, 'GB200 | Healthy EP8+EP8 vs NEP EP8+EP6 | TP1, CP1 | DistOpt + Flex', 23)
    text(70, 216, 'Within-replica NVLink; cross-replica RDMA via separate software cliques', 21, color='#535E67')

    for x, label in [(1120, 'Before'), (1330, 'After')]:
        draw.rectangle((x, 264, x+23, 287), fill=COLORS[label])
        text(x+34, 260, label, 23)
    text(135, 271, 'Owner iteration-time parity (healthy = 100%)', 21)

    left, right, top, bottom = 140, 1700, 327, 837
    def y(value):
        return bottom - value / 110 * (bottom - top)

    for tick in range(0, 101, 20):
        yp = y(tick)
        draw.line((left, yp, right, yp), fill='#DFE4E8', width=2)
        text(left-20, yp, str(tick), 22, anchor='rm')
    for x in range(left, right, 18):
        draw.line((x, y(100), min(x+9, right), y(100)), fill='#727A81', width=2)
    draw.line((left, bottom, right, bottom), fill='#5E666D', width=2)

    for center, row in zip([570, 1300], rows):
        bars = row['bars']
        positions = [center] if len(bars) == 1 else [center-84, center+84]
        for x, (label, values) in zip(positions, bars):
            value = values[window]
            draw.rectangle((x-61, y(value), x+61, bottom-1), fill=COLORS[label])
            text(x, y(value)-14, f'{value:.2f}%', 26, True, anchor='ms', background='white')
        if len(bars) == 2:
            delta = bars[1][1][window] - bars[0][1][window]
            text(center, 310, f'+{delta:.2f} percentage points', 22, True, anchor='mt', color=COLORS['After'])
        else:
            text(center, 310, 'No tuned comparison', 22, anchor='mt', color='#535E67')
        text(center, 855, row['model'], 28, True, anchor='mt')
        text(center, 897, row['change'], 22, True, anchor='mt')
        text(center, 930, row['gbs'], 20, anchor='mt')
        text(center, 961, row['dimensions'], 20, anchor='mt', color='#535E67')
        text(center, 992, row['sampling' if window == 'profiled' else 'post_sampling'], 18, anchor='mt', color='#535E67')

    draw.line((70, 1040, 1730, 1040), fill='#DFE4E8', width=2)
    text(70, 1056, 'Both healthy and NEP use the labeled MBS; reduced-replica work remains proportional. No CPU-cleanup results included.', 20)
    text(70, 1087, 'a500 uses different allocations across MBS settings; a770 shares one allocation. Profiling methods differ between models.', 20)
    text(70, 1118, 'Full-profiler measurements have substantial overhead. Bar changes are observations, not isolated causal estimates.', 20)
    name = f'small_model_parity_{window}'
    canvas.save(OUTPUT / f'{name}.png')
    canvas.save(OUTPUT / f'{name}.pdf', resolution=150)


def main():
    rows = load_rows()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for window in ['profiled', 'post']:
        render(rows, window)
    (OUTPUT / 'data.json').write_text(json.dumps(rows, indent=2) + '\n')
    render_kernel_sums()
    print(OUTPUT)


def render_kernel_sums():
    """Label post-profile parity separately from measured kernel sums."""
    sums = json.loads((OUTPUT / 'final_ga_kernel_sums.json').read_text())
    reshard = json.loads((OUTPUT / 'reshard_arrival_estimates.json').read_text())
    edp = json.loads((OUTPUT / 'edp_arrival_adjusted_delta.json').read_text())
    param_ag = json.loads((OUTPUT / 'param_ag_arrival_adjusted_delta.json').read_text())
    replica = json.loads((OUTPUT / 'compute_replica_host_staging.json').read_text())
    profiled = {r['job']: r for r in json.loads(
        (ROOT / 'analysis/profile_window_parity_20260910.json').read_text())}
    parity = {job: row['parity_percent'] for job, row in profiled.items()}
    gaps = {job: row['nep']['mean_ms'] - row['healthy']['mean_ms'] for job, row in profiled.items()}
    for job in ('2994105', '2994279', '2995392', '2995524'):
        paths = list((ROOT / 'slurm_runs/soft_clique_baseline' / job).glob(f'**/comparison_{job}.json'))
        assert len(paths) == 1
        comparison = json.loads(paths[0].read_text())['comparison']
        parity[job] = comparison['owner_iteration_time_parity_percent']
        gaps[job] = comparison['latency_gap_ms']
    assert round(parity['2995392'], 2) == 85.79
    assert round(parity['2995524'], 2) == 93.50
    assert round(parity['2994105'], 2) == 82.28
    assert round(parity['2994279'], 2) == 94.75
    assert round(gaps['2994105'], 2) == 64.43
    assert round(gaps['2994279'], 2) == 38.77
    cases = [('2994105', 'a500 / 2b: MBS1', 'GBS H/N 48/42'),
             ('2994279', 'a500 / 2b: MBS3', 'GBS H/N 144/126'),
             ('2995392', 'a770 / 4b: MBS1', 'GBS H/N 48/42'),
             ('2995524', 'a770 / 4b: MBS2', 'GBS H/N 96/84')]
    large_path = ROOT / 'slurm_runs/soft_clique_baseline/2994648/ep16/a8b_120b_latentmoe_1t/comparison_2994648.json'
    large = json.loads(large_path.read_text())
    large_parity = large['comparison']['owner_iteration_time_parity_percent']
    large_gap = large['comparison']['latency_gap_ms']
    large_parts = json.loads((OUTPUT / 'a8b_plot_components.json').read_text())
    large_reshard = large_parts['nep']['communications']['reshard']['owner_mean']
    large_gs = sum(large_reshard[k]['adjusted_estimate_ms'] for k in ('gather', 'scatter'))
    large_edp = {kind: {variant: large_parts[variant]['communications'][kind]['owner_mean']['edp']['adjusted_estimate_ms']
                       for variant in ('healthy', 'nep')} for kind in ('grad_rs', 'param_ag')}
    large_delta = sum(v['nep'] - v['healthy'] for v in large_edp.values())
    assert large_delta >= 0, 'Negative delta needs a signed chart'
    large_host = large_parts['nep']['summary']['owners']['host_staging_ms']['host_idle_plus_exposed_staging']
    large_compute = [large_parts['healthy']['summary']['healthy']['final_ga_compute_ms'],
                     large_parts['nep']['summary']['full']['final_ga_compute_ms'],
                     large_parts['nep']['summary']['reduced']['final_ga_compute_ms']]
    assert round(large_parity, 2) == 93.89
    canvas = Image.new('RGB', (2400, 1510), 'white')
    draw = ImageDraw.Draw(canvas)

    def text(x, y, value, size=22, bold=False, anchor=None):
        font = load_font(size, bold)
        bounds = draw.textbbox((x, y), value, font=font, anchor=anchor)
        assert 0 <= bounds[0] and bounds[2] <= 2400 and 0 <= bounds[1] and bounds[3] < 1510, value
        draw.text((x, y), value, font=font, fill='#242B31', anchor=anchor)

    text(70, 25, 'Parity, host/staging exposure and replica compute', 34, True)
    text(70, 83, 'Parity and red gaps: post-profile 15-24 | GPU timings: profiled 10-13 | Blue: estimates', 23, True)
    text(70, 130, 'a500 / a770: Mamba / attention / MoE | 128 experts, top-k 6 | sequence 8192 | GA3', 23)
    text(70, 170, 'GB200 | Healthy EP8+8 vs NEP EP8+6 | TP1, CP1 | DistOpt + Flex', 23)
    text(70, 210, 'a500: H1024, F640, 12 MoE layers | a770: H1280, F768, 15 MoE layers', 23)
    text(70, 250, 'Within-replica NVLink; cross-replica RDMA via separate software cliques', 22)
    draw.line((1760, 120, 1760, 1470), fill='#DFE4E8', width=2)
    text(1830, 130, 'a8b / 120b latent MoE | Original workload', 23, True)
    text(1830, 170, 'GB300 | EP16+16 vs EP16+12 | TP2, CP1', 22)
    text(1830, 210, '512 experts | H4608 | latent width 1152', 22)
    text(1830, 250, 'MBS1, GA8, seq8192 | GBS H/N 128/112', 22)
    centers = [340, 700, 1100, 1460]
    text(70, 300, 'Owner parity (%) | All models: post-profile iterations 15-24', 24, True)
    top, bottom = 355, 610
    for tick in range(0, 101, 20):
        y = bottom - tick / 110 * (bottom - top)
        draw.line((150, y, 1690, y), fill='#DFE4E8', width=2)
        text(130, y, str(tick), 20, anchor='rm')
    for i, (center, (job, label, _)) in enumerate(zip(centers, cases)):
        y = bottom - parity[job] / 110 * (bottom - top)
        draw.rectangle((center-60, y, center+60, bottom), fill=COLORS['Before' if i % 2 == 0 else 'After'])
        text(center, y-8, f'{parity[job]:.2f}%', 24, True, anchor='ms')
        text(center, 625, label, 22, True, anchor='mt')
    for tick in range(0, 101, 20):
        y = bottom - tick / 110 * (bottom - top)
        draw.line((1830, y, 2330, y), fill='#DFE4E8', width=2)
    y = bottom - large_parity / 110 * (bottom - top)
    draw.rectangle((2020, y, 2140, bottom), fill=COLORS['Original only'])
    text(2080, y-8, f'{large_parity:.2f}%', 24, True, anchor='ms')
    text(2080, 625, 'a8b / 120b: MBS1 (original)', 22, True, anchor='mt')
    text(70, 685, 'Time (ms)', 23, True)
    comm_color, edp_color, host_color = '#347BA6', '#91BED8', '#164665'
    healthy_color, full_color, reduced_color, gap_color = '#647A6A', '#947441', '#D1AE71', '#B84C48'
    for x, yp, label, color in [(300, 685, 'Gather + Scatter*', comm_color), (660, 685, 'EDP RS+AG delta*', edp_color),
                              (1030, 685, 'NEP host-idle + exposed staging**', host_color),
                              (300, 723, 'Healthy compute', healthy_color), (660, 723, 'NEP full compute', full_color),
                              (1030, 723, 'NEP reduced compute', reduced_color), (1420, 723, 'Iteration gap', gap_color)]:
        draw.rectangle((x, yp+2, x+24, yp+26), fill=color)
        text(x+34, yp, label, 20)
    top, bottom = 790, 1080
    for tick in range(0, 151, 30):
        y = bottom - tick / 150 * (bottom - top)
        draw.line((150, y, 1690, y), fill='#DFE4E8', width=2)
        text(130, y, str(tick), 20, anchor='rm')
    for center, (job, label, gbs) in zip(centers, cases):
        values = sums[job]['mean_ms_per_owner']
        comm = reshard[job]['owner_mean']['total']
        estimate, raw = comm['adjusted_estimate_ms'], comm['raw_ms']
        assert 0 < estimate <= raw
        delta = edp[job]['delta_ms'] + param_ag[job]['delta_ms']
        # These four measured deltas are positive. Do not silently clip a future negative delta.
        assert delta >= 0, 'Negative EDP delta requires a signed/differencing chart'
        host = replica[job]['nep']['summary']['owners']['host_staging_ms']['host_idle_plus_exposed_staging']
        total = estimate + delta + host
        x = center-128
        y = bottom - estimate / 150 * (bottom - top)
        edp_y = bottom - (estimate + delta) / 150 * (bottom - top)
        total_y = bottom - total / 150 * (bottom - top)
        draw.rectangle((x-21, y, x+21, bottom), fill=comm_color)
        draw.rectangle((x-21, edp_y, x+21, y), fill=edp_color)
        draw.rectangle((x-21, total_y, x+21, edp_y), fill=host_color)
        text(x, total_y-8, f'{total:.1f}*', 18, True, anchor='ms')
        compute_values = [(-64, replica[job]['healthy']['summary']['healthy']['final_ga_compute_ms'], healthy_color),
                          (0, replica[job]['nep']['summary']['full']['final_ga_compute_ms'], full_color),
                          (64, replica[job]['nep']['summary']['reduced']['final_ga_compute_ms'], reduced_color)]
        for offset, value, color in compute_values:
            y = bottom - value / 150 * (bottom - top)
            draw.rectangle((center+offset-21, y, center+offset+21, bottom), fill=color)
            text(center+offset, y-8, f'{value:.1f}', 18, True, anchor='ms')
        value = gaps[job]
        y = bottom - value / 150 * (bottom - top)
        draw.rectangle((center+107, y, center+149, bottom), fill=gap_color)
        text(center+128, y-8, f'{value:.1f}', 18, True, anchor='ms')
        text(center, 1095, label, 22, True, anchor='mt')
        text(center, 1130, gbs, 20, anchor='mt')
        text(center, 1163, f'Job {job}', 19, anchor='mt')
        text(center, 1195, f'Stack: {estimate:.2f} + {delta:.2f} + {host:.2f} ms', 16, anchor='mt')
    text(1830, 685, 'Time (ms) | Same bar definitions and colors', 23, True)
    text(1830, 723, 'Separate scale | Final-GA backward = GA8', 21)
    large_total = large_gs + large_delta + large_host
    limit = (int(max(large_total, large_gap, *large_compute) / 100) + 1) * 100
    for tick in range(0, limit+1, 100):
        y = bottom - tick / limit * (bottom - top)
        draw.line((1870, y, 2350, y), fill='#DFE4E8', width=2)
        text(1850, y, str(tick), 20, anchor='rm')
    running = 0
    for value, color in [(large_gs, comm_color), (large_delta, edp_color), (large_host, host_color)]:
        y0 = bottom - running / limit * (bottom - top)
        running += value
        y1 = bottom - running / limit * (bottom - top)
        draw.rectangle((1880, y1, 1920, y0), fill=color)
    text(1900, y1-8, f'{large_total:.1f}*', 18, True, anchor='ms')
    for center, value, color in zip([2000, 2100, 2200, 2300], large_compute + [large_gap],
                                    [healthy_color, full_color, reduced_color, gap_color]):
        y = bottom - value / limit * (bottom - top)
        draw.rectangle((center-20, y, center+20, bottom), fill=color)
        text(center, y-8, f'{value:.1f}', 18, True, anchor='ms')
    text(2080, 1095, 'a8b / 120b: MBS1 (original)', 22, True, anchor='mt')
    text(2080, 1130, 'GBS H/N 128/112 | Job 2994648', 20, anchor='mt')
    text(2080, 1163, f'Stack: {large_gs:.2f} + {large_delta:.2f} + {large_host:.2f} ms', 20, anchor='mt')
    text(2080, 1195, f"Healthy {large['healthy']['mean_elapsed_ms']:.2f} ms | NEP {large['nep']['mean_elapsed_ms']:.2f} ms", 20, anchor='mt')
    text(1830, 1230, 'No MBS/GA tuning; GB200 OOM -> GB300.', 21)
    text(1830, 1264, 'RDMA-scale cross-replica bandwidth verified.', 21)
    text(1830, 1298, 'Full PyTorch profiles, not low-overhead protocol.', 21)
    text(1830, 1332, 'Compute averages: healthy/full/reduced 32/16/12.', 21)
    text(1830, 1366, 'Blue: full-replica owners 0-11, profiled 10-13.', 21)
    text(1830, 1400, 'Same arrival-adjustment and exposure caveats.', 21)
    text(1830, 1434, 'RS and AG each: healthy 6 vs NEP 8 calls/step.', 21)
    notes = [
        '* Blue: Gather + Scatter, expert gradient reduce-scatter + parameter all-gather deltas, and host-idle/staging exposure.',
        'Trim to last peer start for overlapping peer kernels; retain nonoverlapping/buffered reshard calls. Not exact internal-wait removal.',
        '** GPU-idle intervals inside NEP CPU scopes + staging GPU time with no other GPU activity. Not summed CPU durations.',
        'This host/staging segment is observed coincidence, not proven incremental CPU cost; unannotated host work is not covered.',
        'Compute = final-GA active kernel sums incl. grad accumulation, excluding NEP staging. Healthy/full/reduced means: 16/8/6 ranks.',
        'Blue: owners 0-5, profiled steps 10-13. RS and AG each: healthy 4 vs NEP 8 kernels/step. Scatter redistributes forward parameters.',
        'ALL red iteration gaps use post-profile 15-24. GPU timings use profiled 10-13 and do NOT decompose those post-profile gaps.',
        'The stack is not a causal decomposition of the iteration gap. Residual waits, multi-peer progress and profiling remain.',
    ]
    for i, note in enumerate(notes):
        text(70, 1230 + i * 34, note, 20)
    canvas.save(OUTPUT / 'small_model_parity_final_ga.png')
    canvas.save(OUTPUT / 'small_model_parity_final_ga.pdf', resolution=150)
    (OUTPUT / 'stacked_plot_values.json').write_text(json.dumps({job: {
        'gather_scatter_estimate_ms': reshard[job]['owner_mean']['total']['adjusted_estimate_ms'],
        'edp_healthy_estimate_ms': edp[job]['healthy']['owner_mean']['adjusted_estimate_ms'],
        'edp_nep_estimate_ms': edp[job]['nep']['owner_mean']['adjusted_estimate_ms'],
        'edp_grad_rs_delta_ms': edp[job]['delta_ms'],
        'param_ag_healthy_estimate_ms': param_ag[job]['healthy']['owner_mean']['adjusted_estimate_ms'],
        'param_ag_nep_estimate_ms': param_ag[job]['nep']['owner_mean']['adjusted_estimate_ms'],
        'param_ag_delta_ms': param_ag[job]['delta_ms'],
        'edp_delta_ms': edp[job]['delta_ms'] + param_ag[job]['delta_ms'], 'iteration_gap_ms': gaps[job],
        'host_idle_staging_ms': replica[job]['nep']['summary']['owners']['host_staging_ms'],
        'healthy_compute_ms': replica[job]['healthy']['summary']['healthy']['final_ga_compute_ms'],
        'nep_full_compute_ms': replica[job]['nep']['summary']['full']['final_ga_compute_ms'],
        'nep_reduced_compute_ms': replica[job]['nep']['summary']['reduced']['final_ga_compute_ms'],
        'parity_percent': parity[job], 'iteration_window': 'post 15-24',
        'parity_window': 'post 15-24'
        } for job, _, _ in cases}, indent=2) + '\n')
    values_path = OUTPUT / 'stacked_plot_values.json'
    values = json.loads(values_path.read_text())
    values['2994648'] = {
        'parity_percent': large_parity, 'iteration_gap_ms': large_gap,
        'healthy_iteration_ms': large['healthy']['mean_elapsed_ms'],
        'nep_iteration_ms': large['nep']['mean_elapsed_ms'],
        'iteration_window': 'post 15-24', 'parity_window': 'post 15-24',
        'workload': 'a8b_120b_latentmoe_1t', 'tuning': 'Original MBS1 GA8; GB300',
        'gather_scatter_estimate_ms': large_gs,
        'edp_delta_ms': large_delta, 'edp_components_ms': large_edp,
        'host_idle_staging_ms': large_parts['nep']['summary']['owners']['host_staging_ms'],
        'healthy_compute_ms': large_compute[0], 'nep_full_compute_ms': large_compute[1],
        'nep_reduced_compute_ms': large_compute[2],
    }
    values_path.write_text(json.dumps(values, indent=2) + '\n')


if __name__ == '__main__':
    main()
