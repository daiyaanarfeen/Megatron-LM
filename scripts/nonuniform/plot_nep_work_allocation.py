"""Render work-allocation performance and separately measured profiled phases."""

import csv
import json
import statistics as st
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def plot(job):
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab') / job
    data = json.loads((root / 'work-allocation-summary.json').read_text())
    assert len(data['cases']) == 20
    assert json.loads((root / 'work-controls-audit.json').read_text())['status'] == 'PASS'
    out = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/plots/nep_work_allocation_20260930')
    out.mkdir(parents=True, exist_ok=True)
    cells = ['H1', 'N1', 'R', 'H2', 'N2']
    labels = {'H1': 'Healthy MBS1 GA4', 'N1': 'NEP MBS1 GA4/4', 'R': 'NEP MBS1 GA4/3',
              'H2': 'Healthy MBS2 GA2', 'N2': 'NEP MBS2 GA2/2'}
    rows = []
    for cell in cells:
        runs = [r for r in data['cases'].values() if r['cell'] == cell and not r['profiled']]
        profiles = [r for r in data['cases'].values() if r['cell'] == cell and r['profiled']]
        assert len(runs) == len(profiles) == 2
        timing = data['pooled_run_means'][f'off:{cell}']
        row = dict(cell=cell, label=labels[cell], iteration_ms=timing['mean'], run_means_ms=timing['samples'],
                   true_gbs=runs[0]['true_gbs'], peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in runs + profiles))
        row['tokens_per_second'] = row['true_gbs'] * 1024 * 1000 / row['iteration_ms']
        row['phase_ms'] = {key: st.mean(p['cohorts']['owners'][key]['mean'] for p in profiles)
                           for key in profiles[0]['cohorts']['owners']}
        row['phase_runs'] = [{key: p['cohorts']['owners'][key]['mean'] for key in p['cohorts']['owners']}
                             for p in profiles]
        rows.append(row)
    (out / 'results.json').write_text(json.dumps(dict(rows=rows, comparisons=data['comparisons']), indent=2) + '\n')
    with (out / 'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def new(title, subtitle):
        svg = ET.Element('svg', xmlns='http://www.w3.org/2000/svg', width='1400', height='760', viewBox='0 0 1400 760')

        def node(tag, **attrs):
            return ET.SubElement(svg, tag, {k.replace('_', '-'): str(v) for k, v in attrs.items()})

        def text(x, y, value, size=15, anchor='start', weight='normal'):
            item = node('text', x=x, y=y, font_size=size, text_anchor=anchor,
                        fill='#20242a', font_family='Arial, sans-serif', font_weight=weight)
            item.text = value

        node('rect', x=0, y=0, width=1400, height=760, fill='white')
        text(50, 35, title, 24, weight='bold')
        text(50, 65, '4 latent-MoE layers | H4608 / latent1152 / expert F3072 | E512 / top-k6 | TP2 / CP1 / ETP1', 17)
        text(50, 92, 'S1024 | Healthy EP16/16; NEP EP16/12 | DistOpt + HybridEP + MoE graphs | GB200; cross-replica RDMA', 16)
        text(50, 120, subtitle, 16)
        return svg, node, text

    svg, node, text = new('Work allocation: latency parity and useful throughput',
                          'Profiler-off timings, matched iterations15-20; two allocation blocks, opposite order. Ratios paired within each block.')
    for panel, (x, width, limit, title) in enumerate([(90, 490, 110, 'Owner-time parity (%)'), (735, 580, 1100, 'Aggregate throughput (thousand tokens/s)')]):
        top, bottom = 200, 545
        y = lambda v: bottom - v / limit * (bottom - top)
        text(x, 164, title, 19, weight='bold')
        for tick in range(0, int(limit), 20 if panel == 0 else 200):
            node('line', x1=x, x2=x+width, y1=y(tick), y2=y(tick), stroke='#d9dee3')
            text(x-10, y(tick)+5, str(tick), 13, anchor='end')
        selected = [r for r in rows if not r['cell'].startswith('H')] if panel == 0 else rows
        for i, row in enumerate(selected):
            cell = row['cell']
            value = data['comparisons'][f'off:{cell}']['owner_parity_percent'] if panel == 0 else row['tokens_per_second']/1000
            cx = x + (i+.5)*width/len(selected)
            node('rect', x=cx-30, y=y(value), width=60, height=bottom-y(value),
                 fill='#278b87' if cell.startswith('H') else '#3679b5')
            if panel == 0:
                control = 'H2' if cell == 'N2' else 'H1'
                h = data['pooled_run_means'][f'off:{control}']['samples']
                points = [100*a/b for a, b in zip(h, row['run_means_ms'])]
            else:
                points = [row['true_gbs']*1024/t for t in row['run_means_ms']]
            for dx, point in zip((-5, 5), points):
                node('circle', cx=cx+dx, cy=y(point), r=3, fill='#20242a')
            text(cx, y(value)-12, f'{value:.1f}'+('%' if panel == 0 else ''), 16, anchor='middle')
            text(cx, 573, cell, 16, anchor='middle', weight='bold')
    text(65, 615, 'H1/N1: MBS1 GA4/4, replica batches32/32 or32/24. H2/N2: MBS2 GA2/2, same respective batches.', 16)
    text(65, 643, 'R: MBS1 GA4/3, replica batches32/18. True GBS: healthy64, proportional NEP56, reduced-work NEP50.', 16)
    text(65, 671, 'Parity = healthy owner time / NEP owner time; R uses H1. Throughput accounts for the different true batch sizes.', 15)
    text(65, 699, 'Timing: native rank0 log. Memory checked on every rank. Native buckets: dense1/1, expert2/3 (healthy/NEP).', 15)
    ET.ElementTree(svg).write(out / 'work_allocation.svg', encoding='utf-8', xml_declaration=True)

    svg, node, text = new('Profiled owner phases: where elapsed time changes',
                          'Separate profiled runs; owners0-11, iterations15-20. Paired bars show both blocks; elapsed phases, not pure kernel times.')
    colors = ['#3679b5', '#79b3cf', '#83ad7a', '#bc7e50', '#90979e']
    keys = ['first_forward_ms', 'other_forward_ms', 'nonfinal_backward_ms', 'final_backward_ms', 'outside_fb_ms']
    names = ['First forward', 'Other forwards', 'Other backwards', 'Final backward', 'Outside F/B']
    top, bottom, limit = 205, 535, 150
    y = lambda v: bottom - v / limit * (bottom - top)
    for tick in range(0, 151, 25):
        node('line', x1=90, x2=1310, y1=y(tick), y2=y(tick), stroke='#d9dee3')
        text(75, y(tick)+5, f'{tick} ms', 13, anchor='end')
    for i, row in enumerate(rows):
        cx = 210 + i*240
        for j, phase in enumerate(row['phase_runs']):
            bx = cx-60+j*65
            total = 0
            for key, color in zip(keys, colors):
                value = phase[key]
                node('rect', x=bx, y=y(total+value), width=55, height=y(total)-y(total+value), fill=color)
                total += value
            text(bx+27, y(total)-12, f'{total:.1f}', 13, anchor='middle')
            text(bx+27, 551, f'Block{j+1}', 10, anchor='middle')
        text(cx, 566, row['cell'], 17, anchor='middle', weight='bold')
        text(cx, 591, row['label'], 14, anchor='middle')
    for i, (name, color) in enumerate(zip(names, colors)):
        x = 85 + i*255
        node('rect', x=x, y=625, width=18, height=18, fill=color)
        text(x+26, 640, name, 15)
    text(65, 686, 'The five phases are disjoint and sum to the profiled iteration. Native collective ranges are not added on top.', 16)
    text(65, 714, 'Profiling overhead varies by configuration. Do not use this figure as an exact breakdown of profiler-off timings.', 16)
    ET.ElementTree(svg).write(out / 'profiled_phases.svg', encoding='utf-8', xml_declaration=True)
    (out / 'index.html').write_text('<!doctype html><html><head><title>NEP Work Allocation</title></head><body>'
                                  '<h1>NEP Work Allocation</h1><p><a href="results.csv">CSV</a> | <a href="results.json">JSON</a></p>'
                                  '<img style="width:100%;max-width:1400px" src="work_allocation.svg">'
                                  '<img style="width:100%;max-width:1400px" src="profiled_phases.svg"></body></html>')
    print(out)


if __name__ == '__main__':
    plot(sys.argv[1])
