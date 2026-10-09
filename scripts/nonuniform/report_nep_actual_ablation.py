"""Publish matched actual-a8b policy results without mixing profile protocols."""

import csv
import json
import statistics as st
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

LABELS = {'H': 'Healthy MBS2 GA8', 'H1': 'Healthy MBS1 GA16',
          'A': 'Compact / ascending / 1 stream', 'B': 'Compact / disjoint-first / 1 stream',
          'C': 'Compact / ascending / 2 streams', 'D': 'Compact / disjoint-first / 2 streams',
          'E': 'Spread / ascending / 1 stream', 'F': 'Spread / ascending / 2 streams',
          'R': 'MBS2 GA8/6 (underloaded)', 'O': 'MBS2 GA8/10 (overloaded)',
          'S': 'MBS1 GA16/16 (proportional)'}


def rows(job, cells):
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab') / job
    data = json.loads((root / 'actual-summary.json').read_text())
    assert json.loads((root / 'work-controls-audit.json').read_text())['status'] == 'PASS'
    assert len(data['cases']) == len(cells) * 4
    lookup = {}
    for name, run in data['cases'].items():
        repeat = int(name.split('_')[3][1:])
        key = (run['spec']['cell'], repeat, run['profiled'])
        assert key not in lookup and run['spec']['native_rank_timing']
        lookup[key] = run
    output = []
    for cell in cells:
        for profile in (False, True):
            runs = [lookup[cell, r, profile] for r in (0, 1)]
            control = cell if cell.startswith('H') else 'H1' if cell == 'S' else 'H'
            healthy = [lookup[control, r, profile]['timing']['mean'] for r in (0, 1)]
            times = [x['timing']['mean'] for x in runs]
            parity = [100*h/n for h, n in zip(healthy, times)]
            entry = dict(job=job, cell=cell, label=LABELS[cell], profiled=profile,
                         owner_ms=st.mean(times), owner_run_means=times, healthy_ms=st.mean(healthy),
                         gap_ms=st.mean(n-h for h, n in zip(healthy, times)),
                         parity=st.mean(parity), parity_repeats=parity,
                         true_gbs=runs[0]['spec']['true_gbs'],
                         ktokens_per_second=st.mean(x['tokens_per_second'] for x in runs)/1000,
                         peak_GiB=max(x['peak_GiB'] for x in runs))
            if profile:
                entry['phases_ms'] = {k: st.mean(x['cohorts']['owners'][k]['mean'] for x in runs)
                                      for k in runs[0]['cohorts']['owners']}
            output.append(entry)
    return output


def figure(out, name, entries, title):
    svg = ET.Element('svg', xmlns='http://www.w3.org/2000/svg', width='1450', height='800', viewBox='0 0 1450 800')

    def node(tag, **attrs):
        return ET.SubElement(svg, tag, {k.replace('_', '-'): str(v) for k, v in attrs.items()})

    def text(x, y, label, size=16, anchor='start'):
        item = node('text', x=x, y=y, font_family='Arial, sans-serif', font_size=size,
                    text_anchor=anchor, fill='#24272b')
        item.text = label

    node('rect', width=1450, height=800, fill='white')
    text(45, 36, title, 25)
    text(45, 66, 'a8b latent: 15 MoE layers in hybrid/MTP pattern | H4608 / latent1152 / expert F3072 / E512 / top-k6', 17)
    text(45, 93, 'S8192 | TP2 / CP1 / ETP1 | Healthy EP16/16; NEP EP16/12 | GB300 | DistOpt + HybridEP + CUDA graphs', 16)
    text(45, 120, 'Owners0-11; native timing companions, steps25-30; two reversed-order repeats. Profiles reported separately.', 16)
    for index, (key, label) in enumerate([('parity', 'Owner latency parity (%)'), ('ktokens_per_second', 'Aggregate throughput (thousand tokens/s)')]):
        left = 85 + index*710
        width, bottom, top = 590, 515, 190
        limit = 110 if index == 0 else max(x[key] for x in entries)*1.2
        text(left, 166, label, 20)
        y = lambda value: bottom-value/limit*(bottom-top)
        for tick in range(6):
            value = limit*tick/5
            node('line', x1=left, x2=left+width, y1=y(value), y2=y(value), stroke='#dce1e5')
            text(left-10, y(value)+5, f'{value:.0f}', 12, 'end')
        for i, row in enumerate(entries):
            center = left+(i+.5)*width/len(entries)
            value = row[key]
            node('rect', x=center-24, y=y(value), width=48, height=bottom-y(value),
                 fill='#278b87' if row['cell'].startswith('H') else '#3679b5')
            points = row['parity_repeats'] if index == 0 else [row['true_gbs']*8192/t for t in row['owner_run_means']]
            for offset, point in zip((-5, 5), points):
                node('circle', cx=center+offset, cy=y(point), r=3, fill='#20242a')
            text(center, y(value)-13, f'{value:.1f}', 14, 'middle')
            text(center, 542, row['cell'], 17, 'middle')
    for i, row in enumerate(entries):
        text(55+(i%2)*710, 582+(i//2)*29, f'{row["cell"]}: {row["label"]}', 15)
    text(45, 732, 'Actual buckets: dense3/3, expert5/3 (healthy/NEP); layouts fixed within each topology across policies.', 15)
    text(45, 759, 'Replica-local batches: healthy128/128; NEP128/96, except R128/72 and O128/120. True GBS256/224/200/248.', 15)
    ET.ElementTree(svg).write(out/name, encoding='utf-8', xml_declaration=True)


def main(placement, work):
    out = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/plots/nep_actual_a8b_ablations_20261002')
    out.mkdir(parents=True, exist_ok=True)
    p = rows(placement, ['H', 'A', 'B', 'C', 'D', 'E', 'F'])
    w = rows(work, ['H', 'B', 'R', 'O'])
    for row in w:
        if row['cell'] == 'B':
            row['label'] = 'MBS2 GA8/8 (proportional)'
    figure(out, 'placement.svg', [x for x in p if not x['profiled']], 'Actual a8b: expert placement and transfer scheduling')
    figure(out, 'work_allocation.svg', [x for x in w if not x['profiled']], 'Actual a8b: reduced-replica work allocation')
    data = dict(placement=p, work_allocation=w,
                unresolved=dict(cell='S', job='3215526', reason='MBS1/GA16 NCCL timeout in profiled iteration26; no accepted parity. H1 not run.'))
    (out/'results.json').write_text(json.dumps(data, indent=2)+'\n')
    fields = ['job', 'cell', 'label', 'profiled', 'owner_ms', 'healthy_ms', 'gap_ms', 'parity', 'true_gbs', 'ktokens_per_second', 'peak_GiB']
    with (out/'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(p+w)
    (out/'index.html').write_text('<!doctype html><html><head><title>Actual a8b NEP ablations</title></head><body>'
        '<h1>Actual a8b NEP ablations</h1><p><a href="results.csv">CSV</a> | <a href="results.json">JSON</a> | <a href="README.md">Report</a></p>'
        '<p>MBS1/GA16 remains unresolved: NCCL timeout in profiled iteration26 (job3215526). No parity assigned.</p>'
        '<img style="width:100%;max-width:1450px" src="placement.svg"><img style="width:100%;max-width:1450px" src="work_allocation.svg"></body></html>')
    for path in out.glob('*.svg'):
        ET.parse(path)
    print(out)


if __name__ == '__main__':
    main(*sys.argv[1:])
