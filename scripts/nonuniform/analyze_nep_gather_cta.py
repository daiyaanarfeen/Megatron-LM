"""Audit the Gather-only CTA intervention and matched-window performance."""

import argparse
import bisect
import json
import sqlite3
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from analyze_nep_actual_ablation import summarize
from analyze_nep_backward_interference import analyze as backward
from analyze_nep_norm_headroom import grids

CELLS = {'HGC': 'healthy', 'GBASE': 'nep', 'G16': 'nep'}
PHASES = (
    'first_forward_ms',
    'other_forward_ms',
    'nonfinal_backward_ms',
    'final_backward_ms',
    'outside_fb_ms',
)


def collectives(run, rank):
    """Attribute standalone kernels using host correlation, not stream guesses."""
    events = json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())['traceEvents']
    cpu = defaultdict(list)
    for e in events:
        if e.get('cat') == 'cpu_operation_range' and e['name'].startswith('collective.'):
            label = e['name'] + ':' + e['args']['collective']['group_desc']
            cpu[label].append(e)
    con = sqlite3.connect(f'file:{run}/nsys_profile/rank-{rank}.sqlite?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    strings = dict(con.execute('select id,value from StringIds'))
    by_label, by_thread = defaultdict(list), defaultdict(list)
    for r in con.execute('select * from NVTX_EVENTS where end is not null order by start'):
        name = r['text'] or strings.get(r['textId'], '')
        if name.startswith('collective.'):
            row = dict(start=r['start'], end=r['end'], name=name)
            by_label[name].append(row)
            by_thread[r['globalTid']].append(row)
    assert set(cpu) == set(by_label), (run, rank, set(cpu) ^ set(by_label))
    for label, rows in by_label.items():
        assert len(rows) == len(cpu[label]), (run, rank, label, len(rows), len(cpu[label]))
        for r, e in zip(rows, cpu[label]):
            r['iteration'] = e['args']['iteration']
    starts = {thread: [r['start'] for r in rows] for thread, rows in by_thread.items()}
    correlation = {}
    for r in con.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'):
        thread = r['globalTid']
        if 'Launch' not in strings[r['nameId']] or thread not in by_thread:
            continue
        index = bisect.bisect_right(starts[thread], r['start']) - 1
        if index >= 0 and r['end'] <= by_thread[thread][index]['end']:
            correlation[r['correlationId']] = by_thread[thread][index]
    kernels = []
    for r in con.execute(
        'select * from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is null or graphNodeId=0'
    ):
        host = correlation.get(r['correlationId'])
        if host and 25 <= host['iteration'] <= 30:
            kernels.append(
                dict(
                    iteration=host['iteration'],
                    collective=host['name'],
                    kernel=strings[r['shortName']],
                    grid=[r['gridX'], r['gridY'], r['gridZ']],
                    residency_ms=(r['end'] - r['start']) / 1e6,
                    start_ns=r['start'],
                    end_ns=r['end'],
                )
            )
    con.close()
    assert kernels, (run, rank)
    groups = defaultdict(list)
    for k in kernels:
        groups[k['collective']].append(k)
    return {
        label: dict(
            grids=[
                dict(grid=list(grid), count=n)
                for grid, n in Counter(tuple(k['grid']) for k in ks).items()
            ],
            residency_ms_per_step=sum(k['residency_ms'] for k in ks) / 6,
            records=ks,
        )
        for label, ks in groups.items()
    }


def cache_run(run, output, cell, rep, ranks):
    cache = output / cell
    cache.mkdir(parents=True, exist_ok=True)
    for rank in ranks:
        for name, analyzer in [
            ('backward', backward),
            ('collectives', collectives),
            ('grids', grids),
        ]:
            path = cache / f'{name}-r{rep}-rank{rank}.json'
            if not path.exists():
                path.write_text(json.dumps(analyzer(run, rank)) + '\n')
        print('CACHED', cell, rep, rank, flush=True)


def main(root, cache_only=False, ranks=range(16)):
    output = root / 'gather_cta'
    for cell, topology in CELLS.items():
        for rep in range(2):
            run = root / f'controlled_actual_{cell}_r{rep}_{topology}_moe'
            if (run / 'actual-validation.json').exists():
                cache_run(run, output, cell, rep, ranks)
    if cache_only:
        return
    expected = [
        root / f'controlled_actual_{cell}_r{rep}{suffix}_{topology}_moe'
        for cell, topology in CELLS.items()
        for rep in range(2)
        for suffix in ('', '_off')
    ]
    assert all((run / 'actual-validation.json').exists() for run in expected)
    assert len({(run / 'runtime.sha256').read_text() for run in expected}) == 1
    specs = [
        json.loads((run / 'actual-spec.json').read_text())
        for run in expected
        if '_nep_' in run.name
    ]
    normalized = [
        {k: v for k, v in s.items() if k not in ('cell', 'gather_max_ctas')} for s in specs
    ]
    assert all(s == normalized[0] for s in normalized)
    summarize(root)
    source = json.loads((root / 'actual-summary.json').read_text())['cases']
    results = {}
    for cell, topology in CELLS.items():
        record = {}
        for mode, suffix in [('profiled', ''), ('companion', '_off')]:
            repeats = [
                source[f'controlled_actual_{cell}_r{rep}{suffix}_{topology}_moe']['timing']
                for rep in range(2)
            ]
            record[mode] = dict(repeats=repeats, mean_ms=st.mean(r['mean'] for r in repeats))
        record['phases_ms'] = {
            key: st.mean(
                source[f'controlled_actual_{cell}_r{rep}_{topology}_moe']['cohorts']['owners'][key][
                    'mean'
                ]
                for rep in range(2)
            )
            for key in PHASES
        }
        rows, comm = [], defaultdict(list)
        for rep in range(2):
            for rank in ranks:
                cache = output / cell
                rows.extend(
                    dict(r, repeat=rep)
                    for r in json.loads((cache / f'backward-r{rep}-rank{rank}.json').read_text())
                )
                gs = json.loads((cache / f'grids-r{rep}-rank{rank}.json').read_text())
                assert all(g['grid'] == [360, 1, 1] for g in gs)
                cs = json.loads((cache / f'collectives-r{rep}-rank{rank}.json').read_text())
                for label, item in cs.items():
                    comm[label].append(dict(item, rank=rank, repeat=rep))
                    if label == 'collective.all_to_all:nep_owner_gather':
                        assert (
                            all(g['grid'][0] <= 16 for g in item['grids'])
                            if cell == 'G16'
                            else all(g['grid'][0] <= 32 for g in item['grids'])
                        )
                        if rank < 12:
                            assert all(
                                g['grid'][0] == (16 if cell == 'G16' else 32) for g in item['grids']
                            )
        record['backward'] = {}
        for phase in ('penultimate', 'final'):
            groups = defaultdict(list)
            for row in rows:
                if row['phase'] == phase:
                    groups[row['repeat'], row['rank'], row['iteration']].append(row)
            record['backward'][phase] = {
                key: st.mean(sum(r[key] for r in rs) for rs in groups.values())
                for key in (
                    'norm_no_native_gap_ms',
                    'norm_same_stream_gap_ms',
                    'kernel_span_ms',
                    'native_union_ms',
                )
            }
            record['backward'][phase]['norm_gap_gt_1ms_count'] = sum(
                r['norm_no_native_gap_ms'] > 1 for rs in groups.values() for r in rs
            )
            record['backward'][phase]['norm_gap_max_ms'] = max(
                r['norm_no_native_gap_ms'] for rs in groups.values() for r in rs
            )
            record['backward'][phase]['cohorts'] = {}
            for role, members in [('owners', range(12)), ('followers', range(12, 16))]:
                selected = [rs for (_, rank, _), rs in groups.items() if rank in members]
                if selected:
                    record['backward'][phase]['cohorts'][role] = dict(
                        norm_gap_ms=st.mean(
                            sum(r['norm_no_native_gap_ms'] for r in rs) for rs in selected
                        ),
                        norm_kernel_ms=st.mean(
                            sum(r['kernel_sum_ms'].get('rmsnorm_bwd_general_kernel', 0) for r in rs)
                            for rs in selected
                        ),
                    )
        record['collectives'] = {
            label: dict(
                mean_rank_residency_ms_per_step=st.mean(r['residency_ms_per_step'] for r in rs),
                grids=sorted({tuple(g['grid']) for r in rs for g in r['grids']}),
                cohort_mean_residency_ms={
                    role: st.mean(r['residency_ms_per_step'] for r in rs if r['rank'] in members)
                    for role, members in [('owners', range(12)), ('followers', range(12, 16))]
                    if any(r['rank'] in members for r in rs)
                },
            )
            for label, rs in comm.items()
        }
        results[cell] = record
    for rep in range(2):
        for rank in range(28):
            for cell in ('GBASE', 'G16'):
                run = root / f'controlled_actual_{cell}_r{rep}_nep_moe'
                for name, analyzer in [('collectives', collectives), ('grids', grids)]:
                    path = output / cell / f'{name}-r{rep}-rank{rank}.json'
                    if not path.exists():
                        path.write_text(json.dumps(analyzer(run, rank)) + '\n')
                norm = json.loads((output / cell / f'grids-r{rep}-rank{rank}.json').read_text())
                assert all(g['grid'] == [360, 1, 1] for g in norm)
            controls = [
                json.loads((output / cell / f'collectives-r{rep}-rank{rank}.json').read_text())
                for cell in ('GBASE', 'G16')
            ]
            assert set(controls[0]) == set(controls[1])
            for label in controls[0]:
                before = Counter()
                for g in controls[0][label]['grids']:
                    grid = tuple(g['grid'])
                    if label == 'collective.all_to_all:nep_owner_gather':
                        grid = (min(16, grid[0]), *grid[1:])
                    before[grid] += g['count']
                after = Counter({tuple(g['grid']): g['count'] for g in controls[1][label]['grids']})
                assert before == after, (rep, rank, label, before, after)
    comparisons = {}
    for cell in ('GBASE', 'G16'):
        h, n = results['HGC'], results[cell]
        comparisons[cell] = {
            mode: dict(
                healthy_ms=h[mode]['mean_ms'],
                nep_ms=n[mode]['mean_ms'],
                parity=100 * h[mode]['mean_ms'] / n[mode]['mean_ms'],
            )
            for mode in ('profiled', 'companion')
        }
        comparisons[cell]['phase_delta_ms'] = {
            k: n['phases_ms'][k] - h['phases_ms'][k] for k in PHASES
        }
        comparisons[cell]['differential_instrumentation_ms'] = (
            n['profiled']['mean_ms']
            - n['companion']['mean_ms']
            - h['profiled']['mean_ms']
            + h['companion']['mean_ms']
        )
    result = dict(
        cases=results, comparisons=comparisons, ranks=list(ranks), iterations=list(range(25, 31))
    )
    (root / 'gather-cta-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--cache-only', action='store_true')
    p.add_argument('--ranks', type=int, nargs='+', default=list(range(16)))
    args = p.parse_args()
    main(args.root, args.cache_only, args.ranks)
