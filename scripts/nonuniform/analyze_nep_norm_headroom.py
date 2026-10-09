"""Compare paired normalization margins without changing bucket counts or workload."""

import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from analyze_nep_actual_ablation import summarize
from analyze_nep_backward_interference import analyze as backward


def grids(run, rank):
    con = sqlite3.connect(f'file:{run}/nsys_profile/rank-{rank}.sqlite?mode=ro', uri=True)
    rows = con.execute(
        'select s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.registersPerThread, '
        'count(*), sum(k.end-k.start)/1e6 from CUPTI_ACTIVITY_KIND_KERNEL k '
        'join StringIds s on k.shortName=s.id where s.value like "%rmsnorm_bwd_general%" '
        'group by s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.registersPerThread'
    ).fetchall()
    con.close()
    assert rows, (run, rank, 'no general backward norm kernel')
    return [
        dict(name=name, grid=[x, y, z], block=block, registers=regs, count=count, sum_ms=ms)
        for name, x, y, z, block, regs, count, ms in rows
    ]


def main(root):
    cells = [('H32', 'healthy'), ('N32', 'nep'), ('H64', 'healthy'), ('N64', 'nep')]
    expected = [
        root / f'controlled_actual_{cell}_r{rep}{suffix}_{topology}_moe'
        for cell, topology in cells
        for rep in range(2)
        for suffix in ('', '_off')
    ]
    assert all((run / 'actual-validation.json').exists() for run in expected)
    assert len({(run / 'runtime.sha256').read_text() for run in expected}) == 1
    for topology in ('healthy', 'nep'):
        specs = [
            json.loads((run / 'actual-spec.json').read_text())
            for run in expected
            if run.name.endswith(f'_{topology}_moe')
        ]
        normalized = [
            {k: v for k, v in spec.items() if k not in ('cell', 'bwd_norm_sm_margin')}
            for spec in specs
        ]
        assert all(s == normalized[0] for s in normalized)
        assert {s['bwd_norm_sm_margin'] for s in specs} == {32, 64}
    summarize(root)
    source = json.loads((root / 'actual-summary.json').read_text())['cases']
    output = root / 'norm_headroom'
    output.mkdir(exist_ok=True)
    results = {}
    phase_keys = (
        'first_forward_ms',
        'other_forward_ms',
        'nonfinal_backward_ms',
        'final_backward_ms',
        'outside_fb_ms',
    )
    for cell, topology in cells:
        record = {}
        for mode, suffix in [('profiled', ''), ('companion', '_off')]:
            record[mode] = dict(
                repeats=[
                    dict(
                        repeat=rep,
                        mean_ms=source[f'controlled_actual_{cell}_r{rep}{suffix}_{topology}_moe'][
                            'timing'
                        ]['mean'],
                        sd_ms=source[f'controlled_actual_{cell}_r{rep}{suffix}_{topology}_moe'][
                            'timing'
                        ]['sd'],
                    )
                    for rep in range(2)
                ]
            )
            record[mode]['mean_ms'] = statistics.mean(r['mean_ms'] for r in record[mode]['repeats'])
        record['phases_ms'] = {
            key: statistics.mean(
                source[f'controlled_actual_{cell}_r{rep}_{topology}_moe']['cohorts']['owners'][key][
                    'mean'
                ]
                for rep in range(2)
            )
            for key in phase_keys
        }
        cache = output / cell
        cache.mkdir(exist_ok=True)
        rows, grid_records = [], []
        for rep in range(2):
            run = root / f'controlled_actual_{cell}_r{rep}_{topology}_moe'
            for rank in range(16):
                path = cache / f'backward-r{rep}-rank{rank}.json'
                if not path.exists():
                    path.write_text(json.dumps(backward(run, rank)) + '\n')
                rows.extend(dict(row, repeat=rep) for row in json.loads(path.read_text()))
                grid_records.append(dict(rank=rank, repeat=rep, grids=grids(run, rank)))
            for rank in range(16, 32 if topology == 'healthy' else 28):
                grid_records.append(dict(rank=rank, repeat=rep, grids=grids(run, rank)))
            print('ANALYZED', cell, rep, flush=True)
        expected_grid = 3 * (152 - int(cell[1:]))
        assert all(
            entry['grid'] == [expected_grid, 1, 1]
            for record in grid_records
            for entry in record['grids']
        ), (cell, 'normalization intervention did not match expected kernel grid')
        record['native_backward'] = {}
        for phase in ('penultimate', 'final'):
            grouped = defaultdict(list)
            for row in rows:
                if row['phase'] == phase:
                    grouped[row['repeat'], row['rank'], row['iteration']].append(row)
            keys = (
                'kernel_span_ms',
                'native_union_ms',
                'norm_same_stream_gap_ms',
                'norm_no_native_gap_ms',
            )
            record['native_backward'][phase] = {
                key: statistics.mean(sum(r[key] for r in group) for group in grouped.values())
                for key in keys
            }
            record['native_backward'][phase]['min_host_lead_ms'] = min(
                r['host_lead_ms'] for g in grouped.values() for r in g
            )
            for field in ('category_union_ms', 'kernel_sum_ms'):
                names = set().union(*(row[field] for g in grouped.values() for row in g))
                record['native_backward'][phase][field] = {
                    name: statistics.mean(
                        sum(r[field].get(name, 0) for r in group) for group in grouped.values()
                    )
                    for name in sorted(names)
                }
        record['norm_grid_records'] = grid_records
        results[cell] = record
        (output / 'partial.json').write_text(json.dumps(results, indent=2) + '\n')
    comparisons = {}
    for margin in (32, 64):
        h, n = results[f'H{margin}'], results[f'N{margin}']
        comparisons[margin] = {
            mode: dict(
                healthy_ms=h[mode]['mean_ms'],
                nep_ms=n[mode]['mean_ms'],
                delta_ms=n[mode]['mean_ms'] - h[mode]['mean_ms'],
                parity=100 * h[mode]['mean_ms'] / n[mode]['mean_ms'],
            )
            for mode in ('profiled', 'companion')
        }
        comparisons[margin]['phase_delta_ms'] = {
            k: n['phases_ms'][k] - h['phases_ms'][k] for k in phase_keys
        }
        comparisons[margin]['differential_instrumentation_ms'] = (
            comparisons[margin]['profiled']['delta_ms']
            - comparisons[margin]['companion']['delta_ms']
        )
    (root / 'norm-headroom-summary.json').write_text(
        json.dumps(dict(cases=results, comparisons=comparisons), indent=2) + '\n'
    )
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    main(
        Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab') / sys.argv[1]
    )
