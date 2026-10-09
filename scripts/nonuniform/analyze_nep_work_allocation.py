"""Matched timing and disjoint phase summaries for the five-case work study."""

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from analyze_a8b_graph_attribution import analyze


def stats(values):
    return dict(mean=st.mean(values), sd=st.stdev(values) if len(values) > 1 else 0, samples=values)


def rs_readiness(run):
    """Compare GPU input-ready events, not CPU enqueue calls or wire time."""
    data = {r: json.loads((run / f'cupti_profile/rank-{r}.json').read_text())
            for r in list(range(12)) + list(range(16, 28))}

    def ranges(rank, step):
        trace = data[rank]
        return [(trace['origin_ns']/1000 + e['ts'], trace['origin_ns']/1000 + e['ts'] + e['dur'])
                for e in trace['traceEvents'] if e.get('cat') == 'gpu_operation_range'
                and e['args']['iteration'] == step and e['name'] == 'collective.reduce_scatter'
                and e['args']['collective']['group_desc'] == 'ep_dp']

    records = []
    for step in range(15, 21):
        for rank in range(12):
            full, reduced = ranges(rank, step), ranges(rank+16, step)
            assert len(full) == len(reduced) == 3
            for bucket, (a, b) in enumerate(zip(full, reduced)):
                records.append(dict(iteration=step, owner=rank, bucket=bucket,
                                    reduced_minus_full_ready_ms=(b[0]-a[0])/1000,
                                    reduced_minus_full_completion_ms=(b[1]-a[1])/1000))
    return dict(records=records,
                reduced_minus_full_ready_ms=stats([r['reduced_minus_full_ready_ms'] for r in records]),
                paired_completion_abs_ms=stats([abs(r['reduced_minus_full_completion_ms']) for r in records]),
                late_fraction_over_0_2ms=st.mean(r['reduced_minus_full_ready_ms'] > .2 for r in records),
                max_gpu_host_clock_error_us=max(d['clock_error_bound_ns'] for d in data.values())/1000,
                note='GPU input-ready elapsed timestamps mapped to host clocks. Pair completion spread is an alignment check, not independent clock synchronization.')


def summarize(root):
    cases, pooled = {}, defaultdict(list)
    for run in sorted(root.glob('controlled_work_*')):
        gate = run / 'work-validation.json'
        if not gate.is_file():
            print('INCOMPLETE', run.name, flush=True)
            continue
        validated = json.loads(gate.read_text())
        assert validated['status'] == 'PASS'
        cell = validated['cell']
        healthy = cell.startswith('H')
        ga = 2 if cell.endswith('2') else 4
        world = 32 if healthy else 28
        times = {int(i): float(t) for i, t in re.findall(
            r'iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([0-9.]+)',
            (run / 'driver.log').read_text())}
        result = dict(cell=cell, profiled=validated['profiled'], true_gbs=validated['true_gbs'],
                      logged_matched_ms=stats([times[i] for i in range(15, 21)]),
                      logged_post_ms=stats([times[i] for i in range(22, 26)]))
        result['tokens_per_second'] = validated['true_gbs'] * 1024 * 1000 / result['logged_matched_ms']['mean']
        result['tokens_per_active_gpu_second'] = result['tokens_per_second'] / world
        result['peak_allocated_GiB'] = max(m['peak_allocated'] for m in validated['memory'].values()) / 1024**3
        if validated['profiled']:
            ranks = {rank: analyze(run / f'cupti_profile/rank-{rank}.json', range(15, 21),
                                   num_microbatches={'R': 3, 'O': 5}.get(cell, ga) if rank >= 16 else ga)
                     for rank in range(world)}
            result['cohorts'] = {}
            for role, members in [('owners', range(12)), ('followers', range(12, 16)), ('second', range(16, world))]:
                result['cohorts'][role] = {
                    key: stats([st.mean(ranks[r][i][key] for r in members) for i in range(6)])
                    for key in ('iteration_ms', 'first_forward_ms', 'other_forward_ms',
                                'nonfinal_backward_ms', 'final_backward_ms', 'outside_fb_ms')}
            result['ranks'] = ranks
            if not healthy:
                result['rs_readiness'] = rs_readiness(run)
        cases[run.name] = result
        pooled[f'{"on" if validated["profiled"] else "off"}:{cell}'].append(result['logged_matched_ms']['mean'])
        print(run.name, result['logged_matched_ms']['mean'], 'ms', result['peak_allocated_GiB'], 'GiB', flush=True)
    summary = {key: stats(values) for key, values in pooled.items()}
    comparisons = {}
    for mode in ('on', 'off'):
        for cell, healthy in [('N1', 'H1'), ('N2', 'H2'), ('R', 'H1'), ('O', 'H1')]:
            if f'{mode}:{cell}' not in summary or f'{mode}:{healthy}' not in summary:
                continue
            h, n = summary[f'{mode}:{healthy}']['mean'], summary[f'{mode}:{cell}']['mean']
            batch = {'R': 50, 'O': 62}.get(cell, 56)
            comparisons[f'{mode}:{cell}'] = dict(
                healthy_ms=h, nep_ms=n, gap_ms=n-h, owner_parity_percent=100*h/n,
                aggregate_throughput_relative_percent=100*(batch/64)*h/n,
                per_active_gpu_throughput_relative_percent=100*(batch/64)*(32/28)*h/n)
    output = dict(cases=cases, pooled_run_means=summary, comparisons=comparisons)
    (root / 'work-allocation-summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2))


def combine(first, second):
    """Use two complete blocks; retain interrupted partial repeats separately."""
    base = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab')
    cases, audits, blocks = {}, {}, []
    for repeat, job in enumerate((first, second)):
        root = base / job
        summary = json.loads((root / 'work-allocation-summary.json').read_text())
        audit = json.loads((root / 'work-controls-audit.json').read_text())
        assert audit['status'] == 'PASS'
        selected = {name: row for name, row in summary['cases'].items() if f'_r{repeat}_' in name}
        assert len(selected) == 10, (job, len(selected))
        cells = {(row['cell'], row['profiled']): row for row in selected.values()}
        assert len(cells) == 10
        block = dict(job=job, repeat=repeat, comparisons={})
        for profiled in (False, True):
            mode = 'on' if profiled else 'off'
            for cell, control in [('N1', 'H1'), ('N2', 'H2'), ('R', 'H1')]:
                h = cells[control, profiled]['logged_matched_ms']['mean']
                n = cells[cell, profiled]['logged_matched_ms']['mean']
                batch = cells[cell, profiled]['true_gbs']
                block['comparisons'][f'{mode}:{cell}'] = dict(
                    healthy_ms=h, nep_ms=n, gap_ms=n-h, owner_parity_percent=100*h/n,
                    aggregate_throughput_relative_percent=100*(batch/64)*h/n,
                    per_active_gpu_throughput_relative_percent=100*(batch/64)*(32/28)*h/n)
        blocks.append(block)
        cases.update({name: dict(row, job=job) for name, row in selected.items()})
        audits.update({name: audit['cases'][name] for name in selected})
    pooled = defaultdict(list)
    for row in cases.values():
        pooled[f'{"on" if row["profiled"] else "off"}:{row["cell"]}'].append(row['logged_matched_ms']['mean'])
    comparisons = {key: {metric: st.mean(b['comparisons'][key][metric] for b in blocks)
                         for metric in blocks[0]['comparisons'][key]}
                   for key in blocks[0]['comparisons']}
    output = base / 'work-allocation-20260930'
    output.mkdir(exist_ok=True)
    (output / 'work-allocation-summary.json').write_text(json.dumps(dict(
        cases=cases, blocks=blocks, pooled_run_means={k: stats(v) for k, v in pooled.items()},
        comparisons=comparisons,
        note='Parity/relative throughput are means of within-allocation ratios, not ratios formed by mixing different allocations.'), indent=2)+'\n')
    (output / 'work-controls-audit.json').write_text(json.dumps(dict(status='PASS', cases=audits), indent=2)+'\n')
    print(output)
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('--combine-with')
    args = parser.parse_args()
    if args.combine_with:
        combine(args.job, args.combine_with)
    else:
        summarize(Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab') / args.job)
