"""Actual a8b matched-window timing, phases and input-ready RS comparisons."""

import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from analyze_a8b_graph_attribution import analyze
from audit_nep_work_allocation import audit


def stats(values):
    return dict(mean=st.mean(values), sd=st.stdev(values) if len(values) > 1 else 0, samples=values)


def summarize(root, allow_expert_rebucketing=False):
    cases = {}
    for run in sorted(root.glob('controlled_actual_*')):
        gate = run / 'actual-validation.json'
        if not gate.exists():
            print('INCOMPLETE', run.name, flush=True)
            continue
        spec = json.loads((run / 'actual-spec.json').read_text())
        valid = json.loads(gate.read_text())
        assert valid['status'] == 'PASS'
        healthy = spec['cell'].startswith('H')
        world = 32 if healthy else 28
        steps = range(spec['profile_start'], spec['profile_end'])
        times = {int(i): float(t) for i, t in re.findall(
            r'iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([0-9.]+)',
            (run / 'driver.log').read_text())}
        row = dict(spec=spec, profiled=valid['profiled'], timing=stats([times[i] for i in steps]),
                   peak_GiB=max(m['peak_allocated'] for m in valid['memory'].values()) / 1024**3)
        row['logged_last_rank_timing'] = row['timing']
        if spec.get('native_rank_timing'):
            native = {r: json.loads((run / f'native-timing-{r}.json').read_text()) for r in range(world)}
            assert all(len(x) == spec['train_iters'] for x in native.values())
            row['native_timing_cohorts'] = {
                role: stats([st.mean(native[r][i-1] for r in members) for i in steps])
                for role, members in [('owners', range(12)), ('followers', range(12, 16)), ('second', range(16, world))]}
            row['timing'] = row['native_timing_cohorts']['owners']
        row['tokens_per_second'] = spec['true_gbs'] * spec['seqlen'] * 1000 / row['timing']['mean']
        if row['profiled']:
            ranks = {r: analyze(run / f'cupti_profile/rank-{r}.json', steps,
                                num_microbatches=spec['ga'] if r < 16 else spec['reduced_ga'])
                     for r in range(world)}
            row['cohorts'] = {}
            for role, members in [('owners', range(12)), ('followers', range(12, 16)), ('second', range(16, world))]:
                row['cohorts'][role] = {
                    key: stats([st.mean(ranks[r][i][key] for r in members) for i in range(len(steps))])
                    for key in ('iteration_ms', 'first_forward_ms', 'other_forward_ms',
                                'nonfinal_backward_ms', 'final_backward_ms', 'outside_fb_ms')}
            if not healthy:
                readiness = []
                for owner in range(12):
                    full, reduced = [json.loads((run / f'cupti_profile/rank-{r}.json').read_text())
                                     for r in (owner, owner + 16)]
                    for step in steps:
                        events = [[e for e in data['traceEvents'] if e.get('cat') == 'gpu_operation_range'
                                   and e['args']['iteration'] == step and e['name'] == 'collective.reduce_scatter'
                                   and e['args']['collective']['group_desc'] == 'ep_dp'] for data in (full, reduced)]
                        assert events[0] and len(events[0]) == len(events[1])
                        for bucket, (a, b) in enumerate(zip(*events)):
                            assert a['args']['collective']['operations'] == b['args']['collective']['operations']
                            ready = (reduced['origin_ns'] / 1000 + b['ts'] - full['origin_ns'] / 1000 - a['ts']) / 1000
                            completion = ready + (b['dur'] - a['dur']) / 1000
                            readiness.append(dict(owner=owner, step=step, bucket=bucket,
                                                  reduced_minus_full_ms=ready, completion_delta_ms=completion))
                row['rs_readiness'] = dict(records=readiness,
                    reduced_minus_full_ms=stats([x['reduced_minus_full_ms'] for x in readiness]),
                    completion_abs_ms=stats([abs(x['completion_delta_ms']) for x in readiness]))
        cases[run.name] = row
        print(run.name, row['timing']['mean'], 'ms', row['tokens_per_second'], 'tokens/s', flush=True)
    assert cases
    audit(root, 'controlled_actual_*', allow_expert_rebucketing=allow_expert_rebucketing)
    pooled = defaultdict(list)
    for row in cases.values():
        pooled[f'{"on" if row["profiled"] else "off"}:{row["spec"]["cell"]}'].append(row['timing']['mean'])
    summary = {k: stats(v) for k, v in pooled.items()}
    comparisons = {}
    for mode in ('on', 'off'):
        for cell in ('A', 'B', 'C', 'D', 'E', 'F', 'R', 'O', 'S', 'B3', 'B5', 'B6'):
            healthy_cell = 'H8K' if cell in ('B3', 'B5', 'B6') else 'H1' if cell == 'S' else 'H'
            hkey, nkey = f'{mode}:{healthy_cell}', f'{mode}:{cell}'
            if hkey not in summary or nkey not in summary:
                continue
            h, n = summary[hkey]['mean'], summary[nkey]['mean']
            batch = {'R': 200, 'O': 248}.get(cell, 224)
            comparisons[nkey] = dict(healthy_ms=h, nep_ms=n, gap_ms=n-h, parity=100*h/n,
                                     tokens_per_second=batch*8192*1000/n)
    output = dict(cases=cases, pooled=summary, comparisons=comparisons)
    (root / 'actual-summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    summarize(Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab') / sys.argv[1],
              allow_expert_rebucketing='--expert-buckets' in sys.argv[2:])
