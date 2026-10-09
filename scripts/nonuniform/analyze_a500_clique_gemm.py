"""Measure expert GEMMs in the matched topology-controlled a500 profiles."""

import bisect
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_workload_parity_pair import iter_trace_events


ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen')
OUTPUT = ROOT / 'analysis/a500_clique_gemm_20260910'
WORKLOAD = 'scaling_ladder__a500m_2b_moe_140b'


def union_us(events):
    end = -float('inf')
    total = 0.0
    for start, stop in sorted((e['ts'], e['ts'] + e['dur']) for e in events):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze(path, local_experts, num_moe_layers=12):
    cpu, scopes, kernels, steps = {}, defaultdict(list), [], []
    for event in iter_trace_events(path):
        name, args = event.get('name', ''), event.get('args', {})
        if event.get('cat') == 'cpu_op':
            cpu[args.get('External id')] = (event['tid'], event['ts'])
            if name in ('_GroupedLinear', '_GroupedLinearBackward'):
                scopes[event['tid']].append(
                    (event['ts'], event['ts'] + event['dur'], name)
                )
        if event.get('cat') == 'kernel' and name.startswith('nvjet'):
            kernels.append(event)
        if event.get('cat') == 'user_annotation' and name.startswith('ProfilerStep#'):
            steps.append(event)
    steps.sort(key=lambda e: e['ts'])
    starts = [e['ts'] for e in steps]
    for scope in scopes.values():
        scope.sort()
    groups = defaultdict(list)
    for event in kernels:
        launch = cpu.get(event['args'].get('External id'))
        if launch is None:
            continue
        thread, timestamp = launch
        scope = scopes[thread]
        i = bisect.bisect_right(scope, (timestamp, float('inf'), '')) - 1
        if i < 0 or timestamp >= scope[i][1]:
            continue
        step = bisect.bisect_right(starts, scope[i][0]) - 1
        assert step >= 0
        phase = 'fwd' if scope[i][2] == '_GroupedLinear' else 'bwd'
        groups[step, phase].append(event)
    rows = []
    for i, step in enumerate(steps):
        row = {'step': step['name'], 'host_step_ms': step['dur'] / 1000}
        all_kernels = []
        for phase, multiplier in [('fwd', 2), ('bwd', 4)]:
            events = groups[i, phase]
            expected = num_moe_layers * 3 * local_experts * multiplier
            assert len(events) == expected, (path, step['name'], phase, len(events), expected)
            row[phase + '_count'] = len(events)
            row[phase + '_sum_ms'] = sum(e['dur'] for e in events) / 1000
            row[phase + '_union_ms'] = union_us(events) / 1000
            all_kernels.extend(events)
        row['gemm_union_ms'] = union_us(all_kernels) / 1000
        row['gemm_sum_ms'] = sum(e['dur'] for e in all_kernels) / 1000
        rows.append(row)
    assert len(rows) == 4, path
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for job, mbs in [('2994105', 1), ('2994279', 3)]:
        case = {}
        for mode, world in [('healthy', 16), ('nep', 14)]:
            for rank in range(world):
                role = 'healthy' if mode == 'healthy' else ('full' if rank < 8 else 'reduced')
                # Nondivisible EP distributes 128 true experts as 22,22,21,21,21,21.
                experts = 16 if role != 'reduced' else (22 if rank - 8 < 2 else 21)
                path = ROOT / f'slurm_runs/soft_clique_baseline/{job}/ep8/{WORKLOAD}/{mode}/{job}/torch_profile/rank-{rank}.json.gz'
                cache = OUTPUT / f'{job}_{mode}_{rank}.json'
                if cache.exists():
                    rows = json.loads(cache.read_text())
                else:
                    rows = analyze(path, experts)
                    cache.write_text(json.dumps(rows, indent=2) + '\n')
                case.setdefault(role, []).extend(rows)
                print(job, mode, rank, 'done', flush=True)
        summary = {}
        # Role-average routed assignments are exact: total dispatched tokens / EP.
        # Two expert linears: 4*H*F forward FLOPs, twice that in backward.
        flops = 12 * 12 * 8192 * mbs * 3 * 6 * 1024 * 640
        for role, rows in case.items():
            metrics = {k: statistics.mean(r[k] for r in rows) for k in rows[0] if k != 'step'}
            metrics['samples'] = len(rows)
            metrics['gemm_union_ms_std'] = statistics.pstdev(r['gemm_union_ms'] for r in rows)
            metrics['effective_TF_s'] = flops / (metrics['gemm_union_ms'] * 1e9)
            metrics['fwd_TF_s'] = flops / 3 / (metrics['fwd_union_ms'] * 1e9)
            metrics['bwd_TF_s'] = flops * 2 / 3 / (metrics['bwd_union_ms'] * 1e9)
            summary[role] = metrics
        results[job] = {'mbs': mbs, 'useful_TF_per_gpu': flops / 1e12, 'roles': summary}
        (OUTPUT / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
