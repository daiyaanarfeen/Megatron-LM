"""Sum GPU kernel durations for final-GA backward and expert gradient sync."""

import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from analyze_exact_uniform_nep_slowdown import intersection_duration, merge_intervals, union_duration
from analyze_nep_owner_progress import classify, location
from plot_nep_failure_mechanism_details import cluster_backward

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen')
OUT = ROOT / 'analysis/nep_small_model_parity_20260911'
CASES = [('2994105', 'a500m_2b_moe_140b', 12),
         ('2994279', 'a500m_2b_moe_140b', 12),
         ('2995392', 'a770m_4b_moe_215b', 15),
         ('2995524', 'a770m_4b_moe_215b', 15)]


def scan(path, layers, require_owner_comms=True, include_host_staging=False, ga=3, events=None):
    if events is None:
        with gzip.open(path, 'rt') as handle:
            events = json.load(handle)['traceEvents']
    cpu, launches, comms = {}, {}, {}
    backward, nep = defaultdict(list), defaultdict(list)
    steps, grouped = [], []
    for e in events:
        cat, name, a = e.get('cat'), e.get('name', ''), e.get('args', {})
        if cat == 'user_annotation' and name.startswith('ProfilerStep#'):
            steps.append(e)
        if cat == 'cpu_op':
            if 'External id' in a:
                cpu[a['External id']] = (e['ts'], e['tid'])
            if name.startswith('autograd::engine::evaluate_function:'):
                backward[e['tid']].append((e['ts'], e['ts'] + e['dur']))
                if name.endswith('_GroupedLinearBackward'):
                    grouped.append(e['ts'])
        if cat in ('cuda_runtime', 'cuda_driver') and 'Launch' in name:
            launches[a.get('correlation')] = (e['ts'], e['tid'])
        if name == 'record_param_comms' and a.get('Collective name') != 'wait':
            comms[a.get('External id')] = a
        if cat == 'user_annotation' and name.startswith('nep_'):
            nep[e['tid']].append((e['ts'], e['ts'] + e['dur']))
    backward = {t: merge_intervals(v) for t, v in backward.items()}
    nep = {t: merge_intervals(v) for t, v in nep.items()}
    bs = {t: [a for a, _ in v] for t, v in backward.items()}
    ns = {t: [a for a, _ in v] for t, v in nep.items()}
    result = []
    kernels = [e for e in events if e.get('cat') == 'kernel']
    for step in sorted(steps, key=lambda e: e['ts']):
        lo, hi = step['ts'], step['ts'] + step['dur']
        rows = [(a, b) for v in backward.values() for a, b in v if lo <= a < b <= hi]
        clusters, diagnostics = cluster_backward(rows, ga)
        counts = [sum(a <= t < b for t in grouped) for a, b in clusters]
        assert counts == [2 * layers] * ga, (path, step['name'], counts)
        sums, counts_gpu = defaultdict(float), defaultdict(int)
        unmatched = 0
        for e in kernels:
            if not lo <= e['ts'] < hi:
                continue
            args = dict(e.get('args', {}))
            args.update(comms.get(args.get('External id'), {}))
            category = classify(dict(e, args=args), [], [])
            source = cpu.get(args.get('External id')) or launches.get(args.get('correlation'))
            if category in ('grad_gather', 'expert_grad_rs'):
                if category == 'expert_grad_rs' and 'reduce_scatter' not in str(args.get('Collective name')):
                    continue
                assert e['ts'] >= clusters[-1][0], (path, 'sync before final GA')
                sums[category] += e['dur'] / 1000
                counts_gpu[category] += 1
            if category != 'other_kernel':
                continue
            if source is None:
                unmatched += 1
                continue
            ts, tid = source
            if clusters[-1][0] <= ts < clusters[-1][1]:
                if location(backward.get(tid, []), bs.get(tid, []), ts) is not None:
                    if location(nep.get(tid, []), ns.get(tid, []), ts) is None:
                        sums['backward_compute'] += e['dur'] / 1000
                        counts_gpu['backward_compute'] += 1
        if require_owner_comms:
            assert sums['grad_gather'] > 0 and sums['expert_grad_rs'] > 0
        assert sums['backward_compute'] > 0 and unmatched == 0, (path, unmatched)
        sums['nep_comms'] = sums['grad_gather'] + sums['expert_grad_rs']
        result.append({'step': step['name'], 'sum_ms': dict(sums), 'kernel_counts': dict(counts_gpu),
                       'grouped_backward_counts': counts, 'cluster_diagnostics': diagnostics,
                       'uncorrelated_compute_kernels': unmatched})
        if include_host_staging:
            host_scopes = [(max(lo, a), min(hi, b)) for rows in nep.values()
                           for a, b in rows if a < hi and b > lo]
            staging, other = [], []
            for e in events:
                if e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
                    continue
                if e['ts'] >= hi or e['ts'] + e['dur'] <= lo:
                    continue
                args = e.get('args', {})
                source = cpu.get(args.get('External id')) or launches.get(args.get('correlation'))
                is_staging = False
                if source is not None and classify(e, [], []) in ('other_kernel', 'memory'):
                    ts, tid = source
                    is_staging = location(nep.get(tid, []), ns.get(tid, []), ts) is not None
                pair = (max(lo, e['ts']), min(hi, e['ts'] + e['dur']))
                (staging if is_staging else other).append(pair)
            idle = union_duration(host_scopes) - intersection_duration(host_scopes, staging + other)
            exposed = union_duration(staging) - intersection_duration(staging, other)
            assert idle >= -0.01 and exposed >= -0.01
            result[-1]['host_staging_ms'] = {
                'host_scope_union': union_duration(host_scopes) / 1000,
                'gpu_idle_during_nep_host': max(0, idle) / 1000,
                'gpu_staging_union': union_duration(staging) / 1000,
                'gpu_staging_exposed': max(0, exposed) / 1000,
                'host_idle_plus_exposed_staging': max(0, idle + exposed) / 1000,
            }
    assert len(result) == 4
    return result


def main():
    output = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for job, model, layers in CASES:
        ranks = {}
        for rank in range(6):
            path = ROOT / f'slurm_runs/soft_clique_baseline/{job}/ep8/scaling_ladder__{model}/nep/{job}/torch_profile/rank-{rank}.json.gz'
            ranks[str(rank)] = scan(path, layers)
            print(job, rank, 'validated', flush=True)
        metrics = ranks['0'][0]['sum_ms']
        output[job] = {'ranks': ranks, 'mean_ms_per_owner': {
            k: mean(s['sum_ms'][k] for steps in ranks.values() for s in steps) for k in metrics}}
        (OUT / 'final_ga_kernel_sums.json').write_text(json.dumps(output, indent=2) + '\n')
        print(job, output[job]['mean_ms_per_owner'], flush=True)


if __name__ == '__main__':
    main()
