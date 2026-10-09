"""Compare first-forward/final-backward phases using existing correlated traces."""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, stdev

from analyze_a8b_plot_components import trace_events
from analyze_exact_uniform_nep_slowdown import intersection_duration, merge_intervals
from analyze_nep_final_ga_sums import ROOT
from analyze_nep_owner_progress import classify, location
from plot_nep_failure_mechanism_details import cluster_backward

OUT = ROOT / 'analysis/nep_microbatch_phases_20260914'
CASES = [
    ('2994105', 'scaling_ladder__a500m_2b_moe_140b', 'a500 MBS1', 12, 3, 8, 1),
    ('2994279', 'scaling_ladder__a500m_2b_moe_140b', 'a500 MBS3', 12, 3, 8, 3),
    ('2995392', 'scaling_ladder__a770m_4b_moe_215b', 'a770 MBS1', 15, 3, 8, 1),
    ('2995524', 'scaling_ladder__a770m_4b_moe_215b', 'a770 MBS2', 15, 3, 8, 2),
    ('2994648', 'a8b_120b_latentmoe_1t', 'a8b MBS1', 30, 8, 16, 1),
]


def analyze(case, variant, rank):
    job, model, title, layers, ga, ep, mbs = case
    cache = OUT / f'{job}_{variant}_{rank}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    path = ROOT / f'slurm_runs/soft_clique_baseline/{job}/ep{ep}/{model}/{variant}/{job}/torch_profile/rank-{rank}.json.gz'
    cpu, launches, comms = {}, {}, {}
    backward, nep = defaultdict(list), defaultdict(list)
    steps, grouped, gpu, optimizer, scatter = [], [], [], [], []
    for e in trace_events(path):
        cat, name, a = e.get('cat'), e.get('name', ''), e.get('args', {})
        if 'ts' not in e or 'dur' not in e:
            continue
        ts, end = e['ts'], e['ts'] + e['dur']
        if cat == 'cpu_op':
            if 'External id' in a:
                cpu[a['External id']] = (ts, e['tid'])
            if name.startswith('autograd::engine::evaluate_function:'):
                backward[e['tid']].append((ts, end))
                if name.endswith('_GroupedLinearBackward'):
                    grouped.append(ts)
        if cat in ('cuda_runtime', 'cuda_driver') and 'Launch' in name:
            launches[a.get('correlation')] = (ts, e['tid'])
        if name == 'record_param_comms' and a.get('Collective name') != 'wait':
            comms[a.get('External id')] = {k: a[k] for k in ('Collective name', 'Process Group Description') if k in a}
        if cat == 'user_annotation':
            if name.startswith('ProfilerStep#'):
                steps.append((ts, end, name))
            if name.startswith('nep_'):
                nep[e['tid']].append((ts, end))
            if name == 'nep_distopt_param_scatter':
                scatter.append((ts, end))
        if cat == 'gpu_user_annotation' and name.startswith('Optimizer.step#'):
            optimizer.append((ts, end))
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            # Keep only fields needed for correlation and classification.
            gpu.append((ts, end, sys.intern(cat), sys.intern(name), a.get('External id'), a.get('correlation')))
    steps.sort()
    assert len(steps) == 4, (path, len(steps))
    backward = {t: merge_intervals(v) for t, v in backward.items()}
    nep = {t: merge_intervals(v) for t, v in nep.items()}
    bs = {t: [a for a, _ in v] for t, v in backward.items()}
    ns = {t: [a for a, _ in v] for t, v in nep.items()}
    all_busy = merge_intervals([(a, b) for a, b, *_ in gpu])
    results = []
    for lo, hi, name in steps:
        clusters, diagnostic = cluster_backward([(a, b) for v in backward.values() for a, b in v if lo <= a < b <= hi], ga)
        counts = [sum(a <= t < b for t in grouped) for a, b in clusters]
        assert counts == [2*layers]*ga, (path, name, counts)
        assert diagnostic['minimum_boundary_gap_ms'] > diagnostic['maximum_internal_gap_ms']
        fw = [(lo if i == 0 else clusters[i-1][1], clusters[i][0]) for i in range(ga)]
        fwstarts, bwstarts = [a for a, _ in fw], [a for a, _ in clusters]
        fgpu, bgpu, fcompute, bcompute = [defaultdict(list) for _ in range(4)]
        unmapped = 0
        for ts, end, cat, kernel, ext, corr in gpu:
            if not lo <= ts < hi:
                continue
            source = cpu.get(ext) or launches.get(corr)
            if source is None:
                unmapped += 1
                continue
            host, tid = source
            category = classify({'name': kernel, 'cat': cat, 'args': comms.get(ext, {})}, [], [])
            in_nep = location(nep.get(tid, []), ns.get(tid, []), host) is not None
            i = location(fw, fwstarts, host)
            if i is not None:
                fgpu[i].append((ts, end))
                if category == 'other_kernel' and not in_nep:
                    fcompute[i].append((ts, end))
            i = location(clusters, bwstarts, host)
            if i is not None and location(backward.get(tid, []), bs.get(tid, []), host) is not None:
                if not in_nep and category in ('other_kernel', 'memory', 'model_ep_poll', 'model_ep_other', 'other_collective'):
                    bgpu[i].append((ts, end))
                if not in_nep and category == 'other_kernel':
                    bcompute[i].append((ts, end))
        assert unmapped == 0, (path, name, unmapped)
        assert all(fgpu[i] and bgpu[i] and bcompute[i] for i in range(ga))
        opt = [(a, b) for a, b in optimizer if lo <= a < hi]
        assert opt, (path, name, 'missing GPU optimizer')

        def metrics(rows, compute):
            start, end = min(a for a, _ in rows), max(b for _, b in rows)
            span = (end-start)/1000
            busy = intersection_duration([(start, end)], all_busy)/1000
            return {'start_ms': (start-lo)/1000, 'end_ms': (end-lo)/1000,
                    'span_ms': span, 'all_gpu_busy_ms': busy, 'gpu_idle_ms': max(0, span-busy),
                    'compute_sum_ms': sum(b-a for a, b in compute)/1000}

        forward = [metrics(fgpu[i], fcompute[i]) for i in range(ga)]
        back = [metrics(bgpu[i], bcompute[i]) for i in range(ga)]
        tail = (min(a for a, _ in opt)-lo)/1000-back[-1]['end_ms']
        assert tail >= 0, (path, name, tail)
        scatters_by_fwd = [sum(a <= s < b for s, _ in scatter) for a, b in fw]
        assert not any(scatters_by_fwd[1:]), (path, name, scatters_by_fwd)
        results.append({'step': name, 'forward': forward, 'backward': back,
                        'gradient_finalization_tail_ms': tail, 'scatter_calls_by_forward': scatters_by_fwd,
                        'grouped_backward_counts': counts, 'cluster_diagnostics': diagnostic,
                        'unmapped_gpu_activities': unmapped})
    result = {'job': job, 'title': title, 'variant': variant, 'rank': rank, 'ga': ga,
              'source': str(path), 'steps': results}
    cache.write_text(json.dumps(result, indent=2)+'\n')
    print(job, variant, rank, 'validated', flush=True)
    return result


def aggregate():
    result = []
    for case in CASES:
        job, _, title, _, ga, ep, mbs = case
        series = {}
        for role, variant, rank in [('Healthy', 'healthy', 0), ('NEP full owner', 'nep', 0), ('NEP reduced owner', 'nep', ep)]:
            data = analyze(case, variant, rank)
            old_name = (f'a8b_plot_components_{variant}_{rank}.json' if job == '2994648'
                        else f'compute_host_{job}_{variant}_{rank}.json')
            reference = json.loads((ROOT/'analysis/nep_small_model_parity_20260911'/old_name).read_text())
            reference = reference['steps'] if isinstance(reference, dict) else reference
            assert len(reference) == len(data['steps'])
            for step, previous in zip(data['steps'], reference):
                assert step['step'] == previous['step']
                assert abs(step['backward'][-1]['compute_sum_ms']-previous['sum_ms']['backward_compute']) < 0.002
            values = defaultdict(list)
            for s in data['steps']:
                for key, rows in [('F1', s['forward'][:1]), ('Frest', s['forward'][1:]),
                                  ('Brest', s['backward'][:-1]), ('Blast', s['backward'][-1:])]:
                    for metric in ('span_ms', 'compute_sum_ms', 'gpu_idle_ms'):
                        values[f'{key}_{metric}'].append(mean(r[metric] for r in rows))
                values['tail_ms'].append(s['gradient_finalization_tail_ms'])
            series[role] = {k: {'mean': mean(v), 'std': stdev(v), 'samples': v} for k, v in values.items()}
            series[role]['scatter_calls_by_forward'] = [s['scatter_calls_by_forward'] for s in data['steps']]
        result.append({'job': job, 'title': title, 'ga': ga, 'ep': ep, 'mbs': mbs, 'series': series})
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--variant')
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.job:
        analyze(next(c for c in CASES if c[0] == args.job), args.variant, args.rank)
        return
    tasks = [(c[0], v, r) for c in CASES for v, r in [('healthy', 0), ('nep', 0), ('nep', c[5])]]

    def run(task):
        job, variant, rank = task
        if not (OUT/f'{job}_{variant}_{rank}.json').exists():
            subprocess.run([sys.executable, __file__, '--job', job, '--variant', variant, '--rank', str(rank)], check=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, tasks))
    aggregate()
    print(OUT/'summary.json', flush=True)


if __name__ == '__main__':
    main()
