"""Audit ordinary microbatch gaps with exclusive GPU states and host evidence."""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from analyze_a8b_plot_components import trace_events
from analyze_exact_uniform_nep_slowdown import intersection_duration, merge_intervals, union_duration
from analyze_nep_microbatch_phases import CASES, OUT as PHASES, analyze
from analyze_nep_owner_progress import exclusive, location
from plot_nep_failure_mechanism_details import cluster_backward

OUT = PHASES / 'ordinary_audit'
JOBS = ('2994105', '2995392', '2995524')


def scan(job, variant, rank):
    output = OUT/f'{job}_{variant}_{rank}.json'
    if output.exists():
        result = json.loads(output.read_text())
        # GPU-window-clipped runtime intervals do not partition the full host scope.
        for row in list(result['summary'].values()) + result['microbatches']:
            row['expert_host'] = {'cpu_scope_ms': row['expert_host']['cpu_scope_ms']}
        output.write_text(json.dumps(result, indent=2)+'\n')
        return result
    case = next(c for c in CASES if c[0] == job)
    cached = analyze(case, variant, rank)
    cpu, runtime = {}, {}
    bwd, grouped = defaultdict(list), defaultdict(list)
    steps, gpu, host_events, comms = [], [], [], {}
    for e in trace_events(cached['source']):
        if 'ts' not in e or 'dur' not in e:
            continue
        cat, name, a = e.get('cat'), e.get('name', ''), e.get('args', {})
        ts, end, tid = e['ts'], e['ts']+e['dur'], e.get('tid')
        if cat == 'cpu_op':
            cpu[a.get('External id')] = (ts, end, tid, name)
            host_events.append((ts, end, tid, sys.intern(name), 'cpu'))
            if name.startswith('autograd::engine::evaluate_function:'):
                bwd[tid].append((ts, end))
            if name in ('_GroupedLinear', '_GroupedLinearBackward'):
                grouped[tid].append((ts, end, name))
        if cat in ('cuda_runtime', 'cuda_driver'):
            runtime[a.get('correlation')] = (ts, end, tid, name)
            host_events.append((ts, end, tid, sys.intern(name), 'runtime'))
        if cat == 'user_annotation' and name.startswith('ProfilerStep#'):
            steps.append((ts, end, name))
        if name == 'record_param_comms' and a.get('Collective name') != 'wait':
            comms[a.get('External id')] = (a.get('Process Group Description'), a.get('Collective name'))
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            gpu.append((ts, end, cat, sys.intern(name), a.get('External id'), a.get('correlation')))
    for rows in grouped.values():
        rows.sort()
    gs = {t: [a for a, _, _ in rows] for t, rows in grouped.items()}
    source_steps = {s['step']: s for s in cached['steps']}
    results = []
    for lo, hi, step in sorted(steps):
        clusters, _ = cluster_backward([(a, b) for rows in bwd.values() for a, b in rows if lo <= a < b <= hi], case[4])
        cpu_windows = {'forward': [(lo if i == 0 else clusters[i-1][1], a) for i, (a, b) in enumerate(clusters)],
                       'backward': clusters}
        entries = []
        for ts, end, cat, name, ext, corr in gpu:
            if not lo <= ts < hi:
                continue
            parent = cpu.get(ext) or runtime.get(corr)
            name_lower = name.lower()
            group_index = None
            if parent:
                group_index = location(grouped.get(parent[2], []), gs.get(parent[2], []), parent[0])
            expert_gemm = group_index is not None and ('nvjet' in name_lower or 'gemm' in name_lower or 'cutlass' in name_lower)
            if 'device_sync_kernel' in name_lower and 'hybrid_ep' in name_lower:
                kind = 'model_ep_poll'
            elif 'hybrid_ep' in name_lower or 'hybridep' in name_lower:
                kind = 'model_ep_data'
            elif 'nccl' in name_lower:
                desc, op = comms.get(ext, ('unknown', 'unknown'))
                kind = f'nccl:{desc}:{op}'
            elif cat != 'kernel':
                kind = 'memory'
            elif expert_gemm:
                kind = 'expert_gemm'
            else:
                kind = 'other_compute'
            entries.append((ts, end, kind, name, parent, runtime.get(corr)))
        for phase, indexes in [('forward', range(1, case[4])), ('backward', range(case[4]-1))]:
            for i in indexes:
                previous = source_steps[step][phase][i]
                left, right = lo+previous['start_ms']*1000, lo+previous['end_ms']*1000
                ha, hb = cpu_windows[phase][i]
                intervals, sums, counts = defaultdict(list), defaultdict(float), defaultdict(int)
                active_entries = []
                for a, b, kind, name, parent, launch in entries:
                    a, b = max(left, a), min(right, b)
                    if b <= a:
                        continue
                    intervals[kind].append((a, b))
                    sums[kind] += (b-a)/1000
                    counts[kind] += 1
                    active_entries.append((a, b, kind, name, parent, launch))
                states = exclusive(intervals, left, right)
                assert abs(sum(states.values())-previous['span_ms']) < 0.002
                assert abs(states.get('idle', 0)-previous['gpu_idle_ms']) < 0.002
                busy = merge_intervals([p for rows in intervals.values() for p in rows])
                idle, cursor = [], left
                for a, b in busy:
                    if a > cursor:
                        idle.append((cursor, a))
                    cursor = max(cursor, b)
                if cursor < right:
                    idle.append((cursor, right))
                late, examples = [], []
                begins = defaultdict(list)
                for e in active_entries:
                    begins[e[0]].append(e)
                for a, b in idle:
                    following = begins.get(b, [])
                    launches = [e[5][0] for e in following if e[5] is not None]
                    if launches and len(launches) == len(following):
                        until = min(b, min(launches))
                        if until > a:
                            late.append((a, until))
                    if b-a > 100:
                        examples.append({'gap_start_ms': (a-lo)/1000, 'gap_ms': (b-a)/1000,
                                         'next_kernel': following[0][3] if following else None,
                                         'next_host_op': following[0][4][3] if following and following[0][4] else None,
                                         'launch_after_gap_start_ms': (min(launches)-a)/1000 if launches else None})
                host_by_name, grouped_rows = defaultdict(list), []
                for a, b, tid, name, cat in host_events:
                    a, b = max(left, a), min(right, b)
                    if b <= a:
                        continue
                    if cat == 'runtime' or name in ('_GroupedLinear', '_GroupedLinearBackward', 'aten::split_with_sizes', 'aten::copy_', 'aten::item', 'aten::_local_scalar_dense', 'aten::to'):
                        host_by_name[name].append((a, b))
                for rows in grouped.values():
                    grouped_rows.extend((a, b) for a, b, _ in rows if ha <= a < b <= hb)
                expert = {'cpu_scope_ms': union_duration(grouped_rows)/1000}
                host_idle = {n: intersection_duration(rows, idle)/1000 for n, rows in host_by_name.items()}
                results.append({'step': step, 'phase': phase, 'microbatch': i+1,
                                'span_ms': previous['span_ms'], 'exclusive_ms': states,
                                'sum_ms': dict(sums), 'counts': dict(counts),
                                'expert_host': expert, 'idle_in_host_scopes_ms': host_idle,
                                'idle_before_next_activity_host_submission_ms': union_duration(late)/1000,
                                'largest_idle_gaps': sorted(examples, key=lambda e: e['gap_ms'], reverse=True)[:8]})
    summary = {}
    for phase in ('forward', 'backward'):
        rows = [r for r in results if r['phase'] == phase]
        summary[phase] = {'span_ms': mean(r['span_ms'] for r in rows),
                          'late_submission_idle_ms': mean(r['idle_before_next_activity_host_submission_ms'] for r in rows)}
        for key in ('exclusive_ms', 'sum_ms', 'counts', 'expert_host', 'idle_in_host_scopes_ms'):
            keys = {k for r in rows for k in r[key]}
            summary[phase][key] = {k: mean(r[key].get(k, 0) for r in rows) for k in sorted(keys)}
    result = {'source': cached['source'], 'summary': summary, 'microbatches': results}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(job, variant, rank, 'audited', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--variant')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--extra-ranks', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.job:
        scan(args.job, args.variant, args.rank)
        return
    tasks = [(job, v, r) for job in JOBS for v, r in
             ([('healthy', 1), ('nep', 1), ('nep', 9)] if args.extra_ranks else [('healthy', 0), ('nep', 0), ('nep', 8)])]
    def run(task):
        job, v, r = task
        subprocess.run([sys.executable, __file__, '--job', job, '--variant', v, '--rank', str(r)], check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, tasks))


if __name__ == '__main__':
    main()
