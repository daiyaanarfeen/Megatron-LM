#!/usr/bin/env python3
"""Audit equal-count recurring microbatches using existing all-rank traces."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analyze_exact_uniform_nep_slowdown import (
    intersection_duration,
    merge_intervals,
    union_duration,
)
from analyze_nep_owner_progress import classify, exclusive
from analyze_workload_parity_pair import event_end, iter_trace_events
from plot_nep_failure_mechanism_details import trace_path


def scan(path: Path, cached: dict) -> dict:
    """Measure MB2--8 on local clocks; keep CPU and GPU views non-additive."""
    steps = cached['steps']
    origins = {}
    metadata = {}
    for e in iter_trace_events(path):
        if e.get('cat') == 'user_annotation' and e['name'].startswith('ProfilerStep#'):
            origins[e['name']] = e['ts']
        if e.get('name') == 'record_param_comms':
            metadata[e.get('args', {}).get('External id')] = e.get('args', {})
    windows = [
        (
            origins[s['step']] + s['backward_microbatches_ms'][0][1] * 1000,
            origins[s['step']] + s['backward_microbatches_ms'][7][1] * 1000,
        )
        for s in steps
    ]
    gpu = [defaultdict(list) for _ in windows]
    cpu = [defaultdict(list) for _ in windows]
    kernels = [defaultdict(list) for _ in windows]
    for e in iter_trace_events(path):
        category = e.get('cat')
        if category not in ('kernel', 'gpu_memcpy', 'gpu_memset', 'cpu_op'):
            continue
        for i, (start, end) in enumerate(windows):
            left, right = max(e['ts'], start), min(event_end(e), end)
            if right <= left:
                continue
            if category == 'cpu_op':
                cpu[i][e['name']].append((left, right))
            else:
                e.setdefault('args', {}).update(
                    metadata.get(e.get('args', {}).get('External id'), {})
                )
                gpu[i][classify(e, [], [])].append((left, right))
                kernels[i][e['name']].append((left, right))
    result = []
    for i, (start, end) in enumerate(windows):
        states = exclusive(gpu[i], start, end)
        assert abs(sum(states.values()) - (end - start) / 1000) < 1e-5
        result.append(
            {
                'step': steps[i]['step'],
                'span_ms': (end - start) / 1000,
                'states_ms': states,
                'kernel_families': {
                    name: {'union_ms': union_duration(rows) / 1000, 'calls': len(rows)}
                    for name, rows in kernels[i].items()
                },
                'cpu_union_ms': {
                    name: union_duration(rows) / 1000 for name, rows in cpu[i].items()
                },
            }
        )
    return {'trace': str(path), 'rank': cached['rank'], 'windows': result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host-ranks', type=str, default='')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for case in json.loads(args.manifest.read_text())['cases']:
        for p in sorted(args.cache.glob('*/[hn]*/rank-*.json')):
            cached = json.loads(p.read_text())
            if cached['case'] != case['label']:
                continue
            destination = args.output / p.parent.parent.name / p.parent.name / p.name
            if args.host_ranks:
                if str(cached['rank']) not in args.host_ranks.split(','):
                    continue
                destination = destination.with_name(f'host-{cached["rank"]}.json')
            if destination.exists():
                continue
            print(case['label'], p.parent.name, cached['rank'], flush=True)
            analyzer = scan_host if args.host_ranks else scan
            result = analyzer(trace_path(case, p.parent.name, cached['rank']), cached)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2) + '\n')


def scan_host(path: Path, cached: dict) -> dict:
    """Bound late submission using all GPU events, without assigning all idle to CPU."""
    origins, launches, grouped = {}, {}, []
    for e in iter_trace_events(path):
        cat, name = e.get('cat'), e.get('name', '')
        if cat == 'user_annotation' and name.startswith('ProfilerStep#'):
            origins[name] = e['ts']
        if cat in ('cuda_runtime', 'cuda_driver') and ('Launch' in name or 'Memcpy' in name):
            launches[e.get('args', {}).get('correlation')] = e['ts']
        if cat == 'cpu_op' and name in ('_GroupedLinear', '_GroupedLinearBackward'):
            grouped.append((e['ts'], event_end(e)))
    grouped = merge_intervals(grouped)
    windows = [
        (
            origins[s['step']] + s['backward_microbatches_ms'][0][1] * 1000,
            origins[s['step']] + s['backward_microbatches_ms'][7][1] * 1000,
        )
        for s in cached['steps']
    ]
    gpu = [[] for _ in windows]
    for e in iter_trace_events(path):
        if e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        for i, (start, end) in enumerate(windows):
            left, right = max(start, e['ts']), min(end, event_end(e))
            if right > left:
                gpu[i].append((left, right, launches.get(e.get('args', {}).get('correlation'))))
    result = []
    for (start, end), events in zip(windows, gpu):
        idle, late = [], []
        previous = start
        for left, right, launch in sorted(events, key=lambda x: (x[0], x[1])):
            if left > previous:
                idle.append((previous, left))
                if launch is not None and min(left, launch) > previous:
                    late.append((previous, min(left, launch)))
            previous = max(previous, right)
        if previous < end:
            idle.append((previous, end))
        result.append(
            {
                'idle_ms': union_duration(idle) / 1000,
                'idle_in_grouped_cpu_ms': intersection_duration(idle, grouped) / 1000,
                'idle_before_next_launch_ms': union_duration(late) / 1000,
                'idle_before_next_launch_in_grouped_cpu_ms': intersection_duration(late, grouped)
                / 1000,
            }
        )
    return {'trace': str(path), 'rank': cached['rank'], 'windows': result}


if __name__ == '__main__':
    main()
