#!/usr/bin/env python3
"""Measure work and gaps inside named NEP GPU scopes without counting overlaps twice."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analyze_nep_owner_progress import classify, exclusive
from analyze_workload_parity_pair import event_end, iter_trace_events
from plot_nep_failure_mechanism_details import trace_path


def scan(path: Path) -> dict:
    """Measure all GPU work and host-launch gaps within the named GPU scopes."""
    scopes, steps, comms, launches = [], [], {}, {}
    for event in iter_trace_events(path):
        category, name = event.get('cat'), event.get('name', '')
        if category == 'gpu_user_annotation' and name in (
            'nep_distopt_param_scatter',
            'nep_distopt_stage_owner_grads',
        ):
            scopes.append(event)
        if category == 'user_annotation' and name.startswith('ProfilerStep#'):
            steps.append(event)
        if name == 'record_param_comms' and event.get('args', {}).get('Collective name') != 'wait':
            comms[event['args'].get('External id')] = event['args']
        if category in ('cuda_runtime', 'cuda_driver') and ('Launch' in name or 'Memcpy' in name):
            launches[event.get('args', {}).get('correlation')] = event['ts']
    intervals = defaultdict(list)
    kernel_types = defaultdict(lambda: [0, 0.0])
    ordered = defaultdict(list)
    for event in iter_trace_events(path):
        if event.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        event.setdefault('args', {}).update(comms.get(event.get('args', {}).get('External id'), {}))
        category = classify(event, [], [])
        for index, scope in enumerate(scopes):
            left, right = max(event['ts'], scope['ts']), min(event_end(event), event_end(scope))
            if right <= left:
                continue
            intervals[index, category].append((left, right))
            launch = launches.get(event.get('args', {}).get('correlation'))
            ordered[index].append((left, right, launch))
            key = scope['name'], event['name']
            kernel_types[key][0] += 1
            kernel_types[key][1] += (right - left) / 1000 / len(steps)
    totals = defaultdict(lambda: defaultdict(float))
    late_launch = defaultdict(float)
    for index, scope in enumerate(scopes):
        states = exclusive(
            {cat: rows for (i, cat), rows in intervals.items() if i == index},
            scope['ts'],
            event_end(scope),
        )
        for state, value in states.items():
            totals[scope['name']][state] += value / len(steps)
        previous_end = scope['ts']
        for left, right, launch in sorted(ordered[index]):
            if left > previous_end and launch is not None:
                late_launch[scope['name']] += (
                    max(0, min(left, launch) - previous_end) / 1000 / len(steps)
                )
            previous_end = max(previous_end, right)
    return {
        'trace': str(path),
        'profiled_steps': len(steps),
        'scope_states_mean_ms': totals,
        'idle_before_next_host_launch_mean_ms': dict(late_launch),
        'scoped_gpu_events': [
            {
                'scope': s,
                'name': n,
                'calls_per_step': count / len(steps),
                'residency_mean_ms': duration,
            }
            for (s, n), (count, duration) in kernel_types.items()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for case in json.loads(args.summary.read_text())['cases']:
        print(case['label'], flush=True)
        result = scan(trace_path(case, 'nep', 0))
        result['label'] = case['label']
        results.append(result)
    args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
