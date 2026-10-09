#!/usr/bin/env python3
"""Locate host time inside TE grouped scopes; API rows are inclusive, not additive."""

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

from analyze_exact_uniform_nep_slowdown import union_duration
from analyze_workload_parity_pair import event_end, iter_trace_events


def summarize(events):
    scopes = defaultdict(list)
    runtime = []
    steps = []
    for event in events:
        name = event.get('name', '')
        category = event.get('cat')
        if category == 'user_annotation' and name.startswith('ProfilerStep#'):
            steps.append((event['ts'], event_end(event), name))
        if category == 'cpu_op' and name in ('_GroupedLinear', '_GroupedLinearBackward'):
            scopes[event['tid']].append((event['ts'], event_end(event), name))
        if category in ('cuda_runtime', 'cuda_driver'):
            runtime.append(event)
    steps.sort()
    step_starts = [row[0] for row in steps]
    starts = {}
    for tid, rows in scopes.items():
        rows.sort()
        assert all(a[1] <= b[0] for a, b in zip(rows, rows[1:])), 'Nested grouped scopes'
        starts[tid] = [row[0] for row in rows]
    output = defaultdict(lambda: defaultdict(lambda: {'scope_ms': 0., 'calls': 0,
                                                     'runtime': [], 'launch': [],
                                                     'api': defaultdict(lambda: [0, 0.])}))
    for rows in scopes.values():
        for left, right, name in rows:
            index = bisect.bisect_right(step_starts, left) - 1
            if index >= 0 and right <= steps[index][1]:
                result = output[steps[index][2]][name]
                result['scope_ms'] += (right-left)/1000
                result['calls'] += 1
    for event in runtime:
        tid = event['tid']
        index = bisect.bisect_right(starts.get(tid, []), event['ts']) - 1
        if index < 0:
            continue
        left, right, name = scopes[tid][index]
        step = bisect.bisect_right(step_starts, left) - 1
        if event_end(event) > right or step < 0 or right > steps[step][1]:
            continue
        result = output[steps[step][2]][name]
        interval = (event['ts'], event_end(event))
        result['runtime'].append(interval)
        if 'Launch' in event['name']:
            result['launch'].append(interval)
        api = result['api'][event['name']]
        api[0] += 1
        api[1] += event.get('dur', 0)/1000
    for operations in output.values():
        for result in operations.values():
            result['runtime_union_ms'] = union_duration(result.pop('runtime'))/1000
            result['launch_union_ms'] = union_duration(result.pop('launch'))/1000
            result['outside_runtime_ms'] = result['scope_ms']-result['runtime_union_ms']
            assert result['outside_runtime_ms'] >= -1e-6
    return dict(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(iter_trace_events(args.trace))
    if not result:
        raise ValueError('No complete grouped scopes in profiler steps')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'trace': str(args.trace), 'steps': result}, indent=2)+'\n')
