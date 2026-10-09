#!/usr/bin/env python3
"""Summarize per-thread CPU measurements without adding nested train-step scopes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def summarize(path: Path, first: int, last: int) -> dict:
    payload = json.loads(path.read_text())
    meta = payload['metadata']
    if meta.get('diagnostic_version') != 2:
        raise ValueError(f'{path}: expected diagnostic version 2')
    per_iteration = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    cpus = defaultdict(set)
    for row in payload['rows']:
        it, op, tid, start, wall, cpu, voluntary, involuntary, splits, start_cpu, end_cpu = row
        if not first <= it <= last:
            continue
        out = per_iteration[it][op]
        out['wall_ms'] += wall / 1e6
        out['cpu_ms'] += cpu / 1e6
        out['off_cpu_ms'] += max(0, wall - cpu) / 1e6
        out['calls'] += 1
        out['voluntary_switches'] += voluntary
        out['involuntary_switches'] += involuntary
        out['cpu_migrations_within_call'] += start_cpu != end_cpu
        cpus[op].update((start_cpu, end_cpu))
    expected = set(range(first, last + 1))
    if set(per_iteration) != expected:
        raise ValueError(f'{path}: missing iterations {expected-set(per_iteration)}')
    for it, ops in per_iteration.items():
        if set(ops) != {'forward', 'backward', 'train_step'}:
            raise ValueError(f'{path}: missing operations at iteration {it}: {set(ops)}')
    average = {}
    for op in ('forward', 'backward', 'train_step'):
        values = [per_iteration[it][op] for it in sorted(expected)]
        average[op] = {k: mean(v[k] for v in values) for k in values[0]}
        average[op]['wall_stdev_ms'] = stdev(v['wall_ms'] for v in values) if len(values)>1 else 0
        average[op]['observed_cpus'] = sorted(cpus[op])
    return {'metadata': meta, 'first_iteration_zero_based': first,
            'last_iteration_zero_based': last, 'mean_per_iteration': average,
            'per_iteration': dict(per_iteration)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first', type=int, default=16)
    parser.add_argument('--last', type=int, default=20)
    args = parser.parse_args()
    paths = sorted(args.input.glob('*/rank-*.json'))
    if not paths:
        raise ValueError('No completed CPU diagnostics; do not silently accept empty output')
    rows = []
    for path in paths:
        result = summarize(path, args.first, args.last)
        result['mode'] = path.parent.name
        rows.append(result)
        print(result['mode'], result['metadata']['rank'],
              {op: {k: round(v, 3) for k, v in values.items() if k != 'observed_cpus'}
               for op, values in result['mean_per_iteration'].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
