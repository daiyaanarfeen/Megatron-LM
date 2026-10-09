#!/usr/bin/env python3
"""Correlate owner GPU progress, gradient drain, and host waits in NEP traces."""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

from analyze_exact_uniform_nep_slowdown import (
    intersection_duration,
    merge_intervals,
    union_duration,
)
from analyze_workload_parity_pair import event_end, iter_trace_events
from plot_nep_failure_mechanism_details import cluster_backward, role_counts, trace_path


def location(rows: list, starts: list, timestamp: float) -> int | None:
    """Find the half-open interval containing a timestamp."""
    index = bisect.bisect_right(starts, timestamp) - 1
    return index if index >= 0 and timestamp < rows[index][1] else None


def classify(event: dict, param_scopes: list, param_starts: list) -> str:
    """Classify GPU activity by collective semantics and process-group role."""
    name = event['name'].lower()
    args = event.get('args', {})
    group = str(args.get('Process Group Description', '')).lower()
    collective = str(args.get('Collective name', '')).lower()
    if 'nccl' in name:
        if 'owner_gather' in group:
            return 'grad_gather'
        if 'owner_transfer' in group:
            # Broadcast redistributes parameters; gradient reshard uses all-to-all.
            if (
                'broadcast' in collective
                or location(param_scopes, param_starts, event['ts']) is not None
            ):
                return 'param_redistribution'
            return 'grad_scatter'
        if group == 'ep_dp' or 'expert_data_parallel' in group:
            return 'expert_param_ag' if 'allgather' in collective else 'expert_grad_rs'
        if group == 'dp_cp' or 'data_parallel_group_with_cp' in group:
            return 'dense_param_ag' if 'allgather' in collective else 'dense_dp_sync'
        return 'other_collective'
    if 'hybrid_ep::device_sync_kernel' in name:
        return 'model_ep_poll'
    if 'hybrid_ep::' in name or 'hybridep' in name:
        return 'model_ep_other'
    if event['cat'] != 'kernel':
        return 'memory'
    return 'other_kernel'


def exclusive(intervals: dict, start: float, end: float) -> dict[str, float]:
    """Partition microsecond intervals into disjoint activity masks in milliseconds."""
    changes = defaultdict(lambda: defaultdict(int))
    changes[start]
    changes[end]
    for category, rows in intervals.items():
        for left, right in merge_intervals(rows):
            left, right = max(start, left), min(end, right)
            if right > left:
                changes[left][category] += 1
                changes[right][category] -= 1
    points = sorted(changes)
    active = defaultdict(int)
    totals = defaultdict(float)
    for left, right in zip(points, points[1:]):
        for category, delta in changes[left].items():
            active[category] += delta
        mask = '+'.join(sorted(k for k, count in active.items() if count)) or 'idle'
        totals[mask] += (right - left) / 1000
    assert abs(sum(totals.values()) - (end - start) / 1000) < 1e-5
    return dict(totals)


def scan(path: Path, microbatches: int) -> list[dict]:
    """Correlate GPU backward completion, optimizer execution, and collective activity."""
    steps, backward, cpu, runtime, annotations = [], defaultdict(list), {}, [], []
    launches = {}
    accumulation = defaultdict(list)
    comms = {}
    for event in iter_trace_events(path):
        cat, name = event.get('cat'), str(event.get('name', ''))
        args = event.get('args', {})
        if cat == 'user_annotation' and name.startswith('ProfilerStep#'):
            steps.append(event)
        if cat == 'cpu_op' and name.startswith('autograd::engine::evaluate_function:'):
            backward[event['tid']].append((event['ts'], event_end(event)))
            if name.endswith('AccumulateGrad'):
                accumulation[event['tid']].append((event['ts'], event_end(event)))
        if cat == 'cpu_op' and args.get('External id') is not None:
            cpu[args['External id']] = (event['ts'], event['tid'], name)
        if name == 'record_param_comms' and args.get('Collective name') != 'wait':
            comms[args.get('External id')] = args
        if cat in ('cuda_runtime', 'cuda_driver'):
            if 'Launch' in name:
                launches[args.get('correlation')] = (event['ts'], event['tid'], name)
            elif 'Synchronize' in name:
                runtime.append(event)
        if cat in ('user_annotation', 'gpu_user_annotation') and (
            name.startswith('Optimizer.step#') or name.startswith('nep_')
        ):
            annotations.append(event)
    steps.sort(key=lambda x: x['ts'])
    nep_scopes = defaultdict(list)
    for event in annotations:
        if event['cat'] == 'user_annotation' and event['name'].startswith('nep_'):
            nep_scopes[event['tid']].append((event['ts'], event_end(event)))
    nep_scopes = {tid: merge_intervals(rows) for tid, rows in nep_scopes.items()}
    nep_starts = {tid: [a for a, b in rows] for tid, rows in nep_scopes.items()}
    accumulation = {tid: merge_intervals(rows) for tid, rows in accumulation.items()}
    accumulation_starts = {tid: [a for a, b in rows] for tid, rows in accumulation.items()}
    out = []
    for step in steps:
        start, end = step['ts'], event_end(step)
        host_backward = [
            (a, b) for rows in backward.values() for a, b in rows if start <= a < b <= end
        ]
        clusters, diagnostic = cluster_backward(host_backward, microbatches)
        starts = [a for a, b in clusters]
        thread_rows = {tid: merge_intervals(rows) for tid, rows in backward.items()}
        thread_starts = {tid: [a for a, b in rows] for tid, rows in thread_rows.items()}
        param_scopes = merge_intervals(
            [
                (e['ts'], event_end(e))
                for e in annotations
                if e['cat'] == 'gpu_user_annotation' and e['name'] == 'nep_distopt_param_scatter'
            ]
        )
        param_starts = [a for a, b in param_scopes]
        # Stream GPU events separately so long accumulation traces fit in memory.
        kernels = (
            e
            for e in iter_trace_events(path)
            if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')
            and e['ts'] < end
            and event_end(e) > start
        )
        intervals, bwd_gpu = defaultdict(list), defaultdict(list)
        calls, timeline = [], []
        for e in kernels:
            a = e.get('args', {})
            meta = comms.get(a.get('External id'), {})
            for k, v in meta.items():
                a.setdefault(k, v)
            category = classify(e, param_scopes, param_starts)
            left, right = max(start, e['ts']), min(end, event_end(e))
            source = cpu.get(a.get('External id'))
            launch = launches.get(a.get('correlation'))
            if source is None and launch is not None:
                source = launch
            if source is not None and category in ('other_kernel', 'memory'):
                ts, tid, _ = source
                if location(nep_scopes.get(tid, []), nep_starts.get(tid, []), ts) is not None:
                    category = 'nep_staging'
                elif (
                    location(accumulation.get(tid, []), accumulation_starts.get(tid, []), ts)
                    is not None
                ):
                    category = 'gradient_accumulation'
            intervals[category].append((left, right))
            mb = None
            if source is not None:
                ts, tid, _ = source
                if location(thread_rows.get(tid, []), thread_starts.get(tid, []), ts) is not None:
                    mb = location(clusters, starts, ts)
            if mb is not None and category in ('other_kernel', 'memory'):
                bwd_gpu[mb].append((left, right))
            if 'nccl' in e['name'].lower():
                calls.append(
                    {
                        'category': category,
                        'group': a.get('Process Group Name'),
                        'ranks': a.get('Process Group Ranks'),
                        'op': a.get('Collective name'),
                        'elements': a.get('In msg nelems'),
                        'dtype': a.get('dtype'),
                        'start_ms': (left - start) / 1000,
                        'end_ms': (right - start) / 1000,
                        'launch_ms': (launch[0] - start) / 1000 if launch else None,
                        'stream': a.get('stream'),
                    }
                )
            if category != 'other_kernel' or right - left >= 100:
                timeline.append(
                    [category, (left - start) / 1000, (right - start) / 1000, a.get('stream')]
                )
        if sorted(bwd_gpu) != list(range(microbatches)):
            raise ValueError(
                f'{path} {step["name"]}: missing correlated backward microbatches {sorted(bwd_gpu)}'
            )
        bwd_end = max(right for rows in bwd_gpu.values() for _, right in rows)
        adam = [
            e
            for e in annotations
            if e['cat'] == 'gpu_user_annotation'
            and e['name'].startswith('Optimizer.step#')
            and start <= e['ts'] < end
        ]
        if not adam:
            raise ValueError(f'{path}: no GPU optimizer annotations')
        adam_start, adam_end = min(e['ts'] for e in adam), max(event_end(e) for e in adam)
        if adam_start < bwd_end:
            raise ValueError(f'{path}: optimizer precedes correlated backward completion')
        cpu_adam = [
            e
            for e in annotations
            if e['cat'] == 'user_annotation'
            and e['name'].startswith('Optimizer.step#')
            and start <= e['ts'] < end
        ]
        cpu_adam_end = max(event_end(e) for e in cpu_adam)
        syncs = []
        for e in runtime:
            if 'Synchronize' in e['name'] and start <= e['ts'] < end:
                parent = cpu.get(e.get('args', {}).get('External id'))
                syncs.append(
                    {
                        'name': e['name'],
                        'parent': parent[2] if parent else None,
                        'start_ms': (e['ts'] - start) / 1000,
                        'end_ms': (event_end(e) - start) / 1000,
                    }
                )
        phases = {
            'through_backward': (bwd_end - start) / 1000,
            'gradient_drain': (adam_start - bwd_end) / 1000,
            'adam': (adam_end - adam_start) / 1000,
            'after_adam': (end - adam_end) / 1000,
        }
        assert abs(sum(phases.values()) - (end - start) / 1000) < 1e-5
        operation = {}
        for category, rows in intervals.items():
            other = [pair for k, v in intervals.items() if k != category for pair in v]
            resident = union_duration(rows) / 1000
            operation[category] = {
                'residency_ms': resident,
                'exposed_ms': resident - intersection_duration(rows, other) / 1000,
            }
        out.append(
            {
                'step': step['name'],
                'span_ms': (end - start) / 1000,
                'phases_ms': phases,
                'cpu_adam_end_ms': (cpu_adam_end - start) / 1000,
                'gpu_adam_start_ms': (adam_start - start) / 1000,
                'gpu_adam_end_ms': (adam_end - start) / 1000,
                'backward_end_ms': (bwd_end - start) / 1000,
                'backward_microbatches_ms': [
                    [min(a for a, b in bwd_gpu[i]) - start, max(b for a, b in bwd_gpu[i]) - start]
                    for i in range(microbatches)
                ],
                'drain_states_ms': exclusive(intervals, bwd_end, adam_start),
                'step_states_ms': exclusive(intervals, start, end),
                'operations': operation,
                'collectives': calls,
                'syncs': syncs,
                'timeline': timeline,
                'cluster_diagnostic': diagnostic,
            }
        )
        for row in out[-1]['backward_microbatches_ms']:
            row[:] = [value / 1000 for value in row]
    return out


def normalize_dense_labels(directory: Path) -> None:
    """Correct the historical broad DP label; scalar all-reduces are not gradient RS."""

    def normalize(value):
        if isinstance(value, dict):
            return {
                key.replace('dense_grad_rs', 'dense_dp_sync'): normalize(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str):
            return value.replace('dense_grad_rs', 'dense_dp_sync')
        return value

    count = 0
    for path in directory.glob('*/*/rank-*.json'):
        data = json.loads(path.read_text())
        if data.get('dense_label_version') == 2:
            continue
        data = normalize(data)
        data['dense_label_version'] = 2
        path.write_text(json.dumps(data) + '\n')
        count += 1
    print(f'Normalized dense labels in {count} files')


def validate(cases: list[dict], directory: Path) -> None:
    """Check phase closure and owner/peer expert-collective signatures, not numeric gradients."""
    checks = 0
    for case in cases:
        full, reduced = role_counts(case)
        slug = case['label'].replace('/', '_').replace(' ', '_')
        for mode in ('healthy', 'nep'):
            for owner in range(reduced):
                pair = [
                    json.loads((directory / slug / mode / f'rank-{rank}.json').read_text())
                    for rank in (owner, full + owner)
                ]
                assert len(pair[0]['steps']) == len(pair[1]['steps'])
                for left, right in zip(pair[0]['steps'], pair[1]['steps']):
                    assert left['step'] == right['step']
                    for step in (left, right):
                        assert abs(sum(step['phases_ms'].values()) - step['span_ms']) < 1e-5
                        assert abs(sum(step['step_states_ms'].values()) - step['span_ms']) < 1e-5
                        assert (
                            abs(
                                sum(step['drain_states_ms'].values())
                                - step['phases_ms']['gradient_drain']
                            )
                            < 1e-5
                        )

                    def signature(step):
                        return [
                            (c['category'], c['ranks'], c['op'], c['elements'], c['dtype'])
                            for c in sorted(step['collectives'], key=lambda c: c['start_ms'])
                            if c['category'] in ('expert_grad_rs', 'expert_param_ag')
                        ]

                    assert signature(left) == signature(right), (
                        case['label'],
                        mode,
                        owner,
                        left['step'],
                    )
                    checks += 1
    result = {'owner_peer_step_checks': checks, 'phase_closure_and_expert_signatures': 'passed'}
    (directory / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ranks', help='Comma-separated ranks; default all ranks')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--normalize-dense-labels', action='store_true')
    args = parser.parse_args()
    cases = json.loads(args.summary.read_text())['cases']
    if args.normalize_dense_labels:
        normalize_dense_labels(args.output_dir)
        return
    if args.validate_only:
        validate(cases, args.output_dir)
        return
    for case in cases:
        full, reduced = role_counts(case)
        for mode in ('healthy', 'nep'):
            count = 2 * full if mode == 'healthy' else full + reduced
            ranks = range(count) if args.ranks is None else [int(x) for x in args.ranks.split(',')]
            for rank in ranks:
                safe_label = case['label'].replace('/', '_').replace(' ', '_')
                output = args.output_dir / safe_label / mode / f'rank-{rank}.json'
                if output.exists():
                    print(f'cached {output}', flush=True)
                    continue
                ga = case['batch_configuration'][mode]['replica_num_microbatches'][rank >= full]
                print(f'scan {case["label"]} {mode} rank {rank}', flush=True)
                result = scan(trace_path(case, mode, rank), ga)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            'case': case['label'],
                            'mode': mode,
                            'rank': rank,
                            'full_ranks': full,
                            'reduced_ranks': reduced,
                            'steps': result,
                            'dense_label_version': 2,
                        }
                    )
                    + '\n'
                )


if __name__ == '__main__':
    main()
