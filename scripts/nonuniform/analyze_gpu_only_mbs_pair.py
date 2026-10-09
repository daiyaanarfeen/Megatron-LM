"""Compare GPU-only a770 MBS1/2 traces, transferring labels but never timings."""

import bisect
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_exact_uniform_nep_slowdown import intersection_duration, merge_intervals, union_duration
from analyze_workload_parity_pair import classify_process_group, iter_trace_events

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen')
RUNS = ROOT / 'slurm_runs/soft_clique_baseline'
OUTPUT = ROOT / 'analysis/a770_gpu_only_mbs_pair'
WORKLOAD = 'scaling_ladder__a770m_4b_moe_215b'
JOBS = {1: '2998515', 2: '2997621'}


def trace_path(job, mode, case, rank):
    return RUNS / job / mode / 'ep8' / WORKLOAD / case / job / 'torch_profile' / f'rank-{rank}.json.gz'


def template(path):
    cpu, scopes, gpu = {}, defaultdict(list), []
    for e in iter_trace_events(path):
        if e.get('cat') == 'cpu_op':
            cpu[e.get('args', {}).get('External id')] = (e['tid'], e['ts'])
            if e['name'] in ('_GroupedLinear', '_GroupedLinearBackward'):
                scopes[e['tid']].append((e['ts'], e['ts'] + e['dur']))
        elif e.get('cat') == 'kernel':
            gpu.append(e)
    for spans in scopes.values():
        spans.sort()
    streams = defaultdict(list)
    for e in gpu:
        streams[e['args']['stream']].append(e)
    expert_streams = {s for s, events in streams.items() if all(e['name'].startswith('nvjet') for e in events)}
    assert len(expert_streams) == 4, (path, expert_streams)
    for stream in expert_streams:
        for e in streams[stream]:
            thread, ts = cpu[e['args']['External id']]
            spans = scopes[thread]
            i = bisect.bisect_right(spans, (ts, float('inf'))) - 1
            assert i >= 0 and ts < spans[i][1], (path, e['name'])
    nccl = defaultdict(list)
    for e in sorted(gpu, key=lambda e: e['ts']):
        if 'nccl' in e['name'].lower():
            nccl[e['args']['stream']].append((e['name'], classify_process_group(e['args']['Process Group Description'])))
    return expert_streams, dict(nccl)


def analyze(path, reference, experts):
    expert_streams, expected = reference
    gpu, nccl = [], defaultdict(list)
    for e in iter_trace_events(path):
        assert e.get('cat') != 'cpu_op', path
        if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            gpu.append(e)
            if e['cat'] == 'kernel' and 'nccl' in e['name'].lower():
                nccl[e['args']['stream']].append(e)
    assert set(nccl) == set(expected), (path, set(nccl), set(expected))
    for stream, events in nccl.items():
        events.sort(key=lambda e: e['ts'])
        assert [e['name'] for e in events] == [n for n, _ in expected[stream]], (path, stream)
        for e, (_, label) in zip(events, expected[stream]):
            e['label'] = label
    intervals = defaultdict(list)
    gemms = []
    for e in gpu:
        span = (e['ts'], e['ts'] + e['dur'])
        if 'label' in e:
            label = e['label']
        elif 'hybrid_ep::' in e['name']:
            label = 'hybrid_ep'
        elif e['cat'] != 'kernel':
            label = 'memory_ops'
        else:
            label = 'other_gpu_kernels'
        intervals[label].append(span)
        if e['cat'] == 'kernel' and e['args']['stream'] in expert_streams:
            assert e['name'].startswith('nvjet'), (path, e['name'])
            gemms.append(span)
    assert len(gemms) == 4 * 15 * 3 * experts * 6, (path, len(gemms), experts)
    merged = {k: merge_intervals(v) for k, v in intervals.items()}
    all_spans = [v for spans in merged.values() for v in spans]
    row = {'gpu_busy_ms': union_duration(all_spans) / 4000, 'expert_gemm_ms': union_duration(gemms) / 4000, 'expert_kernels_per_step': len(gemms) / 4, 'categories': {}}
    for label, spans in merged.items():
        others = [v for key, values in merged.items() if key != label for v in values]
        residency = union_duration(spans)
        hidden = intersection_duration(spans, others)
        row['categories'][label] = {'union_ms': residency / 4000, 'exposed_ms': (residency - hidden) / 4000}
    return row


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {mbs: {'ranks': {}, 'timing': {}} for mbs in JOBS}
    for case, size in (('healthy', 16), ('nep', 14)):
        for rank in range(size):
            ref = None
            experts = 16 if case == 'healthy' or rank < 8 else (22 if rank < 10 else 21)
            for mbs, job in JOBS.items():
                cache = OUTPUT / f'{job}_{case}_{rank}.json'
                if cache.exists():
                    row = json.loads(cache.read_text())
                else:
                    if ref is None:
                        ref = template(trace_path(JOBS[2], 'cpu_cuda', case, rank))
                    row = analyze(trace_path(job, 'cuda', case, rank), ref, experts)
                    cache.write_text(json.dumps(row, indent=2) + '\n')
                result[mbs]['ranks'][f'{case}_{rank}'] = row
                print(job, case, rank, row['expert_gemm_ms'], flush=True)
        for mbs, job in JOBS.items():
            log = trace_path(job, 'cuda', case, 0).parent.parent / 'driver.log'
            timings = {int(i): float(t) for i, t in re.findall(r'iteration\s+(\d+)/\s*24.*?elapsed time per iteration \(ms\):\s*([\d.]+)', log.read_text())}
            result[mbs]['timing'][case] = {window: {'ms': [timings[i] for i in indices], 'mean_ms': statistics.mean(timings[i] for i in indices)} for window, indices in (('recorded', range(10, 14)), ('post', range(15, 25)))}
    (OUTPUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
