"""Partition GPU activity during NEP host scopes, without inferring causality."""

import gzip
import json
from collections import defaultdict
from statistics import mean

from analyze_nep_final_ga_sums import CASES, OUT, ROOT
from analyze_nep_owner_progress import exclusive, location
from analyze_exact_uniform_nep_slowdown import merge_intervals, intersection_duration, union_duration


def scan(path):
    with gzip.open(path, 'rt') as handle:
        events = json.load(handle)['traceEvents']
    scopes, named, cpu, launches = defaultdict(list), defaultdict(list), {}, {}
    metadata = {}
    steps = []
    for e in events:
        cat, name, a = e.get('cat'), e.get('name', ''), e.get('args', {})
        if cat == 'user_annotation' and name.startswith('ProfilerStep#'):
            steps.append(e)
        if cat == 'user_annotation' and name.startswith('nep_'):
            pair = (e['ts'], e['ts'] + e['dur'])
            scopes[e['tid']].append(pair)
            named[name].append(pair)
        if cat == 'cpu_op' and 'External id' in a:
            cpu[a['External id']] = (e['ts'], e['tid'])
        if name == 'record_param_comms' and a.get('Collective name') != 'wait':
            metadata[a.get('External id')] = a
        if cat in ('cuda_runtime', 'cuda_driver') and 'Launch' in name:
            launches[a.get('correlation')] = (e['ts'], e['tid'])
    scopes = {t: merge_intervals(v) for t, v in scopes.items()}
    starts = {t: [a for a, _ in v] for t, v in scopes.items()}
    out = []
    for s in steps:
        lo, hi = s['ts'], s['ts'] + s['dur']
        intervals = defaultdict(list)
        param_comms = defaultdict(list)
        non_scatter_activity = []
        intervals['host'] = [(a, b) for v in scopes.values() for a, b in v if a < hi and b > lo]
        for e in events:
            if e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset') or e['ts'] >= hi or e['ts'] + e['dur'] <= lo:
                continue
            name = e['name'].lower()
            args = e.get('args', {})
            meta = metadata.get(args.get('External id'), {})
            group, op = meta.get('Process Group Description'), meta.get('Collective name', '')
            is_scatter = 'nccl' in name and group == 'nep_owner_transfer' and op == 'broadcast'
            if not is_scatter:
                non_scatter_activity.append((max(lo, e['ts']), min(hi, e['ts'] + e['dur'])))
            if 'nccl' in name:
                if group == 'ep_dp' and 'allgather' in op:
                    param_comms['native_param_all_gather'].append((e['ts'], e['ts'] + e['dur']))
                elif group == 'nep_owner_transfer' and op == 'broadcast':
                    param_comms['nep_param_scatter'].append((e['ts'], e['ts'] + e['dur']))
            if 'nccl' in name or 'hybrid_ep::' in name or 'hybridep' in name:
                category = 'comm'
            else:
                source = cpu.get(args.get('External id')) or launches.get(args.get('correlation'))
                stage = source is not None and location(scopes.get(source[1], []), starts.get(source[1], []), source[0]) is not None
                category = 'staging' if stage else 'compute'
            intervals[category].append((max(lo, e['ts']), min(hi, e['ts'] + e['dur'])))
        states = exclusive(intervals, lo, hi)
        selected = {mask.replace('host+', '').replace('+host', ''): ms for mask, ms in states.items() if 'host' in mask.split('+')}
        # Per-name scopes can nest; retain individually but never add their durations.
        activity = [v for k, rows in intervals.items() if k != 'host' for v in rows]
        by_name = {}
        for name, rows in named.items():
            clipped = [(max(lo, a), min(hi, b)) for a, b in rows if a < hi and b > lo]
            duration = union_duration(clipped) / 1000
            idle = duration - intersection_duration(clipped, activity) / 1000
            by_name[name] = {'scope_ms': duration, 'gpu_idle_ms': idle}
        gpu_comms = {}
        for name, calls in param_comms.items():
            residency = sum(b - a for a, b in calls) / 1000
            overlap = sum(intersection_duration([pair], intervals['compute']) for pair in calls) / 1000
            gpu_comms[name] = {'sum_ms': residency, 'compute_overlap_ms': overlap,
                               'without_compute_ms': residency - overlap, 'calls': len(calls)}
            if name == 'nep_param_scatter':
                other_overlap = sum(intersection_duration([pair], non_scatter_activity) for pair in calls) / 1000
                gpu_comms[name]['exposed_any_ms'] = residency - other_overlap
                gpu_comms[name]['other_work_overlap_ms'] = other_overlap
        assert set(gpu_comms) == {'native_param_all_gather', 'nep_param_scatter'}
        out.append({'step': s['name'], 'states_ms': selected, 'by_name': by_name, 'param_gpu_comms': gpu_comms})
    assert len(out) == 4
    return out


def main():
    result = {}
    for job, model, _ in CASES:
        rows = []
        for rank in range(6):
            cache = OUT / f'host_scope_states_v3_{job}_{rank}.json'
            if cache.exists():
                steps = json.loads(cache.read_text())
            else:
                path = ROOT / f'slurm_runs/soft_clique_baseline/{job}/ep8/scaling_ladder__{model}/nep/{job}/torch_profile/rank-{rank}.json.gz'
                steps = scan(path)
                cache.write_text(json.dumps(steps, indent=2) + '\n')
            rows.extend(steps)
            print(job, rank, 'validated', flush=True)
        keys = {k for s in rows for k in s['states_ms']}
        names = {k for s in rows for k in s['by_name']}
        result[job] = {'states_ms': {k: mean(s['states_ms'].get(k, 0) for s in rows) for k in keys},
                       'by_name': {k: {m: mean(s['by_name'][k][m] for s in rows) for m in ('scope_ms', 'gpu_idle_ms')} for k in names},
                       'param_gpu_comms': {k: {m: mean(s['param_gpu_comms'][k][m] for s in rows)
                                              for m in rows[0]['param_gpu_comms'][k]}
                                           for k in rows[0]['param_gpu_comms']}}
        (OUT / 'host_scope_overlap_summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(job, result[job], flush=True)


if __name__ == '__main__':
    main()
