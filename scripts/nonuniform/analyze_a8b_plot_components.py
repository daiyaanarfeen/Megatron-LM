"""Apply the small-model plot definitions to all ranks of job 2994648."""

import argparse
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from analyze_nep_final_ga_sums import OUT, ROOT, scan
from analyze_nep_reshard_estimate import extract
from analyze_workload_parity_pair import iter_trace_events

JOB = '2994648'
BASE = ROOT / f'slurm_runs/soft_clique_baseline/{JOB}/ep16/a8b_120b_latentmoe_1t'


def trace_events(path):
    try:
        from ijson import items
    except ImportError:
        yield from iter_trace_events(path)
        return
    with gzip.open(path, 'rb') as stream:
        yield from items(stream, 'traceEvents.item', use_float=True)


def rank_data(variant, rank):
    cache = OUT / f'a8b_plot_components_{variant}_{rank}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    path = BASE / f'{variant}/{JOB}/torch_profile/rank-{rank}.json.gz'
    with gzip.open(path, 'rt') as handle:
        prefix = handle.read(1024 * 1024).split('"traceEvents"', 1)[0]
    data = json.loads(prefix + '"traceEvents": []}')
    events = data['traceEvents']
    for event in trace_events(path):
        cat, name = event.get('cat'), event.get('name', '')
        if cat in ('cpu_op', 'kernel', 'gpu_memcpy', 'gpu_memset', 'user_annotation') or name == 'record_param_comms' or (cat in ('cuda_runtime', 'cuda_driver') and 'Launch' in name):
            compact = {k: event[k] for k in ('cat', 'name', 'ts', 'dur', 'tid') if k in event}
            compact['name'] = sys.intern(name)
            if cat is not None:
                compact['cat'] = sys.intern(cat)
            args = event.get('args', {})
            compact['args'] = args if name == 'record_param_comms' else {
                k: args[k] for k in ('External id', 'correlation', 'Process Group Description', 'Collective name') if k in args}
            events.append(compact)
    steps = scan(path, 30, require_owner_comms=False, include_host_staging=True, ga=8, events=events)
    calls = {}
    for kind in ('grad_rs', 'param_ag'):
        calls[kind] = extract(JOB, '', rank, variant=variant, edp_only=True,
                              param_ag=kind == 'param_ag', trace_data=data)
    if variant == 'nep':
        calls['reshard'] = extract(JOB, '', rank, trace_data=data)
    result = {'steps': steps, 'calls': calls}
    cache.write_text(json.dumps(result) + '\n')
    print(variant, rank, 'validated', flush=True)
    return result


def match_calls(ranks, kind, owner_ids):
    groups = defaultdict(dict)
    origins = set()
    for rank, data in ranks.items():
        item = data['calls'][kind]
        origins.add(item['base_time_ns'])
        counts = defaultdict(int)
        for call in sorted(item['calls'], key=lambda c: c['host_start']):
            key = (call['category'], call['pg'], call['step'])
            ordinal = counts[key]
            counts[key] += 1
            groups[key + (ordinal,)][rank] = call
    assert len(origins) == 1
    totals = defaultdict(lambda: defaultdict(float))
    records = []
    denominator = len(owner_ids) * 4
    for (category, pg, step, ordinal), peers in groups.items():
        sample = next(iter(peers.values()))
        assert set(peers) == set(sample['ranks']), (kind, pg, peers.keys(), sample['ranks'])
        if category == 'gather':
            assert sum(p['input_elements'] for p in peers.values()) == sum(p['output_elements'] for p in peers.values())
        else:
            assert len({p['input_elements'] for p in peers.values()}) == 1
        latest = max(p['start'] for p in peers.values())
        intersect = latest < min(p['end'] for p in peers.values())
        for rank, p in peers.items():
            removed = latest - p['start'] if intersect else 0
            adjusted = p['duration'] - removed
            assert -0.01 <= adjusted <= p['duration'] + 0.01
            if rank in owner_ids:
                for metric, value in [('raw_ms', p['duration']/1000), ('adjusted_estimate_ms', adjusted/1000),
                                      ('removed_initial_skew_ms', removed/1000), ('calls_per_step', 1),
                                      ('nonintersecting_calls_per_step', int(not intersect))]:
                    totals[category][metric] += value / denominator
            records.append(dict(p, ordinal=ordinal, adjusted_estimate_ms=adjusted/1000, intersecting_peers=intersect))
    return {'owner_mean': {k: dict(v) for k, v in totals.items()}, 'records': records}


def main():
    result = {}
    pending = [(variant, rank) for variant, count in [('healthy', 32), ('nep', 28)]
               for rank in range(count) if not (OUT / f'a8b_plot_components_{variant}_{rank}.json').exists()]
    def run_rank(item):
        variant, rank = item
        subprocess.run([sys.executable, __file__, '--variant', variant, '--rank', str(rank)], check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run_rank, pending))
    for variant, count in [('healthy', 32), ('nep', 28)]:
        ranks = {}
        for rank in range(count):
            cache = OUT / f'a8b_plot_components_{variant}_{rank}.json'
            if not cache.exists():
                subprocess.run([sys.executable, __file__, '--variant', variant, '--rank', str(rank)], check=True)
            ranks[rank] = json.loads(cache.read_text())
        roles = {'healthy': list(range(32))} if variant == 'healthy' else {
            'full': list(range(16)), 'reduced': list(range(16, 28)), 'owners': list(range(12))}
        summary = {}
        for role, ids in roles.items():
            steps = [s for rank in ids for s in ranks[rank]['steps']]
            summary[role] = {'final_ga_compute_ms': mean(s['sum_ms']['backward_compute'] for s in steps),
                             'host_staging_ms': {k: mean(s['host_staging_ms'][k] for s in steps)
                                                 for k in steps[0]['host_staging_ms']}}
        result[variant] = {'summary': summary, 'communications': {
            kind: match_calls(ranks, kind, list(range(12)))
            for kind in (('grad_rs', 'param_ag', 'reshard') if variant == 'nep' else ('grad_rs', 'param_ag'))}}
        (OUT / 'a8b_plot_components.json').write_text(json.dumps(result, indent=2) + '\n')
        print(variant, summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['healthy', 'nep'])
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    if args.variant is None:
        main()
    else:
        rank_data(args.variant, args.rank)
