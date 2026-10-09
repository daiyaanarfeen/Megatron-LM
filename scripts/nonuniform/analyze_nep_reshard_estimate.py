"""Estimate arrival-adjusted reshard residency from matched peer GPU kernels.

This is not an exact measurement of internal NCCL wait or transfer time.
For overlapping peer kernels, trim the interval preceding the last peer start.
Retain disjoint peer durations: buffered sends can complete before receivers run.
Multi-peer kernels may do useful work before the last arrival, and may still
wait internally afterward. Keep raw residency and per-call records for auditing.
"""

import gzip
import json
from collections import defaultdict

from analyze_nep_final_ga_sums import CASES, OUT, ROOT


def extract(job, model, rank, variant='nep', edp_only=False, param_ag=False, trace_data=None):
    cache = OUT / (f'edp_calls_{job}_{variant}_{rank}.json' if edp_only else f'reshard_calls_{job}_{rank}.json')
    if param_ag:
        assert edp_only
        cache = OUT / f'param_ag_calls_{job}_{variant}_{rank}.json'
    if cache.exists():
        cached = json.loads(cache.read_text())
        if not edp_only or cached['calls']:
            return cached
    path = ROOT / f'slurm_runs/soft_clique_baseline/{job}/ep8/scaling_ladder__{model}/{variant}/{job}/torch_profile/rank-{rank}.json.gz'
    if trace_data is None:
        with gzip.open(path, 'rt') as handle:
            data = json.load(handle)
    else:
        data = trace_data
    events = data['traceEvents']
    config = {p['pg_name']: p for p in data['distributedInfo']['pg_config']}
    metadata = {e['args']['External id']: e for e in events
                if e.get('name') == 'record_param_comms' and e['args'].get('Collective name') != 'wait'}
    steps = sorted([e for e in events if e.get('cat') == 'user_annotation'
                    and e.get('name', '').startswith('ProfilerStep#')], key=lambda e: e['ts'])
    assert len(steps) == 4
    calls = []
    for e in events:
        if e.get('cat') != 'kernel' or 'nccl' not in e.get('name', '').lower():
            continue
        a = dict(e.get('args', {}))
        m = metadata.get(a.get('External id'))
        if m is None:
            continue
        a.update(m['args'])
        desc = a.get('Process Group Description', '')
        category = 'gather' if 'owner_gather' in desc else ('scatter' if 'owner_transfer' in desc else None)
        if edp_only:
            collective = 'allgather' if param_ag else 'reduce_scatter'
            category = 'edp' if (desc == 'ep_dp' or desc == 'EXPERT_DATA_PARALLEL_GROUP') and collective in a['Collective name'] else None
        if category is None:
            continue
        if not edp_only:
            assert a['Collective name'] == ('all_to_allv' if category == 'gather' else 'broadcast'), a
        step = next((s['name'] for s in steps if s['ts'] <= m['ts'] < s['ts'] + s['dur']), None)
        assert step is not None
        calls.append({'category': category, 'pg': a['Process Group Name'],
                      'ranks': config[a['Process Group Name']]['ranks'], 'step': step,
                      'start': e['ts'], 'end': e['ts'] + e['dur'], 'duration': e['dur'],
                      'host_start': m['ts'], 'external_id': a['External id'],
                      'input_elements': a['In msg nelems'], 'output_elements': a['Out msg nelems'],
                      'dtype': a['dtype'], 'rank': rank})
    result = {'base_time_ns': data['baseTimeNanoseconds'], 'calls': calls}
    cache.write_text(json.dumps(result) + '\n')
    return result


def main():
    output = {}
    for job, model, _ in CASES:
        groups = defaultdict(dict)
        base = None
        for rank in range(14):
            data = extract(job, model, rank)
            base = data['base_time_ns'] if base is None else base
            assert base == data['base_time_ns'], 'Trace origins differ; align before matching'
            counts = defaultdict(int)
            for call in sorted(data['calls'], key=lambda c: c['host_start']):
                key = (call['category'], call['pg'], call['step'])
                ordinal = counts[key]
                counts[key] += 1
                groups[key + (ordinal,)][rank] = call
            print(job, rank, 'cached', flush=True)
        totals = defaultdict(lambda: defaultdict(float))
        records = []
        for (category, pg, step, ordinal), peers in groups.items():
            example = next(iter(peers.values()))
            assert set(peers) == set(example['ranks']), (job, pg, step, ordinal, peers.keys(), example['ranks'])
            if category == 'gather':
                assert sum(p['input_elements'] for p in peers.values()) == sum(p['output_elements'] for p in peers.values())
            else:
                assert len({p['input_elements'] for p in peers.values()}) == 1
            assert len({p['dtype'] for p in peers.values()}) == 1
            latest = max(p['start'] for p in peers.values())
            earliest_end = min(p['end'] for p in peers.values())
            overlapping = latest < earliest_end
            for rank, p in peers.items():
                removed = latest - p['start'] if overlapping else 0
                adjusted = p['duration'] - removed
                assert -0.01 <= adjusted <= p['duration'] + 0.01
                if rank < 6:
                    totals[category]['raw_ms'] += p['duration'] / 24000
                    totals[category]['adjusted_estimate_ms'] += adjusted / 24000
                    totals[category]['removed_initial_skew_ms'] += removed / 24000
                    totals[category]['nonoverlapping_calls'] += int(not overlapping)
                    totals[category]['calls'] += 1
                records.append(dict(p, ordinal=ordinal, overlapping_peers=overlapping,
                                    estimated_removed_us=removed, adjusted_estimate_us=adjusted))
        summary = {k: dict(v) for k, v in totals.items()}
        summary['total'] = {metric: sum(summary[c][metric] for c in ('gather', 'scatter'))
                            for metric in ('raw_ms', 'adjusted_estimate_ms', 'removed_initial_skew_ms')}
        output[job] = {'owner_mean': summary, 'records': records,
                       'method': 'Last-peer-start trimming only for intersecting peer kernels; retain buffered/nonintersecting calls. Estimate, not exact wait-free transfer time.'}
        (OUT / 'reshard_arrival_estimates.json').write_text(json.dumps(output, indent=2) + '\n')
        print(job, summary, flush=True)


if __name__ == '__main__':
    main()
