"""Reproduce matched-owner first-forward accounting from existing job3122515 data.

Phase partitions are disjoint. Cross-rank kernel-start/ready prefixes are
diagnostics, not independent additive costs or pure transport measurements.
"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_nep_first_forward_delivery import OUT, ROOT


def load_case(case, size):
    return {r: json.loads((OUT / f'delivery-{case}-rank-{r}.json').read_text()) for r in range(size)}


def shifts(data):
    origin = min(d['origin_ns'] for d in data.values())
    return {r: (d['origin_ns'] - origin) / 1e6 for r, d in data.items()}


def expert_ag(row):
    return [x for x in row['ops'] if x['name'] == 'collective.all_gather'
            and ('expert' in x['meta']['group_desc'].lower() or x['meta']['group_desc'] == 'ep_dp')]


def redistribution(row):
    return [x for x in row['ops'] if x['name'] == 'nep.redistribute_params']


def phase_summary(data):
    shift = shifts(data)
    parts = {'graph_parts': defaultdict(float), 'gap_parts': defaultdict(float)}
    peers = defaultdict(float)
    phase = 0.0
    for rank in range(12):
        for i, row in enumerate(data[rank]['rows']):
            phase += (row['end'] - row['start']) / 96
            for label, dt in row['gap_parts'].items():
                parts['gap_parts'][label] += dt / 96
            for gi, graph in enumerate(row['graphs']):
                for label, dt in graph['parts'].items():
                    parts['graph_parts'][label] += dt / 96
                ordinal = Counter()
                for k in graph['model_ep']:
                    name = k['name']
                    if name not in ('ag_nvl_kernel', 'device_sync_kernel'):
                        continue
                    n = ordinal[name]
                    ordinal[name] += 1
                    starts = []
                    for other in range(16):
                        g = data[other]['rows'][i]['graphs'][gi]
                        assert g['name'] == graph['name']
                        matching = [x for x in g['model_ep'] if x['name'] == name]
                        assert len(matching) > n
                        starts.append(matching[n]['start'] + shift[other])
                    a, b = k['start'] + shift[rank], k['end'] + shift[rank]
                    peers[name + '_residency'] += (b - a) / 96
                    peers[name + '_before_last_peer_start'] += max(0, min(b, max(starts)) - a) / 96
    assert abs(sum(sum(p.values()) for p in parts.values()) - phase) < 1e-5
    return dict(phase_ms=phase, peer_diagnostics_ms=peers, **parts)


def broadcast_accounting(data, run):
    shift = shifts(data)
    matched = defaultdict(dict)
    for rank in data:
        raw = json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())
        for i, row in enumerate(data[rank]['rows']):
            events = [e for e in raw['traceEvents'] if e['cat'] == 'gpu_operation_range'
                      and e['args'].get('iteration') == 20 + i and e['name'] == 'collective.broadcast'
                      and e['args'].get('collective', {}).get('group_desc') == 'nep_owner_transfer']
            ks = sorted([k for k in row['kernels'] if k['label'] == 'NEP_broadcast'], key=lambda k: k['launch'])
            assert len(ks) == len(events)
            counts = Counter()
            for k, e in zip(ks, events):
                assert abs(k['start'] - e['ts'] / 1000) < 1
                assert abs(k['end'] - (e['ts'] + e['dur']) / 1000) < 0.1
                meta = e['args']['collective']
                gid = meta['group_id']
                key = (i, gid, counts[gid])
                counts[gid] += 1
                matched[key][rank] = dict(k=k, ready=e['ts'] / 1000 + shift[rank], meta=meta)
    result = defaultdict(float)
    for (i, _, _), group in matched.items():
        assert set(group) == set(next(iter(group.values()))['meta']['ranks'])
        last = max(x['ready'] for x in group.values())
        for rank, p in group.items():
            if rank >= 12:
                continue
            k, row = p['k'], data[rank]['rows'][i]
            ready = last - shift[rank]
            result['residency_ms'] += (k['end'] - k['start']) / 96
            result['before_last_input_ready_ms'] += max(0, min(k['end'], ready) - k['start']) / 96
            result['after_last_input_ready_ms'] += max(0, k['end'] - max(k['start'], ready)) / 96
            for ga, gb in row['gaps']:
                a, b = max(ga, k['start']), min(gb, k['end'])
                if b <= a:
                    continue
                edges = [(a, 0), (b, 0)]
                for other in row['kernels']:
                    if other['label'] == 'NEP_broadcast':
                        continue
                    x, y = max(a, other['start']), min(b, other['end'])
                    if y > x:
                        edges.extend(((x, 1), (y, -1)))
                active, previous = 0, a
                for t, delta in sorted(edges):
                    if active == 0:
                        cut = max(previous, min(t, ready))
                        result['exposed_wait_ms'] += (cut - previous) / 96
                        result['exposed_after_ready_ms'] += (t - cut) / 96
                    active += delta
                    previous = t
    return dict(result)


def scheduling(data):
    shift = shifts(data)
    leads, next_leads, next_gaps, start_gaps, spreads, last_ranks = [], [], [], [], [], Counter()
    for rank in range(12):
        for row in data[rank]['rows']:
            red, ag = redistribution(row), expert_ag(row)
            assert len(red) == len(ag) == 8
            for b, (r, a) in enumerate(zip(red, ag)):
                leads.append(a['end'] - r['cpu_end'])
                start_gaps.append(max(0, r['start'] - a['end']))
                if b:
                    next_leads.append(red[b - 1]['end'] - a['cpu_end'])
                    next_gaps.append(a['start'] - red[b - 1]['end'])
    for i in range(8):
        for b in range(8):
            ends = [(redistribution(data[r]['rows'][i])[b]['end'] + shift[r], r) for r in range(16)]
            spreads.append(max(ends)[0] - min(ends)[0])
            last_ranks[max(ends)[1]] += 1
    return dict(redistribution_cpu_lead_ms=dict(min=min(leads), mean=statistics.mean(leads), max=max(leads)),
                redistribution_cpu_early_calls=sum(x > 0 for x in leads), redistribution_calls=len(leads),
                AG_to_redistribution_positive_gap_per_rank_step_ms=sum(start_gaps) / 96,
                next_AG_cpu_early_calls=sum(x > 0 for x in next_leads), next_AG_calls=len(next_leads),
                next_AG_cpu_lead_min_ms=min(next_leads), next_AG_gpu_gap_min_ms=min(next_gaps),
                bucket_ready_spread_ms=dict(mean=statistics.mean(spreads), min=min(spreads), max=max(spreads)),
                last_ready_ranks=last_ranks,
                max_clock_calibration_error_ms=max(x['clock_error_ns'] for x in data.values()) / 1e6,
                max_nsight_alignment_halfwidth_ms=max((x['alignment_interval_ms'][1]-x['alignment_interval_ms'][0])/2 for x in data.values()))


def main():
    cases = json.loads((ROOT / 'cleanup_phase_analysis.json').read_text())
    healthy = phase_summary(load_case('healthy_margin', 32))
    data = load_case('cleanup_margin', 28)
    nep = phase_summary(data)
    b = broadcast_accounting(data, Path(cases['cleanup_ab_cleanup_margin']['run']))
    h, n = healthy, nep
    delta = n['phase_ms'] - h['phase_ms']
    costs = dict(model_EP_inside_graphs=n['graph_parts']['model_EP']-h['graph_parts']['model_EP'],
                 broadcast_wait_outside_graphs=b['exposed_wait_ms'],
                 broadcast_after_ready_outside_graphs=b['exposed_after_ready_ms'],
                 staging_only_outside_graphs=n['gap_parts']['NEP_staging'],
                 compute_TP_inside_graphs=sum(n['graph_parts'][k]-h['graph_parts'][k] for k in ('model_compute', 'model_TP')),
                 AG_only_outside_graphs=sum(n['gap_parts'].get(k, 0)-h['gap_parts'].get(k, 0) for k in ('dense_AG', 'expert_AG', 'dense_AG+expert_AG')))
    assert abs(b['exposed_wait_ms']+b['exposed_after_ready_ms']-n['gap_parts']['NEP_broadcast']) < 1e-5
    selected_graph = {'model_EP', 'model_compute', 'model_TP'}
    selected_gap = {'NEP_broadcast', 'NEP_staging', 'dense_AG', 'expert_AG', 'dense_AG+expert_AG'}
    costs['other_disjoint_intervals'] = sum(n['graph_parts'].get(k, 0)-h['graph_parts'].get(k, 0) for k in (n['graph_parts'].keys() | h['graph_parts'].keys())-selected_graph)
    costs['other_disjoint_intervals'] += sum(n['gap_parts'].get(k, 0)-h['gap_parts'].get(k, 0) for k in (n['gap_parts'].keys() | h['gap_parts'].keys())-selected_gap)
    assert abs(sum(costs.values())-delta) < 1e-5
    result = dict(healthy=healthy, nep=nep, broadcast=b, scheduling=scheduling(data),
                  phase_delta_ms=delta, additive_delta_ms=costs)
    (OUT / 'first_forward_findings.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(delta_ms=delta, additive_delta_ms=costs, scheduling=result['scheduling']), indent=2))


if __name__ == '__main__':
    main()
