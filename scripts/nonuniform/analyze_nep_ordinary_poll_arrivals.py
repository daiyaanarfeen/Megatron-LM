"""Match reduced-replica HybridEP polling points across existing rank traces."""

import argparse
import gzip
import json
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from analyze_a8b_plot_components import trace_events
from analyze_nep_microbatch_phases import CASES, ROOT
from analyze_nep_ordinary_microbatches import OUT
from analyze_nep_owner_progress import location
from plot_nep_failure_mechanism_details import cluster_backward


def scan(job, rank):
    cache = OUT/f'poll_{job}_{rank}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    case = next(c for c in CASES if c[0] == job)
    path = ROOT/f'slurm_runs/soft_clique_baseline/{job}/ep8/{case[1]}/nep/{job}/torch_profile/rank-{rank}.json.gz'
    with gzip.open(path, 'rt') as f:
        prefix = f.read(1024*1024).split('"traceEvents"', 1)[0]
    base = json.loads(prefix+'"traceEvents": []}')['baseTimeNanoseconds']
    steps, backward, polls, groups = [], [], [], []
    cpu, launches = {}, {}
    for e in trace_events(path):
        if 'ts' not in e or 'dur' not in e:
            continue
        cat, name, a = e.get('cat'), e.get('name', ''), e.get('args', {})
        ts, end = e['ts'], e['ts']+e['dur']
        if cat == 'user_annotation' and name.startswith('ProfilerStep#'):
            steps.append((ts, end, name))
        if cat == 'cpu_op':
            cpu[a.get('External id')] = ts
            if name.startswith('autograd::engine::evaluate_function:'):
                backward.append((ts, end))
            if name == '_GroupedLinearBackward':
                groups.append((ts, end))
        if cat in ('cuda_runtime', 'cuda_driver') and 'Launch' in name:
            launches[a.get('correlation')] = ts
        if cat == 'kernel' and 'hybrid_ep::device_sync_kernel' in name:
            polls.append((ts, end, a.get('External id'), a.get('correlation')))
    result = {'base_time_ns': base, 'calls': []}
    for lo, hi, step in sorted(steps):
        bwd, _ = cluster_backward([(a, b) for a, b in backward if lo <= a < b <= hi], 3)
        starts = [a for a, _ in bwd]
        order = Counter()
        for a, b, ext, corr in sorted(polls):
            if not lo <= a < b <= hi:
                continue
            host = cpu.get(ext, launches.get(corr))
            assert host is not None
            mb = location(bwd, starts, host)
            if mb is None or mb == 2:
                continue
            ordinal = order[mb]
            order[mb] += 1
            prior = [(x, y) for x, y in groups if bwd[mb][0] <= x < y <= host]
            result['calls'].append({'step': step, 'mb': mb+1, 'ordinal': ordinal,
                                    'start': a, 'end': b, 'launch': launches.get(corr),
                                    'previous_grouped_cpu_end': max((y for _, y in prior), default=None)})
        assert order[0] == order[1] and order[0] > 0, (job, rank, step, order)
    cache.write_text(json.dumps(result)+'\n')
    print(job, rank, 'polls extracted', flush=True)
    return result


def aggregate(job):
    ranks = {r: scan(job, r) for r in range(8, 14)}
    assert len({d['base_time_ns'] for d in ranks.values()}) == 1
    calls = defaultdict(dict)
    for rank, data in ranks.items():
        for c in data['calls']:
            calls[(c['step'], c['mb'], c['ordinal'])][rank] = c
    totals, last_counts, records = Counter(), Counter(), []
    for key, peers in sorted(calls.items()):
        assert set(peers) == set(ranks)
        last = max(peers, key=lambda r: peers[r]['start'])
        latest = peers[last]['start']
        intersect = latest < min(p['end'] for p in peers.values())
        p = peers[8]
        totals['poll_ms'] += (p['end']-p['start'])/1000
        totals['calls'] += 1
        launch = peers[last]['launch']
        totals['before_last_peer_launch_observed_ms'] += (
            max(0, min(p['end'], launch)-p['start'])/1000 if launch is not None else 0
        )
        totals['after_last_peer_start_observed_ms'] += max(0, p['end']-latest)/1000
        if not intersect:
            totals['nonintersecting'] += 1
            totals['nonintersection_gap_sum_ms'] += (latest-min(q['end'] for q in peers.values()))/1000
            totals['nonintersection_gap_max_ms'] = max(
                totals['nonintersection_gap_max_ms'],
                (latest-min(q['end'] for q in peers.values()))/1000,
            )
            continue
        wait = (latest-p['start'])/1000
        launch = peers[last]['launch']
        host_part = max(0, min(latest, launch)-p['start'])/1000 if launch is not None else 0
        totals['before_last_peer_starts_ms'] += wait
        totals['after_last_peer_starts_ms'] += (p['end']-latest)/1000
        totals['before_last_peer_launch_call_ms'] += host_part
        last_counts[last] += 1
        records.append({'step': key[0], 'mb': key[1], 'ordinal': key[2], 'last_rank': last,
                        'rank8_poll_ms': (p['end']-p['start'])/1000, 'arrival_spread_ms': wait,
                        'before_last_peer_launch_call_ms': host_part, 'peers': peers})
    result = {'job': job, 'microbatches': 8,
              'rank8_mean_per_microbatch': {k: v/8 for k, v in totals.items() if k != 'nonintersection_gap_max_ms'},
              'nonintersection_gap_max_ms': totals['nonintersection_gap_max_ms'],
              'latest_rank_counts': dict(last_counts),
              'largest_waits': sorted(records, key=lambda c: c['arrival_spread_ms'], reverse=True)[:30]}
    (OUT/f'poll_summary_{job}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(job, result['rank8_mean_per_microbatch'], last_counts, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    if args.rank is not None:
        scan(args.job, args.rank)
        return
    tasks = [(job, rank) for job in ('2994105', '2995392') for rank in range(8, 14)]
    def run(task):
        job, rank = task
        subprocess.run([sys.executable, __file__, '--job', job, '--rank', str(rank)], check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, tasks))
    for job in ('2994105', '2995392'):
        aggregate(job)


if __name__ == '__main__':
    main()
