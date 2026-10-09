"""Match all a770 NEP collective participants to bound pre-arrival residency."""

import gzip
import json
from collections import defaultdict

from analyze_nep_final_ga_sums import OUT, ROOT


def main():
    output = {}
    for job in ('2995392', '2995524'):
        groups = defaultdict(dict)
        for rank in range(14):
            path = ROOT / f'slurm_runs/soft_clique_baseline/{job}/ep8/scaling_ladder__a770m_4b_moe_215b/nep/{job}/torch_profile/rank-{rank}.json.gz'
            with gzip.open(path, 'rt') as handle:
                events = json.load(handle)['traceEvents']
            metadata = {e.get('args', {}).get('External id'): e['args'] for e in events
                        if e.get('name') == 'record_param_comms' and e.get('args', {}).get('Collective name') != 'wait'}
            steps = sorted([e for e in events if e.get('cat') == 'user_annotation'
                            and e.get('name', '').startswith('ProfilerStep#')], key=lambda e: e['ts'])
            assert len(steps) == 4
            counters = defaultdict(int)
            for e in sorted(events, key=lambda e: e.get('ts', 0)):
                if e.get('cat') != 'kernel' or 'nccl' not in e.get('name', '').lower():
                    continue
                a = dict(e.get('args', {}))
                a.update(metadata.get(a.get('External id'), {}))
                pg = a.get('Process Group Description', '')
                category = 'gather' if 'owner_gather' in pg else (
                    'edp' if pg == 'ep_dp' and 'reduce_scatter' in str(a.get('Collective name')) else None)
                if category is None:
                    continue
                step = next((s['name'] for s in steps if s['ts'] <= e['ts'] < s['ts'] + s['dur']), None)
                if step is None:
                    continue
                ranks = a['Process Group Ranks']
                ranks = json.loads(ranks) if isinstance(ranks, str) else ranks
                key = (category, step, tuple(ranks), str(a['Process Group Name']))
                index = counters[key]
                counters[key] += 1
                groups[key + (index,)][rank] = {'start': e['ts'], 'end': e['ts'] + e['dur'],
                                               'duration': e['dur'], 'elements': a.get('In msg nelems')}
            print(job, rank, 'read', flush=True)
        totals = defaultdict(lambda: defaultdict(float))
        max_end_spread = defaultdict(float)
        records = []
        for (category, step, ranks, pg_id, index), participants in groups.items():
            if category == 'edp':
                assert set(participants) == set(ranks), (job, category, ranks, participants.keys())
            latest = max(p['start'] for p in participants.values())
            end_spread = max(p['end'] for p in participants.values()) - min(p['end'] for p in participants.values())
            max_end_spread[category] = max(max_end_spread[category], end_spread / 1000)
            for rank, p in participants.items():
                # Send/receive Gather kernels can finish independently; their
                # starts do not establish an all-participant waiting boundary.
                if category == 'edp':
                    assert latest <= p['end'] + 100, (job, ranks, pg_id, step, index, participants)
                wait = max(0, min(latest, p['end']) - p['start']) / 1000 if category == 'edp' else 0
                duration = p['duration'] / 1000
                if rank < 6:
                    totals[category]['residency_ms'] += duration / 24
                    if category == 'edp':
                        totals[category]['pre_latest_start_ms'] += wait / 24
                        totals[category]['remaining_ms'] += (duration - wait) / 24
                records.append({'category': category, 'step': step, 'ranks': ranks, 'bucket': index,
                                'rank': rank, 'residency_ms': duration, 'pre_latest_start_ms': wait if category == 'edp' else None})
        output[job] = {'owner_mean': dict(totals), 'max_peer_end_spread_ms': dict(max_end_spread), 'records': records}
        print(job, dict(totals), dict(max_end_spread), flush=True)
        (OUT / 'a770_collective_arrivals.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
