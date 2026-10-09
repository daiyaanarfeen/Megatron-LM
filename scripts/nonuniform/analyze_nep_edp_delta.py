"""Compare matched healthy/NEP EDP residency after peer kernel arrival."""

import json
from collections import defaultdict

from analyze_nep_reshard_estimate import CASES, OUT, extract


def main(param_ag=False):
    output = {}
    for job, model, _ in CASES:
        output[job] = {}
        for variant in ('healthy', 'nep'):
            groups = defaultdict(dict)
            base = None
            for rank in list(range(6)) + list(range(8, 14)):
                data = extract(job, model, rank, variant=variant, edp_only=True, param_ag=param_ag)
                assert data['calls'], (job, variant, rank, 'No EDP kernels found')
                base = data['base_time_ns'] if base is None else base
                assert base == data['base_time_ns']
                counts = defaultdict(int)
                for c in sorted(data['calls'], key=lambda c: c['host_start']):
                    key = (c['pg'], c['step'])
                    i = counts[key]
                    counts[key] += 1
                    groups[key + (i,)][rank] = c
                print(job, variant, rank, 'cached', flush=True)
            totals = defaultdict(float)
            records = []
            for (pg, step, ordinal), peers in groups.items():
                sample = next(iter(peers.values()))
                assert set(peers) == set(sample['ranks']) and len(peers) == 2
                assert len({p['input_elements'] for p in peers.values()}) == 1
                latest = max(p['start'] for p in peers.values())
                end_spread = max(p['end'] for p in peers.values()) - min(p['end'] for p in peers.values())
                for rank, p in peers.items():
                    assert p['end'] > latest, (job, variant, peers)
                    adjusted = (p['end'] - latest) / 1000
                    if rank < 6:
                        totals['raw_ms'] += p['duration'] / 24000
                        totals['adjusted_estimate_ms'] += adjusted / 24
                        totals['removed_initial_skew_ms'] += (latest - p['start']) / 24000
                        totals['owner_calls'] += 1
                        totals['max_peer_end_spread_ms'] = max(totals['max_peer_end_spread_ms'], end_spread / 1000)
                    records.append(dict(p, ordinal=ordinal, adjusted_estimate_ms=adjusted))
            output[job][variant] = {'owner_mean': dict(totals), 'records': records}
            print(job, variant, dict(totals), flush=True)
        output[job]['delta_ms'] = output[job]['nep']['owner_mean']['adjusted_estimate_ms'] - output[job]['healthy']['owner_mean']['adjusted_estimate_ms']
        filename = 'param_ag_arrival_adjusted_delta.json' if param_ag else 'edp_arrival_adjusted_delta.json'
        (OUT / filename).write_text(json.dumps(output, indent=2) + '\n')
        print(job, 'delta', output[job]['delta_ms'], flush=True)


if __name__ == '__main__':
    main()
