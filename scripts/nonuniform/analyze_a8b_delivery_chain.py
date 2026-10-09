"""Match participant readiness and audit the post-AG delivery chain."""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_a8b_current_overheads import category, stats

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3078207/mbs2_overhead_analysis')


def main():
    result = {}
    ranks_by_case = {}
    for case, size in [('healthy', 32), ('nep', 28)]:
        ranks = {r: json.loads((ROOT/f'{case}-rank-{r}.json').read_text()) for r in range(size)}
        ranks_by_case[case] = ranks
        origin = min(d['origin_ns'] for d in ranks.values())
        matched = defaultdict(dict)
        for rank, data in ranks.items():
            for step, row in data['steps'].items():
                counts = Counter()
                for event in row['events']:
                    meta = event['args'].get('collective')
                    if meta is None:
                        continue
                    key = (step, meta['group_id'], event['name'])
                    ordinal = counts[key]
                    counts[key] += 1
                    start = (data['origin_ns']-origin)/1e6 + event['ts']/1000
                    matched[key+(ordinal,)][rank] = dict(start=start, end=start+event['dur']/1000,
                                                          category=category(event), metadata=meta)
        totals = defaultdict(lambda: defaultdict(float))
        calls = []
        for key, peers in matched.items():
            meta = next(iter(peers.values()))['metadata']
            assert set(peers) == set(meta['ranks']), (key, peers.keys(), meta['ranks'])
            latest = max(v['start'] for v in peers.values())
            for rank, event in peers.items():
                role = 'owners' if rank < 12 else 'followers' if rank < 16 else 'second'
                name = event['category']
                if name.startswith(('expert.', 'nep_')):
                    calls.append(dict(iteration=int(key[0]), ordinal=key[-1], rank=rank,
                                      latest_start=latest, **event))
                    # Zero-output Gather senders can finish before another rank's
                    # input-ready marker. Do not call their signed remainder wire time.
                    if name == 'nep_grad.all_to_all':
                        continue
                    row = totals[(role, name, int(key[0]))]
                    row['waiting_ms'] += latest-event['start']
                    row['post_last_ready_ms'] += event['end']-latest
                    row['elapsed_ms'] += event['end']-event['start']
        summaries = {}
        for role, count in [('owners',12),('followers',4),('second',size-16)]:
            names = {k[1] for k in totals if k[0] == role}
            summaries[role] = {name:{metric:stats([totals[(role,name,s)][metric]/count for s in range(25,35)])
                                   for metric in ['waiting_ms','post_last_ready_ms','elapsed_ms']} for name in names}
        result[case] = dict(readiness=summaries, matched_calls=calls)
    nep = ranks_by_case['nep']
    origin = min(d['origin_ns'] for d in nep.values())
    delivery = []
    for step in range(25,35):
        bucket_count = sum(e['name'] == 'nep.redistribute_params'
                           for e in nep[0]['steps'][str(step)]['events'])
        for bucket in range(bucket_count):
            ag_ends, transfer_ends = [], []
            for rank in range(16):
                data = nep[rank]
                row = data['steps'][str(step)]
                offset = (data['origin_ns']-origin)/1e6
                transfer = [e for e in row['events'] if e['name']=='nep.redistribute_params'][bucket]
                transfer_ends.append(offset+(transfer['ts']+transfer['dur'])/1000)
                if rank < 12:
                    ag = [e for e in row['events'] if category(e)=='expert.all_gather'][bucket]
                    ag_ends.append(offset+(ag['ts']+ag['dur'])/1000)
            delivery.append(dict(iteration=step, bucket=bucket,
                                 latest_transfer_after_latest_ag_ms=max(transfer_ends)-max(ag_ends),
                                 transfer_completion_spread_ms=max(transfer_ends)-min(transfer_ends)))
    result['delivery_buckets'] = delivery
    result['first_forward_gap_penalty_vs_next_microbatch'] = {}
    for case, ranks in ranks_by_case.items():
        gaps = []
        layer_count = len(ranks[0]['steps']['25']['micros'][0]['layers']['f'])
        for layer in range(1,layer_count):
            samples = []
            for step in range(25,35):
                differences = []
                for rank in range(12):
                    micro = ranks[rank]['steps'][str(step)]['micros']
                    a, b = micro[0]['layers']['f'], micro[1]['layers']['f']
                    differences.append(a[layer]['start_ms']-a[layer-1]['end_ms']-
                                       b[layer]['start_ms']+b[layer-1]['end_ms'])
                samples.append(statistics.mean(differences))
            gaps.append(dict(layer_ordinal_zero_based=layer, **stats(samples)))
        result['first_forward_gap_penalty_vs_next_microbatch'][case] = gaps
    (ROOT/'delivery_chain.json').write_text(json.dumps(result,indent=2)+'\n')
    print(ROOT/'delivery_chain.json')
    for case in ['healthy','nep']:
        print(case,{name:{k:round(v['mean'],3) for k,v in metrics.items()}
                    for name,metrics in result[case]['readiness']['owners'].items()})


if __name__ == '__main__':
    main()
