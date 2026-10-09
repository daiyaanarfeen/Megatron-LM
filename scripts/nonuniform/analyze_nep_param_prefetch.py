"""Same-allocation first-forward and iteration comparison with unchanged capture."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from analyze_exact_uniform_nep_slowdown import intersection_duration
from analyze_nep_collective_probe import category, interval
from analyze_nep_sync_workload import ROOT, SLUG, compare, read_case


def profile(path):
    data = json.loads(path.read_text())
    assert data['collective_schema_version'] == 1
    rows = []
    for step in range(40, 50):
        selected = [e for e in data['traceEvents'] if e.get('args', {}).get('iteration') == step]
        gpu = [e for e in selected if e['cat'] == 'gpu_operation_range']
        cpu = [e for e in selected if e['cat'] == 'cpu_operation_range']
        names = Counter(e['name'] for e in gpu)
        assert names == Counter(e['name'] for e in cpu)
        forward = sorted((e for e in gpu if e['name'] == 'expert.forward'), key=lambda e: e['ts'])[:30]
        redistribute = sorted((e for e in gpu if e['name'] == 'nep.redistribute_params'), key=lambda e: e['ts'])
        ag = sorted((e for e in gpu if category(e) == 'expert_all_gather'), key=lambda e: e['ts'])
        cpu_r = sorted((e for e in cpu if e['name'] == 'nep.redistribute_params'), key=lambda e: e['ts'])
        assert len(ag) == len(redistribute) == len(cpu_r) == 8
        row = dict(iteration=step, buckets=[])
        for j, (a, r, c) in enumerate(zip(ag, redistribute, cpu_r)):
            target = 0 if j == 0 else 2*j-1
            previous = forward[2*target-1] if target else None
            row['buckets'].append(dict(
                bucket=j, target_expert=target, ag_ms=a['dur']/1000,
                redistribution_ms=r['dur']/1000,
                ag_completion_to_redistribution_start_ms=interval(r)[0]-interval(a)[1],
                overlap_recorded_forward_expert_ms=intersection_duration([interval(r)], [interval(e) for e in forward]),
                cpu_submit_after_previous_expert_ms=(c['ts']/1000-interval(previous)[1]) if previous else None,
                redistribution_end_before_target_expert_ms=interval(forward[2*target])[0]-interval(r)[1],
                same_stream_as_target=r['args']['stream'] == forward[2*target]['args']['stream'],
            ))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    root = ROOT / args.job
    cases = {tag: read_case(root, tag, 'healthy' if tag == 'healthy' else 'nep') for tag in ('healthy', 'delayed', 'prefetch')}
    result = dict(job=args.job, cases=cases, comparisons={tag: compare({'healthy': cases['healthy'], 'nep': cases[tag]}) for tag in ('delayed', 'prefetch')}, profiles={})
    for tag in ('delayed', 'prefetch'):
        ranks = {r: profile(root/tag/'cupti/ep8'/SLUG/'nep'/args.job/'cupti_profile'/f'rank-{r}.json') for r in range(6)}
        summaries = []
        for j in range(8):
            values = [ranks[r][i]['buckets'][j] for r in range(6) for i in range(10)]
            summaries.append({key: statistics.mean(v[key] for v in values) for key in values[0] if values[0][key] is not None})
        result['profiles'][tag] = dict(ranks=ranks, bucket_means=summaries)
    output = root/'prefetch_analysis'
    output.mkdir(exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for tag, case in cases.items():
        owner = case['cohorts']['full_owners']
        print(tag, 'iteration', owner['iteration']['mean'], 'sd', owner['iteration']['sd'], 'first_forward', owner['comparisons']['first_forward']['mean'])
        print('windows', case['windows'])
    for tag, comparison in result['comparisons'].items():
        print(tag, json.dumps(comparison, indent=2))
        print('prefetch_means', json.dumps(result['profiles'][tag]['bucket_means'], indent=2))
    print(output/'summary.json')


if __name__ == '__main__':
    main()
