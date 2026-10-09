"""Audit compact-parameter before/after profiles without calling waiting wire time.

Input caches come from analyze_a8b_current_overheads.py. Ordinary microbatches
exclude the first and final microbatches. CPU duration is measured thread time,
not inferred from a GPU gap. These operation ranges cannot identify GPU kernels.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_a8b_current_overheads import category

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline')


def audit(job: str, tag: str) -> dict:
    """Match RS participants and partition changes in replica arrival times."""
    directory = ROOT / job / f'{tag}_overhead_analysis'
    ranks = {r: json.loads((directory / f'nep-rank-{r}.json').read_text()) for r in range(28)}
    origin = min(rank['origin_ns'] for rank in ranks.values())
    offsets = {r: (rank['origin_ns'] - origin) / 1e6 for r, rank in ranks.items()}
    rows = []
    for step in range(25, 35):
        data = {r: rank['steps'][str(step)] for r, rank in ranks.items()}
        matched = defaultdict(dict)
        for rank, row in data.items():
            counts = Counter()
            for event in row['events']:
                if category(event) != 'expert.reduce_scatter':
                    continue
                group = event['args']['collective']
                key = (group['group_id'], counts[group['group_id']])
                counts[group['group_id']] += 1
                a = offsets[rank] + event['ts'] / 1000
                matched[key][rank] = (a, a + event['dur'] / 1000, group['ranks'])
        rs = defaultdict(lambda: defaultdict(float))
        for peers in matched.values():
            assert set(peers) == set(next(iter(peers.values()))[2])
            latest = max(row[0] for row in peers.values())
            for rank, (start, end, _) in peers.items():
                rs[rank]['waiting_ms'] += latest - start
                rs[rank]['post_last_ready_ms'] += end - latest
                rs[rank]['elapsed_ms'] += end - start
        groups = {}
        for role, members in [('owners', range(12)), ('full', range(16)), ('reduced', range(16, 28))]:
            phases = {
                key: statistics.mean(data[r]['phases'][key] for r in members)
                for key in data[0]['phases']
            }
            micros = [{
                key: statistics.mean(data[r]['micros'][mb][key] for r in members)
                for key in ['forward_ms', 'backward_ms', 'b_cpu_thread_ms',
                            'recompute_cpu_thread_ms', 'b_gpu_range_sum_ms']
            } for mb in range(8)]
            cpu = {
                r: statistics.mean(m['b_cpu_thread_ms'] + m['recompute_cpu_thread_ms']
                                   for m in data[r]['micros'][1:7])
                for r in members
            }
            last = Counter()
            for mb in range(1, 7):
                for layer in range(15):
                    finish = {r: offsets[r] + data[r]['micros'][mb]['layers']['b'][layer]['end_ms']
                              for r in members}
                    last[max(finish, key=finish.get)] += 1
            groups[role] = dict(
                phases=phases, micros=micros, ordinary_expert_cpu_ms_by_rank=cpu,
                latest_expert_backward_rank_counts=dict(last),
                rs={key: statistics.mean(rs[r][key] for r in members)
                    for key in ['waiting_ms', 'post_last_ready_ms', 'elapsed_ms']},
            )
        skew = [{
            key: statistics.mean(
                offsets[r + 16] - offsets[r] + data[r + 16]['micros'][mb][key]
                - data[r]['micros'][mb][key] for r in range(12)
            ) for key in ['f_start_ms', 'f_end_ms', 'b_start_ms', 'b_end_ms']
        } for mb in range(8)]
        rows.append(dict(iteration=step, groups=groups, reduced_minus_full=skew))
    summary = {}
    for role in ['owners', 'full', 'reduced']:
        summary[role] = {
            'phases': {key: statistics.mean(row['groups'][role]['phases'][key] for row in rows)
                       for key in rows[0]['groups'][role]['phases']},
            'rs': {key: statistics.mean(row['groups'][role]['rs'][key] for row in rows)
                   for key in rows[0]['groups'][role]['rs']},
            'ordinary_backward_ms': statistics.mean(
                m['backward_ms'] for row in rows for m in row['groups'][role]['micros'][1:7]
            ),
            'critical_expert_cpu_ms': statistics.mean(
                max(row['groups'][role]['ordinary_expert_cpu_ms_by_rank'].values()) for row in rows
            ),
        }
    summary['reduced_minus_full'] = [{
        key: statistics.mean(row['reduced_minus_full'][mb][key] for row in rows)
        for key in rows[0]['reduced_minus_full'][mb]
    } for mb in range(8)]
    return dict(job=job, tag=tag, summary=summary, iterations=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-job', required=True)
    parser.add_argument('--after-job', required=True)
    parser.add_argument('--before-tag', default='gb200_before')
    parser.add_argument('--after-tag', default='gb200_after')
    args = parser.parse_args()
    result = {
        label: audit(getattr(args, f'{label}_job'), getattr(args, f'{label}_tag'))
        for label in ['before', 'after']
    }
    output = ROOT / args.after_job / f'backward_skew_from_{args.before_job}.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    for label, data in result.items():
        print(label, json.dumps(data['summary'], indent=2))
    print(output)


if __name__ == '__main__':
    main()
