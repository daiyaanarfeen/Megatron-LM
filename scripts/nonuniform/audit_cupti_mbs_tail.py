"""Partition measured iteration latency at observable GPU milestones, without overlap sums."""

import argparse
import json
import statistics
from pathlib import Path


def audit(rank):
    raw = json.loads(Path(rank['path']).read_text())
    assert raw['dropped_records'] == raw['invalid_records'] == 0
    kernels = sorted((e for e in raw['traceEvents'] if e['cat'] == 'kernel'),
                     key=lambda e: e['ts'])
    rows = []
    for row in rank['iterations']:
        start, end = row['window_start_ms'], row['window_end_ms']
        expert_end = start + row['last_expert_ms']
        adam = [e for e in kernels if expert_end <= e['ts']/1000 < end
                and 'AdamFunctor' in e['name']]
        assert adam, (rank['path'], row['iteration'])
        first_adam = min(e['ts'] for e in adam)/1000
        last_adam = max(e['ts'] + e['dur'] for e in adam)/1000
        # These final two operations map to default_pg barriers in the reference
        # trace. The parent analyzer checks every stream's complete NCCL sequence.
        barrier = sorted(row['comm_ops'], key=lambda e: e['b'])[-2]
        assert barrier['category'] == 'other_comm'
        bounds = [start, expert_end, first_adam, last_adam, barrier['a'], barrier['b'], end]
        assert bounds == sorted(bounds), (rank['path'], row['iteration'], bounds)
        names = ['through_last_expert', 'expert_end_to_adam_start', 'adam_span',
                 'adam_end_to_barrier_start', 'barrier_kernel', 'barrier_end_to_step_end']
        parts = dict(zip(names, (b-a for a, b in zip(bounds, bounds[1:]))))
        assert abs(sum(parts.values())-row['range_ms']) < 1e-7
        rows.append(dict(iteration=row['iteration'], parts_ms=parts,
                         expert_gemm_ms=row['expert_gemm_ms'],
                         barrier_start_ms=barrier['a']-start,
                         adam_kernels=len(adam),
                         adam_kernel_sum_ms=sum(e['dur'] for e in adam)/1000))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline')
    output = root / args.job / 'cupti_mbs_analysis'
    summary = json.loads((output/'summary.json').read_text())
    results = {}
    for mbs, data in summary.items():
        results[mbs] = {'ranks': {}, 'owners': {}}
        for name, rank in data['ranks'].items():
            cache = output/f'tail_mbs{mbs}_{name}.json'
            rows = json.loads(cache.read_text()) if cache.exists() else audit(rank)
            cache.write_text(json.dumps(rows, indent=2)+'\n')
            results[mbs]['ranks'][name] = rows
            print(mbs, name, 'tail validated', flush=True)
        for case in ['healthy', 'nep']:
            rank_rows = [results[mbs]['ranks'][f'{case}_{i}'] for i in range(6)]
            keys = rank_rows[0][0]['parts_ms']
            per_iteration = [{k: statistics.mean(r[i]['parts_ms'][k] for r in rank_rows)
                              for k in keys} for i in range(len(rank_rows[0]))]
            results[mbs]['owners'][case] = {
                k: {'mean_ms': statistics.mean(r[k] for r in per_iteration),
                    'std_ms': statistics.stdev(r[k] for r in per_iteration)} for k in keys}
        print(mbs, json.dumps(results[mbs]['owners']), flush=True)
    (output/'tail_summary.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
