"""Matched timing, phase sums, and parameter-demand spans for actual a8b studies."""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from analyze_nep_actual_ablation import summarize

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab')


def deadlines(run, spec):
    """Keep elapsed demand-to-ready spans separate from transfer service time."""
    healthy = spec['cell'].startswith('H')
    records = []
    for rank in range(32 if healthy else 28):
        data = json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())
        events = data['traceEvents']
        for step in range(spec['profile_start'], spec['profile_end']):
            gpu = [e for e in events if e.get('cat') == 'gpu_operation_range' and e['args']['iteration'] == step]
            demand = [e for e in gpu if e['name'].startswith('parameter.demand.expert.')]
            ready = [e for e in gpu if e['name'].startswith('parameter.ready.expert.')]
            assert demand and len(demand) == len(ready), (run, rank, step)
            for before, after in zip(demand, ready):
                bucket = before['name'].rsplit('bucket', 1)[1]
                assert after['name'].endswith(f'bucket{bucket}')
                duration = (after['ts'] - before['ts']) / 1000
                assert duration >= -0.001
                records.append(dict(rank=rank, iteration=step, bucket=int(bucket),
                                    demand_ms=before['ts']/1000, ready_ms=after['ts']/1000,
                                    demand_to_ready_ms=max(0, duration)))
    cohorts = {}
    for role, members in [('owners', range(12)), ('followers', range(12, 16)),
                          ('second', range(16, 32 if healthy else 28))]:
        grouped = defaultdict(list)
        for row in records:
            if row['rank'] in members:
                grouped[row['bucket']].append(row['demand_to_ready_ms'])
        cohorts[role] = {str(bucket): dict(mean_ms=st.mean(values), samples=values)
                         for bucket, values in sorted(grouped.items())}
    return dict(records=records, cohorts=cohorts,
                note='Elapsed compute-stream demand-to-ready spans, not pure transfer time.')


def analyze(job):
    root = ROOT / job
    summarize(root)
    summary = json.loads((root / 'actual-summary.json').read_text())
    lookup = {}
    for name, run in summary['cases'].items():
        repeat = int(name.split('_')[3][1:])
        spec = run['spec']
        key = (spec['cell'], repeat, run['profiled'])
        assert key not in lookup
        lookup[key] = run
        if run['profiled']:
            run['delivery_deadlines'] = deadlines(root / name, spec)
            for role in run['cohorts'].values():
                for total, left, right in [('all_forward_ms', 'first_forward_ms', 'other_forward_ms'),
                                           ('all_backward_ms', 'nonfinal_backward_ms', 'final_backward_ms')]:
                    values = [a+b for a, b in zip(role[left]['samples'], role[right]['samples'])]
                    role[total] = dict(mean=st.mean(values), samples=values)
    comparisons = []
    for (cell, repeat, profiled), nep in sorted(lookup.items()):
        if cell.startswith('H'):
            continue
        if cell in ('M1', 'M2', 'M4'):
            healthy_cell = 'H' + cell[1:]
        elif cell in ('N4K', 'D4K', 'P4K', 'N8K', 'D8K', 'P8K'):
            healthy_cell = 'H' + cell[1:]
        else:
            raise ValueError(cell)
        healthy = lookup.get((healthy_cell, repeat, profiled))
        if healthy is None:
            print('NO_MATCHED_CONTROL', cell, repeat, profiled, flush=True)
            continue
        n, h = nep['timing']['mean'], healthy['timing']['mean']
        spec, hs = nep['spec'], healthy['spec']
        assert (spec['mbs'], spec['ga'], spec['seqlen']) == (hs['mbs'], hs['ga'], hs['seqlen'])
        row = dict(cell=cell, healthy_cell=healthy_cell, repeat=repeat, profiled=profiled,
                   healthy_ms=h, nep_ms=n, gap_ms=n-h, parity=100*h/n,
                   mbs=spec['mbs'], ga=spec['ga'], seqlen=spec['seqlen'],
                   throughput=nep['tokens_per_second'], peak_GiB=nep['peak_GiB'])
        if profiled:
            row['phase_deltas_ms'] = {
                key: nep['cohorts']['owners'][key]['mean'] - healthy['cohorts']['owners'][key]['mean']
                for key in ('first_forward_ms', 'other_forward_ms', 'nonfinal_backward_ms',
                            'final_backward_ms', 'outside_fb_ms', 'all_forward_ms', 'all_backward_ms')}
        comparisons.append(row)
        print('MATCHED', json.dumps(row), flush=True)
    summary['matched_comparisons'] = comparisons
    (root / 'delivery-batching-summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    analyze(sys.argv[1])
