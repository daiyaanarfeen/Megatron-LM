"""Factorial scheduling effects and rank-local parameter-delivery dependencies.

Event spans include queueing/waiting. Kernel unions include participant waits.
Neither is interchangeable with pure network service or causal iteration cost.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analyze_nep_actual_ablation import stats, summarize
from analyze_nep_ag_red_overlap import duration, union
from analyze_nep_first_forward_delivery import read_rank

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab')
EXPERT_PGS = ('ep_dp', 'EXPERT_DATA_PARALLEL_GROUP')


def intersection_duration(left, right):
    """Intersect merged intervals without a quadratic Cartesian product."""
    left, right = union(left), union(right)
    i, j, total = 0, 0, 0.0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def role(rank):
    return 'owners' if rank < 12 else 'followers' if rank < 16 else 'second'


def event_dependencies(run, spec):
    """Pair buckets in canonical forward order; never sum across ranks."""
    rows = []
    healthy = spec['cell'].startswith('H')
    for rank in range(32 if healthy else 28):
        trace = json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())
        events = trace['traceEvents']
        for step in range(spec['profile_start'], spec['profile_end']):
            gpu = [
                e
                for e in events
                if e.get('cat') == 'gpu_operation_range' and e['args']['iteration'] == step
            ]
            cpu = [
                e
                for e in events
                if e.get('cat') == 'cpu_operation_range' and e['args']['iteration'] == step
            ]
            assert [e['name'] for e in gpu] == [e['name'] for e in cpu]
            pairs = list(zip(gpu, cpu))
            first = next(g for g, c in pairs if g['name'] == 'microbatch.forward')
            origin = first['ts'] / 1000

            def select(predicate):
                return [(g, c) for g, c in pairs if predicate(g)]

            demand = select(lambda e: e['name'].startswith('parameter.demand.expert.'))
            ready = select(lambda e: e['name'].startswith('parameter.ready.expert.'))
            ag = select(
                lambda e: e['name'] == 'collective.all_gather'
                and e['args']['collective']['group_desc'] in EXPERT_PGS
            )
            red = select(lambda e: e['name'] == 'nep.redistribute_params')
            assert len(demand) == len(ready) and demand
            owner = healthy or rank < 12 or rank >= 16
            assert len(ag) == (len(demand) if owner else 0), (run, rank, step, 'AG count')
            assert len(red) == (0 if healthy else len(demand)), (run, rank, step, 'delivery count')
            for index, ((dg, dc), (rg, rc)) in enumerate(zip(demand, ready)):
                bucket = int(dg['name'].rsplit('bucket', 1)[1])
                assert rg['name'].endswith(f'bucket{bucket}')
                row = dict(
                    rank=rank,
                    role=role(rank),
                    iteration=step,
                    bucket=bucket,
                    position=index,
                    first_forward_ms=first['dur'] / 1000,
                    demand_ms=dg['ts'] / 1000 - origin,
                    ready_ms=rg['ts'] / 1000 - origin,
                    demand_host_ms=dc['ts'] / 1000 - origin,
                    wait_ms=(rg['ts'] - dg['ts']) / 1000,
                    clock_error_ms=trace['clock_error_bound_ns'] / 1e6,
                )
                for prefix, sequence in [('ag', ag), ('red', red)]:
                    if not sequence:
                        continue
                    g, c = sequence[index]
                    row.update(
                        {
                            prefix + '_start_ms': g['ts'] / 1000 - origin,
                            prefix + '_end_ms': (g['ts'] + g['dur']) / 1000 - origin,
                            prefix + '_host_ms': c['ts'] / 1000 - origin,
                            prefix + '_host_end_ms': (c['ts'] + c['dur']) / 1000 - origin,
                            prefix + '_host_duration_ms': c['dur'] / 1000,
                        }
                    )
                if index and spec['prefetch_depth'] == 0:
                    # A control that did not actually remove AG lookahead is invalid.
                    if owner:
                        assert row['ag_host_ms'] >= row['demand_host_ms'], row
                if not healthy and spec['delivery'] == 'consumer':
                    assert row['red_host_ms'] >= row['demand_host_ms'], row
                if ag:
                    row['wait_before_ag_completion_ms'] = max(
                        0, min(row['ready_ms'], row['ag_end_ms']) - row['demand_ms']
                    )
                    row['wait_after_ag_completion_ms'] = (
                        row['wait_ms'] - row['wait_before_ag_completion_ms']
                    )
                if red:
                    row['red_span_ms'] = row['red_end_ms'] - row['red_start_ms']
                    row['red_ready_slack_ms'] = row['demand_ms'] - row['red_end_ms']
                    if ag:
                        row['ag_to_red_start_ms'] = row['red_start_ms'] - row['ag_end_ms']
                        row['host_submission_after_ag_ms'] = max(
                            0, row['red_host_ms'] - row['ag_end_ms']
                        )
                rows.append(row)
    return rows


def kernel_dependencies(run, spec, out):
    """Kernel-level unions for the same first-forward windows and iterations."""
    result = []
    healthy = spec['cell'].startswith('H')
    for rank in range(32 if healthy else 28):
        cache = out / f'{run.name}-rank-{rank}.json'
        if not cache.exists():
            data = read_rank(run, rank, True, range(spec['profile_start'], spec['profile_end']))
            cache.write_text(json.dumps(data) + '\n')
        data = json.loads(cache.read_text())
        for step in data['rows']:
            a, b = step['start'], step['end']
            groups = defaultdict(list)
            for kernel in step['kernels']:
                groups[kernel['label']].append((max(a, kernel['start']), min(b, kernel['end'])))
            # Historical parser label NEP_broadcast also matches destination-exact A2A.
            red = groups['NEP_broadcast'] + groups['NEP_staging']
            ag = groups['expert_AG']
            compute = groups['model_compute']
            other = [
                x
                for label, spans in groups.items()
                if label not in ('NEP_broadcast', 'NEP_staging')
                for x in spans
            ]
            result.append(
                dict(
                    rank=rank,
                    role=role(rank),
                    iteration=step['iteration'],
                    first_forward_ms=b - a,
                    redistribution_union_ms=duration(red),
                    redistribution_exclusive_ms=duration(red) - intersection_duration(red, other),
                    expert_ag_union_ms=duration(ag),
                    ag_red_overlap_ms=intersection_duration(ag, red),
                    red_compute_overlap_ms=intersection_duration(red, compute),
                    ag_compute_overlap_ms=intersection_duration(ag, compute),
                    idle_ms=(b - a) - duration(x for spans in groups.values() for x in spans),
                )
            )
        print('KERNELS', run.name, rank, flush=True)
    return result


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for group, values in sorted(groups.items()):
        output = dict(zip(keys, group))
        for key in values[0]:
            if key in keys or key in ('rank', 'iteration', 'bucket', 'position', 'role'):
                continue
            if all(key in v and isinstance(v[key], (int, float)) for v in values):
                output[key] = stats([v[key] for v in values])
        result.append(output)
    return result


def analyze(job, kernels=False):
    root = ROOT / job
    summarize(root)
    output = root / 'delivery_factorial'
    output.mkdir(exist_ok=True)
    summary = json.loads((root / 'actual-summary.json').read_text())
    lookup, comparisons = {}, []
    for name, case in summary['cases'].items():
        spec = case['spec']
        repeat = int(name.split('_')[3][1:])
        lookup[spec['cell'], repeat, case['profiled']] = case
        if case['profiled']:
            deps = event_dependencies(root / name, spec)
            case['bucket_dependencies'] = aggregate(deps, ('role', 'position'))
            (output / f'{name}-dependencies.json').write_text(json.dumps(deps) + '\n')
            if kernels:
                ks = kernel_dependencies(root / name, spec, output)
                case['kernel_unions'] = aggregate(ks, ('role',))
    contrasts = [
        ('AG_prefetch_consumer', 'Q00', 'Q10'),
        ('AG_prefetch_early', 'Q01', 'Q11'),
        ('early_delivery_no_prefetch', 'Q00', 'Q01'),
        ('early_delivery_prefetch', 'Q10', 'Q11'),
        ('native_AG_prefetch', 'H0', 'H1P'),
        ('complete_NEP_pipeline', 'Q00', 'Q11'),
    ]
    for repeat in (0, 1):
        for profiled in (False, True):
            for cell in ('Q00', 'Q01', 'Q10', 'Q11'):
                hc = 'H0' if cell[1] == '0' else 'H1P'
                n, h = lookup.get((cell, repeat, profiled)), lookup.get((hc, repeat, profiled))
                if n is None or h is None:
                    continue
                reference = lookup.get(('H1P', repeat, profiled))
                row = dict(
                    cell=cell,
                    repeat=repeat,
                    profiled=profiled,
                    healthy_ms=h['timing']['mean'],
                    nep_ms=n['timing']['mean'],
                    matched_policy_parity=100 * h['timing']['mean'] / n['timing']['mean'],
                )
                if reference is not None:
                    row['fixed_healthy_ms'] = reference['timing']['mean']
                    row['fixed_healthy_parity'] = (
                        100 * reference['timing']['mean'] / n['timing']['mean']
                    )
                comparisons.append(row)
            for label, before, after in contrasts:
                left, right = lookup.get((before, repeat, profiled)), lookup.get(
                    (after, repeat, profiled)
                )
                if left is None or right is None:
                    continue
                row = dict(
                    effect=label,
                    repeat=repeat,
                    profiled=profiled,
                    saved_iteration_ms=left['timing']['mean'] - right['timing']['mean'],
                )
                if profiled:
                    row['saved_phases_ms'] = {
                        k: left['cohorts']['owners'][k]['mean']
                        - right['cohorts']['owners'][k]['mean']
                        for k in left['cohorts']['owners']
                    }
                comparisons.append(row)
                print('EFFECT', json.dumps(row), flush=True)
            quartet = [
                lookup.get((cell, repeat, profiled)) for cell in ('Q00', 'Q01', 'Q10', 'Q11')
            ]
            if all(quartet):
                q00, q01, q10, q11 = [x['timing']['mean'] for x in quartet]
                comparisons.append(
                    dict(
                        effect='early_release_prefetch_interaction',
                        repeat=repeat,
                        profiled=profiled,
                        saved_iteration_ms=(q10 - q11) - (q00 - q01),
                    )
                )
                h0, h1 = lookup.get(('H0', repeat, profiled)), lookup.get(('H1P', repeat, profiled))
                if h0 is not None and h1 is not None:
                    comparisons.append(
                        dict(
                            effect='pipeline_gain_beyond_healthy_prefetch',
                            repeat=repeat,
                            profiled=profiled,
                            saved_iteration_ms=(q00 - q11)
                            - (h0['timing']['mean'] - h1['timing']['mean']),
                        )
                    )
    summary['factorial_comparisons'] = comparisons
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('--kernels', action='store_true')
    args = parser.parse_args()
    analyze(args.job, args.kernels)
