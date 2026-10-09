"""Offline schedule sensitivity, not a measured or guaranteed speedup.

Replay recorded GPU-event dependencies with unchanged local service/copy intervals.
The coupled scenario advances full-replica compute but freezes reduced-peer readiness.
No runtime code, communication backend, or training configuration is changed.
"""

import json
import statistics
from pathlib import Path


ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3131835')
CONSUMERS = [1, 3, 8, 12, 16, 21, 25, 29]


def waves(members):
    batches, occupied = [], []
    for owner, ranks in sorted(members.items()):
        ranks = set(ranks)
        for batch, used in zip(batches, occupied):
            if used.isdisjoint(ranks):
                batch.append(owner)
                used.update(ranks)
                break
        else:
            batches.append([owner])
            occupied.append(set(ranks))
    assert sorted(x for batch in batches for x in batch) == sorted(members)
    for batch in batches:
        assert sum(len(members[x]) for x in batch) == len(set().union(*(set(members[x]) for x in batch)))
    return batches


def interval(event, offset):
    return event['ts'] / 1000 + offset, (event['ts'] + event['dur']) / 1000 + offset


def prepare(raw, step):
    origin = raw[0]['origin_ns']
    events = {}
    offsets = {r: (d['origin_ns'] - origin) / 1e6 for r, d in raw.items()}
    for rank, d in raw.items():
        events[rank] = [e for e in d['traceEvents'] if e['cat'] == 'gpu_operation_range'
                        and e['args']['iteration'] == step]
    anchor = next(e['ts']/1000 for e in events[0] if e['name'] == 'microbatch.forward')
    offsets = {r: s - anchor for r, s in offsets.items()}
    red, ag, graphs, first_f, comm = {}, {}, {}, {}, {}
    members = {}
    for rank, es in events.items():
        first_f[rank] = interval(next(e for e in es if e['name'] == 'microbatch.forward'), offsets[rank])
        ag[rank] = [interval(e, offsets[rank]) for e in es if e['name'] == 'collective.all_gather'
                    and e['args']['collective']['group_desc'] == 'ep_dp']
        if rank >= 16:
            assert len(ag[rank]) == 8
            continue
        red[rank] = [interval(e, offsets[rank]) for e in es if e['name'] == 'nep.redistribute_params']
        graphs[rank] = [interval(e, offsets[rank]) for e in es
                        if e['name'].startswith('graph.') and e['name'].endswith('.forward')][:32]
        assert len(graphs[rank]) == 32 and len(red[rank]) == 8
        comm[rank] = {}
        for e in es:
            if e['name'] != 'collective.broadcast' or e['args']['collective']['group_desc'] != 'nep_owner_transfer':
                continue
            meta = e['args']['collective']
            owner = meta['operations'][0]['source_rank']
            assert members.setdefault(owner, meta['ranks']) == meta['ranks']
            comm[rank].setdefault(owner, []).append(interval(e, offsets[rank]))
        assert all(len(values) == 8 for values in comm[rank].values())
    assert sorted(members) == list(range(12))
    assert all(sorted(comm[r]) == [o for o, rs in members.items() if r in rs] for r in range(16))

    service, after, initial = [], [], []
    negative_gaps = []
    for bucket in range(8):
        service.append({})
        after.append({})
        initial.append({})
        for owner, ranks in members.items():
            latest = max(comm[r][owner][bucket][0] for r in ranks)
            service[bucket][owner] = {r: max(0, comm[r][owner][bucket][1] - latest) for r in ranks}
        for rank in range(16):
            order = sorted(comm[rank])
            initial[bucket][rank] = comm[rank][order[0]][bucket][0]
            for i, owner in enumerate(order):
                next_start = comm[rank][order[i+1]][bucket][0] if i+1 < len(order) else red[rank][bucket][1]
                gap = next_start - comm[rank][owner][bucket][1]
                negative_gaps.append(min(0, gap))
                after[bucket][rank, owner] = max(0, gap)
    # Timer/observer jitter must be small compared with the modeled millisecond effects.
    assert min(negative_gaps) > -0.1, min(negative_gaps)
    return dict(red=red, ag=ag, graphs=graphs, first_f=first_f, members=members,
                service=service, after=after, initial=initial,
                max_negative_gap_ms=-min(negative_gaps))


def replay(d, order, coupled=False, service_scale=1.0):
    ag, red, graphs = d['ag'], d['red'], d['graphs']
    ref = 11
    assert all(max(range(16), key=lambda r: red[r][b][1]) == ref for b in range(8))
    produced, ag_done = {}, {}
    stage_errors = []

    def deliver(bucket, boundary_shift):
        ready = {}
        for rank in range(16):
            if bucket == 0 or not coupled:
                ready[rank] = d['initial'][bucket][rank]
                if rank < 12:
                    ag_done[bucket, rank] = ag[rank][bucket][1]
                continue
            old_prev = red[rank][bucket-1][1]
            new_prev = produced[bucket-1][rank]
            j = CONSUMERS[bucket-1]-1
            old_boundary = graphs[rank][j][1]
            new_boundary = old_boundary + boundary_shift
            if rank < 12:
                old_ag_release = max(ag[rank][bucket][0], ag[rank+16][bucket][0], ag[rank][bucket-1][1])
                ag_service = ag[rank][bucket][1] - old_ag_release
                assert ag_service > -0.1
                new_ag_release = max(ag[rank][bucket][0] + boundary_shift,
                                     ag[rank+16][bucket][0], ag_done[bucket-1, rank])
                ag_done[bucket, rank] = new_ag_release + max(0, ag_service)
                old_gate = max(old_prev, old_boundary, ag[rank][bucket][1])
                new_gate = max(new_prev, new_boundary, ag_done[bucket, rank])
            else:
                old_gate = max(old_prev, old_boundary)
                new_gate = max(new_prev, new_boundary)
            preparation = d['initial'][bucket][rank] - old_gate
            stage_errors.append(min(0, preparation))
            ready[rank] = new_gate + max(0, preparation)
        for owner in order:
            ranks = d['members'][owner]
            start = max(ready[r] for r in ranks)
            for rank in ranks:
                ready[rank] = start + d['service'][bucket][owner][rank]*service_scale + d['after'][bucket][rank,owner]
        produced[bucket] = ready

    deliver(0, 0)
    previous_old = d['first_f'][ref][0]
    previous_new = previous_old
    for j, (start, end) in enumerate(graphs[ref]):
        bucket = CONSUMERS.index(j) if j in CONSUMERS else None
        if bucket is not None and bucket < 7:
            deliver(bucket + 1, previous_new - previous_old)
        old_gate = max(previous_old, max(red[r][bucket][1] for r in range(16))) if bucket is not None else previous_old
        new_gate = max(previous_new, max(produced[bucket].values())) if bucket is not None else previous_new
        overhead = start - old_gate
        assert overhead > -0.1, (j, overhead)
        previous_new = new_gate + max(0, overhead) + end - start
        previous_old = end
    shift = previous_new - previous_old
    return dict(first_forward_delta_ms=shift,
                final_bucket_ready_shift_ms=max(produced[7].values()) - max(red[r][7][1] for r in range(16)),
                bucket_ready_shift_ms=[max(produced[b].values())-max(red[r][b][1] for r in range(16)) for b in range(8)],
                preparation_fit_error_ms=max([-x for x in stage_errors], default=0),
                predicted_param_ready=produced)


def main():
    cases = json.loads((ROOT/'cleanup_phase_analysis.json').read_text())
    case = cases['cleanup_ab_ag_overlap_margin']
    run = Path(case['run'])
    raw = {r: json.loads((run/f'cupti_profile/rank-{r}.json').read_text()) for r in range(28)}
    rows = []
    for step in range(20, 28):
        d = prepare(raw, step)
        batches = waves(d['members'])
        order = [x for batch in batches for x in batch]
        variants = {}
        for label, sequence, coupled, scale in [
            ('original_fixed', list(range(12)), False, 1),
            ('original_coupled', list(range(12)), True, 1),
            ('waves_fixed_releases', order, False, 1),
            ('waves_full_compute_coupled', order, True, 1),
            ('waves_coupled_service_1.25', order, True, 1.25),
            ('waves_coupled_service_1.5', order, True, 1.5),
        ]:
            variants[label] = replay(d, sequence, coupled, scale)
        rows.append(dict(step=step, waves=batches, variants=variants))
        print(step, {k: round(v['first_forward_delta_ms'], 3) for k,v in variants.items()}, flush=True)
    summary = {}
    healthy = cases['cleanup_ab_healthy_margin']['cohorts']['owners']['summary']
    current = case['cohorts']['owners']['summary']
    for key in rows[0]['variants']:
        deltas = [row['variants'][key]['first_forward_delta_ms'] for row in rows]
        delta = statistics.mean(deltas)
        first = current['first_forward_ms']['mean'] + delta
        iteration = current['iteration_ms']['mean'] + delta
        summary[key] = dict(mean_delta_ms=delta, min_delta_ms=min(deltas), max_delta_ms=max(deltas),
                            projected_first_forward_ms=first,
                            projected_first_forward_parity=100*healthy['first_forward_ms']['mean']/first,
                            projected_iteration_ms=iteration,
                            projected_iteration_parity=100*healthy['iteration_ms']['mean']/iteration,
                            max_preparation_fit_error_ms=max(row['variants'][key]['preparation_fit_error_ms'] for row in rows))
    result = dict(waves=rows[0]['waves'], summary=summary, rows=rows,
                  assumptions=['same local post-ready service and copy/launch gaps',
                               'no additional native-kernel interference',
                               'reduced-peer AG readiness fixed',
                               'unchanged critical-lane graph service and later phases',
                               'no new streams or extra simultaneous transfers per rank'],
                  warning='Schedule sensitivity, not a measured gain or a lower bound.')
    (ROOT/'ag_redistribution_analysis/parameter_wave_estimate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
