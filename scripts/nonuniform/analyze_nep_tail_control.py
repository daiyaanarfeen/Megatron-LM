"""Compare zero-profile controls and sparse/full final-sync milestones."""

import argparse
import json
import re
import statistics
from pathlib import Path

from analyze_cuda_event_mbs_pair import timing
from analyze_nep_collective_probe import category
from analyze_nep_sync_workload import ROOT, SLUG


def full_to_markers(data):
    events = []
    for step in range(40, 50):
        gpu = [e for e in data['traceEvents'] if e.get('cat') == 'gpu_operation_range'
               and e.get('args', {}).get('iteration') == step]
        groups = {}
        for event in gpu:
            groups.setdefault(category(event), []).append(event)
        for group in groups.values():
            group.sort(key=lambda e: e['ts'])
        backward = groups['expert.backward']
        assert len(backward) == 90
        selected = [('final_backward_start', backward[60], False),
                    ('final_backward_end', backward[-1], True),
                    ('optimizer_start', groups['optimizer.step'][0], False)]
        if 'nep.stage_grads' in groups:
            selected += [('last_stage_start', groups['nep.stage_grads'][-1], False),
                         ('last_stage_end', groups['nep.stage_grads'][-1], True)]
        if 'expert_reduce_scatter' in groups:
            last = groups['expert_reduce_scatter'][-1]
            selected += [('expert_rs_ready', last, False), ('expert_rs_done', last, True)]
        dense = [e for e in gpu if e['name'] == 'collective.reduce_scatter'
                 and category(e) != 'expert_reduce_scatter']
        assert len(dense) == 2
        last = max(dense, key=lambda e: e['ts'])
        selected += [('dense_rs_ready', last, False), ('dense_rs_done', last, True)]
        events += [dict(iteration=step, name=name, ms=(e['ts'] + (e['dur'] if end else 0))/1000,
                        metadata=e['args'].get('collective')) for name, e, end in selected]
    return dict(origin_ns=data['origin_ns'], clock_error_bound_ns=data['clock_error_bound_ns'], events=events)


def milestones(run, mode, size):
    path = run / ('tail_markers' if mode == 'sparse' else 'cupti_profile')
    assert len(list(path.glob('rank-*.json'))) == size
    ranks = {r: json.loads((path / f'rank-{r}.json').read_text()) for r in range(size)}
    if mode == 'full':
        ranks = {r: full_to_markers(data) for r, data in ranks.items()}
    origin = min(data['origin_ns'] for data in ranks.values())
    rows = []
    for step in range(40, 50):
        values = {}
        for rank, data in ranks.items():
            events = [e for e in data['events'] if e['iteration'] == step]
            assert len({e['name'] for e in events}) == len(events)
            expected = 7 if size == 16 else (5 if rank in (6, 7) else 9)
            assert len(events) == expected, (run, rank, step, len(events))
            values[rank] = {e['name']: (data['origin_ns']-origin)/1e6 + e['ms'] for e in events}
        latest_dense_rank = max(values, key=lambda r: values[r]['dense_rs_ready'])
        for rank in range(6):
            owner, peer = values[rank], values[rank+8]
            end = owner['final_backward_end']
            row = dict(iteration=step, rank=rank, latest_dense_rank=latest_dense_rank)
            for key in ('expert_rs_ready', 'expert_rs_done', 'optimizer_start', 'dense_rs_done'):
                row[key + '_after_local_backward_ms'] = owner[key]-end
            row.update(
                final_backward_ms=end-owner['final_backward_start'],
                peer_backward_ms=peer['final_backward_end']-peer['final_backward_start'],
                peer_backward_start_lag_ms=peer['final_backward_start']-owner['final_backward_start'],
                peer_backward_end_lag_ms=peer['final_backward_end']-end,
                peer_rs_ready_after_local_backward_ms=peer['expert_rs_ready']-end,
                final_rs_peer_wait_ms=max(0, peer['expert_rs_ready']-owner['expert_rs_ready']),
                final_rs_after_both_ready_ms=owner['expert_rs_done']-max(owner['expert_rs_ready'], peer['expert_rs_ready']),
                final_rs_completion_skew_ms=owner['expert_rs_done']-peer['expert_rs_done'],
                dense_rs_after_all_ready_ms=owner['dense_rs_done']-values[latest_dense_rank]['dense_rs_ready'],
                optimizer_after_dense_rs_ms=owner['optimizer_start']-owner['dense_rs_done'],
            )
            if 'last_stage_end' in owner:
                row.update(
                    local_stage_end_after_backward_ms=owner['last_stage_end']-end,
                    peer_stage_ms=peer['last_stage_end']-peer['last_stage_start'],
                    peer_backward_after_local_stage_ms=peer['final_backward_end']-owner['last_stage_end'],
                )
            rows.append(row)
    stats = {}
    for key in rows[0]:
        if not key.endswith('_ms'):
            continue
        samples = [statistics.mean(row[key] for row in rows if row['iteration'] == step)
                   for step in range(40, 50)]
        stats[key] = dict(mean=statistics.mean(samples), sd=statistics.stdev(samples), samples=samples)
    return dict(stats=stats, rows=rows,
                clock_bounds_ns={rank: d['clock_error_bound_ns'] for rank, d in ranks.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    root = ROOT / args.job
    result = dict(job=args.job, cases={}, parity={})
    for mode in ('none', 'sparse', 'full'):
        result['cases'][mode] = {}
        for case, size in (('healthy', 16), ('nep', 14)):
            run = root/f'{mode}_{case}'/'cupti/ep8'/SLUG/case/args.job
            log = (run/'driver.log').read_text()
            statuses = re.findall(r'iteration\s+(\d+)/\s*60.*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)', log)
            assert len(statuses) == 60 and all(s == n == '0' for _, s, n in statuses)
            manifests = list((run/'launch_manifest').glob('rank-*.json'))
            assert len(manifests) == size
            for path in manifests:
                manifest = json.loads(path.read_text())
                assert ('--profile' in manifest['options']) == (mode == 'full')
                env = manifest['environment']
                assert env['REPLICA_NUM_MICROBATCHES'] == '3 3'
                assert env['REPLICA_MICRO_BATCH_SIZES'] == '2 2'
                assert env['NCCL_MNNVL_CLIQUE_ID'] == str(1000 + int(int(env['RANK']) >= 8))
            entry = dict(path=str(run), windows={name: timing(run/'driver.log', a, b)
                         for name, a, b in (('pre', 29, 39), ('measured', 40, 50), ('post', 51, 61))})
            if mode != 'none':
                entry['milestones'] = milestones(run, mode, size)
            else:
                assert 'TAIL_CONTROL_NO_BROAD_PROFILER none' in log
                assert 'GPU_EVENT_TRACE_RESULT' not in log and 'TAIL_MARKERS_RESULT' not in log
            result['cases'][mode][case] = entry
        result['parity'][mode] = {w: result['cases'][mode]['healthy']['windows'][w]['mean_ms'] /
                                  result['cases'][mode]['nep']['windows'][w]['mean_ms']
                                  for w in ('pre', 'measured', 'post')}
    output = root/'tail_control_analysis'
    output.mkdir(exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('parity', result['parity'])
    for mode, cases in result['cases'].items():
        for case, entry in cases.items():
            print(mode, case, entry['windows']['measured'])
            if 'milestones' in entry:
                print({k: round(v['mean'], 3) for k, v in entry['milestones']['stats'].items()})
    print(output/'summary.json')


if __name__ == '__main__':
    main()
