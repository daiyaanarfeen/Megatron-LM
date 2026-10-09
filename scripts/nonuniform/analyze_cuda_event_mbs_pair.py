"""Audit targeted operation traces and calibrate their same-window iteration timing."""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from analyze_exact_uniform_nep_slowdown import union_duration

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline')


def timing(path, start, end):
    text = path.read_text()
    values = {int(i): float(v) for i, v in re.findall(
        r'iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+)', text)}
    selected = [values[i] for i in range(start, end)]
    return dict(ms=selected, mean_ms=statistics.mean(selected), std_ms=statistics.stdev(selected))


def analyze(path, first, last, case):
    data = json.loads(path.read_text())
    assert data['backend'] == 'cuda_events'
    assert data['clock_error_bound_ns'] < 200000
    events = data['traceEvents']
    rows = []
    for step in range(first, last):
        windows = [e for e in events if e['cat']=='iteration' and e['name']==f'iteration_{step}']
        assert len(windows) == 1, (path, step)
        window = windows[0]
        selected = [e for e in events if e.get('args', {}).get('iteration') == step]
        gpu = [e for e in selected if e['cat']=='gpu_operation_range']
        cpu = [e for e in selected if e['cat']=='cpu_operation_range']
        counts = Counter(e['name'] for e in gpu)
        assert counts == Counter(e['name'] for e in cpu)
        assert counts['expert.forward'] == counts['expert.backward'] == 90, (path, step, counts)
        assert counts['optimizer.step'] == 1, (path, step, counts)
        assert counts['optimizer.copy_params'] in (1, 2), (path, step, counts)
        if case == 'nep':
            assert counts['nep.stage_grads'] in (0, 8), (path, step, counts)
            assert counts['nep.redistribute_params'] in (0, 8), (path, step, counts)
        else:
            assert not any(name.startswith('nep.') for name in counts)
        metrics = {}
        for name, count in counts.items():
            g = [e for e in gpu if e['name'] == name]
            c = [e for e in cpu if e['name'] == name]
            assert all(e['dur'] >= 0 for e in g+c)
            metrics[name] = dict(
                count=count,
                gpu_elapsed_union_ms=union_duration([(e['ts']/1000, (e['ts']+e['dur'])/1000) for e in g]),
                gpu_elapsed_sum_ms=sum(e['dur'] for e in g)/1000,
                cpu_wall_ms=sum(e['dur'] for e in c)/1000,
                cpu_thread_ms=sum(e['args']['thread_cpu_us'] for e in c)/1000,
                mean_gpu_start_lag_ms=statistics.mean((ge['ts']-ce['ts'])/1000 for ge,ce in zip(g,c)),
                mean_gpu_end_lag_ms=statistics.mean((ge['ts']+ge['dur']-ce['ts']-ce['dur'])/1000
                                                  for ge,ce in zip(g,c)),
                launch_done_before_gpu_start_fraction=statistics.mean(
                    ge['ts'] > ce['ts']+ce['dur']+3*data['clock_error_bound_ns']/1000
                    for ge,ce in zip(g,c)),
            )
        expert_end = max(e['ts']+e['dur'] for e in gpu if e['name'].startswith('expert.'))
        optimizer = next(e for e in gpu if e['name']=='optimizer.step')
        # Disjoint temporal spans, not a causal attribution to the kernels inside.
        forward = sorted((e for e in gpu if e['name']=='expert.forward'), key=lambda e: e['ts'])
        backward = sorted((e for e in gpu if e['name']=='expert.backward'), key=lambda e: e['ts'])
        boundaries = [('iteration_start', window['ts'])]
        for microbatch in range(3):
            f = forward[30*microbatch:30*(microbatch+1)]
            b = backward[30*microbatch:30*(microbatch+1)]
            boundaries.extend([
                (f'mb{microbatch}_first_forward', f[0]['ts']),
                (f'mb{microbatch}_last_forward', f[-1]['ts']+f[-1]['dur']),
                (f'mb{microbatch}_first_backward', b[0]['ts']),
                (f'mb{microbatch}_last_backward', b[-1]['ts']+b[-1]['dur']),
            ])
        boundaries.extend([('optimizer_start', optimizer['ts']),
                           ('optimizer_end', optimizer['ts']+optimizer['dur']),
                           ('iteration_end', window['ts']+window['dur'])])
        spans = {f'{a[0]} -> {b[0]}': (b[1]-a[1])/1000
                 for a,b in zip(boundaries, boundaries[1:])}
        assert min(spans.values()) >= 0, (path, step, spans)
        assert abs(sum(spans.values())-window['dur']/1000) < 1e-6
        rows.append(dict(iteration=step, range_ms=window['dur']/1000, operations=metrics,
                         temporal_spans_ms=spans,
                         last_expert_ms=(expert_end-window['ts'])/1000,
                         after_expert_ms=(window['ts']+window['dur']-expert_end)/1000,
                         optimizer_start_ms=(optimizer['ts']-window['ts'])/1000,
                         after_optimizer_ms=(window['ts']+window['dur']-optimizer['ts']-optimizer['dur'])/1000))
    return dict(path=str(path), iterations=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    root = ROOT/args.job
    output = root/'cuda_event_analysis'
    output.mkdir(exist_ok=True)
    result = {}
    for mbs in (1, 2):
        result[mbs] = {'timing': {}, 'ranks': {}}
        for case, size in [('healthy', 16), ('nep', 14)]:
            run = root/f'mbs{mbs}/cupti/ep8/scaling_ladder__a770m_4b_moe_215b'/case/args.job
            log = run/'driver.log'
            assert 'backend=cuda_events' in log.read_text()
            assert 'GPU_TRACE_CUPTI_VERSION' not in log.read_text()
            control = root/f'control_mbs{mbs}/cupti/ep8/scaling_ladder__a770m_4b_moe_215b'/case/args.job/'driver.log'
            assert 'mode=control_no_cupti' in control.read_text()
            assert 'GPU_TRACE_CUPTI_VERSION' not in control.read_text()
            result[mbs]['timing'][case] = {k: timing(log, a, b) for k, a, b in
                                           [('pre',29,39), ('recorded',40,50), ('post',51,61)]}
            result[mbs]['timing'][case]['fresh_control'] = timing(control,40,50)
            for rank in range(size):
                data = analyze(run/f'cupti_profile/rank-{rank}.json',40,50,case)
                result[mbs]['ranks'][f'{case}_{rank}'] = data
                print(mbs,case,rank,'validated',flush=True)
        result[mbs]['parity'] = {w: result[mbs]['timing']['healthy'][w]['mean_ms'] /
                                result[mbs]['timing']['nep'][w]['mean_ms']
                                for w in ('pre','recorded','post','fresh_control')}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{'timing':v['timing'],'parity':v['parity']} for k,v in result.items()},indent=2))


if __name__ == '__main__':
    main()
