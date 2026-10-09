"""Disjoint phase accounting and graph-scope attribution, not kernel service time."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(path, steps, num_microbatches=8):
    data = json.loads(path.read_text())
    grouped = defaultdict(list)
    iterations = {}
    for event in data['traceEvents']:
        if event['cat'] == 'iteration':
            iterations[int(event['name'].split('_')[1])] = event
        else:
            grouped[event['args']['iteration']].append(event)
    rows = []
    for step in steps:
        events = grouped[step]
        gpu = [e for e in events if e['cat'] == 'gpu_operation_range']
        cpu = [e for e in events if e['cat'] == 'cpu_operation_range']
        named = lambda items, name: sorted([e for e in items if e['name'] == name], key=lambda e: e['ts'])
        forwards = named(gpu, 'microbatch.forward')
        backwards = named(gpu, 'microbatch.backward')
        cf = named(cpu, 'microbatch.forward')
        cb = named(cpu, 'microbatch.backward')
        assert len(forwards) == len(backwards) == len(cf) == len(cb) == num_microbatches
        end = lambda e: e['ts'] + e['dur']
        opt, = named(gpu, 'optimizer.step')
        it = iterations[step]
        # Preserve GPU+CPU pairs from the export, rather than joining by timestamp:
        # the host may already be several GPU microbatches ahead.
        pairs = list(zip(gpu, cpu))
        assert all(g['name'] == c['name'] for g, c in pairs)
        row = dict(iteration=step, iteration_ms=it['dur']/1000,
                   first_forward_ms=forwards[0]['dur']/1000,
                   other_forward_ms=sum(e['dur'] for e in forwards[1:])/1000,
                   nonfinal_backward_ms=sum(e['dur'] for e in backwards[:-1])/1000,
                   final_backward_ms=backwards[-1]['dur']/1000,
                   before_first_forward_ms=(forwards[0]['ts']-it['ts'])/1000,
                   backward_to_optimizer_ms=(opt['ts']-end(backwards[-1]))/1000,
                   optimizer_ms=opt['dur']/1000,
                   after_optimizer_ms=(end(it)-end(opt))/1000)
        row['inter_microbatch_gaps_ms'] = (
            sum(b['ts']-end(f) for f,b in zip(forwards,backwards))
            + sum(f['ts']-end(b) for f,b in zip(forwards[1:],backwards[:-1])))/1000
        row['outside_fb_ms'] = sum(row[k] for k in (
            'before_first_forward_ms','backward_to_optimizer_ms','optimizer_ms',
            'after_optimizer_ms','inter_microbatch_gaps_ms'))
        assert abs(row['iteration_ms'] - sum(row[k] for k in (
            'first_forward_ms','other_forward_ms','nonfinal_backward_ms',
            'final_backward_ms','outside_fb_ms'))) < 0.001
        row['graphs'] = {}
        for label, parent in [('first_forward',cf[0]),('last_backward',cb[-1])]:
            enclosed = [g for g,c in pairs if c['name'].startswith('graph.')
                        and parent['ts'] <= c['ts'] < end(parent)]
            row['graphs'][label] = [dict(name=e['name'], ms=e['dur']/1000,
                                       start_ms=(e['ts']-(forwards[0] if label=='first_forward' else backwards[-1])['ts'])/1000)
                                    for e in enclosed]
        row['extra_scopes_ms'] = {
            name:sum(e['dur'] for e in gpu if e['name']==name)/1000
            for name in sorted({e['name'] for e in gpu if e['name'].startswith(('finalize.','ddp.','nep.scale','optimizer.'))})}
        row['clock_error_us'] = data['clock_error_bound_ns']/1000
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--healthy-job')
    args = parser.parse_args()
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline')/args.job
    result = {}
    for case, size in [('healthy',32),('nep',28)]:
        tag = f'graph_attribution_{case}' if args.diagnostic else f'graph_moe_{case}'
        mode = 'nsys' if args.diagnostic else 'cupti'
        job = args.healthy_job if case == 'healthy' and args.healthy_job else args.job
        case_root = root.parent/job
        run = case_root/tag/mode/'ep16/a8b_120b_latentmoe_1t'/case/job/'cupti_profile'
        steps = range(25,28) if args.diagnostic else range(25,35)
        ranks = {str(r):analyze(run/f'rank-{r}.json',steps) for r in range(size)}
        roles = {}
        for role, members in [('owners',range(12)),('followers',range(12,16)),('second',range(16,size))]:
            rows = [row for r in members for row in ranks[str(r)]]
            roles[role] = {key:statistics.mean([row[key] for row in rows]) for key in rows[0]
                           if isinstance(rows[0][key],(float,int))}
            roles[role]['extra_scopes_ms'] = {key:statistics.mean(row['extra_scopes_ms'][key] for row in rows)
                                            for key in rows[0]['extra_scopes_ms']}
        result[case] = dict(roles=roles,ranks=ranks)
        print(case,json.dumps(roles,indent=2))
    out = root/'graph_residual_attribution.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
