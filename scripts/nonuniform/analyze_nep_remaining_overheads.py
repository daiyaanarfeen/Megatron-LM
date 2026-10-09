"""Matched phase accounting and fixed-schedule drain sensitivity, not causal replay."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3215521')
OUT = ROOT / 'remaining_overheads_20261006'
EXPERT = ('ep_dp', 'EXPERT_DATA_PARALLEL_GROUP')
DENSE = ('dp_cp', 'DATA_PARALLEL_GROUP_WITH_CP')


def span(event: dict) -> tuple[float, float]:
    """Return recorded GPU event endpoints in milliseconds."""
    return event['ts'] / 1000, (event['ts'] + event['dur']) / 1000


def analyze(path: Path, rank: int) -> list[dict]:
    """Account for one rank without double counting nested optimizer scopes."""
    trace = json.loads((path / f'cupti_profile/rank-{rank}.json').read_text())
    events = trace['traceEvents']
    rows = []
    for step in range(25, 31):
        gpu = [
            e
            for e in events
            if e.get('args', {}).get('iteration') == step and e['cat'] == 'gpu_operation_range'
        ]
        cpu = [
            e
            for e in events
            if e.get('args', {}).get('iteration') == step and e['cat'] == 'cpu_operation_range'
        ]
        assert [e['name'] for e in gpu] == [e['name'] for e in cpu]
        pairs = list(zip(gpu, cpu))

        def named(name: str) -> list[dict]:
            return [e for e in gpu if e['name'] == name]

        def comm(name: str, desc: tuple) -> list[dict]:
            return [e for e in named(name) if e['args']['collective']['group_desc'] in desc]

        def one(name: str) -> dict:
            candidates = named(name)
            assert len(candidates) == 1, (path, rank, step, name)
            return candidates[0]

        fw, bw = named('microbatch.forward'), named('microbatch.backward')
        assert len(fw) == len(bw) == 8
        first, last = fw[0], bw[-1]
        sync, opt = one('ddp.finish_grad_sync'), one('optimizer.step')
        scale = next(e for e in gpu if e['name'] in ('ddp.scale_gradients', 'nep.scale_gradients'))
        iteration = next(
            e for e in events if e.get('cat') == 'iteration' and e['name'] == f'iteration_{step}'
        )
        row = dict(rank=rank, iteration=step, iteration_ms=iteration['dur'] / 1000)
        row.update(
            first_forward_ms=first['dur'] / 1000,
            final_backward_ms=last['dur'] / 1000,
            other_forward_ms=sum(e['dur'] / 1000 for e in fw[1:]),
            nonfinal_backward_ms=sum(e['dur'] / 1000 for e in bw[:-1]),
        )
        row['outside_fb_ms'] = row['iteration_ms'] - sum(e['dur'] / 1000 for e in fw + bw)
        for key, e in [
            ('finish_grad_sync_ms', sync),
            ('scale_gradients_ms', scale),
            ('optimizer_ms', opt),
        ]:
            row[key] = e['dur'] / 1000
        row['sync_to_scale_gap_ms'] = span(scale)[0] - span(sync)[1]
        gap_start, gap_end = span(sync)[1], span(scale)[0]
        for label, desc in [
            ('tp_gradient', ('tp', 'TENSOR_MODEL_PARALLEL_GROUP')),
            ('token_count', DENSE),
        ]:
            selected = [
                e for e in comm('collective.all_reduce', desc) if gap_start <= span(e)[0] < gap_end
            ]
            if label == 'token_count':
                assert all(
                    op['input_bytes'] == 4 and op['dtype'] == 'torch.int32'
                    for e in selected
                    for op in e['args']['collective']['operations']
                )
            row['syncgap_' + label + '_allreduce_ms'] = sum(
                min(gap_end, span(e)[1]) - span(e)[0] for e in selected
            )
        row['scale_to_optimizer_gap_ms'] = span(opt)[0] - span(scale)[1]
        accounted = sum(
            row[k]
            for k in (
                'finish_grad_sync_ms',
                'scale_gradients_ms',
                'optimizer_ms',
                'sync_to_scale_gap_ms',
                'scale_to_optimizer_gap_ms',
            )
        )
        row['outside_other_ms'] = row['outside_fb_ms'] - accounted
        prepare, norm, adam = (
            one(n)
            for n in ('optimizer.prepare_grads', 'optimizer.grad_norm', 'optimizer.adam_step')
        )
        row['optimizer_prepare_ms'] = prepare['dur'] / 1000
        row['optimizer_norm_ms'] = norm['dur'] / 1000
        row['optimizer_adam_including_param_copy_ms'] = adam['dur'] / 1000
        row['optimizer_other_ms'] = row['optimizer_ms'] - sum(
            e['dur'] / 1000 for e in (prepare, norm, adam)
        )
        row['optimizer_param_copy_nested_ms'] = sum(
            e['dur'] / 1000 for e in named('optimizer.copy_params')
        )
        phase_graphs = {}
        for label, phase in [('first_forward', first), ('final_backward', last)]:
            pc = next(c for g, c in pairs if g is phase)
            graphs = [
                g
                for g, c in pairs
                if g['name'].startswith('graph.') and pc['ts'] <= c['ts'] < pc['ts'] + pc['dur']
            ]
            phase_graphs[label] = graphs
            assert len(graphs) == 32, (path, rank, step, label, len(graphs))
            row[label + '_graphs_ms'] = sum(g['dur'] / 1000 for g in graphs)
            row[label + '_outside_graphs_ms'] = phase['dur'] / 1000 - row[label + '_graphs_ms']
            for kind in ('MoETransformerLayer', 'MambaLayer', 'TransformerLayer'):
                row[label + '_' + kind + '_ms'] = sum(
                    g['dur'] / 1000 for g in graphs if g['name'].split('.')[1] == kind
                )
        ag, rs = comm('collective.all_gather', EXPERT), comm('collective.reduce_scatter', EXPERT)
        dense = comm('collective.reduce_scatter', DENSE)
        for label, seq in [('expert_ag', ag), ('expert_rs', rs)]:
            row[label + '_count'] = len(seq)
            row[label + '_sum_ranges_ms'] = sum(e['dur'] / 1000 for e in seq)
            row[label + '_input_MB'] = sum(
                op['input_bytes'] / 1e6 for e in seq for op in e['args']['collective']['operations']
            )
        graphs = phase_graphs['first_forward']
        first_moe = next(g for g in graphs if g['name'] == 'graph.MoETransformerLayer.forward')
        previous_end = span(graphs[graphs.index(first_moe) - 1])[1]
        row['first_ag_after_preceding_graph_ms'] = max(0, span(ag[0])[1] - previous_end)
        row['first_ag_input_MB'] = sum(
            op['input_bytes'] / 1e6 for op in ag[0]['args']['collective']['operations']
        )
        # Keep backward and non-expert sync fixed. This counts drain only, not
        # interference or any graph waits that have already occurred.
        b_end = span(last)[1]
        graph_end = max(span(g)[1] for g in phase_graphs['final_backward'])
        e_end, d_end = max(span(e)[1] for e in rs), max(span(e)[1] for e in dense)
        row['expert_rs_pending_after_backward_ms'] = max(0, e_end - b_end)
        row['dense_rs_pending_after_backward_ms'] = max(0, d_end - b_end)
        row['zero_expert_rs_drain_saving_ms'] = max(b_end, e_end, d_end) - max(b_end, d_end)
        row['zero_dense_rs_drain_saving_ms'] = max(b_end, e_end, d_end) - max(b_end, e_end)
        row['zero_all_rs_drain_saving_ms'] = max(b_end, e_end, d_end) - b_end
        row['expert_rs_after_last_graph_ms'] = max(0, e_end - graph_end)
        gather = comm('collective.all_to_all', ('nep_owner_gather',))
        stage = named('nep.stage_grads')
        for label, seq in [('gather', gather), ('stage', stage)]:
            row[label + '_sum_ranges_ms'] = sum(e['dur'] / 1000 for e in seq)
            row[label + '_pending_after_backward_ms'] = max(
                0, max((span(e)[1] for e in seq), default=b_end) - b_end
            )
        rows.append(row)
    return rows


def main() -> None:
    """Compare the same owner cohort and steps in both repeats."""
    OUT.mkdir(exist_ok=True)
    result = dict(
        cases={},
        deltas=[],
        notes=[
            'GPU operation ranges include queueing and waiting, not pure kernel service.',
            'Native expert buckets are 5 healthy versus 3 NEP; individual buckets are not matched.',
            'Drain deletion holds graph timing fixed; it cannot estimate contention removal.',
            'Nested optimizer.copy_params is already included in optimizer.adam_step.',
        ],
    )
    for repeat in range(2):
        cases = {}
        for cell, topology in [('H', 'healthy'), ('A', 'nep')]:
            path = ROOT / f'controlled_actual_{cell}_r{repeat}_{topology}_moe'
            rows = [row for rank in range(12) for row in analyze(path, rank)]
            means = {
                key: statistics.mean(row[key] for row in rows)
                for key in rows[0]
                if key not in ('rank', 'iteration')
            }
            cases[topology] = means
            result['cases'][f'{topology}_r{repeat}'] = dict(means=means, rows=rows)
        delta = {key: cases['nep'][key] - cases['healthy'][key] for key in cases['nep']}
        result['deltas'].append(delta)
    result['mean_delta'] = {
        key: statistics.mean(d[key] for d in result['deltas']) for key in result['deltas'][0]
    }
    for topology in ('healthy', 'nep'):
        result[topology] = {
            key: statistics.mean(
                result['cases'][f'{topology}_r{r}']['means'][key] for r in range(2)
            )
            for key in result['mean_delta']
        }
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for key, value in result['mean_delta'].items():
        print(key, round(result['healthy'][key], 4), round(result['nep'][key], 4), round(value, 4))


if __name__ == '__main__':
    main()
