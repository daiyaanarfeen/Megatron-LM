"""Real a8b workload arguments and fail-closed ablation controls; no runtime edits."""

import hashlib
import json
import os
import sys
from pathlib import Path

from validate_nep_randomized_controls import validate as compare_placement
from validate_nep_study import audit as audit_transfers
from validate_nep_work_allocation import validate as validate_work

TEMPLATE = Path(
    '/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3120733/graph_moe_nep/cupti/ep16/a8b_120b_latentmoe_1t/nep/3120733/launch_manifest/rank-0.json'
)


def parse_options(tokens):
    result = {}
    for token in tokens:
        if token.startswith('--'):
            assert token not in result, token
            key = token
            result[key] = []
        else:
            result[key].append(token)
    return result


def options():
    """Preserve the successful model configuration; change only study controls."""
    original = parse_options(json.loads(TEMPLATE.read_text())['options'])
    flags = {key: value.copy() for key, value in original.items()}
    root = Path(os.environ['GRAPH_OUTPUT'])
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    mbs, ga, reduced_ga = [
        int(os.environ[key]) for key in ('ABLATION_MBS', 'ABLATION_GA', 'ABLATION_REDUCED_GA')
    ]
    local_ga = ga if rank < 16 else reduced_ga
    seqlen = int(os.environ.get('ABLATION_SEQLEN', '8192'))
    assert seqlen in (2048, 4096, 8192, 32768)
    replacements = {
        '--micro-batch-size': [str(mbs)],
        '--global-batch-size': [str(world // 2 * mbs * local_ga)],
        '--train-iters': ['35'],
        '--lr-decay-iters': ['35'],
        '--profile-step-start': ['25'],
        '--profile-step-end': ['31'],
        '--profile-ranks': list(map(str, range(world))),
        '--tensorboard-dir': [str(root / 'tensorboard')],
        '--log-memory-interval': ['1'],
        '--distributed-timeout-minutes': ['3'],
        '--seq-length': [str(seqlen)],
    }
    flags.update(replacements)
    if seqlen == 32768:
        assert mbs == 2 and ga == reduced_ga == 2
        flags['--max-position-embeddings'] = [str(seqlen)]
    if os.environ['GRAPH_CASE'] == 'healthy':
        for key in (
            '--nonuniform-mode',
            '--nonuniform-ep-ddp-approach',
            '--nonuniform-ep-num-tp-cp-per-replica',
        ):
            del flags[key]
    assert flags['--hybrid-layer-pattern'] == ['MEMEMEM*EMEMEMEMEM*EMEMEMEME/*E/*E']
    assert flags['--vocab-size'] == ['131072']
    spec = dict(
        cell=os.environ['NEP_ACTUAL_CELL'],
        mbs=mbs,
        ga=ga,
        reduced_ga=reduced_ga,
        true_gbs=int(os.environ['TRUE_GLOBAL_BATCH_SIZE']),
        train_iters=35,
        profile_start=25,
        profile_end=31,
        native_rank_timing=True,
        delivery_markers=True,
        seqlen=seqlen,
        delivery=os.environ.get('NEP_ABLATION_PARAM_RELEASE', 'early'),
        prefetch_depth=int(os.environ.get('NEP_ABLATION_PREFETCH_DEPTH', '1')),
        placement=os.environ['NEP_ABLATION_PLACEMENT'],
        order=os.environ['NEP_ABLATION_PARAM_ORDER'],
        lanes=int(os.environ['NEP_ABLATION_PARAM_LANES']),
        template=str(TEMPLATE),
        template_sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
    )
    if rank == 0:
        (root / 'actual-spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    effective = [item for key, values in flags.items() for item in [key, *values]]
    changes = {
        key: dict(before=original.get(key), after=flags.get(key))
        for key in original.keys() | flags.keys()
        if original.get(key) != flags.get(key)
    }
    (root / f'actual-options-{rank}.json').write_text(
        json.dumps(dict(options=effective, changes=changes), indent=2) + '\n'
    )
    sys.stdout.buffer.write(('\0'.join(effective) + '\0').encode())


def validate(run):
    """Reuse the established memory/profile and exact placement/traffic gates."""
    spec = json.loads((run / 'actual-spec.json').read_text())
    validate_work(run, spec)
    healthy = spec['cell'].startswith('H')
    result = json.loads((run / 'work-validation.json').read_text())
    profiled = result['profiled']
    if not healthy:
        transfer = audit_transfers(run, 16, 12, 'training', profiled, 35, range(25, 31))
    else:
        transfer = None
    current = [
        json.loads((run / f'controls/initialization-{r}.json').read_text())
        for r in range(32 if healthy else 28)
    ]
    if spec.get('native_rank_timing'):
        for rank in range(len(current)):
            timings = json.loads((run / f'native-timing-{rank}.json').read_text())
            assert len(timings) == spec['train_iters'] and all(t > 0 for t in timings), (
                rank,
                timings,
            )
    if profiled and spec.get('delivery_markers'):
        for rank in range(len(current)):
            events = json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())[
                'traceEvents'
            ]
            for step in range(spec['profile_start'], spec['profile_end']):
                names = [
                    e['name']
                    for e in events
                    if e.get('cat') == 'gpu_operation_range' and e['args']['iteration'] == step
                ]
                demand = [
                    name.removeprefix('parameter.demand.')
                    for name in names
                    if name.startswith('parameter.demand.')
                ]
                ready = [
                    name.removeprefix('parameter.ready.')
                    for name in names
                    if name.startswith('parameter.ready.')
                ]
                assert demand and demand == ready, (rank, step, 'missing delivery deadlines')
    for previous in run.parent.glob('controlled_actual_*/actual-validation.json'):
        other = previous.parent
        old_spec = json.loads((other / 'actual-spec.json').read_text())
        old = json.loads(previous.read_text())
        old_healthy = old_spec['cell'].startswith('H')
        same_topology = healthy == old_healthy
        # Expert identity is global, not the physical rank or padded local slot.
        old_manifests = [
            json.loads(p.read_text()) for p in (other / 'controls').glob('initialization-*.json')
        ]
        old_experts = {
            name: value
            for m in old_manifests
            for name, value in m['parameters'].items()
            if '.experts.' in name
        }
        for r, now in enumerate(current):
            previous_rank = r if r < len(old_manifests) else r % 16
            before = json.loads(
                (other / f'controls/initialization-{previous_rank}.json').read_text()
            )
            for name, value in now['parameters'].items():
                if '.experts.' in name:
                    assert old_experts[name] == value, (r, name, 'weight drift')
                elif name in before['parameters']:
                    assert before['parameters'][name] == value, (r, name, 'dense weight drift')
            if same_topology:
                assert before['dense_buckets'] == now['dense_buckets']
                assert before['expert_buckets'] == now['expert_buckets']
            routes = lambda parent: [
                json.loads(p.read_text()) for p in (parent / 'controls').glob(f'routing-{r}-*.json')
            ]
            if r < len(old_manifests):
                assert {x['unit_route_sha256'] for x in routes(other)} == {
                    x['unit_route_sha256'] for x in routes(run)
                }
        if (
            not healthy
            and not old_healthy
            and old_spec['mbs'] == spec['mbs']
            and old_spec['seqlen'] == spec['seqlen']
        ):
            compare_placement(other, run, 16, 12)
        if profiled and old['profiled'] and same_topology:

            def fingerprint(records):
                result = {
                    key: sorted(
                        json.dumps(
                            {k: x[k] for k in ('group_desc', 'ranks', 'operations')}, sort_keys=True
                        )
                        for x in calls
                        if x['group_desc']
                        in (
                            'dp_cp',
                            'ep_dp',
                            'DATA_PARALLEL_GROUP_WITH_CP',
                            'EXPERT_DATA_PARALLEL_GROUP',
                        )
                    )
                    for key, calls in records.items()
                }
                assert result and all(result.values()), 'Missing native optimizer collectives'
                return result

            assert fingerprint(old['native_collectives']) == fingerprint(
                result['native_collectives']
            ), 'Native AG/RS payload drift'
            if not healthy:
                assert (
                    old['transfer']['bytes_per_step'] == transfer['bytes_per_step']
                ), 'Transfer volume drift'
    result['transfer'] = transfer
    (run / 'actual-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ACTUAL_ABLATION_PASS', run, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'options':
        options()
    else:
        assert sys.argv[1] == 'validate'
        validate(Path(sys.argv[2]))
