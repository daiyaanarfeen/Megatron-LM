"""Startup-only bucket controls for the expert-count ablation; no runtime changes."""

import json
import os


def install_bucket_control():
    """Give native expert buffers five equal layer groups using the native layout code."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    original = DistributedOptimizer._compute_per_buffer_param_layout

    def layout(params, bucket_size, data_parallel_world_size, ddp_config, param_indices=None):
        expert = [not getattr(param, 'allreduce', True) for param in params]
        assert all(expert) or not any(expert), 'Mixed dense/expert buffer'
        # NEP's already-partitioned owner buffers request None and must remain intact.
        if all(expert) and bucket_size is not None:
            total = sum(param.numel() for param in params)
            assert total % 5 == 0 and all(param.numel() % 128 == 0 for param in params)
            bucket_size = total // 5
        return original(params, bucket_size, data_parallel_world_size, ddp_config, param_indices)

    DistributedOptimizer._compute_per_buffer_param_layout = staticmethod(layout)


def validate_bucket_control(ddp, param_names, root, rank):
    """Fail before training if expert grouping differs from three MoE layers per bucket."""
    groups = ddp.expert_parallel_bucket_groups
    signatures = []
    for group in groups:
        modules = sorted(
            {
                param_names[param].split('.experts.')[0]
                for bucket in group.buckets
                for param in bucket.params_list
            }
        )
        assert len(modules) == 3, (rank, modules)
        signatures.append(modules)
    assert len(signatures) == 5 and len({m for s in signatures for m in s}) == 15
    dense = [
        sorted(param_names[p] for bucket in group.buckets for p in bucket.params_list)
        for group in ddp.bucket_groups
    ]
    record = dict(expert_modules=signatures, dense_names=dense)
    for previous in root.parent.parent.glob('controlled_actual_*/controls/bucket-control-*.json'):
        if previous.parent == root:
            continue
        old = json.loads(previous.read_text())
        assert record == old, (rank, previous, 'Layer-to-bucket grouping changed')
    (root / f'bucket-control-{rank}.json').write_text(json.dumps(record, indent=2) + '\n')
    print('EXPERT_COUNT_BUCKET_CONTROL_PASS', rank, os.environ['ABLATION_EXPERTS'], flush=True)
