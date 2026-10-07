"""Opt-in benchmark hooks; native dispatcher and optimizer are unmodified."""

import hashlib
import json
import os
from pathlib import Path

from nep_balanced_routing import balanced_routes, identity_seed, route_summary


def install() -> None:
    """Install startup identity checks and cached randomized exact routing."""
    import torch

    from megatron.core.distributed import nonuniform_common as common
    from megatron.core.distributed import nonuniform_ep as nep

    root = Path(os.environ['GRAPH_OUTPUT']) / 'controls'
    root.mkdir(exist_ok=True)
    rank = int(os.environ['RANK'])
    seed = int(os.environ['NEP_CONTROLLED_ROUTING_SEED'])
    cache = {}

    def forward(ctx, logits, topk):
        experts = logits.shape[-1]
        tokens = logits.numel() // experts
        key = (tokens, experts, topk, logits.device)
        if key not in cache:
            assert (
                not torch.cuda.is_current_stream_capturing()
            ), 'Routing cache missed during capture'
            unit = int(os.environ.get('NEP_CONTROLLED_ROUTE_UNIT', tokens))
            assert tokens % unit == 0
            rows = balanced_routes(unit, experts, topk, identity_seed(f'route:{rank}', seed))
            rows = rows * (tokens // unit)
            cache[key] = torch.tensor(rows, device=logits.device, dtype=torch.long)
            runtime = common.get_nonuniform_ep_runtime_config()
            if runtime is None:
                from megatron.core import parallel_state

                ep = parallel_state.get_expert_model_parallel_world_size()
                assert experts % ep == 0
                placement = [
                    list(range(i * experts // ep, (i + 1) * experts // ep)) for i in range(ep)
                ]
            else:
                placement = runtime['expert_placement']
            manifest = route_summary(rows, placement)
            manifest['unit_route_sha256'] = route_summary(rows[:unit], placement)['route_sha256']
            manifest.update(tokens=tokens, experts=experts, topk=topk, seed=seed, rank=rank)
            (root / f'routing-{rank}-{tokens}.json').write_text(
                json.dumps(manifest, indent=2) + '\n'
            )
        result = torch.full_like(logits, -1.0)
        result.view(tokens, experts).scatter_(1, cache[key], 1.0)
        return result

    common.ExactUniformRoutingSTE.forward = staticmethod(forward)
    original = nep.NonuniformEPDistributedDataParallel._synchronize_nondivisible_expert_parameters

    def digest(param):
        return hashlib.sha256(
            param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()

    @torch.no_grad()
    def synchronize(self, healthy=False):
        if healthy:
            from megatron.core import parallel_state

            ep = parallel_state.get_expert_model_parallel_world_size()
            local_rank = parallel_state.get_expert_model_parallel_rank()
            count = self.config.num_moe_experts
            assert count % ep == 0
            placement = [list(range(i * count // ep, (i + 1) * count // ep)) for i in range(ep)]
            runtime = dict(local_expert_indices=placement[local_rank], expert_placement=placement)
            pattern = nep.NonuniformEPConfig().expert_name_pattern
        else:
            runtime = self._nonuniform_ep_runtime_config
            pattern = self.nonuniform_ep_config.expert_name_pattern
        assert (
            self.config.expert_tensor_parallel_size == 1
        ), 'Identity control currently requires ETP1'
        names = {}
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for name, param in self.module.named_parameters():
                expert = nep._local_expert_id_from_name(
                    name, pattern, runtime['local_expert_indices']
                )
                if expert is None:
                    names[name] = param
                    continue
                identity = nep._expert_slot_key_from_name(name, pattern).format(expert=expert)
                if '.linear_fc1.' in name:
                    initialize = self.config.init_method
                elif '.linear_fc2.' in name:
                    initialize = self.config.output_layer_init_method
                else:
                    raise RuntimeError(f'Unsupported expert initializer: {name}')
                assert 'weight' in name and initialize is not None, name
                torch.cuda.manual_seed(identity_seed(identity))
                initialize(param)
                names[identity] = param
        expected = {name: digest(param) for name, param in names.items()}
        if not healthy:
            original(self)
        observed = {name: digest(param) for name, param in names.items()}
        assert expected == observed, 'Startup synchronization changed canonical parameter values'
        groups = self.expert_parallel_bucket_groups
        param_names = {param: name for name, param in self.module.named_parameters()}
        if os.environ.get('NEP_EXPERT_COUNT_STUDY') == '1':
            from nep_expert_count_controls import validate_bucket_control

            validate_bucket_control(self, param_names, root, rank)
        manifest = dict(
            rank=rank,
            placement=runtime['expert_placement'],
            parameters={
                name: dict(sha256=observed[name], shape=list(param.shape), dtype=str(param.dtype))
                for name, param in names.items()
            },
            expert_buckets=(
                [
                    dict(slots=group._nep_nccl_slot_keys, numels=group._nep_nccl_slot_numels)
                    for group in groups
                ]
                if not healthy
                else [
                    dict(numel=bucket.grad_data.numel(), parameters=len(bucket.params_list))
                    for group in groups
                    for bucket in group.buckets
                ]
            ),
            delivery_buckets=(
                [
                    dict(
                        edp_bucket=group._nep_nccl_edp_bucket_index,
                        names=[
                            param_names[param]
                            for bucket in group.buckets
                            for param in bucket.params_list
                        ],
                    )
                    for group in groups
                ]
                if not healthy
                else []
            ),
            dense_buckets=[
                dict(
                    numel=bucket.grad_data.numel(),
                    names=[param_names[param] for param in bucket.params_list],
                )
                for group in self.bucket_groups
                for bucket in group.buckets
            ],
        )
        (root / f'initialization-{rank}.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if os.environ.get('NEP_CONTROLLED_REPLAY') == '1':
            from nep_all_ready_replay import run

            run(self)

    nep.NonuniformEPDistributedDataParallel._synchronize_nondivisible_expert_parameters = (
        synchronize
    )
    if os.environ.get('NEP_WORK_ALLOCATION') == '1':
        from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel

        native_init = DistributedDataParallel.__init__

        def initialize_healthy(self, *args, **kwargs):
            native_init(self, *args, **kwargs)
            if type(self) is DistributedDataParallel:
                synchronize(self, healthy=True)

        DistributedDataParallel.__init__ = initialize_healthy
