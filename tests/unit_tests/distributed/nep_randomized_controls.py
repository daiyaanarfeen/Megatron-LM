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
            rows = balanced_routes(tokens, experts, topk, identity_seed(f'route:{rank}', seed))
            cache[key] = torch.tensor(rows, device=logits.device, dtype=torch.long)
            runtime = common.get_nonuniform_ep_runtime_config()
            assert runtime is not None
            manifest = route_summary(rows, runtime['expert_placement'])
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
    def synchronize(self):
        runtime = self._nonuniform_ep_runtime_config
        assert (
            self.config.expert_tensor_parallel_size == 1
        ), 'Identity control currently requires ETP1'
        pattern = self.nonuniform_ep_config.expert_name_pattern
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
        original(self)
        observed = {name: digest(param) for name, param in names.items()}
        assert expected == observed, 'Startup synchronization changed canonical parameter values'
        groups = self.expert_parallel_bucket_groups
        manifest = dict(
            rank=rank,
            placement=runtime['expert_placement'],
            parameters={
                name: dict(sha256=observed[name], shape=list(param.shape), dtype=str(param.dtype))
                for name, param in names.items()
            },
            expert_buckets=[
                dict(slots=group._nep_nccl_slot_keys, numels=group._nep_nccl_slot_numels)
                for group in groups
            ],
            dense_buckets=[
                dict(
                    numel=bucket.grad_data.numel(),
                    names=[self._param_to_name[param] for param in bucket.params_list],
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
