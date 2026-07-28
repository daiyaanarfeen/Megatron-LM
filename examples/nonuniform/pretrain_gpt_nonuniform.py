# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""GPT pretraining entrypoint for opt-in nonuniform EP training.

This script intentionally keeps the generic Megatron training loop untouched.  It imports the
standard GPT pretraining providers, patches the DDP class used by ``megatron.training.training``
to the opt-in nonuniform EP wrapper, and rejects distributed optimizer for nonuniform EP runs
benchmark modes.  With the non-distributed optimizer, synced gradients are present on the ranks
that own the local parameters before the normal optimizer step runs.
"""

import json
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pretrain_gpt as gpt
from megatron.core import parallel_state
from megatron.core.distributed.nonuniform_common import (
    get_global_rank,
    get_nonuniform_ep_runtime_config,
    set_nonuniform_ep_runtime_config,
)
from megatron.core.distributed.nonuniform_ep import (
    NonuniformEPConfig,
    NonuniformEPDistributedDataParallel,
    initialize_nonuniform_ep_process_groups,
)
import megatron.training.training as training_module


_EP_CONFIG_CACHE = {}
_ORIGINAL_INITIALIZE_MODEL_PARALLEL = parallel_state.initialize_model_parallel
_ORIGINAL_DDP = training_module.DDP


def _load_json_arg(value: Optional[str], path: Optional[str], default=None):
    if value is not None:
        return json.loads(value)
    if path is not None:
        with Path(path).open() as stream:
            return json.load(stream)
    return default










def _build_ep_runtime_config(args):
    registered_runtime_config = get_nonuniform_ep_runtime_config()
    placement = _load_json_arg(
        args.nonuniform_ep_placement_json,
        args.nonuniform_ep_placement_path,
        None,
    )
    if placement is None:
        return dict(registered_runtime_config) if registered_runtime_config is not None else None

    ep_group = parallel_state.get_expert_model_parallel_group()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    local_ep_size = parallel_state.get_expert_model_parallel_world_size()
    if len(placement) != local_ep_size:
        raise ValueError(
            f"NEP placement must have one expert list per EP rank: "
            f"got {len(placement)}, expected {local_ep_size}"
        )
    min_ep_size = args.nonuniform_ep_min_size or local_ep_size
    if min_ep_size <= 0 or min_ep_size > local_ep_size:
        raise ValueError(
            f"--nonuniform-ep-min-size must be in [1, {local_ep_size}], got {min_ep_size}"
        )

    owner_global_ranks = [get_global_rank(ep_group, rank) for rank in range(min_ep_size)]
    edp_group = dist.new_group(ranks=owner_global_ranks)
    local_expert_indices = [int(expert) for expert in placement[ep_rank]]
    return {
        'needs_reshard': local_ep_size != min_ep_size,
        'local_ep_size': local_ep_size,
        'min_ep_size': min_ep_size,
        'num_replicas': max(1, parallel_state.get_data_parallel_world_size()),
        'dp_size': max(
            1, parallel_state.get_data_parallel_world_size(with_context_parallel=True)
        ),
        'ep_group': ep_group,
        'edp_group': edp_group,
        'ep_rank': ep_rank,
        'local_expert_indices': local_expert_indices,
        'expert_placement': [[int(expert) for expert in experts] for experts in placement],
    }


def _create_gloo_process_groups_arg(args):
    return getattr(
        args,
        "enable_gloo_process_groups",
        getattr(args, "use_gloo_process_groups", True),
    )


def _initialize_model_parallel(*args, **kwargs):
    """Initialize the nonuniform EP topology before model construction."""
    megatron_args = gpt.get_args()
    nep_topology = megatron_args.nonuniform_ep_num_tp_cp_per_replica


    if megatron_args.nonuniform_mode != "ep" or nep_topology is None:
        return _ORIGINAL_INITIALIZE_MODEL_PARALLEL(*args, **kwargs)

    if megatron_args.num_experts is None:
        raise RuntimeError("num_experts is required for nonuniform EP topology mode")
    if megatron_args.pipeline_model_parallel_size != 1:
        raise RuntimeError("Nonuniform EP topology mode currently supports PP=1 only")
    if megatron_args.virtual_pipeline_model_parallel_size is not None:
        raise RuntimeError("Nonuniform EP topology mode does not support virtual PP")
    if megatron_args.use_torch_fsdp2:
        raise RuntimeError("--use-torch-fsdp2 is not supported with nonuniform EP")
    if megatron_args.num_distributed_optimizer_instances != 1:
        raise RuntimeError("Nonuniform EP topology mode does not support partial DistOpt")

    etp = megatron_args.expert_tensor_parallel_size or megatron_args.tensor_model_parallel_size
    tp_cp = megatron_args.tensor_model_parallel_size * megatron_args.context_parallel_size
    computed_min_ep_size = min(nep_topology) * tp_cp // etp
    if (
        megatron_args.nonuniform_ep_min_size is not None
        and megatron_args.nonuniform_ep_min_size != computed_min_ep_size
    ):
        raise RuntimeError(
            "--nonuniform-ep-min-size must match the topology-derived min EP size "
            f"({computed_min_ep_size}) when --nonuniform-ep-num-tp-cp-per-replica is set"
        )
    for num_tp_cp in nep_topology:
        if num_tp_cp * tp_cp % etp != 0:
            raise RuntimeError(
                "Each nonuniform EP replica must produce an integer EP size: "
                f"num_tp_cp={num_tp_cp}, TP*CP={tp_cp}, ETP={etp}"
            )
        ep_size = num_tp_cp * tp_cp // etp
        if megatron_args.num_experts % ep_size != 0:
            raise RuntimeError(
                f"num_experts ({megatron_args.num_experts}) must be divisible by "
                f"local EP size {ep_size}"
            )

    runtime_config = initialize_nonuniform_ep_process_groups(
        tensor_model_parallel_size=megatron_args.tensor_model_parallel_size,
        context_parallel_size=megatron_args.context_parallel_size,
        num_tp_cp_per_replica=nep_topology,
        expert_tensor_parallel_size=megatron_args.expert_tensor_parallel_size,
        num_moe_experts=megatron_args.num_experts,
        nccl_communicator_config_path=megatron_args.nccl_communicator_config_path,
        distributed_timeout_minutes=megatron_args.distributed_timeout_minutes,
        create_gloo_process_groups=_create_gloo_process_groups_arg(megatron_args),
        get_embedding_ranks=kwargs.get("get_embedding_ranks"),
        get_position_embedding_ranks=kwargs.get("get_position_embedding_ranks"),
    )
    megatron_args.expert_model_parallel_size = runtime_config['local_ep_size']
    return None


def _build_ep_config(args) -> NonuniformEPConfig:
    expert_owner = _load_json_arg(
        args.nonuniform_ep_expert_owner_json,
        args.nonuniform_ep_expert_owner_path,
        None,
    )
    if expert_owner is not None:
        expert_owner = {int(expert): int(owner) for expert, owner in expert_owner.items()}

    kwargs = {
        'approach': args.nonuniform_ep_ddp_approach,
        'runtime_config': _build_ep_runtime_config(args),
        'expert_owner': expert_owner,
        'require_owner_local_expert': True,
    }
    if args.nonuniform_ep_expert_name_pattern is not None:
        kwargs['expert_name_pattern'] = args.nonuniform_ep_expert_name_pattern
    return NonuniformEPConfig(**kwargs)


def _get_ep_config(args) -> NonuniformEPConfig:
    if 'config' not in _EP_CONFIG_CACHE:
        _EP_CONFIG_CACHE['config'] = _build_ep_config(args)
        set_nonuniform_ep_runtime_config(_EP_CONFIG_CACHE['config'].runtime_config)
    return _EP_CONFIG_CACHE['config']


def _ep_model_provider(builder, args, *provider_args, **kwargs):
    _get_ep_config(args)
    return gpt.model_provider(builder, *provider_args, **kwargs)


def _install_opt_in_ddp(args):
    training_module.DDP = _ORIGINAL_DDP
    if args.nonuniform_mode == "none":
        set_nonuniform_ep_runtime_config(None)
        parallel_state.initialize_model_parallel = _ORIGINAL_INITIALIZE_MODEL_PARALLEL
        return
    if args.use_distributed_optimizer:
        raise RuntimeError(
            "Nonuniform EP requires the non-distributed optimizer. "
            "Remove --use-distributed-optimizer."
        )

    class NonuniformEPDDP(NonuniformEPDistributedDataParallel):
        def __init__(self, *ddp_args, **kwargs):
            super().__init__(
                *ddp_args,
                nonuniform_ep_config=_get_ep_config(args),
                **kwargs,
            )

    training_module.DDP = NonuniformEPDDP
    parallel_state.initialize_model_parallel = _initialize_model_parallel


def _add_nonuniform_args(parser):
    if gpt.has_nvidia_modelopt:
        maybe_parser = gpt.add_modelopt_args(parser)
        if maybe_parser is not None:
            parser = maybe_parser

    group = parser.add_argument_group(title="nonuniform EP")
    group.add_argument(
        "--nonuniform-mode",
        choices=["none", "ep"],
        default="none",
        help="Opt into nonuniform EP training.",
    )
    group.add_argument("--nonuniform-ep-min-size", type=int, default=None)
    group.add_argument(
        "--nonuniform-ep-num-tp-cp-per-replica", nargs="+", type=int, default=None
    )
    group.add_argument("--nonuniform-ep-placement-json", default=None)
    group.add_argument("--nonuniform-ep-placement-path", default=None)
    group.add_argument("--nonuniform-ep-expert-owner-json", default=None)
    group.add_argument("--nonuniform-ep-expert-owner-path", default=None)
    group.add_argument("--nonuniform-ep-expert-name-pattern", default=None)
    group.add_argument(
        "--nonuniform-ep-ddp-approach",
        choices=["nccl"],
        default="nccl",
        help="NCCL nonuniform EP gradient synchronization.",
    )
    return parser


if __name__ == "__main__":
    _MAIN_ENTRY_TIME = gpt.time.time()
    gpt.set_startup_timestamps(
        program_start=gpt._PROGRAM_START_TIME,
        main_entry=_MAIN_ENTRY_TIME,
    )

    gpt.train_valid_test_datasets_provider.is_distributed = True
    pretrain, store = gpt.inprocess_restart.maybe_wrap_for_inprocess_restart(gpt.pretrain)

    args = gpt.parse_and_validate_args(
        extra_args_provider=_add_nonuniform_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    full_config = gpt.pretrain_cfg_container_from_args(args)
    _install_opt_in_ddp(args)
    if args.nonuniform_mode == "ep":
        model_provider = partial(_ep_model_provider, gpt.gpt_builder, args)
    else:
        model_provider = partial(gpt.model_provider, gpt.gpt_builder)

    pretrain(
        full_config,
        gpt.train_valid_test_datasets_provider,
        model_provider,
        gpt.ModelType.encoder_or_decoder,
        gpt.forward_step,
        store=store,
        get_embedding_ranks=gpt.get_embedding_ranks,
    )
