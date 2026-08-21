# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""GPT pretraining entrypoint for opt-in nonuniform expert parallelism."""

import json
import sys
from pathlib import Path
from typing import Optional

import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import megatron.training.models.dist_utils as model_dist_utils
import megatron.training.training as training_module
import pretrain_gpt as gpt
from megatron.core import parallel_state
from megatron.core.distributed.nonuniform_common import set_nonuniform_ep_runtime_config
from megatron.core.distributed.nonuniform_ep import (
    NonuniformEPConfig,
    NonuniformEPDistributedDataParallel,
    initialize_nonuniform_ep_process_groups_from_args,
)

_EP_CONFIG_CACHE = {}
_ORIGINAL_INITIALIZE_MODEL_PARALLEL = parallel_state.initialize_model_parallel
_ORIGINAL_TRAINING_DDP = training_module.DDP
_ORIGINAL_MODEL_BUILDER_DDP = model_dist_utils.DistributedDataParallel


def _load_json_arg(value: Optional[str], path: Optional[str], default=None):
    if value is not None:
        return json.loads(value)
    if path is not None:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    return default


def _build_ep_runtime_config(args):
    placement = _load_json_arg(
        args.nonuniform_ep_placement_json, args.nonuniform_ep_placement_path, None
    )
    if placement is None:
        return None

    ep_group = parallel_state.get_expert_model_parallel_group()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    local_ep_size = parallel_state.get_expert_model_parallel_world_size()
    if len(placement) != local_ep_size:
        raise ValueError(
            "NEP placement must have one expert list per EP rank: "
            f"got {len(placement)}, expected {local_ep_size}"
        )
    min_ep_size = args.nonuniform_ep_min_size or local_ep_size
    if min_ep_size <= 0 or min_ep_size > local_ep_size:
        raise ValueError(
            f"--nonuniform-ep-min-size must be in [1, {local_ep_size}], got {min_ep_size}"
        )

    owner_global_ranks = [dist.get_global_rank(ep_group, rank) for rank in range(min_ep_size)]
    edp_group = dist.new_group(ranks=owner_global_ranks)
    local_expert_indices = [int(expert) for expert in placement[ep_rank]]
    return {
        "local_ep_size": local_ep_size,
        "min_ep_size": min_ep_size,
        "num_replicas": max(1, parallel_state.get_data_parallel_world_size()),
        "dp_size": max(1, parallel_state.get_data_parallel_world_size(with_context_parallel=True)),
        "ep_group": ep_group,
        "edp_group": edp_group,
        "ep_rank": ep_rank,
        "local_expert_indices": local_expert_indices,
        "expert_placement": [[int(expert) for expert in experts] for experts in placement],
    }


def _build_ep_config(args) -> NonuniformEPConfig:
    kwargs = {"runtime_config": _build_ep_runtime_config(args)}
    if args.nonuniform_ep_expert_name_pattern is not None:
        kwargs["expert_name_pattern"] = args.nonuniform_ep_expert_name_pattern
    return NonuniformEPConfig(**kwargs)


def _get_ep_config(args) -> NonuniformEPConfig:
    if "config" not in _EP_CONFIG_CACHE:
        _EP_CONFIG_CACHE["config"] = _build_ep_config(args)
        if _EP_CONFIG_CACHE["config"].runtime_config is not None:
            set_nonuniform_ep_runtime_config(_EP_CONFIG_CACHE["config"].runtime_config)
    return _EP_CONFIG_CACHE["config"]


def _validate_nonuniform_ep_args(args) -> None:
    if args.gtp_weight_remat_size != 1 or args.expert_gtp_weight_remat_size != 1:
        raise RuntimeError("Nonuniform EP does not support GTP/EGTP rematerialization")
    if args.use_megatron_fsdp or args.use_torch_fsdp2:
        raise RuntimeError("Nonuniform EP currently requires Megatron DDP")


def _initialize_model_parallel(*args, **kwargs):
    """Replace standard MPU initialization only for an opted-in NEP topology."""
    megatron_args = gpt.get_args()
    _validate_nonuniform_ep_args(megatron_args)

    if initialize_nonuniform_ep_process_groups_from_args(
        megatron_args,
        get_embedding_ranks=kwargs.get("get_embedding_ranks"),
        get_position_embedding_ranks=kwargs.get("get_position_embedding_ranks"),
    ):
        return None

    result = _ORIGINAL_INITIALIZE_MODEL_PARALLEL(*args, **kwargs)
    # Explicit placement is known only after native EP groups exist, but it must be
    # registered before ModelBuilder constructs its MoE layers.
    _get_ep_config(megatron_args)
    return result


def _install_opt_in_ddp(args) -> None:
    training_module.DDP = _ORIGINAL_TRAINING_DDP
    model_dist_utils.DistributedDataParallel = _ORIGINAL_MODEL_BUILDER_DDP
    parallel_state.initialize_model_parallel = _ORIGINAL_INITIALIZE_MODEL_PARALLEL

    if args.nonuniform_mode == "none":
        set_nonuniform_ep_runtime_config(None)
        return

    _validate_nonuniform_ep_args(args)
    parallel_state.initialize_model_parallel = _initialize_model_parallel

    class BenchmarkNonuniformEPDDP(NonuniformEPDistributedDataParallel):
        """Bind the parsed benchmark NEP configuration to Megatron DDP."""

        def __init__(self, *ddp_args, **kwargs):
            super().__init__(*ddp_args, nonuniform_ep_config=_get_ep_config(args), **kwargs)

    training_module.DDP = BenchmarkNonuniformEPDDP
    model_dist_utils.DistributedDataParallel = BenchmarkNonuniformEPDDP


def _add_nonuniform_args(parser):
    if gpt.has_nvidia_modelopt:
        maybe_parser = gpt.add_modelopt_args(parser)
        if maybe_parser is not None:
            parser = maybe_parser

    group = parser.add_argument_group(title="nonuniform benchmark")
    group.add_argument(
        "--nonuniform-mode",
        choices=["none", "ep"],
        default="none",
        help="Opt into a nonuniform expert-parallel DDP wrapper for this GPT training run.",
    )
    group.add_argument("--nonuniform-ep-min-size", type=int, default=None)
    group.add_argument("--nonuniform-ep-num-tp-cp-per-replica", nargs="+", type=int, default=None)
    group.add_argument("--nonuniform-ep-placement-json", default=None)
    group.add_argument("--nonuniform-ep-placement-path", default=None)
    group.add_argument("--nonuniform-ep-expert-name-pattern", default=None)
    group.add_argument(
        "--nonuniform-ep-ddp-approach",
        choices=["nccl"],
        default="nccl",
        help="Compatibility spelling for the NCCL nonuniform-EP implementation.",
    )
    return parser


if __name__ == "__main__":
    main_entry_time = gpt.time.time()
    gpt.print_rank_0(f"> PyTorch version ................ {gpt.get_torch_version()}")
    gpt.print_rank_0(f"> Megatron-Core version .......... {gpt.mcore_version}")
    gpt.print_rank_0(f"> Transformer Engine version ... {gpt.get_te_version()}")
    gpt.set_startup_timestamps(program_start=gpt._PROGRAM_START_TIME, main_entry=main_entry_time)

    gpt.train_valid_test_datasets_provider.is_distributed = True
    pretrain, store = gpt.inprocess_restart.maybe_wrap_for_inprocess_restart(gpt.pretrain)
    args = gpt.parse_and_validate_args(
        extra_args_provider=_add_nonuniform_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    if gpt.has_nvidia_modelopt:
        gpt.maybe_enable_modelopt(args)
    _install_opt_in_ddp(args)

    if gpt.has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        model_cfg = gpt.gpt_config_from_args(args, model_config_cls=gpt.ModelOptModelConfig)
    else:
        model_cfg = gpt.gpt_config_from_args(args)
    full_config = gpt.pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        gpt.train_valid_test_datasets_provider,
        gpt.ModelType.encoder_or_decoder,
        gpt.forward_step,
        store=store,
        get_embedding_ranks=gpt.get_embedding_ranks,
    )
