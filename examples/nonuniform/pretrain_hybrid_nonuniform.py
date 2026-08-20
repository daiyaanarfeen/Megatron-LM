# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Hybrid pretraining entrypoint for opt-in nonuniform expert parallelism."""

import json
import runpy
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import megatron.training.argument_utils as argument_utils
import megatron.training.arguments as training_arguments
import megatron.training.models.dist_utils as model_dist_utils
import megatron.training.training as training_module
from megatron.core import parallel_state
from megatron.core.distributed.nonuniform_common import set_nonuniform_ep_runtime_config
from megatron.core.distributed.nonuniform_ep import (
    NonuniformEPConfig,
    NonuniformEPDistributedDataParallel,
    initialize_nonuniform_ep_process_groups_from_args,
)
from megatron.training import get_args

_EP_CONFIG_CACHE = {}
_ORIGINAL_HYBRID_CONFIG_FROM_ARGS = argument_utils.hybrid_config_from_args
_ORIGINAL_PARSE_AND_VALIDATE_ARGS = training_arguments.parse_and_validate_args
_ORIGINAL_INITIALIZE_MODEL_PARALLEL = parallel_state.initialize_model_parallel
_ORIGINAL_TRAINING_DDP = training_module.DDP
_ORIGINAL_MODEL_BUILDER_DDP = model_dist_utils.DistributedDataParallel
_ORIGINAL_LOGICAL_AND = training_module.logical_and_across_model_parallel_group
_ORIGINAL_REDUCE_MAX = training_module.reduce_max_stat_across_model_parallel_group
_ORIGINAL_GET_MOE_METRICS_TRACKER = training_module.get_moe_metrics_tracker


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
    """Replace MPU initialization only for an opted-in nonuniform EP topology."""
    megatron_args = get_args()
    _validate_nonuniform_ep_args(megatron_args)
    if initialize_nonuniform_ep_process_groups_from_args(
        megatron_args,
        get_embedding_ranks=kwargs.get("get_embedding_ranks"),
        get_position_embedding_ranks=kwargs.get("get_position_embedding_ranks"),
    ):
        return None
    result = _ORIGINAL_INITIALIZE_MODEL_PARALLEL(*args, **kwargs)
    _get_ep_config(megatron_args)
    return result


class _LocalMoEMetricsTracker:
    """Delegate metric recording but clear locally instead of reducing at report time."""

    def __init__(self, tracker):
        self._tracker = tracker

    def __getattr__(self, name):
        return getattr(self._tracker, name)

    def report(self, *args, **kwargs):
        self._tracker.clear()
        return ""


def _local_identity(value, *args, **kwargs):
    return value


def _local_scalar(value, *args, **kwargs):
    return value.item() if isinstance(value, torch.Tensor) else value


def _install_nonuniform_ep_ddp(args) -> None:
    training_module.DDP = _ORIGINAL_TRAINING_DDP
    model_dist_utils.DistributedDataParallel = _ORIGINAL_MODEL_BUILDER_DDP
    parallel_state.initialize_model_parallel = _ORIGINAL_INITIALIZE_MODEL_PARALLEL
    training_module.logical_and_across_model_parallel_group = _ORIGINAL_LOGICAL_AND
    training_module.reduce_max_stat_across_model_parallel_group = _ORIGINAL_REDUCE_MAX
    training_module.get_moe_metrics_tracker = _ORIGINAL_GET_MOE_METRICS_TRACKER

    if args.nonuniform_disable_nongrad_sync_collectives:
        args.log_throughput = False
        args.log_progress = False
        training_module.logical_and_across_model_parallel_group = _local_identity
        training_module.reduce_max_stat_across_model_parallel_group = _local_scalar
        tracker = _LocalMoEMetricsTracker(_ORIGINAL_GET_MOE_METRICS_TRACKER())
        training_module.get_moe_metrics_tracker = lambda: tracker

    if args.nonuniform_mode == "none":
        set_nonuniform_ep_runtime_config(None)
        return

    _validate_nonuniform_ep_args(args)

    class BenchmarkNonuniformEPDDP(NonuniformEPDistributedDataParallel):
        def __init__(self, *ddp_args, **kwargs):
            super().__init__(*ddp_args, nonuniform_ep_config=_get_ep_config(args), **kwargs)

    training_module.DDP = BenchmarkNonuniformEPDDP
    model_dist_utils.DistributedDataParallel = BenchmarkNonuniformEPDDP
    parallel_state.initialize_model_parallel = _initialize_model_parallel


def _add_nonuniform_args(parser):
    group = parser.add_argument_group(title="nonuniform benchmark")
    group.add_argument(
        "--nonuniform-mode",
        choices=["none", "ep"],
        default="none",
        help="Opt into a nonuniform EP DDP wrapper for this hybrid training run.",
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
    group.add_argument(
        "--nonuniform-disable-nongrad-sync-collectives",
        action="store_true",
        help="Disable reporting-only collectives for communication-isolated benchmarks.",
    )
    return parser


def _parse_with_nonuniform_args(*args, **kwargs):
    """Compose NEP arguments around the native entrypoint parser."""
    native_extra_args_provider = kwargs.get("extra_args_provider")

    def _combined_extra_args_provider(parser):
        if native_extra_args_provider is not None:
            maybe_parser = native_extra_args_provider(parser)
            if maybe_parser is not None:
                parser = maybe_parser
        return _add_nonuniform_args(parser)

    kwargs["extra_args_provider"] = _combined_extra_args_provider
    return _ORIGINAL_PARSE_AND_VALIDATE_ARGS(*args, **kwargs)


def _hybrid_config_with_nonuniform_ep(args, *config_args, **config_kwargs):
    """Install NEP at the same point as the validated hybrid entrypoint integration."""
    _install_nonuniform_ep_ddp(args)
    return _ORIGINAL_HYBRID_CONFIG_FROM_ARGS(args, *config_args, **config_kwargs)


if __name__ == "__main__":
    training_arguments.parse_and_validate_args = _parse_with_nonuniform_args
    argument_utils.hybrid_config_from_args = _hybrid_config_with_nonuniform_ep
    runpy.run_path(str(_REPO_ROOT / "pretrain_hybrid.py"), run_name="__main__")
