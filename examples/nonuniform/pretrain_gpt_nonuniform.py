# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""GPT pretraining entrypoint for opt-in nonuniform TP/EP benchmarks."""

import json
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import megatron.training.arguments as training_arguments
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
from megatron.core.distributed.nonuniform_tp import (
    NonuniformTPConfig,
    NonuniformTPDistributedDataParallel,
    initialize_nonuniform_tp_process_groups,
    ntp_init,
    ntp_map,
)

_NTP_GROUPS_INITIALIZED = False
_NTP_CONFIG_CACHE = {}
_EP_CONFIG_CACHE = {}
_ORIGINAL_INITIALIZE_MODEL_PARALLEL = parallel_state.initialize_model_parallel
_ORIGINAL_TRAINING_DDP = training_module.DDP
_ORIGINAL_MODEL_BUILDER_DDP = model_dist_utils.DistributedDataParallel
_ORIGINAL_VALIDATE_ARGS = training_arguments.validate_args


def _load_json_arg(value: Optional[str], path: Optional[str], default=None):
    if value is not None:
        return json.loads(value)
    if path is not None:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    return default


def _validate_args_with_nonuniform_tp_topology(args, defaults=None):
    """Validate an active-rank NTP topology using its logical replica count."""
    defaults = {} if defaults is None else defaults
    domain_sizes = getattr(args, "nonuniform_tp_domain_sizes", None)
    if getattr(args, "nonuniform_mode", "none") != "tp" or domain_sizes is None:
        return _ORIGINAL_VALIDATE_ARGS(args, defaults)

    if args.tensor_model_parallel_size != args.nonuniform_tp_base:
        raise RuntimeError(
            "--tensor-model-parallel-size must match --nonuniform-tp-base in NTP topology mode"
        )
    expected_active_world_size = (
        sum(domain_sizes) * args.context_parallel_size * args.pipeline_model_parallel_size
    )
    if args.world_size != expected_active_world_size:
        raise RuntimeError(
            f"NTP topology active world size ({args.world_size}) must equal "
            f"sum(tp_domain_sizes) * CP * PP ({expected_active_world_size})"
        )

    # Native validation assumes every replica has the full TP width. Temporarily present that
    # logical world so it derives the correct DP replica count and batch semantics, then restore
    # the actual active-rank world used by torch.distributed and the topology process groups.
    active_world_size = args.world_size
    args.world_size = (
        len(domain_sizes)
        * args.tensor_model_parallel_size
        * args.context_parallel_size
        * args.pipeline_model_parallel_size
    )
    try:
        return _ORIGINAL_VALIDATE_ARGS(args, defaults)
    finally:
        args.world_size = active_world_size


def _parse_and_validate_args(*args, **kwargs):
    """Use native argument parsing with an NTP-topology-only validation adapter."""
    training_arguments.validate_args = _validate_args_with_nonuniform_tp_topology
    try:
        return gpt.parse_and_validate_args(*args, **kwargs)
    finally:
        training_arguments.validate_args = _ORIGINAL_VALIDATE_ARGS


def _parse_ntp_non_active_map(args) -> Optional[Dict[object, list]]:
    raw_map = _load_json_arg(
        args.nonuniform_tp_non_active_ranks_json, args.nonuniform_tp_non_active_ranks_path, None
    )
    if raw_map is None:
        return None

    parsed = {}
    for key, value in raw_map.items():
        if "," in str(key):
            parsed_key = tuple(int(part.strip()) for part in str(key).split(","))
            if len(parsed_key) != 3:
                raise ValueError("NTP non-active rank map tuple keys must be 'dp,cp,pp' triples")
        else:
            parsed_key = int(key)
        parsed[parsed_key] = [int(rank) for rank in value]
    return parsed


def _build_ntp_config(args) -> NonuniformTPConfig:
    return NonuniformTPConfig(
        tp_base=args.nonuniform_tp_base,
        tp_spares=args.nonuniform_tp_spares,
        num_reduced_tp_dp_ranks=args.nonuniform_tp_num_reduced_dp_ranks,
        non_active_ranks_per_dp=_parse_ntp_non_active_map(args),
        tp_domain_sizes=args.nonuniform_tp_domain_sizes,
    )


def _apply_ntp_mappings_to_gpt(model, ntp_config: NonuniformTPConfig) -> None:
    for module in model.modules():
        if module.__class__.__name__ == "TransformerLayer":
            ntp_init(module, ntp_config)

    vocab_shards = getattr(gpt.get_args(), "padded_vocab_size", None)
    if vocab_shards is None:
        return
    if hasattr(model, "embedding") and hasattr(model.embedding, "word_embeddings"):
        ntp_map(model.embedding.word_embeddings, ntp_config, vocab_shards)
    if hasattr(model, "output_layer"):
        ntp_map(model.output_layer, ntp_config, vocab_shards)


def _apply_ntp_pre_wrap_hook(models, ntp_config: NonuniformTPConfig, exit_inactive_ranks: bool):
    global _NTP_GROUPS_INITIALIZED
    if not _NTP_GROUPS_INITIALIZED:
        initialize_nonuniform_tp_process_groups(ntp_config, exit_spares=exit_inactive_ranks)
        _NTP_GROUPS_INITIALIZED = True
    for model in models:
        _apply_ntp_mappings_to_gpt(model, ntp_config)
    return models


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
        "needs_reshard": local_ep_size != min_ep_size,
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


def _validate_common_nonuniform_args(args) -> None:
    if args.gtp_weight_remat_size != 1 or args.expert_gtp_weight_remat_size != 1:
        raise RuntimeError("Nonuniform TP/EP does not support GTP/EGTP rematerialization")
    if args.use_megatron_fsdp or args.use_torch_fsdp2:
        raise RuntimeError("Nonuniform TP/EP currently requires Megatron DDP")


def _initialize_model_parallel(*args, **kwargs):
    """Replace standard MPU initialization only for an opted-in nonuniform topology."""
    global _NTP_GROUPS_INITIALIZED
    megatron_args = gpt.get_args()
    _validate_common_nonuniform_args(megatron_args)

    if megatron_args.nonuniform_mode == "tp" and megatron_args.nonuniform_tp_domain_sizes:
        if megatron_args.tensor_model_parallel_size != megatron_args.nonuniform_tp_base:
            raise RuntimeError(
                "--tensor-model-parallel-size must match --nonuniform-tp-base in NTP topology mode"
            )
        if megatron_args.pipeline_model_parallel_size != 1:
            raise RuntimeError("Nonuniform TP topology mode currently supports PP=1 only")
        if megatron_args.virtual_pipeline_model_parallel_size is not None:
            raise RuntimeError("Nonuniform TP topology mode does not support virtual PP")
        if megatron_args.num_distributed_optimizer_instances != 1:
            raise RuntimeError("Nonuniform TP topology mode does not support partial DistOpt")
        if megatron_args.expert_model_parallel_size != 1:
            raise RuntimeError("Nonuniform TP topology mode currently requires EP=1")

        ntp_config = _NTP_CONFIG_CACHE.get("config")
        if ntp_config is None:
            ntp_config = _build_ntp_config(megatron_args)
            _NTP_CONFIG_CACHE["config"] = ntp_config
        initialize_nonuniform_tp_process_groups(
            ntp_config,
            exit_spares=not megatron_args.nonuniform_tp_keep_inactive_ranks,
            context_parallel_size=megatron_args.context_parallel_size,
            nccl_communicator_config_path=megatron_args.nccl_communicator_config_path,
            distributed_timeout_minutes=megatron_args.distributed_timeout_minutes,
            create_gloo_process_groups=megatron_args.use_gloo_process_groups,
            get_embedding_ranks=kwargs.get("get_embedding_ranks"),
            get_position_embedding_ranks=kwargs.get("get_position_embedding_ranks"),
        )
        _NTP_GROUPS_INITIALIZED = True
        return None

    if initialize_nonuniform_ep_process_groups_from_args(
        megatron_args,
        get_embedding_ranks=kwargs.get("get_embedding_ranks"),
        get_position_embedding_ranks=kwargs.get("get_position_embedding_ranks"),
    ):
        return None

    result = _ORIGINAL_INITIALIZE_MODEL_PARALLEL(*args, **kwargs)
    # Explicit placement is known only after native EP groups exist, but the placement must be
    # registered before ModelBuilder constructs its MoE layers.
    if megatron_args.nonuniform_mode == "ep":
        _get_ep_config(megatron_args)
    return result


def _install_opt_in_ddp(args):
    training_module.DDP = _ORIGINAL_TRAINING_DDP
    model_dist_utils.DistributedDataParallel = _ORIGINAL_MODEL_BUILDER_DDP
    parallel_state.initialize_model_parallel = _ORIGINAL_INITIALIZE_MODEL_PARALLEL

    if args.nonuniform_mode == "none":
        set_nonuniform_ep_runtime_config(None)
        return None

    _validate_common_nonuniform_args(args)
    if args.use_distributed_optimizer and args.nonuniform_mode == "tp":
        raise RuntimeError("Nonuniform TP benchmark mode does not support distributed optimizer")

    parallel_state.initialize_model_parallel = _initialize_model_parallel
    if args.nonuniform_mode == "tp":
        set_nonuniform_ep_runtime_config(None)
        ntp_config = _build_ntp_config(args)
        _NTP_CONFIG_CACHE["config"] = ntp_config

        class BenchmarkNonuniformTPDDP(NonuniformTPDistributedDataParallel):
            def __init__(self, *ddp_args, **kwargs):
                super().__init__(*ddp_args, ntp_config=ntp_config, **kwargs)

        training_module.DDP = BenchmarkNonuniformTPDDP
        model_dist_utils.DistributedDataParallel = BenchmarkNonuniformTPDDP
        return ntp_config

    class BenchmarkNonuniformEPDDP(NonuniformEPDistributedDataParallel):
        def __init__(self, *ddp_args, **kwargs):
            super().__init__(*ddp_args, nonuniform_ep_config=_get_ep_config(args), **kwargs)

    training_module.DDP = BenchmarkNonuniformEPDDP
    model_dist_utils.DistributedDataParallel = BenchmarkNonuniformEPDDP
    return None


def _add_nonuniform_args(parser):
    if gpt.has_nvidia_modelopt:
        maybe_parser = gpt.add_modelopt_args(parser)
        if maybe_parser is not None:
            parser = maybe_parser

    group = parser.add_argument_group(title="nonuniform benchmark")
    group.add_argument(
        "--nonuniform-mode",
        choices=["none", "tp", "ep"],
        default="none",
        help="Opt into a nonuniform DDP wrapper for this GPT training run.",
    )
    group.add_argument("--nonuniform-tp-base", type=int, default=8)
    group.add_argument("--nonuniform-tp-spares", type=int, default=0)
    group.add_argument("--nonuniform-tp-num-reduced-dp-ranks", type=int, default=1)
    group.add_argument("--nonuniform-tp-non-active-ranks-json", default=None)
    group.add_argument("--nonuniform-tp-non-active-ranks-path", default=None)
    group.add_argument("--nonuniform-tp-domain-sizes", nargs="+", type=int, default=None)
    group.add_argument("--nonuniform-tp-keep-inactive-ranks", action="store_true")
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
    args = _parse_and_validate_args(
        extra_args_provider=_add_nonuniform_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    if gpt.has_nvidia_modelopt:
        gpt.maybe_enable_modelopt(args)
    ntp_config = _install_opt_in_ddp(args)

    if gpt.has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        model_cfg = gpt.gpt_config_from_args(args, model_config_cls=gpt.ModelOptModelConfig)
    else:
        model_cfg = gpt.gpt_config_from_args(args)
    if args.nonuniform_mode == "tp":
        model_cfg.pre_wrap_hooks.append(
            partial(
                _apply_ntp_pre_wrap_hook,
                ntp_config=ntp_config,
                exit_inactive_ranks=not args.nonuniform_tp_keep_inactive_ranks,
            )
        )

    full_config = gpt.pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        gpt.train_valid_test_datasets_provider,
        gpt.ModelType.encoder_or_decoder,
        gpt.forward_step,
        store=store,
        get_embedding_ranks=gpt.get_embedding_ranks,
    )
