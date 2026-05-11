# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""GPT pretraining entrypoint for opt-in nonuniform TP/EP benchmarks.

This script intentionally keeps the generic Megatron training loop untouched.  It imports the
standard GPT pretraining providers, patches the DDP class used by ``megatron.training.training``
to one of the opt-in nonuniform wrappers, and rejects distributed optimizer for nonuniform
benchmark modes.  With the non-distributed optimizer, synced gradients are present on the ranks
that own the local parameters before the normal optimizer step runs.
"""

from functools import partial
import json
from pathlib import Path
from typing import Dict, Optional

import torch.distributed as dist

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
from megatron.core.distributed.nonuniform_tp import (
    NonuniformTPConfig,
    NonuniformTPDistributedDataParallel,
    initialize_nonuniform_tp_process_groups,
    ntp_init,
    ntp_map,
)
import megatron.training.training as training_module


_NTP_GROUPS_INITIALIZED = False
_NTP_CONFIG_CACHE = {}
_EP_CONFIG_CACHE = {}
_ORIGINAL_INITIALIZE_MODEL_PARALLEL = parallel_state.initialize_model_parallel


def _load_json_arg(value: Optional[str], path: Optional[str], default=None):
    if value is not None:
        return json.loads(value)
    if path is not None:
        with Path(path).open() as stream:
            return json.load(stream)
    return default


def _parse_ntp_non_active_map(args) -> Optional[Dict[object, list]]:
    raw_map = _load_json_arg(
        args.nonuniform_tp_non_active_ranks_json,
        args.nonuniform_tp_non_active_ranks_path,
        None,
    )
    if raw_map is None:
        return None

    parsed = {}
    for key, value in raw_map.items():
        if "," in str(key):
            parsed_key = tuple(int(part.strip()) for part in str(key).split(","))
            if len(parsed_key) != 3:
                raise ValueError(
                    "NTP non-active rank map tuple keys must be 'dp,cp,pp' triples"
                )
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


def _ntp_model_provider(builder, ntp_config, exit_inactive_ranks, *provider_args, **kwargs):
    global _NTP_GROUPS_INITIALIZED
    if not _NTP_GROUPS_INITIALIZED:
        initialize_nonuniform_tp_process_groups(ntp_config, exit_spares=exit_inactive_ranks)
        _NTP_GROUPS_INITIALIZED = True

    model = gpt.model_provider(builder, *provider_args, **kwargs)
    _apply_ntp_mappings_to_gpt(model, ntp_config)
    return model


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


def _initialize_model_parallel(*args, **kwargs):
    """Runtime replacement for standard MPU init in nonuniform topology modes."""
    global _NTP_GROUPS_INITIALIZED
    megatron_args = gpt.get_args()
    ntp_topology = megatron_args.nonuniform_tp_domain_sizes
    nep_topology = megatron_args.nonuniform_ep_num_tp_cp_per_replica

    if megatron_args.nonuniform_mode == "tp" and ntp_topology is not None:
        if megatron_args.tensor_model_parallel_size != megatron_args.nonuniform_tp_base:
            raise RuntimeError(
                "--tensor-model-parallel-size must match --nonuniform-tp-base "
                "in NTP topology mode"
            )
        if megatron_args.pipeline_model_parallel_size != 1:
            raise RuntimeError("Nonuniform TP topology mode currently supports PP=1 only")
        if megatron_args.virtual_pipeline_model_parallel_size is not None:
            raise RuntimeError("Nonuniform TP topology mode does not support virtual PP")
        if megatron_args.use_torch_fsdp2:
            raise RuntimeError("--use-torch-fsdp2 is not supported with nonuniform TP")
        if megatron_args.num_distributed_optimizer_instances != 1:
            raise RuntimeError("Nonuniform TP topology mode does not support partial DistOpt")
        if megatron_args.expert_model_parallel_size != 1:
            raise RuntimeError("Nonuniform TP topology mode currently requires EP=1")

        ntp_config = _NTP_CONFIG_CACHE.get('config')
        if ntp_config is None:
            ntp_config = _build_ntp_config(megatron_args)
            _NTP_CONFIG_CACHE['config'] = ntp_config
        initialize_nonuniform_tp_process_groups(
            ntp_config,
            exit_spares=not megatron_args.nonuniform_tp_keep_inactive_ranks,
            context_parallel_size=megatron_args.context_parallel_size,
            nccl_communicator_config_path=megatron_args.nccl_communicator_config_path,
            distributed_timeout_minutes=megatron_args.distributed_timeout_minutes,
            create_gloo_process_groups=megatron_args.enable_gloo_process_groups,
            get_embedding_ranks=kwargs.get("get_embedding_ranks"),
            get_position_embedding_ranks=kwargs.get("get_position_embedding_ranks"),
        )
        _NTP_GROUPS_INITIALIZED = True
        return None

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
        create_gloo_process_groups=megatron_args.enable_gloo_process_groups,
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
    if args.nonuniform_mode == "none":
        set_nonuniform_ep_runtime_config(None)
        parallel_state.initialize_model_parallel = _ORIGINAL_INITIALIZE_MODEL_PARALLEL
        return None
    if args.use_distributed_optimizer:
        raise RuntimeError(
            "Nonuniform TP/EP benchmark modes intentionally use the non-distributed "
            "optimizer. Remove --use-distributed-optimizer."
        )

    if args.nonuniform_mode == "tp":
        set_nonuniform_ep_runtime_config(None)
        ntp_config = _build_ntp_config(args)
        _NTP_CONFIG_CACHE['config'] = ntp_config
        parallel_state.initialize_model_parallel = (
            _initialize_model_parallel
            if args.nonuniform_tp_domain_sizes is not None
            else _ORIGINAL_INITIALIZE_MODEL_PARALLEL
        )

        class BenchmarkNonuniformTPDDP(NonuniformTPDistributedDataParallel):
            def __init__(self, *ddp_args, **kwargs):
                super().__init__(*ddp_args, ntp_config=ntp_config, **kwargs)

        training_module.DDP = BenchmarkNonuniformTPDDP
        return ntp_config

    class BenchmarkNonuniformEPDDP(NonuniformEPDistributedDataParallel):
        def __init__(self, *ddp_args, **kwargs):
            super().__init__(
                *ddp_args,
                nonuniform_ep_config=_get_ep_config(args),
                **kwargs,
            )

    training_module.DDP = BenchmarkNonuniformEPDDP
    parallel_state.initialize_model_parallel = _initialize_model_parallel
    return None


def _add_nonuniform_args(parser):
    if gpt.has_nvidia_modelopt:
        maybe_parser = gpt.add_modelopt_args(parser)
        if maybe_parser is not None:
            parser = maybe_parser

    group = parser.add_argument_group(title='nonuniform benchmark')
    group.add_argument(
        '--nonuniform-mode',
        choices=['none', 'tp', 'ep'],
        default='none',
        help='Opt into a nonuniform DDP wrapper for this GPT training run.',
    )

    group.add_argument('--nonuniform-tp-base', type=int, default=8)
    group.add_argument('--nonuniform-tp-spares', type=int, default=0)
    group.add_argument('--nonuniform-tp-num-reduced-dp-ranks', type=int, default=1)
    group.add_argument('--nonuniform-tp-non-active-ranks-json', default=None)
    group.add_argument('--nonuniform-tp-non-active-ranks-path', default=None)
    group.add_argument(
        '--nonuniform-tp-domain-sizes',
        nargs="+",
        type=int,
        default=None,
        help=(
            "Topology-aware active TP size per replica. Each value creates one "
            "contiguous TP domain block and must be either tp_base or "
            "tp_base - tp_spares."
        ),
    )
    group.add_argument(
        '--nonuniform-tp-keep-inactive-ranks',
        action='store_true',
        help='Do not exit inactive reduced-TP ranks after process-group reconfiguration.',
    )

    group.add_argument('--nonuniform-ep-min-size', type=int, default=None)
    group.add_argument(
        '--nonuniform-ep-num-tp-cp-per-replica',
        nargs="+",
        type=int,
        default=None,
    )
    group.add_argument('--nonuniform-ep-placement-json', default=None)
    group.add_argument('--nonuniform-ep-placement-path', default=None)
    group.add_argument('--nonuniform-ep-expert-owner-json', default=None)
    group.add_argument('--nonuniform-ep-expert-owner-path', default=None)
    group.add_argument('--nonuniform-ep-expert-name-pattern', default=None)
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
    ntp_config = _install_opt_in_ddp(args)
    if args.nonuniform_mode == "tp":
        model_provider = partial(
            _ntp_model_provider,
            gpt.gpt_builder,
            ntp_config,
            not args.nonuniform_tp_keep_inactive_ranks,
        )
    elif args.nonuniform_mode == "ep":
        model_provider = partial(_ep_model_provider, gpt.gpt_builder, args)
    else:
        model_provider = partial(gpt.model_provider, gpt.gpt_builder)

    pretrain(
        gpt.train_valid_test_datasets_provider,
        model_provider,
        gpt.ModelType.encoder_or_decoder,
        gpt.forward_step,
        store=store,
        get_embedding_ranks=gpt.get_embedding_ranks,
    )
