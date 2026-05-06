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
from megatron.core.distributed.nonuniform_common import get_global_rank
from megatron.core.distributed.nonuniform_ep import (
    NonuniformEPConfig,
    NonuniformEPDistributedDataParallel,
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
    placement = _load_json_arg(
        args.nonuniform_ep_placement_json,
        args.nonuniform_ep_placement_path,
        None,
    )
    if placement is None:
        return None

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


def _install_opt_in_ddp(args):
    if args.nonuniform_mode == "none":
        return None
    if args.use_distributed_optimizer:
        raise RuntimeError(
            "Nonuniform TP/EP benchmark modes intentionally use the non-distributed "
            "optimizer. Remove --use-distributed-optimizer."
        )

    if args.nonuniform_mode == "tp":
        ntp_config = _build_ntp_config(args)

        class BenchmarkNonuniformTPDDP(NonuniformTPDistributedDataParallel):
            def __init__(self, *ddp_args, **kwargs):
                super().__init__(*ddp_args, ntp_config=ntp_config, **kwargs)

        training_module.DDP = BenchmarkNonuniformTPDDP
        return ntp_config

    ep_config_cache = {}

    class BenchmarkNonuniformEPDDP(NonuniformEPDistributedDataParallel):
        def __init__(self, *ddp_args, **kwargs):
            if 'config' not in ep_config_cache:
                ep_config_cache['config'] = _build_ep_config(args)
            super().__init__(
                *ddp_args,
                nonuniform_ep_config=ep_config_cache['config'],
                **kwargs,
            )

    training_module.DDP = BenchmarkNonuniformEPDDP
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
        '--nonuniform-tp-keep-inactive-ranks',
        action='store_true',
        help='Do not exit inactive reduced-TP ranks after process-group reconfiguration.',
    )

    group.add_argument('--nonuniform-ep-min-size', type=int, default=None)
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
