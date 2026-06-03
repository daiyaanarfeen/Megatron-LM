# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPT pretraining entrypoint with opt-in heterogeneous EP DDP.

This script intentionally keeps the standard Megatron training library unchanged.
It reuses ``pretrain_gpt.py`` providers and installs runtime hooks that select
heterogeneous EP process groups and the opt-in DDP wrapper for this entrypoint.
"""

import time
from functools import partial

_PROGRAM_START_TIME = time.time()

import pretrain_gpt as gpt_entry

import torch

from megatron.core import parallel_state
from megatron.core.distributed.heterogeneous_ep import (
    HeterogeneousEPConfig,
    HeterogeneousEPDistributedDataParallel,
)
from megatron.core.enums import ModelType
from megatron.training import get_args, inprocess_restart, set_startup_timestamps
import megatron.training.training as training_module


_ORIGINAL_INITIALIZE_MODEL_PARALLEL = parallel_state.initialize_model_parallel


def add_heterogeneous_ep_args(parser):
    """Add entrypoint-local heterogeneous EP flags."""
    group = parser.add_argument_group(title="heterogeneous expert parallel training")
    group.add_argument(
        "--heterogeneous-ep-num-tp-cp-per-replica",
        nargs="+",
        type=int,
        required=True,
        help=(
            "Number of TP*CP units in each MoE replica. For TP2 CP2 ETP2, "
            "'4 3' gives EP8/EP6 and '4 4' gives uniform EP8."
        ),
    )
    group.add_argument(
        "--heterogeneous-ep-ddp-approach",
        type=str,
        required=True,
        choices=["nccl", "nvshmem", "phased"],
        help="Heterogeneous EP gradient sync approach for the opt-in DDP wrapper.",
    )
    group.add_argument(
        "--heterogeneous-ep-num-pipeline-chunks",
        type=int,
        default=None,
        help="Number of pipeline chunks for the heterogeneous EP NVSHMEM approach.",
    )
    return parser


def extra_args_provider(parser):
    """Compose this entrypoint's args with optional ModelOpt args."""
    if getattr(gpt_entry, "has_nvidia_modelopt", False):
        parser = gpt_entry.add_modelopt_args(parser)
    return add_heterogeneous_ep_args(parser)


def _initialize_model_parallel(*args, **kwargs):
    """Runtime replacement for standard MPU init in this entrypoint."""
    megatron_args = get_args()
    topology = megatron_args.heterogeneous_ep_num_tp_cp_per_replica
    if topology is None:
        return _ORIGINAL_INITIALIZE_MODEL_PARALLEL(*args, **kwargs)

    assert megatron_args.num_experts is not None, (
        "num_experts must be non None to use heterogeneous expert parallelism"
    )
    assert megatron_args.pipeline_model_parallel_size == 1, (
        "Heterogeneous expert parallelism currently supports pipeline parallel size 1"
    )
    assert megatron_args.virtual_pipeline_model_parallel_size is None, (
        "Heterogeneous expert parallelism does not support virtual pipeline stages"
    )
    assert not megatron_args.use_torch_fsdp2, (
        "--use-torch-fsdp2 is not supported with heterogeneous expert parallelism"
    )
    assert megatron_args.num_distributed_optimizer_instances == 1, (
        "Heterogeneous expert parallelism does not support partial distributed optimizer"
    )

    tp_cp = megatron_args.tensor_model_parallel_size * megatron_args.context_parallel_size
    for num_tp_cp in topology:
        assert num_tp_cp * tp_cp % megatron_args.expert_tensor_parallel_size == 0, (
            "Each heterogeneous replica size must produce an integer EP size"
        )
        ep_size = num_tp_cp * tp_cp // megatron_args.expert_tensor_parallel_size
        assert megatron_args.num_experts % ep_size == 0, (
            "num_experts must be divisible by each heterogeneous EP size"
        )

    parallel_state.initialize_heterogeneous_model_parallel(
        tensor_model_parallel_size=megatron_args.tensor_model_parallel_size,
        context_parallel_size=megatron_args.context_parallel_size,
        num_tp_cp_per_replica=topology,
        expert_tensor_parallel_size=megatron_args.expert_tensor_parallel_size,
        num_moe_experts=megatron_args.num_experts,
        hidden_size=megatron_args.hidden_size,
        ffn_hidden_size=megatron_args.ffn_hidden_size,
        heterogeneous_ep_approach=megatron_args.heterogeneous_ep_ddp_approach,
        nccl_communicator_config_path=megatron_args.nccl_communicator_config_path,
        distributed_timeout_minutes=megatron_args.distributed_timeout_minutes,
        create_gloo_process_groups=megatron_args.enable_gloo_process_groups,
    )
    # The heterogeneous initializer intentionally supports only one distributed
    # optimizer instance, but standard optimizer setup still expects this
    # process-group slot to exist. With one instance, it is the full world.
    parallel_state._INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = torch.distributed.group.WORLD
    megatron_args.expert_model_parallel_size = (
        parallel_state.get_expert_model_parallel_world_size()
    )


class HeterogeneousEPDDP(HeterogeneousEPDistributedDataParallel):
    """DDP shim that reads entrypoint-local heterogeneous EP args."""

    def __init__(self, config, ddp_config, module, disable_bucketing=False, pg_collection=None):
        megatron_args = get_args()
        heterogeneous_ep_config = HeterogeneousEPConfig(
            approach=megatron_args.heterogeneous_ep_ddp_approach,
            num_pipeline_chunks=megatron_args.heterogeneous_ep_num_pipeline_chunks,
        )
        super().__init__(
            config=config,
            ddp_config=ddp_config,
            module=module,
            heterogeneous_ep_config=heterogeneous_ep_config,
            disable_bucketing=disable_bucketing,
            pg_collection=pg_collection,
        )


def install_runtime_hooks():
    """Install entrypoint-local hooks before Megatron initialization."""
    parallel_state.initialize_model_parallel = _initialize_model_parallel
    training_module.DDP = HeterogeneousEPDDP


if __name__ == "__main__":
    _MAIN_ENTRY_TIME = time.time()
    install_runtime_hooks()

    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    gpt_entry.train_valid_test_datasets_provider.is_distributed = True

    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(
        training_module.pretrain
    )

    pretrain(
        gpt_entry.train_valid_test_datasets_provider,
        partial(gpt_entry.model_provider, gpt_entry.gpt_builder),
        ModelType.encoder_or_decoder,
        gpt_entry.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=extra_args_provider,
        store=store,
        get_embedding_ranks=gpt_entry.get_embedding_ranks,
    )
