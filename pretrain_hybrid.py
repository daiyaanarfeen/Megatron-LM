# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain and SFT Hybrid."""

# Capture the true program start time BEFORE any heavy imports.
import time

_PROGRAM_START_TIME = time.time()

import json

# Suppress warnings on all ranks but rank 0.
import os
import warnings

rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist

from hybrid_builders import hybrid_builder
from megatron.core import mpu, parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
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
from megatron.core.enums import ModelType
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_hybrid_data_context_parallel_groups,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.multi_token_prediction import (
    mtp_on_this_rank as mtp_on_this_rank_func,
)
from megatron.core.utils import (
    StragglerDetector,
    get_attr_wrapped_model,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training import (
    get_args,
    get_timers,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.argument_utils import (
    hybrid_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.datasets.sft_dataset import SFTDataset
import megatron.training.training as training_module
from megatron.training.training import update_seqlen_stats_from_cu_seqlens
from megatron.training.utils import get_blend_and_blend_per_split, is_first_or_last_pipeline_stage
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False

stimer = StragglerDetector()
_EP_CONFIG_CACHE = {}
_ORIGINAL_DDP = training_module.DDP
_ORIGINAL_INITIALIZE_MODEL_PARALLEL = parallel_state.initialize_model_parallel
_ORIGINAL_GET_MEGATRON_OPTIMIZER = training_module.get_megatron_optimizer
_ORIGINAL_GET_OPTIMIZER_PARAM_SCHEDULER = training_module.get_optimizer_param_scheduler
_ORIGINAL_TRAIN_STEP = training_module.train_step


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""

    BATCH_KEYS = ["attention_mask", "cu_seqlens", "cu_seqlens_padded", "hybrid_cp_group", "labels", "local_cp_size", "loss_mask", "max_seqlen", "position_ids", "tokens"]

    args = get_args()
    config = core_transformer_config_from_args(args)

    cp_size = args.context_parallel_size
    tp_rank = mpu.get_tensor_model_parallel_rank()
    is_sft = args.sft
    create_attention_mask_in_dataloader = args.create_attention_mask_in_dataloader
    mtp_on_this_rank = mtp_on_this_rank_func(layout=config.pipeline_model_parallel_layout, mtp_num_layers=config.mtp_num_layers, ignore_virtual=False, vp_stage=vp_stage)
    is_hybrid_cp = args.hybrid_context_parallel

    if not is_first_or_last_pipeline_stage(vp_stage) and not mtp_on_this_rank and not is_sft:
        return [None for _ in BATCH_KEYS]

    batch = {}
    if tp_rank == 0:
        batch = next(data_iterator)
        for key in BATCH_KEYS:
            batch[key] = batch[key].cuda(non_blocking=True) if key in batch and batch[key] is not None else None

    batch = get_batch_on_this_tp_rank(batch, broadcast_src_rank=mpu.get_tensor_model_parallel_src_rank(), broadcast_group=mpu.get_tensor_model_parallel_group(), is_sft=is_sft, is_hybrid_cp=is_hybrid_cp, create_attention_mask_in_dataloader=create_attention_mask_in_dataloader, cp_size=cp_size, tp_rank=tp_rank, micro_batch_size=args.micro_batch_size, seq_length=args.seq_length, mtp_on_this_rank=mtp_on_this_rank, pipeline_model_parallel_size=args.pipeline_model_parallel_size, is_pipeline_first_stage=mpu.is_pipeline_first_stage(), is_pipeline_last_stage=mpu.is_pipeline_last_stage())

    if not is_first_or_last_pipeline_stage(vp_stage) and not mtp_on_this_rank:
        assert is_sft
        return None, batch['cu_seqlens'], batch['cu_seqlens_padded'], None, None, None, None, batch['max_seqlen'], None, None

    batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=is_hybrid_cp, cp_group=get_context_parallel_group(), hybrid_cp_group_func=get_hybrid_data_context_parallel_groups)

    return [batch[key] for key in sorted(batch.keys())]


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[HybridModel] = None):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are deterministic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are deterministic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are deterministic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: HybridModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (HybridModel): The Hybrid Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()

    global stimer

    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        (
            attention_mask,
            cu_seqlens,
            cu_seqlens_padded,
            hybrid_cp_group,
            labels,
            local_cp_size,
            loss_mask,
            max_seqlen,
            position_ids,
            tokens,
        ) = get_batch(data_iterator, vp_stage)

    packed_seq_params = None
    if cu_seqlens is not None:
        # cu_seqlens / cu_seqlens_padded carry the dataloader's batch dim (1, n).
        # PackedSeqParams (and TE attention) expect 1-D, so squeeze before use.
        cu_seqlens = cu_seqlens[0]
        if cu_seqlens_padded is not None:
            cu_seqlens_padded = cu_seqlens_padded[0]
        # Use real (unpadded) cu_seqlens to feed the FLOPs accounting: varlen
        # attention only computes work for real tokens within each chunk.
        update_seqlen_stats_from_cu_seqlens(cu_seqlens)
        cu_seqlens_for_params = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_for_params,
            cu_seqlens_kv=cu_seqlens_for_params,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=int(max_seqlen.item()),
            max_seqlen_kv=int(max_seqlen.item()),
            local_cp_size=int(local_cp_size.item()) if local_cp_size is not None else None,
            cp_group=hybrid_cp_group,
            total_tokens=int(cu_seqlens_for_params[-1].item()),
        )

    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask
        )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
    args = get_args()
    config = core_transformer_config_from_args(args)
    if mpu.get_tensor_model_parallel_rank() != 0:
        return False
    elif is_packed_sequence:
        return True
    return (
        is_first_or_last_pipeline_stage(vp_stage)
        or mtp_on_this_rank_func(layout=config.pipeline_model_parallel_layout, mtp_num_layers=config.mtp_num_layers, ignore_virtual=False, vp_stage=vp_stage)
    )


def core_gpt_dataset_config_from_args(args: Any) -> GPTDatasetConfig:
    tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        sequences_per_dataset=sequences_per_dataset,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        context_parallel_size=args.context_parallel_size,
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=args.tensor_model_parallel_size * args.sequence_parallel,
        hybrid_context_parallel=args.hybrid_context_parallel,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    is_packed_sequence = False
    if args.sft:
        dataset_type = SFTDataset
        is_packed_sequence = True  # SFT always uses packed sequence
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        partial(is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence),
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def _load_json_arg(value: Optional[str], path: Optional[str], default=None):
    if value is not None:
        return json.loads(value)
    if path is not None:
        with open(path, "r", encoding="utf-8") as stream:
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
    """Runtime replacement for standard MPU init in nonuniform EP topology mode."""
    megatron_args = get_args()
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
        enable_edp_ready_gate=megatron_args.nonuniform_ep_ddp_approach == "nccl",
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
    return model_provider(builder, *provider_args, **kwargs)


def _no_op_optimizer_step_for_nonuniform_benchmark():
    return True, None, None


class _NonuniformBenchmarkNoOpOptimizer:
    is_stub_optimizer = True
    chained_optimizers = []

    def __init__(self, args):
        self.param_groups = [{'default_config': True, 'lr': args.lr}]

    def zero_grad(self):
        return None

    def step(self):
        return _no_op_optimizer_step_for_nonuniform_benchmark()

    def scale_loss(self, loss):
        return loss

    def get_loss_scale(self):
        device = (
            torch.device('cuda', torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device('cpu')
        )
        return torch.tensor(1.0, device=device)

    def reload_model_params(self):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


class _NonuniformBenchmarkNoOpParamScheduler:
    def step(self, increment):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


def _get_no_op_optimizer_for_nonuniform_benchmark(*args, **kwargs):
    return _NonuniformBenchmarkNoOpOptimizer(get_args())


def _get_no_op_param_scheduler_for_nonuniform_benchmark(*args, **kwargs):
    return _NonuniformBenchmarkNoOpParamScheduler()


def _log_nonuniform_local_grad_checksum(model, iteration):
    """Write a per-rank fingerprint for same-topology scheduler comparisons."""
    stats = torch.zeros(5, dtype=torch.float64, device='cuda')
    for model_chunk in model:
        unwrapped_model = training_module.unwrap_model(model_chunk)
        for name, param in unwrapped_model.named_parameters():
            grad = getattr(param, 'main_grad', None)
            if grad is None or getattr(param, 'shared', False):
                continue
            name_weight = 1 + sum(
                (index + 1) * ord(character) for index, character in enumerate(name)
            ) % 104729
            stats[0] += grad.sum(dtype=torch.float64) * name_weight
            stats[1] += grad.abs().sum(dtype=torch.float64) * name_weight
            stats[2] += grad.square().sum(dtype=torch.float64) * name_weight
            stats[3] += grad.numel() * name_weight
            stats[4] += grad.numel()

    checksum_line = (
        "[nonuniform-local-grad-checksum] "
        f"iteration={iteration} rank={dist.get_rank()} "
        f"weighted_sum={stats[0].item():.17e} "
        f"weighted_abs={stats[1].item():.17e} "
        f"weighted_sq={stats[2].item():.17e} "
        f"weighted_numel={stats[3].item():.0f} "
        f"numel={stats[4].item():.0f}"
    )
    print(checksum_line, flush=True)
    checksum_dir = get_args().nonuniform_grad_checksum_dir
    if checksum_dir is not None:
        os.makedirs(checksum_dir, exist_ok=True)
        checksum_path = os.path.join(checksum_dir, f"rank_{dist.get_rank()}.log")
        with open(checksum_path, 'a', encoding='utf-8') as stream:
            stream.write(f"{checksum_line}\n")


def _train_step_without_optimizer_step(*args, **kwargs):
    optimizer = args[3] if len(args) > 3 else kwargs.get('optimizer')
    if optimizer is None:
        return _ORIGINAL_TRAIN_STEP(*args, **kwargs)

    original_step = optimizer.step
    optimizer.step = _no_op_optimizer_step_for_nonuniform_benchmark
    try:
        result = _ORIGINAL_TRAIN_STEP(*args, **kwargs)
        benchmark_args = get_args()
        iteration = args[7] if len(args) > 7 else kwargs.get('iteration')
        if (
            benchmark_args.nonuniform_log_grad_checksum
            and iteration is not None
            and iteration >= benchmark_args.train_iters - 1
        ):
            model = args[2] if len(args) > 2 else kwargs['model']
            _log_nonuniform_local_grad_checksum(model, iteration)
        return result
    finally:
        optimizer.step = original_step


def _install_nonuniform_ep_ddp(args):
    training_module.train_step = _ORIGINAL_TRAIN_STEP
    training_module.get_megatron_optimizer = _ORIGINAL_GET_MEGATRON_OPTIMIZER
    training_module.get_optimizer_param_scheduler = _ORIGINAL_GET_OPTIMIZER_PARAM_SCHEDULER
    if args.nonuniform_skip_optimizer_step:
        training_module.get_megatron_optimizer = _get_no_op_optimizer_for_nonuniform_benchmark
        training_module.get_optimizer_param_scheduler = (
            _get_no_op_param_scheduler_for_nonuniform_benchmark
        )
        training_module.train_step = _train_step_without_optimizer_step
    if args.nonuniform_mode == "none":
        set_nonuniform_ep_runtime_config(None)
        training_module.DDP = _ORIGINAL_DDP
        parallel_state.initialize_model_parallel = _ORIGINAL_INITIALIZE_MODEL_PARALLEL
        return
    if args.use_distributed_optimizer:
        raise RuntimeError(
            "Nonuniform EP benchmark mode intentionally uses the non-distributed optimizer. "
            "Remove --use-distributed-optimizer."
        )

    class BenchmarkNonuniformEPDDP(NonuniformEPDistributedDataParallel):
        def __init__(self, *ddp_args, **kwargs):
            super().__init__(
                *ddp_args,
                nonuniform_ep_config=_get_ep_config(args),
                **kwargs,
            )

    training_module.DDP = BenchmarkNonuniformEPDDP
    parallel_state.initialize_model_parallel = _initialize_model_parallel


def _add_nonuniform_args(parser):
    if has_nvidia_modelopt:
        maybe_parser = add_modelopt_args(parser)
        if maybe_parser is not None:
            parser = maybe_parser

    group = parser.add_argument_group(title='nonuniform benchmark')
    group.add_argument(
        '--nonuniform-mode',
        choices=['none', 'ep'],
        default='none',
        help='Opt into a nonuniform EP DDP wrapper for this hybrid training run.',
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
    group.add_argument(
        '--nonuniform-ep-ddp-approach',
        choices=['p2p', 'nccl'],
        default='p2p',
        help='Nonuniform EP expert-gradient sync approach. `nccl` is Approach A.',
    )
    group.add_argument(
        '--nonuniform-skip-optimizer-step',
        action='store_true',
        help=(
            'Run forward/backward and nonuniform EP grad sync with a no-op '
            'optimizer for performance-only validation.'
        ),
    )
    group.add_argument(
        '--nonuniform-log-grad-checksum',
        action='store_true',
        help='Log a local post-sync gradient fingerprint on every rank.',
    )
    group.add_argument(
        '--nonuniform-grad-checksum-dir',
        type=str,
        default=None,
        help='Also write each local gradient fingerprint to a per-rank file.',
    )
    return parser


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    setattr(train_valid_test_datasets_provider, "is_distributed", True)

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = parse_and_validate_args(
        extra_args_provider=_add_nonuniform_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    _install_nonuniform_ep_ddp(args)
    model_cfg = hybrid_config_from_args(args)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    provider = partial(model_provider, hybrid_builder)
    if args.nonuniform_mode == "ep":
        provider = partial(_ep_model_provider, hybrid_builder, args)
    pretrain(full_config,
             train_valid_test_datasets_provider,
             provider,
             ModelType.encoder_or_decoder,
             forward_step,
             store=store,
             )
