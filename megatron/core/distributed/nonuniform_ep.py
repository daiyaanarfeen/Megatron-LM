# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in nonuniform expert-parallel gradient ownership transfer.

This module keeps nonuniform EP out of generic Megatron DDP. Expert params
are wrapped into expert-level bucket groups. Non-owner ranks transfer expert
gradients to owner ranks before native expert-data-parallel synchronization.
Non-distributed optimizers receive the synchronized gradients back on every
physical holder. Distributed optimizers instead update persistent owner-layout
parameter buffers and redistribute the updated parameters to physical holders.
"""

import copy
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .. import parallel_state
from ..optimizer.param_layout import (
    FullParamLayout,
    PerBufferParamLayout,
    pad_bucket_end,
    pad_param_start,
)
from ..process_groups_config import ProcessGroupCollection
from ..transformer.transformer_config import TransformerConfig
from .distributed_data_parallel import DistributedDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .nonuniform_common import (
    NonuniformEPRankGenerator,
    compute_nonuniform_ep_dispatch_slots,
    compute_nonuniform_ep_expert_placement,
    compute_nonuniform_ep_owner_expert_slots,
    configure_ordered_bucket_group_scheduler,
    filter_kwargs_for_callable,
    get_global_rank,
    get_nonuniform_ep_runtime_config,
    reset_ordered_bucket_group_scheduler,
    set_nonuniform_ep_runtime_config,
)
from .param_and_grad_buffer import (
    _ParamAndGradBucket,
    _ParamAndGradBucketGroup,
    _ParamAndGradBuffer,
)

logger = logging.getLogger(__name__)
_NEP_NCCL_DEFAULT_MAX_GATHER_BYTES = 8 * 1024 * 1024 * 1024
_NEP_NCCL_DEFAULT_ASYNC_CHUNK_WINDOW = 2
_NEP_NCCL_DEFAULT_EXPERT_BUCKET_GROUPS = 3

















def _nep_owner_ddp_config(
    ddp_config: DistributedDataParallelConfig,
) -> DistributedDataParallelConfig:
    """Return the config used by owner DDP groups without redundant gradient checks."""
    # The live config may have both num_buckets and its resolved bucket_size.
    # Do not invoke __post_init__ again while disabling checks already performed by native DDP.
    native_ddp_config = copy.copy(ddp_config)
    native_ddp_config.check_for_nan_in_grad = False
    native_ddp_config.check_for_large_grads = False
    return native_ddp_config


def _default_expert_name_pattern() -> re.Pattern:
    return re.compile(
        r"(?:^|\.)local_experts\.(\d+)(?:\.|$)"
        r"|(?:^|\.)experts\.linear_fc[12]\.(?:weight|bias)(\d+)$"
    )


def _get_nep_nccl_max_gather_bytes() -> int:
    value = os.getenv("MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES")
    if value is None:
        return _NEP_NCCL_DEFAULT_MAX_GATHER_BYTES
    max_gather_bytes = int(value)
    if max_gather_bytes <= 0:
        raise RuntimeError("MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES must be positive")
    return max_gather_bytes


def _get_nep_nccl_target_chunks() -> Optional[int]:
    value = os.getenv("MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS")
    if value in (None, ""):
        return None
    target_chunks = int(value)
    if target_chunks <= 0:
        raise RuntimeError("MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS must be positive")
    return target_chunks


def _get_nep_nccl_scatter_chunks() -> int:
    value = os.getenv("MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS", "1")
    scatter_chunks = int(value)
    if scatter_chunks <= 0:
        raise RuntimeError("MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS must be positive")
    return scatter_chunks


def _build_nep_nccl_scatter_chunk_ranges(
    chunk_start: int, chunk_end: int, remote_segments: List[Tuple[int, int]], scatter_chunks: int
) -> List[Tuple[int, int]]:
    """Partition cumulative remote payload while covering the full owner-layout range."""
    if chunk_start < 0 or chunk_end < chunk_start:
        raise RuntimeError(
            "NEP Scatter chunk range must satisfy " f"0 <= {chunk_start} <= {chunk_end}"
        )
    if scatter_chunks <= 0:
        raise RuntimeError("NEP Scatter chunk count must be positive")
    if chunk_start == chunk_end:
        return []

    remote_segments = sorted(remote_segments)
    previous_end = chunk_start
    for segment_start, segment_end in remote_segments:
        if (
            segment_start < previous_end
            or segment_start < chunk_start
            or segment_end < segment_start
            or segment_end > chunk_end
        ):
            raise RuntimeError(
                "NEP Scatter remote segments must be ordered, disjoint, and "
                "contained in the owner-layout chunk"
            )
        previous_end = segment_end

    remote_numel = sum(end - start for start, end in remote_segments)
    num_chunks = min(scatter_chunks, remote_numel)
    if num_chunks <= 1:
        return [(chunk_start, chunk_end)]

    boundaries = [chunk_start]
    segment_index = 0
    cumulative_before_segment = 0
    for partition_index in range(1, num_chunks):
        target = (remote_numel * partition_index + num_chunks - 1) // num_chunks
        while (
            cumulative_before_segment
            + remote_segments[segment_index][1]
            - remote_segments[segment_index][0]
            < target
        ):
            cumulative_before_segment += (
                remote_segments[segment_index][1] - remote_segments[segment_index][0]
            )
            segment_index += 1
        segment_start, _ = remote_segments[segment_index]
        boundaries.append(segment_start + target - cumulative_before_segment)
    boundaries.append(chunk_end)
    return list(zip(boundaries, boundaries[1:]))


def _build_nep_nccl_chunk_ranges(
    owner_numel: int, max_chunk_numel: int, target_chunks: Optional[int]
) -> List[Tuple[int, int]]:
    """Partition one owner payload without changing the one-chunk task."""
    if owner_numel == 0:
        return []
    byte_cap_chunks = (owner_numel + max_chunk_numel - 1) // max_chunk_numel
    num_chunks = max(byte_cap_chunks, target_chunks or 1)
    chunk_numel = (owner_numel + num_chunks - 1) // num_chunks
    return [
        (start, min(start + chunk_numel, owner_numel))
        for start in range(0, owner_numel, chunk_numel)
    ]


def _get_nep_nccl_async_chunk_window() -> int:
    value = os.getenv("MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW")
    if value is None:
        return _NEP_NCCL_DEFAULT_ASYNC_CHUNK_WINDOW
    chunk_window = int(value)
    if chunk_window <= 0:
        raise RuntimeError("MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW must be positive")
    return chunk_window




def _get_nep_nccl_expert_bucket_group_count() -> int:
    value = os.getenv("MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS")
    if value is None:
        return _NEP_NCCL_DEFAULT_EXPERT_BUCKET_GROUPS
    group_count = int(value)
    if group_count <= 0:
        raise RuntimeError("MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS must be positive")
    return group_count






def _nep_block_current_stream(work) -> None:
    """Order later CUDA work after a NCCL work item without a CPU wait when supported."""
    if work is None:
        return
    block_current_stream = getattr(work, "block_current_stream", None)
    if block_current_stream is not None:
        block_current_stream()
    else:
        # Older torch Work objects may not expose block_current_stream. This keeps the
        # path correct, but removes overlap on those builds.
        work.wait()




@dataclass
class NonuniformEPConfig:
    """Configuration for nonuniform EP gradient ownership transfer."""

    runtime_config: Optional[dict] = None
    expert_name_pattern: Union[str, re.Pattern] = field(
        default_factory=_default_expert_name_pattern
    )

    def __post_init__(self):
        if isinstance(self.expert_name_pattern, str):
            self.expert_name_pattern = re.compile(self.expert_name_pattern)


@dataclass
class _ExpertBucketSpec:
    buffer: object
    source_bucket_index: int
    expert_id: int
    params: List[torch.nn.Parameter]
    start: int
    end: int
    slot_key: Tuple[str, ...]
    synthetic_owner: bool = False


def _copy_nep_optimizer_parameter_attributes(
    source: torch.nn.Parameter, target: torch.nn.Parameter
) -> None:
    """Copy optimizer- and model-parallel metadata to an owner proxy parameter."""
    attribute_names = (
        "allreduce",
        "expert_model_parallel",
        "is_embedding_or_output_parameter",
        "is_embedding_parameter",
        "partition_dim",
        "partition_stride",
        "sequence_parallel",
        "shared",
        "shared_embedding",
        "tensor_model_parallel",
    )
    for attribute_name in attribute_names:
        if hasattr(source, attribute_name):
            setattr(target, attribute_name, getattr(source, attribute_name))
    # Owner proxies always belong to the expert optimizer.
    target.allreduce = False


def _compute_nep_distopt_owner_layout(
    params: List[torch.nn.Parameter],
    data_parallel_world_size: int,
    ddp_config: DistributedDataParallelConfig,
) -> PerBufferParamLayout:
    """Build one native-compatible distributed-optimizer bucket for owner params."""
    if not params:
        raise RuntimeError("NEP owner DistOpt layout requires at least one logical parameter")

    param_index_map = {}
    param_end_index = 0
    for param in params[::-1]:
        param_start_index = pad_param_start(param_end_index)
        param_end_index = param_start_index + param.numel()
        param_index_map[param] = (param_start_index, param_end_index, 0)

    bucket_end_index = pad_bucket_end(
        param_end_index, data_parallel_world_size, ddp_config.pad_buckets_for_high_nccl_busbw
    )
    return PerBufferParamLayout(
        param_index_map=param_index_map,
        bucket_indices=[(0, bucket_end_index)],
        per_bucket_numel_unpadded=[param_end_index],
        param_indices=list(range(len(params))),
    )


def _nep_distopt_proxy_name(slot_key: Tuple[str, ...], expert_id: int) -> str:
    """Return a stable logical-expert name for one owner proxy parameter."""
    if len(slot_key) != 1:
        raise RuntimeError(f"NEP DistOpt expects one name per parameter slot; got {slot_key}")
    return slot_key[0].replace("{expert}", str(expert_id))


def _source_ep_ranks_for_owner(
    expert_placement: List[List[int]], owner_ep_rank: int, num_experts: int, min_ep_size: int
) -> List[int]:
    """Return EP ranks that physically hold one owner's logical experts."""
    owner_slots = compute_nonuniform_ep_owner_expert_slots(num_experts, min_ep_size)
    owner_expert_ids = {
        expert_id for expert_id in owner_slots[owner_ep_rank] if expert_id is not None
    }
    return [
        source_ep_rank
        for source_ep_rank, expert_ids in enumerate(expert_placement)
        if owner_expert_ids.intersection(expert_ids)
    ]


def _runtime_config_from_parallel_state() -> dict:
    runtime_config = get_nonuniform_ep_runtime_config()
    if runtime_config is not None:
        return dict(runtime_config)

    if hasattr(parallel_state, "is_nonuniform_ep") and parallel_state.is_nonuniform_ep():
        return dict(parallel_state.get_nonuniform_ep_config())

    ep_group = parallel_state.get_expert_model_parallel_group(check_initialized=False)
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    local_ep_size = parallel_state.get_expert_model_parallel_world_size()
    local_ep_size = max(1, local_ep_size)
    return {
        "needs_reshard": False,
        "local_ep_size": local_ep_size,
        "min_ep_size": local_ep_size,
        "num_replicas": 1,
        "dp_size": 1,
        "ep_group": ep_group if ep_group is not None else dist.group.WORLD,
        "nep_owner_gather_groups": {},
        "nep_owner_transfer_groups": {},
        "nep_owner_transfer_group_ranks": {},
        "nep_owner_source_ranks": {},
        "edp_group": None,
        "ep_rank": ep_rank,
        "local_expert_indices": None,
        "expert_placement": None,
    }


def _get_runtime_config(config: NonuniformEPConfig) -> dict:
    if config.runtime_config is not None:
        return dict(config.runtime_config)
    return _runtime_config_from_parallel_state()




def _set_parallel_state_attr(name: str, value) -> None:
    setattr(parallel_state, name, value)


def _get_nccl_communicator_configs(path: Optional[str]) -> dict:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import `yaml`. Setting custom NCCL communicator configs "
            "requires the yaml package."
        ) from exc
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _create_group(ranks, timeout, nccl_comm_cfgs, desc, backend=None):
    pg_options = (
        None if backend == "gloo" else parallel_state.get_nccl_options(desc, nccl_comm_cfgs)
    )
    group = parallel_state.create_group(
        ranks, timeout=timeout, backend=backend, pg_options=pg_options, group_desc=desc
    )
    return group


def initialize_nonuniform_ep_process_groups(
    tensor_model_parallel_size: int,
    context_parallel_size: int,
    num_tp_cp_per_replica: List[int],
    expert_tensor_parallel_size: Optional[int] = None,
    num_moe_experts: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    create_gloo_process_groups: bool = True,
    get_embedding_ranks: Optional[Callable[[List[int]], List[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[List[int]], List[int]]] = None,
) -> dict:
    """Initialize opt-in nonuniform EP process groups and runtime metadata.

    This is the shared-branch equivalent of the older nonuniform EP rank
    generator path, but it lives with the NEP opt-in implementation. It supports
    PP=1, non-distributed optimizer runs where attention uses a uniform
    TP/CP/DP layout and expert groups vary by replica size.
    """
    if num_moe_experts is None:
        raise RuntimeError("num_moe_experts is required for nonuniform EP placement")
    if get_embedding_ranks is None:
        get_embedding_ranks = parallel_state.default_embedding_ranks
    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = parallel_state.default_position_embedding_ranks

    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    tp = tensor_model_parallel_size
    cp = context_parallel_size
    etp = expert_tensor_parallel_size if expert_tensor_parallel_size is not None else tp

    generator = NonuniformEPRankGenerator(
        tp=tp, cp=cp, num_tp_cp_per_replica=num_tp_cp_per_replica, etp=etp
    )
    if generator.world_size != world_size:
        raise RuntimeError(
            f"NonuniformEPRankGenerator world_size ({generator.world_size}) != "
            f"distributed world_size ({world_size}). Expected "
            "sum(num_tp_cp_per_replica) * TP * CP ranks."
        )

    for replica_index, num_tp_cp in enumerate(num_tp_cp_per_replica):
        local_ep_size = num_tp_cp * tp * cp // etp
        if num_moe_experts < local_ep_size:
            raise RuntimeError(
                "Nonuniform EP currently requires at least one logical expert per local EP "
                f"rank; replica {replica_index} has num_moe_experts={num_moe_experts}, "
                f"local_ep_size={local_ep_size}"
            )

    timeout = timedelta(minutes=distributed_timeout_minutes)
    nccl_comm_cfgs = _get_nccl_communicator_configs(nccl_communicator_config_path)
    # Attention/data groups.
    for ranks in generator.get_ranks("dp-cp"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "dp_cp")
        group_gloo = (
            _create_group(
                ranks, timeout, nccl_comm_cfgs, "DATA_PARALLEL_GROUP_WITH_CP_GLOO", "gloo"
            )
            if create_gloo_process_groups
            else None
        )
        if rank in ranks:
            _set_parallel_state_attr("_DATA_PARALLEL_GROUP_WITH_CP", group)
            _set_parallel_state_attr("_DATA_PARALLEL_GROUP_WITH_CP_GLOO", group_gloo)
            _set_parallel_state_attr("_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP", ranks)
            _set_parallel_state_attr("_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP", group)
            _set_parallel_state_attr("_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO", group_gloo)

    for ranks in generator.get_ranks("dp"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "dp")
        group_gloo = (
            _create_group(ranks, timeout, nccl_comm_cfgs, "DATA_PARALLEL_GROUP_GLOO", "gloo")
            if create_gloo_process_groups
            else None
        )
        if rank in ranks:
            _set_parallel_state_attr("_DATA_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_DATA_PARALLEL_GROUP_GLOO", group_gloo)
            _set_parallel_state_attr("_DATA_PARALLEL_GLOBAL_RANKS", ranks)

    for ranks in generator.get_ranks("cp"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "cp")
        if rank in ranks:
            _set_parallel_state_attr("_CONTEXT_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_CONTEXT_PARALLEL_GLOBAL_RANKS", ranks)

    for ranks in generator.get_ranks("tp"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp")
        if rank in ranks:
            _set_parallel_state_attr("_TENSOR_MODEL_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS", ranks)

    for ranks in generator.get_ranks("tp"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "mp")
        if rank in ranks:
            _set_parallel_state_attr("_MODEL_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_MODEL_PARALLEL_GLOBAL_RANKS", ranks)

    for ranks in [[global_rank] for global_rank in range(world_size)]:
        pp_group = _create_group(ranks, timeout, nccl_comm_cfgs, "pp")
        if rank in ranks:
            _set_parallel_state_attr("_PIPELINE_MODEL_PARALLEL_GROUP", pp_group)
            _set_parallel_state_attr("_PIPELINE_GLOBAL_RANKS", ranks)

        embedding_ranks = get_embedding_ranks(ranks)
        embedding_group = _create_group(embedding_ranks, timeout, nccl_comm_cfgs, "embd")
        if rank in embedding_ranks:
            _set_parallel_state_attr("_EMBEDDING_GROUP", embedding_group)
            _set_parallel_state_attr("_EMBEDDING_GLOBAL_RANKS", embedding_ranks)

        position_embedding_ranks = get_position_embedding_ranks(ranks)
        position_embedding_group = _create_group(
            position_embedding_ranks, timeout, nccl_comm_cfgs, "pos_embd"
        )
        if rank in position_embedding_ranks:
            _set_parallel_state_attr("_POSITION_EMBEDDING_GROUP", position_embedding_group)
            _set_parallel_state_attr("_POSITION_EMBEDDING_GLOBAL_RANKS", position_embedding_ranks)

    for ranks in generator.get_ranks("tp-dp-cp"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp_dp_cp")
        if rank in ranks:
            _set_parallel_state_attr("_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP", group)

    for ranks in generator.get_ranks("tp-dp"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp_dp")
        if rank in ranks:
            _set_parallel_state_attr("_TENSOR_AND_DATA_PARALLEL_GROUP", group)

    for ranks in generator.get_ranks("tp-cp"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp_cp")
        if rank in ranks:
            _set_parallel_state_attr("_TENSOR_AND_CONTEXT_PARALLEL_GROUP", group)

    # Expert groups.
    min_ep_size = generator.min_k * tp * cp // etp
    nep_owner_gather_groups = {}
    nep_owner_transfer_groups = {}
    nep_owner_transfer_group_ranks = {}
    nep_owner_source_ranks = {}
    for ranks in generator.get_ranks("ep"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "ep")
        group_expert_placement, _ = compute_nonuniform_ep_expert_placement(
            num_moe_experts,
            len(ranks),
            min_ep_size,
            preferred_follower_fanout=1,
        )
        source_ranks_by_owner = {
            owner_ep_rank: _source_ep_ranks_for_owner(
                group_expert_placement, owner_ep_rank, num_moe_experts, min_ep_size
            )
            for owner_ep_rank in range(min_ep_size)
        }
        transfer_ranks_by_owner = {
            owner_ep_rank: [owner_ep_rank]
            + [source_rank for source_rank in source_ranks if source_rank != owner_ep_rank]
            for owner_ep_rank, source_ranks in source_ranks_by_owner.items()
        }
        for owner_ep_rank in range(min_ep_size):
            source_ep_ranks = source_ranks_by_owner[owner_ep_rank]
            transfer_ep_ranks = transfer_ranks_by_owner[owner_ep_rank]
            transfer_global_ranks = [ranks[ep_rank] for ep_rank in transfer_ep_ranks]
            owner_transfer_group = None
            owner_gather_group = None
            if len(transfer_global_ranks) > 1:
                owner_transfer_group = _create_group(
                    transfer_global_ranks,
                    timeout,
                    nccl_comm_cfgs,
                    "nep_owner_transfer",
                )
                owner_gather_group = _create_group(
                    transfer_global_ranks, timeout, nccl_comm_cfgs, "nep_owner_gather"
                )
            if rank in ranks:
                nep_owner_source_ranks[owner_ep_rank] = source_ep_ranks
                nep_owner_transfer_group_ranks[owner_ep_rank] = transfer_ep_ranks
                if rank in transfer_global_ranks:
                    if owner_gather_group is not None:
                        nep_owner_gather_groups[owner_ep_rank] = owner_gather_group
                    nep_owner_transfer_groups[owner_ep_rank] = owner_transfer_group
        if rank in ranks:
            _set_parallel_state_attr("_EXPERT_MODEL_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_EXPERT_MODEL_PARALLEL_RANKS", ranks)

    for ranks in generator.get_ranks("etp"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "ep_tp")
        if rank in ranks:
            _set_parallel_state_attr("_EXPERT_TENSOR_PARALLEL_GROUP", group)

    for ranks in generator.get_ranks("etp-ep"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp_ep_mp")
        if rank in ranks:
            _set_parallel_state_attr("_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP", group)

    for ranks in generator.get_ranks("etp-ep"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp_ep_pp")
        if rank in ranks:
            _set_parallel_state_attr("_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP", group)

    edp_groups = generator.get_ranks("edp")
    covered_edp_ranks = {covered_rank for group in edp_groups for covered_rank in group}
    all_edp_groups = edp_groups + [
        [uncovered_rank] for uncovered_rank in sorted(set(range(world_size)) - covered_edp_ranks)
    ]
    for ranks in all_edp_groups:
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "ep_dp")
        group_gloo = (
            _create_group(ranks, timeout, nccl_comm_cfgs, "EXPERT_DATA_PARALLEL_GROUP_GLOO", "gloo")
            if create_gloo_process_groups
            else None
        )
        if rank in ranks:
            _set_parallel_state_attr("_EXPERT_DATA_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_EXPERT_DATA_PARALLEL_GROUP_GLOO", group_gloo)
            _set_parallel_state_attr("_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO", group_gloo)
            _set_parallel_state_attr("_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP", None)

    _set_parallel_state_attr("_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP", dist.group.WORLD)
    parallel_state._set_global_memory_buffer()

    replica_ep_sizes = [num_tp_cp * tp * cp // etp for num_tp_cp in num_tp_cp_per_replica]
    has_nondivisible_expert_placement = any(
        num_moe_experts % replica_ep_size != 0 for replica_ep_size in replica_ep_sizes
    )

    local_ep_size = None
    for replica_index, num_tp_cp in enumerate(num_tp_cp_per_replica):
        replica_start = generator.replica_offsets[replica_index]
        replica_end = generator.replica_offsets[replica_index + 1]
        if replica_start <= rank < replica_end:
            local_ep_size = num_tp_cp * tp * cp // etp
            break
    if local_ep_size is None:
        raise RuntimeError(f"Rank {rank} is not in any nonuniform EP replica")

    ep_group = parallel_state.get_expert_model_parallel_group()
    ep_rank = ep_group.rank()
    expert_placement, expert_gather_map = compute_nonuniform_ep_expert_placement(
        num_moe_experts,
        local_ep_size,
        min_ep_size,
        preferred_follower_fanout=1,
    )
    owner_expert_slots = compute_nonuniform_ep_owner_expert_slots(num_moe_experts, min_ep_size)
    expert_to_owner = {}
    expert_to_owner_slot = {}
    for owner_ep_rank, owner_slots in enumerate(owner_expert_slots):
        for owner_slot, expert_id in enumerate(owner_slots):
            if expert_id is None:
                continue
            expert_to_owner[expert_id] = owner_ep_rank
            expert_to_owner_slot[expert_id] = owner_slot
    dispatch_expert_slots = compute_nonuniform_ep_dispatch_slots(expert_placement, num_moe_experts)
    runtime_config = {
        "needs_reshard": local_ep_size > min_ep_size,
        "local_ep_size": local_ep_size,
        "min_ep_size": min_ep_size,
        "num_replicas": generator.num_replicas,
        "dp_size": sum(num_tp_cp_per_replica),
        "ep_group": ep_group,
        "nep_owner_gather_groups": nep_owner_gather_groups,
        "nep_owner_transfer_groups": nep_owner_transfer_groups,
        "nep_owner_transfer_group_ranks": nep_owner_transfer_group_ranks,
        "nep_owner_source_ranks": nep_owner_source_ranks,
        "edp_group": parallel_state.get_expert_data_parallel_group(),
        "ep_rank": ep_rank,
        "is_edp_eligible": ep_rank < min_ep_size,
        "is_b_leader": ep_rank < min_ep_size,
        "local_expert_indices": expert_placement[ep_rank],
        "expert_placement": expert_placement,
        "dispatch_expert_slots": dispatch_expert_slots,
        "owner_expert_slots": owner_expert_slots,
        "has_nondivisible_expert_placement": has_nondivisible_expert_placement,
        "expert_to_owner": expert_to_owner,
        "expert_to_owner_slot": expert_to_owner_slot,
        "expert_gather_map": expert_gather_map,
    }
    set_nonuniform_ep_runtime_config(runtime_config)
    return runtime_config


def _local_expert_id_from_name(
    name: str, pattern: re.Pattern, local_expert_indices: Optional[List[int]]
) -> Optional[int]:
    match = pattern.search(name)
    if match is None:
        return None
    local_idx = None
    for group_index in range(1, len(match.groups()) + 1):
        if match.group(group_index) is not None:
            local_idx = int(match.group(group_index))
            break
    if local_idx is None:
        raise RuntimeError(f"NEP expert name pattern matched without a capture group: {name}")
    if local_expert_indices is None:
        return local_idx
    if local_idx >= len(local_expert_indices):
        raise RuntimeError(f"Local expert index {local_idx} is out of range for {name}")
    return int(local_expert_indices[local_idx])


def _expert_slot_key_from_name(name: str, pattern: re.Pattern) -> str:
    match = pattern.search(name)
    if match is None:
        return name
    for group_index in range(1, len(match.groups()) + 1):
        if match.group(group_index) is not None:
            start, end = match.span(group_index)
            return f"{name[:start]}{{expert}}{name[end:]}"
    return name


class NonuniformEPNCCLParamAndGradBucketGroup(_ParamAndGradBucketGroup):
    """Approach A: NCCL owner-layout reshard/allreduce/reshard for nonuniform EP."""

    def configure_nonuniform_ep_nccl(
        self,
        runtime_config: dict,
        nonuniform_ep_config: NonuniformEPConfig,
        entries: Optional[List[dict]] = None,
        slot_key: Optional[Tuple[str, ...]] = None,
        slot_numel: Optional[int] = None,
        slot_keys: Optional[Tuple[Tuple[str, ...], ...]] = None,
        slot_numels: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self._nep_runtime_config = runtime_config
        self._nep_config = nonuniform_ep_config
        self._nep_nccl_entries = entries or []
        if slot_keys is None:
            slot_keys = (slot_key,) if slot_key is not None else ()
        if slot_numels is None:
            slot_numels = (slot_numel,) if slot_numel is not None else ()
        if len(slot_keys) != len(slot_numels):
            raise RuntimeError("NEP expert bucket metadata must have matching slot keys and sizes")
        self._nep_nccl_slot_keys = tuple(slot_keys)
        self._nep_nccl_slot_numels = tuple(slot_numels)
        self._nep_nccl_slot_offsets = tuple(
            sum(self._nep_nccl_slot_numels[:slot_index])
            for slot_index in range(len(self._nep_nccl_slot_numels))
        )
        self._nep_nccl_expert_stride = sum(self._nep_nccl_slot_numels)
        self._nep_nccl_slot_key = (
            self._nep_nccl_slot_keys[0]
            if len(self._nep_nccl_slot_keys) == 1
            else self._nep_nccl_slot_keys
        )
        self._nep_nccl_slot_numel = (
            self._nep_nccl_slot_numels[0]
            if len(self._nep_nccl_slot_numels) == 1
            else self._nep_nccl_expert_stride
        )
        self._nep_nccl_grad_sync_started = False
        self._nep_nccl_ready = len(self.params) == 0
        self._nep_nccl_bucket_numels_cache = {}
        self._nep_nccl_async_handles = []
        self._nep_nccl_async_tensors = []
        self._nep_nccl_streams = {}
        self._nep_nccl_logical_grad_data_cache = {}
        self._nep_nccl_send_chunk_cache = {}
        self._nep_nccl_gather_buf_cache = {}
        self._nep_nccl_gather_list_cache = {}
        self._nep_nccl_buffer_state = {}
        self._nep_nccl_segment_cache = {}
        self._nep_nccl_tensor_view_cache = {}
        self._nep_nccl_native_edp_bucket_groups = {}
        self._nep_nccl_active_native_edp_states = []
        self._nep_nccl_entry_by_key = {
            entry.get("entry_key", (entry["expert_id"], entry.get("slot_index", 0))): entry
            for entry in self._nep_nccl_entries
        }
        self._nep_nccl_owner_layout = None
        self._nep_nccl_started_tasks = set()
        self._nep_nccl_prepped_experts = set()
        self._nep_nccl_task_count = 0
        self._nep_dispatch_boundary_launch = False
        self._nep_dispatch_boundary_ready = False
        self._nep_dispatch_boundary_launched = False
        self._nep_dispatch_boundary_launching = False
        self._nep_dispatch_boundary_wait_logged = False
        self._nep_dispatch_boundary_callback = None
        self._nep_dispatch_boundary_groups = ()
        self._nep_dispatch_boundary_module_label = None

    def _get_nep_nccl_shared_buffer_state(self) -> dict:
        state = getattr(self, "_nep_nccl_scheduler_state", None)
        if state is None:
            state = self._nep_nccl_buffer_state
        state.setdefault("gather_buf_cache", {})
        state.setdefault("buffer_slot_handles", {})
        state.setdefault("buffer_slot_events", {})
        return state

    def _order_nep_nccl_buffer_slot(self, slot_key: tuple) -> None:
        """Order slot reuse after prior NCCL work without blocking the host thread."""
        state = self._get_nep_nccl_shared_buffer_state()
        slot_handles = state["buffer_slot_handles"]
        handles = slot_handles.pop(slot_key, [])
        for work in handles:
            _nep_block_current_stream(work)

        for event in state["buffer_slot_events"].pop(slot_key, []):
            torch.cuda.current_stream().wait_event(event)

    def _record_nep_nccl_work(
        self, work, buffer_slot_key: Optional[tuple] = None, block_current_stream: bool = True
    ) -> None:
        if work is None:
            return
        self._nep_nccl_async_handles.append(work)
        if buffer_slot_key is not None:
            state = self._get_nep_nccl_shared_buffer_state()
            state["buffer_slot_handles"].setdefault(buffer_slot_key, []).append(work)
        if block_current_stream:
            _nep_block_current_stream(work)












    def _drain_nep_nccl_async_window(self, force_all: bool = False) -> None:
        if not self._nep_nccl_async_handles:
            return

        if force_all:
            drain_count = len(self._nep_nccl_async_handles)
        else:
            chunk_window = _get_nep_nccl_async_chunk_window()
            drain_count = max(0, len(self._nep_nccl_async_handles) - chunk_window)
        if drain_count == 0:
            return

        for work in self._nep_nccl_async_handles[:drain_count]:
            work.wait()
        del self._nep_nccl_async_handles[:drain_count]

    def _get_nep_nccl_cached_tensor(
        self, cache: dict, key: tuple, numel: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        tensor = cache.get(key)
        if (
            tensor is None
            or tensor.numel() != numel
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            tensor = torch.empty(numel, dtype=dtype, device=device)
            cache[key] = tensor
        return tensor

    def _get_nep_nccl_comm_stream(self, stream_slot: int) -> torch.cuda.Stream:
        state = getattr(self, "_nep_nccl_scheduler_state", None)
        if self._nep_dispatch_boundary_launch and not self.is_first_batch:
            stream_key = "dispatch"
        else:
            stream_key = ("end_iteration", stream_slot % _get_nep_nccl_async_chunk_window())
        if state is not None:
            streams = state.setdefault("comm_streams", {})
            stream = streams.get(stream_key)
            if stream is None:
                stream = torch.cuda.Stream(device=torch.cuda.current_device())
                streams[stream_key] = stream
            self._nep_nccl_streams[stream_key] = stream
            return stream

        stream = self._nep_nccl_streams.get(stream_key)
        if stream is None:
            stream = torch.cuda.Stream(device=torch.cuda.current_device())
            self._nep_nccl_streams[stream_key] = stream
        return stream

    def _get_nep_nccl_ordered_edp_stream(self) -> torch.cuda.Stream:
        """Return the shared stream that preserves native EDP bucket order."""
        stream_key = "edp"
        state = getattr(self, "_nep_nccl_scheduler_state", None)
        if state is not None:
            streams = state.setdefault("comm_streams", {})
            stream = streams.get(stream_key)
            if stream is None:
                stream = torch.cuda.Stream(device=torch.cuda.current_device())
                streams[stream_key] = stream
            self._nep_nccl_streams[stream_key] = stream
            return stream

        stream = self._nep_nccl_streams.get(stream_key)
        if stream is None:
            stream = torch.cuda.Stream(device=torch.cuda.current_device())
            self._nep_nccl_streams[stream_key] = stream
        return stream

    def _get_nep_nccl_task_buffer_slot(self, owner_ep_rank: int, chunk_index: int) -> int:
        layout = self._get_nep_nccl_owner_layout()
        group_index = max(0, getattr(self, "_nep_nccl_group_index", 0))
        state = getattr(self, "_nep_nccl_scheduler_state", None)
        group_slot_offsets = None if state is None else state.get("group_slot_offsets")
        if group_slot_offsets is None:
            raise RuntimeError("End-of-iteration NEP slots are not configured")
        return (
            group_slot_offsets[group_index]
            + owner_ep_rank * max(1, layout["num_chunks"])
            + chunk_index
        )

    def _flush_nep_nccl_pending_scatters(
        self, buffer_slot: Optional[int] = None, force_all: bool = False
    ) -> None:
        state = getattr(self, "_nep_nccl_scheduler_state", None)
        if state is None:
            return
        pending_scatters = state.setdefault("pending_scatters", [])
        remaining_scatters = []
        for context in pending_scatters:
            if not force_all and context["buffer_slot"] != buffer_slot:
                remaining_scatters.append(context)
                continue
            group = context["group"]
            nccl_stream = group._get_nep_nccl_comm_stream(context["buffer_slot"])
            with torch.cuda.stream(nccl_stream):
                group._start_nep_nccl_owner_task_scatter(context)
        state["pending_scatters"] = remaining_scatters

    def _get_nep_nccl_owner_layout(self) -> dict:
        """Return cached owner-layout metadata for this expert slot bucket group."""
        if self._nep_nccl_owner_layout is not None:
            return self._nep_nccl_owner_layout

        cfg = self._nep_runtime_config
        local_ep_size = cfg["local_ep_size"]
        ep_rank = cfg["ep_rank"]
        min_ep_size = cfg.get("min_ep_size", local_ep_size)
        if min_ep_size < 2:
            raise RuntimeError(
                "NEP NCCL Approach A requires at least two owner EP ranks "
                f"(min_ep_size >= 2); got min_ep_size={min_ep_size}."
            )

        placement = cfg.get("expert_placement")
        if placement is None:
            num_experts = local_ep_size
        else:
            num_experts = sum(len(experts) for experts in placement)
        owner_expert_slots = cfg.get("owner_expert_slots")
        if owner_expert_slots is None:
            owner_expert_slots = compute_nonuniform_ep_owner_expert_slots(num_experts, min_ep_size)
        if len(owner_expert_slots) != min_ep_size:
            raise RuntimeError(
                "NEP owner-slot row count must equal min_ep_size: "
                f"got {len(owner_expert_slots)} and {min_ep_size}"
            )
        experts_per_owner = len(owner_expert_slots[0])
        if any(len(slots) != experts_per_owner for slots in owner_expert_slots):
            raise RuntimeError("NEP owner-layout communication requires fixed-width owner slots")
        expert_stride = getattr(self, "_nep_nccl_expert_stride", None)
        if expert_stride is None:
            expert_stride = self._nep_nccl_slot_numel
        if expert_stride is None:
            raise RuntimeError("NEP NCCL bucket group is missing slot-size metadata")

        owner_numel = experts_per_owner * expert_stride
        target_chunks = _get_nep_nccl_target_chunks()
        max_gather_bytes = _get_nep_nccl_max_gather_bytes()
        byte_cap_numel = max(1, max_gather_bytes // self.buckets[0].grad_data.element_size())
        chunk_ranges = _build_nep_nccl_chunk_ranges(owner_numel, byte_cap_numel, target_chunks)
        num_chunks = len(chunk_ranges)
        max_chunk_numel = max((end - start for start, end in chunk_ranges), default=0)

        self._nep_nccl_experts_per_owner = experts_per_owner
        self._nep_nccl_owner_layout = {
            "ep_rank": ep_rank,
            "local_ep_size": local_ep_size,
            "min_ep_size": min_ep_size,
            "num_experts": num_experts,
            "experts_per_owner": experts_per_owner,
            "owner_expert_slots": owner_expert_slots,
            "expert_stride": expert_stride,
            "owner_numel": owner_numel,
            "target_chunks": target_chunks,
            "chunk_ranges": chunk_ranges,
            "max_chunk_numel": max_chunk_numel,
            "num_chunks": num_chunks,
        }
        return self._nep_nccl_owner_layout

    def _get_nep_nccl_transfer_group_info(
        self, owner_ep_rank: int, group_key: str = "nep_owner_transfer_groups"
    ) -> tuple:
        """Return the owner-source communicator used for NEP reshard all-to-alls."""
        cfg = self._nep_runtime_config
        transfer_ranks = self._nep_nccl_owner_transfer_ranks(owner_ep_rank)
        ep_rank = cfg["ep_rank"]
        if ep_rank not in transfer_ranks:
            return None, -1, len(transfer_ranks), transfer_ranks
        if len(transfer_ranks) <= 1:
            return None, 0, len(transfer_ranks), transfer_ranks

        transfer_group = cfg.get(group_key, {}).get(owner_ep_rank)
        if transfer_group is None:
            raise RuntimeError(
                f"Missing NEP owner group {group_key} for owner {owner_ep_rank} "
                f"with transfer EP ranks {transfer_ranks}"
            )
        transfer_rank = dist.get_rank(group=transfer_group)
        transfer_size = dist.get_world_size(group=transfer_group)
        if transfer_size != len(transfer_ranks):
            raise RuntimeError(
                "NEP owner transfer group size must match transfer-rank count; got "
                f"transfer_size={transfer_size}, transfer_ranks={transfer_ranks}"
            )
        return transfer_group, transfer_rank, transfer_size, transfer_ranks

    def _nep_nccl_owner_expert_ids(self, owner_ep_rank: int) -> List[int]:
        """Return logical experts assigned to one fixed-width owner row."""
        layout = self._get_nep_nccl_owner_layout()
        return [
            expert_id
            for expert_id in layout["owner_expert_slots"][owner_ep_rank]
            if expert_id is not None
        ]

    def _nep_nccl_owner_slot_for_expert(self, expert_id: int) -> Tuple[int, int]:
        """Return ``(owner rank, owner-local physical slot)`` for a logical expert."""
        cfg = self._nep_runtime_config
        expert_to_owner = cfg.get("expert_to_owner")
        expert_to_owner_slot = cfg.get("expert_to_owner_slot")
        if expert_to_owner is not None and expert_to_owner_slot is not None:
            try:
                return int(expert_to_owner[expert_id]), int(expert_to_owner_slot[expert_id])
            except KeyError as exc:
                raise RuntimeError(f"NEP has no owner slot for expert {expert_id}") from exc

        layout = self._get_nep_nccl_owner_layout()
        for owner_ep_rank, owner_slots in enumerate(layout["owner_expert_slots"]):
            if expert_id in owner_slots:
                return owner_ep_rank, owner_slots.index(expert_id)
        raise RuntimeError(f"NEP has no owner slot for expert {expert_id}")

    def _nep_nccl_owner_entries(self, owner_ep_rank: int) -> List[dict]:
        """Return local expert-slot entries that contribute to an owner-layout chunk."""
        owner_expert_ids = set(self._nep_nccl_owner_expert_ids(owner_ep_rank))
        return [entry for entry in self._nep_nccl_entries if entry["expert_id"] in owner_expert_ids]

    @staticmethod
    def _nep_nccl_entry_key(entry: dict) -> tuple:
        return entry.get("entry_key", (entry["expert_id"], entry.get("slot_index", 0)))

    def _nep_nccl_entry_owner_start(self, entry: dict, owner_ep_rank: int) -> int:
        expert_stride = getattr(self, "_nep_nccl_expert_stride", self._nep_nccl_slot_numel)
        mapped_owner, owner_slot = self._nep_nccl_owner_slot_for_expert(entry["expert_id"])
        if mapped_owner != owner_ep_rank:
            raise RuntimeError(
                f"Expert {entry['expert_id']} belongs to owner {mapped_owner}, "
                f"not requested owner {owner_ep_rank}"
            )
        return owner_slot * expert_stride + entry.get("slot_offset", 0)

    def _nep_nccl_owner_task_ready(
        self, owner_ep_rank: int, respect_dispatch_boundary: bool = True
    ) -> bool:
        """Return True when this rank's local inputs for an owner task are ready."""
        if self.is_first_batch:
            return False
        if (
            respect_dispatch_boundary
            and self._nep_dispatch_boundary_launch
            and not self._nep_dispatch_boundary_ready
        ):
            return False
        for entry in self._nep_nccl_owner_entries(owner_ep_rank):
            for param in entry["bucket"].params_list:
                ready_count = self.per_param_grad_ready_counts.get(param, 0)
                expected_count = self.golden_per_param_grad_ready_counts.get(param)
                if expected_count is None or ready_count < expected_count:
                    return False
        return True

    def _nep_dispatch_boundary_inputs_ready(self) -> bool:
        """Return True when every local source needed by this group is ready."""
        layout = self._get_nep_nccl_owner_layout()
        return all(
            self._nep_nccl_owner_task_ready(owner_ep_rank, respect_dispatch_boundary=False)
            for owner_ep_rank in range(layout["min_ep_size"])
        )

    def _prep_nep_nccl_owner_entries_for_sync(self, owner_ep_rank: int) -> None:
        """Copy locally accumulated grads into their communication-buffer views."""
        copy_destinations = []
        copy_sources = []
        for entry in self._nep_nccl_owner_entries(owner_ep_rank):
            entry_key = self._nep_nccl_entry_key(entry)
            if entry_key in self._nep_nccl_prepped_experts:
                continue
            bucket = entry["bucket"]
            for param in bucket.params_with_extra_main_grads:
                if getattr(param, "main_grad_copy_in_grad_buffer", None) is not None:
                    copy_destinations.append(param.main_grad_copy_in_grad_buffer)
                    copy_sources.append(param.main_grad)
            self._nep_nccl_prepped_experts.add(entry_key)

        self._foreach_copy_(copy_destinations, copy_sources)

    def _pack_nep_nccl_owner_chunk(
        self, owner_ep_rank: int, chunk_start: int, chunk_end: int, chunk: torch.Tensor
    ) -> None:
        """Pack local source grads into a common owner-rank layout chunk."""

        def build_views():
            destinations = []
            sources = []
            for entry in self._nep_nccl_owner_entries(owner_ep_rank):
                entry_start = self._nep_nccl_entry_owner_start(entry, owner_ep_rank)
                entry_end = entry_start + entry["numel"]
                overlap_start = max(chunk_start, entry_start)
                overlap_end = min(chunk_end, entry_end)
                if overlap_start >= overlap_end:
                    continue

                chunk_offset = overlap_start - chunk_start
                entry_offset = overlap_start - entry_start
                numel = overlap_end - overlap_start
                destinations.append(chunk[chunk_offset : chunk_offset + numel])
                sources.append(entry["bucket"].grad_data[entry_offset : entry_offset + numel])
            return destinations, sources

        chunk.zero_()
        destinations, sources = self._get_nep_nccl_cached_tensor_views(
            ("pack_owner", owner_ep_rank, chunk_start, chunk_end, chunk.data_ptr()), build_views
        )
        self._foreach_copy_(destinations, sources)

    def _nep_nccl_owner_source_segments(
        self, owner_ep_rank: int, source_ep_rank: int, chunk_start: int, chunk_end: int
    ) -> List[Tuple[tuple, int, int, int]]:
        """Map one source rank's dense payload to offsets in an owner-layout chunk."""
        cache = getattr(self, "_nep_nccl_segment_cache", None)
        if cache is None:
            cache = self._nep_nccl_segment_cache = {}
        cache_key = (owner_ep_rank, source_ep_rank, chunk_start, chunk_end)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        placement = self._nep_runtime_config.get("expert_placement")
        if placement is None:
            source_expert_ids = sorted(
                {
                    entry["expert_id"]
                    for entry in self._nep_nccl_entries
                    if source_ep_rank == owner_ep_rank
                }
            )
        else:
            source_expert_ids = placement[source_ep_rank]

        slot_numels = getattr(self, "_nep_nccl_slot_numels", (self._nep_nccl_slot_numel,))
        slot_offsets = getattr(self, "_nep_nccl_slot_offsets", (0,))
        expert_stride = getattr(self, "_nep_nccl_expert_stride", self._nep_nccl_slot_numel)
        owner_expert_ids = set(self._nep_nccl_owner_expert_ids(owner_ep_rank))
        segments = []
        for expert_id in source_expert_ids:
            if expert_id not in owner_expert_ids:
                continue
            mapped_owner, owner_slot = self._nep_nccl_owner_slot_for_expert(expert_id)
            if mapped_owner != owner_ep_rank:
                raise RuntimeError(
                    f"Expert {expert_id} maps to owner {mapped_owner}, not {owner_ep_rank}"
                )
            expert_start = owner_slot * expert_stride
            for slot_index, (slot_offset, slot_numel) in enumerate(zip(slot_offsets, slot_numels)):
                entry_start = expert_start + slot_offset
                overlap_start = max(chunk_start, entry_start)
                overlap_end = min(chunk_end, entry_start + slot_numel)
                if overlap_start >= overlap_end:
                    continue
                segments.append(
                    (
                        (expert_id, slot_index),
                        overlap_start - chunk_start,
                        overlap_start - entry_start,
                        overlap_end - overlap_start,
                    )
                )
        cache[cache_key] = segments
        return segments

    def _get_nep_nccl_cached_tensor_views(self, key: tuple, build_views) -> tuple:
        cache = getattr(self, "_nep_nccl_tensor_view_cache", None)
        if cache is None:
            cache = self._nep_nccl_tensor_view_cache = {}
        views = cache.get(key)
        if views is None:
            views = build_views()
            cache[key] = views
        return views

    def _foreach_copy_(self, destinations: List[torch.Tensor], sources: List[torch.Tensor]) -> None:
        if not destinations:
            return
        for destination, source in zip(destinations, sources):
            destination.copy_(source)

    def _foreach_add_(self, destinations: List[torch.Tensor], sources: List[torch.Tensor]) -> None:
        if not destinations:
            return
        for destination, source in zip(destinations, sources):
            destination.add_(source)

    def _nep_nccl_owner_source_payload_numel(
        self, owner_ep_rank: int, source_ep_rank: int, chunk_start: int, chunk_end: int
    ) -> int:
        return sum(
            numel
            for _, _, _, numel in self._nep_nccl_owner_source_segments(
                owner_ep_rank, source_ep_rank, chunk_start, chunk_end
            )
        )

    def _nep_nccl_scatter_chunk_ranges(
        self, owner_ep_rank: int, chunk_start: int, chunk_end: int, scatter_chunks: int
    ) -> List[Tuple[int, int]]:
        """Return ranges with balanced network payload, including owner-local gaps."""
        remote_segments = []
        for source_ep_rank in self._nep_nccl_owner_source_ranks(owner_ep_rank):
            if source_ep_rank == owner_ep_rank:
                continue
            remote_segments.extend(
                (chunk_start + chunk_offset, chunk_start + chunk_offset + numel)
                for _, chunk_offset, _, numel in self._nep_nccl_owner_source_segments(
                    owner_ep_rank, source_ep_rank, chunk_start, chunk_end
                )
            )
        return _build_nep_nccl_scatter_chunk_ranges(
            chunk_start, chunk_end, remote_segments, scatter_chunks
        )

    def _pack_nep_nccl_source_payload(
        self,
        owner_ep_rank: int,
        source_ep_rank: int,
        chunk_start: int,
        chunk_end: int,
        payload: torch.Tensor,
    ) -> None:
        entry_by_key = getattr(self, "_nep_nccl_entry_by_key", None)
        if entry_by_key is None:
            entry_by_key = self._nep_nccl_entry_by_key = {
                self._nep_nccl_entry_key(entry): entry for entry in self._nep_nccl_entries
            }

        def build_views():
            destinations = []
            sources = []
            payload_offset = 0
            for entry_key, _, entry_offset, numel in self._nep_nccl_owner_source_segments(
                owner_ep_rank, source_ep_rank, chunk_start, chunk_end
            ):
                entry = entry_by_key.get(entry_key)
                if entry is None:
                    raise RuntimeError(
                        f"NEP source rank {source_ep_rank} is missing expert slot {entry_key}"
                    )
                destinations.append(payload[payload_offset : payload_offset + numel])
                sources.append(entry["bucket"].grad_data[entry_offset : entry_offset + numel])
                payload_offset += numel
            if payload_offset != payload.numel():
                raise RuntimeError(
                    f"NEP packed {payload_offset} elements into a "
                    f"{payload.numel()}-element payload"
                )
            return destinations, sources

        destinations, sources = self._get_nep_nccl_cached_tensor_views(
            (
                "pack_source",
                owner_ep_rank,
                source_ep_rank,
                chunk_start,
                chunk_end,
                payload.data_ptr(),
            ),
            build_views,
        )
        self._foreach_copy_(destinations, sources)

    def _accumulate_nep_nccl_source_payload(
        self,
        owner_ep_rank: int,
        source_ep_rank: int,
        chunk_start: int,
        chunk_end: int,
        payload: torch.Tensor,
        chunk: torch.Tensor,
    ) -> None:
        def build_views():
            destinations = []
            sources = []
            payload_offset = 0
            for _, chunk_offset, _, numel in self._nep_nccl_owner_source_segments(
                owner_ep_rank, source_ep_rank, chunk_start, chunk_end
            ):
                destinations.append(chunk[chunk_offset : chunk_offset + numel])
                sources.append(payload[payload_offset : payload_offset + numel])
                payload_offset += numel
            return destinations, sources

        destinations, sources = self._get_nep_nccl_cached_tensor_views(
            (
                "accumulate_source",
                owner_ep_rank,
                source_ep_rank,
                chunk_start,
                chunk_end,
                payload.data_ptr(),
                chunk.data_ptr(),
            ),
            build_views,
        )
        self._foreach_add_(destinations, sources)

    def _pack_nep_nccl_scatter_payload(
        self,
        owner_ep_rank: int,
        destination_ep_rank: int,
        chunk_start: int,
        chunk_end: int,
        chunk: torch.Tensor,
        payload: torch.Tensor,
    ) -> None:
        def build_views():
            destinations = []
            sources = []
            payload_offset = 0
            for _, chunk_offset, _, numel in self._nep_nccl_owner_source_segments(
                owner_ep_rank, destination_ep_rank, chunk_start, chunk_end
            ):
                destinations.append(payload[payload_offset : payload_offset + numel])
                sources.append(chunk[chunk_offset : chunk_offset + numel])
                payload_offset += numel
            return destinations, sources

        destinations, sources = self._get_nep_nccl_cached_tensor_views(
            (
                "pack_scatter",
                owner_ep_rank,
                destination_ep_rank,
                chunk_start,
                chunk_end,
                chunk.data_ptr(),
                payload.data_ptr(),
            ),
            build_views,
        )
        self._foreach_copy_(destinations, sources)

    def _copy_nep_nccl_scatter_payload_to_local_grads(
        self,
        owner_ep_rank: int,
        source_ep_rank: int,
        chunk_start: int,
        chunk_end: int,
        payload: torch.Tensor,
    ) -> None:
        entry_by_key = getattr(self, "_nep_nccl_entry_by_key", None)
        if entry_by_key is None:
            entry_by_key = self._nep_nccl_entry_by_key = {
                self._nep_nccl_entry_key(entry): entry for entry in self._nep_nccl_entries
            }

        def build_views():
            destinations = []
            sources = []
            payload_offset = 0
            for entry_key, _, entry_offset, numel in self._nep_nccl_owner_source_segments(
                owner_ep_rank, source_ep_rank, chunk_start, chunk_end
            ):
                entry = entry_by_key.get(entry_key)
                if entry is None:
                    raise RuntimeError(
                        f"NEP source rank {source_ep_rank} is missing expert slot {entry_key}"
                    )
                destinations.append(entry["bucket"].grad_data[entry_offset : entry_offset + numel])
                sources.append(payload[payload_offset : payload_offset + numel])
                payload_offset += numel
            return destinations, sources

        destinations, sources = self._get_nep_nccl_cached_tensor_views(
            (
                "copy_scatter",
                owner_ep_rank,
                source_ep_rank,
                chunk_start,
                chunk_end,
                payload.data_ptr(),
            ),
            build_views,
        )
        self._foreach_copy_(destinations, sources)

    def _copy_nep_nccl_owner_chunk_to_local_grads(
        self, owner_ep_rank: int, chunk_start: int, chunk_end: int, chunk: torch.Tensor
    ) -> None:
        """Copy a reduced common owner-rank layout chunk back to local source grads."""

        def build_views():
            destinations = []
            sources = []
            for entry in self._nep_nccl_owner_entries(owner_ep_rank):
                entry_start = self._nep_nccl_entry_owner_start(entry, owner_ep_rank)
                entry_end = entry_start + entry["numel"]
                overlap_start = max(chunk_start, entry_start)
                overlap_end = min(chunk_end, entry_end)
                if overlap_start >= overlap_end:
                    continue

                chunk_offset = overlap_start - chunk_start
                entry_offset = overlap_start - entry_start
                numel = overlap_end - overlap_start
                destinations.append(entry["bucket"].grad_data[entry_offset : entry_offset + numel])
                sources.append(chunk[chunk_offset : chunk_offset + numel])
            return destinations, sources

        destinations, sources = self._get_nep_nccl_cached_tensor_views(
            ("copy_owner", owner_ep_rank, chunk_start, chunk_end, chunk.data_ptr()), build_views
        )
        self._foreach_copy_(destinations, sources)

    def _nep_nccl_owner_source_ranks(self, owner_ep_rank: int) -> List[int]:
        """Return EP ranks that physically hold experts for an owner-layout chunk."""
        source_ranks_by_owner = self._nep_runtime_config.get("nep_owner_source_ranks")
        if source_ranks_by_owner is not None and owner_ep_rank in source_ranks_by_owner:
            return list(source_ranks_by_owner[owner_ep_rank])

        legacy_source_ranks = self._nep_runtime_config.get("nep_owner_transfer_group_ranks")
        if legacy_source_ranks is not None and owner_ep_rank in legacy_source_ranks:
            return list(legacy_source_ranks[owner_ep_rank])

        placement = self._nep_runtime_config.get("expert_placement")
        if placement is None:
            return [owner_ep_rank]

        owner_expert_ids = set(self._nep_nccl_owner_expert_ids(owner_ep_rank))
        source_ranks = []
        for source_ep_rank, expert_ids in enumerate(placement):
            if owner_expert_ids.intersection(expert_ids):
                source_ranks.append(source_ep_rank)
        return source_ranks

    def _nep_nccl_owner_transfer_ranks(self, owner_ep_rank: int) -> List[int]:
        """Return all ranks that participate in an owner's reshard communicator."""
        transfer_ranks = self._nep_runtime_config.get("nep_owner_transfer_group_ranks")
        if transfer_ranks is not None and owner_ep_rank in transfer_ranks:
            return list(transfer_ranks[owner_ep_rank])
        return self._nep_nccl_owner_source_ranks(owner_ep_rank)





    def _start_nep_nccl_owner_all_to_all_gather(
        self,
        owner_ep_rank: int,
        chunk_index: int,
        chunk_start: int,
        chunk_end: int,
        chunk: torch.Tensor,
        buffer_slot_key: tuple,
        async_op: bool,
    ) -> None:
        """Reshard source-rank expert grads into one owner-layout chunk."""
        cfg = self._nep_runtime_config
        ep_rank = cfg["ep_rank"]
        source_ranks = self._nep_nccl_owner_source_ranks(owner_ep_rank)
        transfer_ranks = self._nep_nccl_owner_transfer_ranks(owner_ep_rank)
        if ep_rank not in transfer_ranks:
            return

        remote_source_ranks = [rank for rank in source_ranks if rank != owner_ep_rank]

        if ep_rank == owner_ep_rank:
            self._pack_nep_nccl_owner_chunk(owner_ep_rank, chunk_start, chunk_end, chunk)

        if not remote_source_ranks:
            return

        gather_groups = cfg.get("nep_owner_gather_groups", {})
        gather_group_key = (
            "nep_owner_gather_groups"
            if owner_ep_rank in gather_groups
            else "nep_owner_transfer_groups"
        )
        transfer_group, _, transfer_size, transfer_source_ranks = (
            self._get_nep_nccl_transfer_group_info(owner_ep_rank, gather_group_key)
        )
        owner_transfer_rank = transfer_source_ranks.index(owner_ep_rank)
        remote_transfer_ranks = [transfer_source_ranks.index(rank) for rank in remote_source_ranks]

        cache = self._get_nep_nccl_shared_buffer_state()["gather_buf_cache"]
        empty = self._get_nep_nccl_cached_tensor(
            cache, ("empty", chunk.dtype, chunk.device), 0, chunk.dtype, chunk.device
        )

        input_split_sizes = [0] * transfer_size
        if ep_rank in remote_source_ranks:
            gather_input_numel = self._nep_nccl_owner_source_payload_numel(
                owner_ep_rank, ep_rank, chunk_start, chunk_end
            )
            input_split_sizes[owner_transfer_rank] = gather_input_numel
            gather_input = self._get_nep_nccl_cached_tensor(
                cache,
                (
                    "owner_layout_a2a_gather_input",
                    buffer_slot_key[0],
                    gather_input_numel,
                    chunk.dtype,
                    chunk.device,
                ),
                gather_input_numel,
                chunk.dtype,
                chunk.device,
            )
            self._pack_nep_nccl_source_payload(
                owner_ep_rank, ep_rank, chunk_start, chunk_end, gather_input
            )
        else:
            gather_input = empty

        output_split_sizes = [0] * transfer_size
        if ep_rank == owner_ep_rank:
            gather_output_numel = 0
            for source_transfer_rank in remote_transfer_ranks:
                source_ep_rank = transfer_source_ranks[source_transfer_rank]
                source_numel = self._nep_nccl_owner_source_payload_numel(
                    owner_ep_rank, source_ep_rank, chunk_start, chunk_end
                )
                output_split_sizes[source_transfer_rank] = source_numel
                gather_output_numel += source_numel
            gather_output = self._get_nep_nccl_cached_tensor(
                cache,
                (
                    "owner_layout_a2a_gather_output",
                    buffer_slot_key[0],
                    gather_output_numel,
                    chunk.dtype,
                    chunk.device,
                ),
                gather_output_numel,
                chunk.dtype,
                chunk.device,
            )
        else:
            gather_output = empty

        work = dist.all_to_all_single(
            gather_output,
            gather_input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=transfer_group,
            async_op=async_op,
        )
        self._record_nep_nccl_work(work, buffer_slot_key)
        if async_op:
            if gather_input.numel() > 0:
                self._nep_nccl_async_tensors.append(gather_input)
            if gather_output.numel() > 0:
                self._nep_nccl_async_tensors.append(gather_output)

        if ep_rank == owner_ep_rank:
            gather_offset = 0
            for source_ep_rank in remote_source_ranks:
                source_numel = self._nep_nccl_owner_source_payload_numel(
                    owner_ep_rank, source_ep_rank, chunk_start, chunk_end
                )
                self._accumulate_nep_nccl_source_payload(
                    owner_ep_rank,
                    source_ep_rank,
                    chunk_start,
                    chunk_end,
                    gather_output[gather_offset : gather_offset + source_numel],
                    chunk,
                )
                gather_offset += source_numel

    def _prepare_nep_nccl_owner_all_to_all_scatter(
        self,
        owner_ep_rank: int,
        chunk_index: int,
        chunk_start: int,
        chunk_end: int,
        chunk: torch.Tensor,
        buffer_slot_key: tuple,
        async_op: bool,
        scatter_chunk_index: int = 0,
    ) -> Optional[dict]:
        """Pack one owner-layout Scatter chunk without launching its collective."""
        cfg = self._nep_runtime_config
        ep_rank = cfg["ep_rank"]
        group_index = getattr(self, "_nep_nccl_group_index", -1)
        chunk_size = chunk_end - chunk_start
        source_ranks = self._nep_nccl_owner_source_ranks(owner_ep_rank)
        transfer_ranks = self._nep_nccl_owner_transfer_ranks(owner_ep_rank)
        if ep_rank not in transfer_ranks:
            return None

        remote_source_ranks = [rank for rank in source_ranks if rank != owner_ep_rank]
        transfer_group, _, transfer_size, transfer_source_ranks = (
            self._get_nep_nccl_transfer_group_info(owner_ep_rank)
        )
        owner_transfer_rank = transfer_source_ranks.index(owner_ep_rank)
        remote_transfer_ranks = [transfer_source_ranks.index(rank) for rank in remote_source_ranks]

        if not remote_source_ranks:
            return {
                "kind": "local",
                "owner_ep_rank": owner_ep_rank,
                "chunk_index": chunk_index,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "chunk": chunk,
            }

        cache = self._get_nep_nccl_shared_buffer_state()["gather_buf_cache"]
        empty = self._get_nep_nccl_cached_tensor(
            cache, ("empty", chunk.dtype, chunk.device), 0, chunk.dtype, chunk.device
        )

        input_split_sizes = [0] * transfer_size
        if ep_rank == owner_ep_rank:
            scatter_input_numel = 0
            for destination_transfer_rank in remote_transfer_ranks:
                destination_ep_rank = transfer_source_ranks[destination_transfer_rank]
                destination_numel = self._nep_nccl_owner_source_payload_numel(
                    owner_ep_rank, destination_ep_rank, chunk_start, chunk_end
                )
                input_split_sizes[destination_transfer_rank] = destination_numel
                scatter_input_numel += destination_numel
            scatter_input = self._get_nep_nccl_cached_tensor(
                cache,
                (
                    "owner_layout_a2a_scatter_input",
                    buffer_slot_key[0],
                    chunk_index,
                    scatter_chunk_index,
                    scatter_input_numel,
                    chunk.dtype,
                    chunk.device,
                ),
                scatter_input_numel,
                chunk.dtype,
                chunk.device,
            )
            scatter_offset = 0
            for destination_ep_rank in remote_source_ranks:
                destination_numel = self._nep_nccl_owner_source_payload_numel(
                    owner_ep_rank, destination_ep_rank, chunk_start, chunk_end
                )
                self._pack_nep_nccl_scatter_payload(
                    owner_ep_rank,
                    destination_ep_rank,
                    chunk_start,
                    chunk_end,
                    chunk,
                    scatter_input[scatter_offset : scatter_offset + destination_numel],
                )
                scatter_offset += destination_numel
        else:
            scatter_input = empty

        output_split_sizes = [0] * transfer_size
        if ep_rank in remote_source_ranks:
            scatter_output_numel = self._nep_nccl_owner_source_payload_numel(
                owner_ep_rank, ep_rank, chunk_start, chunk_end
            )
            output_split_sizes[owner_transfer_rank] = scatter_output_numel
            scatter_output = self._get_nep_nccl_cached_tensor(
                cache,
                (
                    "owner_layout_a2a_scatter_output",
                    buffer_slot_key[0],
                    chunk_index,
                    scatter_chunk_index,
                    scatter_output_numel,
                    chunk.dtype,
                    chunk.device,
                ),
                scatter_output_numel,
                chunk.dtype,
                chunk.device,
            )
        else:
            scatter_output = empty

        return {
            "kind": "all_to_all",
            "owner_ep_rank": owner_ep_rank,
            "chunk_index": chunk_index,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "chunk": chunk,
            "buffer_slot_key": buffer_slot_key,
            "async_op": async_op,
            "ep_rank": ep_rank,
            "group_index": group_index,
            "chunk_size": chunk_size,
            "source_ranks": source_ranks,
            "remote_source_ranks": remote_source_ranks,
            "transfer_source_ranks": transfer_source_ranks,
            "transfer_group": transfer_group,
            "scatter_input": scatter_input,
            "scatter_output": scatter_output,
            "input_split_sizes": input_split_sizes,
            "output_split_sizes": output_split_sizes,
            "work": None,
        }

    def _prepare_nep_nccl_owner_all_to_all_scatter_batch(
        self, contexts: List[dict]
    ) -> Optional[dict]:
        """Pack one Scatter collective for multiple contexts owned by one rank."""
        if not contexts:
            return None
        if len(contexts) == 1:
            context = contexts[0]
            return context["group"]._prepare_nep_nccl_owner_all_to_all_scatter(
                context["owner_ep_rank"],
                context["chunk_index"],
                context["chunk_start"],
                context["chunk_end"],
                context["chunk"],
                context["buffer_slot_key"],
                async_op=context["async_op"],
            )

        contexts = sorted(
            contexts,
            key=lambda context: (
                getattr(context["group"], "_nep_nccl_group_index", -1),
                context["chunk_index"],
            ),
        )
        owner_ep_rank = contexts[0]["owner_ep_rank"]
        if any(context["owner_ep_rank"] != owner_ep_rank for context in contexts):
            raise RuntimeError("Two-level NEP Scatter cannot combine different owner ranks")
        if any(context["async_op"] != contexts[0]["async_op"] for context in contexts):
            raise RuntimeError("Two-level NEP Scatter contexts disagree on async mode")

        representative = contexts[0]["group"]
        cfg = representative._nep_runtime_config
        ep_rank = cfg["ep_rank"]
        source_ranks = representative._nep_nccl_owner_source_ranks(owner_ep_rank)
        transfer_ranks = representative._nep_nccl_owner_transfer_ranks(owner_ep_rank)
        for context in contexts[1:]:
            group = context["group"]
            if group._nep_nccl_owner_source_ranks(owner_ep_rank) != source_ranks:
                raise RuntimeError("Two-level NEP Scatter contexts disagree on source ranks")
            if group._nep_nccl_owner_transfer_ranks(owner_ep_rank) != transfer_ranks:
                raise RuntimeError("Two-level NEP Scatter contexts disagree on transfer ranks")
        if ep_rank not in transfer_ranks:
            return None

        remote_source_ranks = [rank for rank in source_ranks if rank != owner_ep_rank]
        total_chunk_numel = sum(context["chunk"].numel() for context in contexts)
        if not remote_source_ranks:
            return {
                "kind": "local",
                "owner_ep_rank": owner_ep_rank,
                "chunk_index": 0,
                "chunk_start": 0,
                "chunk_end": total_chunk_numel,
                "scatter_contexts": contexts,
            }

        transfer_group, _, transfer_size, transfer_source_ranks = (
            representative._get_nep_nccl_transfer_group_info(owner_ep_rank)
        )
        owner_transfer_rank = transfer_source_ranks.index(owner_ep_rank)
        remote_transfer_ranks = [transfer_source_ranks.index(rank) for rank in remote_source_ranks]
        dtype = contexts[0]["chunk"].dtype
        device = contexts[0]["chunk"].device
        if any(
            context["chunk"].dtype != dtype or context["chunk"].device != device
            for context in contexts
        ):
            raise RuntimeError("Two-level NEP Scatter contexts disagree on dtype or device")

        edp_bucket_indices = tuple(
            sorted({context["group"]._nep_nccl_edp_bucket_index for context in contexts})
        )
        scatter_bucket_key = (
            edp_bucket_indices[0]
            if len(edp_bucket_indices) == 1
            else ("end_iteration", edp_bucket_indices)
        )
        cache = representative._get_nep_nccl_shared_buffer_state()["gather_buf_cache"]
        empty = representative._get_nep_nccl_cached_tensor(
            cache, ("empty", dtype, device), 0, dtype, device
        )
        cache_prefix = (scatter_bucket_key, owner_ep_rank, dtype, device)

        input_split_sizes = [0] * transfer_size
        if ep_rank == owner_ep_rank:
            destination_numels = {}
            for destination_transfer_rank in remote_transfer_ranks:
                destination_ep_rank = transfer_source_ranks[destination_transfer_rank]
                destination_numel = sum(
                    context["group"]._nep_nccl_owner_source_payload_numel(
                        owner_ep_rank,
                        destination_ep_rank,
                        context["chunk_start"],
                        context["chunk_end"],
                    )
                    for context in contexts
                )
                destination_numels[destination_ep_rank] = destination_numel
                input_split_sizes[destination_transfer_rank] = destination_numel
            scatter_input_numel = sum(destination_numels.values())
            scatter_input = representative._get_nep_nccl_cached_tensor(
                cache,
                ("owner_layout_a2a_scatter_input_edp",) + cache_prefix,
                scatter_input_numel,
                dtype,
                device,
            )
            scatter_offset = 0
            for destination_ep_rank in remote_source_ranks:
                for context in contexts:
                    group = context["group"]
                    destination_numel = group._nep_nccl_owner_source_payload_numel(
                        owner_ep_rank,
                        destination_ep_rank,
                        context["chunk_start"],
                        context["chunk_end"],
                    )
                    group._pack_nep_nccl_scatter_payload(
                        owner_ep_rank,
                        destination_ep_rank,
                        context["chunk_start"],
                        context["chunk_end"],
                        context["chunk"],
                        scatter_input[scatter_offset : scatter_offset + destination_numel],
                    )
                    scatter_offset += destination_numel
        else:
            scatter_input = empty

        output_split_sizes = [0] * transfer_size
        if ep_rank in remote_source_ranks:
            scatter_output_numel = sum(
                context["group"]._nep_nccl_owner_source_payload_numel(
                    owner_ep_rank, ep_rank, context["chunk_start"], context["chunk_end"]
                )
                for context in contexts
            )
            output_split_sizes[owner_transfer_rank] = scatter_output_numel
            scatter_output = representative._get_nep_nccl_cached_tensor(
                cache,
                ("owner_layout_a2a_scatter_output_edp",) + cache_prefix,
                scatter_output_numel,
                dtype,
                device,
            )
        else:
            scatter_output = empty

        return {
            "kind": "all_to_all",
            "owner_ep_rank": owner_ep_rank,
            "chunk_index": 0,
            "chunk_start": 0,
            "chunk_end": total_chunk_numel,
            "buffer_slot_key": (
                "two_level_scatter",
                scatter_bucket_key,
                owner_ep_rank,
                dtype,
                device,
            ),
            "async_op": contexts[0]["async_op"],
            "ep_rank": ep_rank,
            "group_index": representative._nep_nccl_group_index,
            "chunk_size": total_chunk_numel,
            "source_ranks": source_ranks,
            "remote_source_ranks": remote_source_ranks,
            "transfer_source_ranks": transfer_source_ranks,
            "transfer_group": transfer_group,
            "scatter_input": scatter_input,
            "scatter_output": scatter_output,
            "input_split_sizes": input_split_sizes,
            "output_split_sizes": output_split_sizes,
            "scatter_contexts": contexts,
            "work": None,
        }

    def _submit_nep_nccl_owner_all_to_all_scatter(self, descriptor: Optional[dict]) -> None:
        """Launch one prepared Scatter descriptor without ordering copyback."""
        if descriptor is None:
            return

        kind = descriptor["kind"]
        if kind == "completed_local":
            descriptor["submitted"] = True
            return
        if kind == "local":
            descriptor["submitted"] = True
            return
        if kind != "all_to_all":
            raise RuntimeError(f"Unknown NEP Scatter descriptor kind: {kind}")

        work = dist.all_to_all_single(
            descriptor["scatter_output"],
            descriptor["scatter_input"],
            output_split_sizes=descriptor["output_split_sizes"],
            input_split_sizes=descriptor["input_split_sizes"],
            group=descriptor["transfer_group"],
            async_op=descriptor["async_op"],
        )
        descriptor["work"] = work
        descriptor["submitted"] = True
        self._record_nep_nccl_work(work, descriptor["buffer_slot_key"], block_current_stream=False)
        if descriptor["async_op"]:
            if descriptor["scatter_input"].numel() > 0:
                self._nep_nccl_async_tensors.append(descriptor["scatter_input"])
            if descriptor["scatter_output"].numel() > 0:
                self._nep_nccl_async_tensors.append(descriptor["scatter_output"])

    def _order_nep_nccl_owner_all_to_all_scatter_completion(
        self, descriptor: Optional[dict]
    ) -> None:
        """Order the launch stream after one submitted Scatter descriptor."""
        if descriptor is None:
            return
        if not descriptor.get("submitted", False):
            raise RuntimeError("NEP Scatter completion was ordered before submission")
        if descriptor["kind"] == "all_to_all":
            _nep_block_current_stream(descriptor["work"])
        descriptor["completion_ordered"] = True

    def _finish_nep_nccl_owner_all_to_all_scatter(self, descriptor: Optional[dict]) -> None:
        """Copy one completed Scatter descriptor into its physical gradients."""
        if descriptor is None:
            return
        if not descriptor.get("completion_ordered", False):
            raise RuntimeError("NEP Scatter copyback was queued before collective completion")

        kind = descriptor["kind"]
        if kind == "completed_local":
            return
        if kind == "local":
            scatter_contexts = descriptor.get("scatter_contexts")
            if scatter_contexts is None:
                self._copy_nep_nccl_owner_chunk_to_local_grads(
                    descriptor["owner_ep_rank"],
                    descriptor["chunk_start"],
                    descriptor["chunk_end"],
                    descriptor["chunk"],
                )
            else:
                for context in scatter_contexts:
                    context["group"]._copy_nep_nccl_owner_chunk_to_local_grads(
                        context["owner_ep_rank"],
                        context["chunk_start"],
                        context["chunk_end"],
                        context["chunk"],
                    )
            return
        if kind != "all_to_all":
            raise RuntimeError(f"Unknown NEP Scatter descriptor kind: {kind}")

        scatter_contexts = descriptor.get("scatter_contexts")
        if scatter_contexts is None and descriptor["ep_rank"] == descriptor["owner_ep_rank"]:
            self._copy_nep_nccl_owner_chunk_to_local_grads(
                descriptor["owner_ep_rank"],
                descriptor["chunk_start"],
                descriptor["chunk_end"],
                descriptor["chunk"],
            )
        elif (
            scatter_contexts is None and descriptor["ep_rank"] in descriptor["remote_source_ranks"]
        ):
            self._copy_nep_nccl_scatter_payload_to_local_grads(
                descriptor["owner_ep_rank"],
                descriptor["ep_rank"],
                descriptor["chunk_start"],
                descriptor["chunk_end"],
                descriptor["scatter_output"],
            )
        elif scatter_contexts is not None and descriptor["ep_rank"] == descriptor["owner_ep_rank"]:
            for context in scatter_contexts:
                context["group"]._copy_nep_nccl_owner_chunk_to_local_grads(
                    context["owner_ep_rank"],
                    context["chunk_start"],
                    context["chunk_end"],
                    context["chunk"],
                )
        elif (
            scatter_contexts is not None
            and descriptor["ep_rank"] in descriptor["remote_source_ranks"]
        ):
            scatter_offset = 0
            for context in scatter_contexts:
                group = context["group"]
                source_numel = group._nep_nccl_owner_source_payload_numel(
                    context["owner_ep_rank"],
                    descriptor["ep_rank"],
                    context["chunk_start"],
                    context["chunk_end"],
                )
                group._copy_nep_nccl_scatter_payload_to_local_grads(
                    context["owner_ep_rank"],
                    descriptor["ep_rank"],
                    context["chunk_start"],
                    context["chunk_end"],
                    descriptor["scatter_output"][scatter_offset : scatter_offset + source_numel],
                )
                scatter_offset += source_numel

    def _start_nep_nccl_owner_all_to_all_scatter(
        self,
        owner_ep_rank: int,
        chunk_index: int,
        chunk_start: int,
        chunk_end: int,
        chunk: torch.Tensor,
        buffer_slot_key: tuple,
        async_op: bool,
        scatter_chunk_index: int = 0,
    ) -> None:
        """Reshard one chunk through the same phased path used by chunk trains."""
        descriptor = self._prepare_nep_nccl_owner_all_to_all_scatter(
            owner_ep_rank,
            chunk_index,
            chunk_start,
            chunk_end,
            chunk,
            buffer_slot_key,
            async_op,
            scatter_chunk_index,
        )
        self._submit_nep_nccl_owner_all_to_all_scatter(descriptor)
        self._order_nep_nccl_owner_all_to_all_scatter_completion(descriptor)
        self._finish_nep_nccl_owner_all_to_all_scatter(descriptor)

    def _mark_nep_nccl_task_started(self, owner_ep_rank: int, chunk_index: int) -> None:
        self._nep_nccl_grad_sync_started = True
        self._nep_nccl_started_tasks.add((owner_ep_rank, chunk_index))
        if len(self._nep_nccl_started_tasks) == self._nep_nccl_task_count:
            self._nep_nccl_ready = True

    def _copy_nep_nccl_contexts_to_distopt_grads(self, contexts: List[dict]) -> None:
        """Stage compact Gather results into native owner DistOpt gradient views."""
        if not self.ddp_config.use_distributed_optimizer:
            return
        if not contexts:
            return

        bundle = contexts[0]["group"]._nep_distopt_owner_bundle
        owner_ep_rank = contexts[0]["owner_ep_rank"]
        if self._nep_runtime_config["ep_rank"] != owner_ep_rank:
            return
        for context in contexts:
            group = context["group"]
            if group._nep_distopt_owner_bundle is not bundle:
                raise RuntimeError("NEP DistOpt EDP batch spans different owner buffers")
            layout = group._get_nep_nccl_owner_layout()
            chunk_start = context["chunk_start"]
            chunk_end = context["chunk_end"]
            group_index = group._nep_nccl_group_index
            for expert_id in group._nep_nccl_owner_expert_ids(owner_ep_rank):
                _, owner_slot = group._nep_nccl_owner_slot_for_expert(expert_id)
                for slot_index, (slot_offset, slot_numel) in enumerate(
                    zip(group._nep_nccl_slot_offsets, group._nep_nccl_slot_numels)
                ):
                    entry_start = owner_slot * layout["expert_stride"] + slot_offset
                    overlap_start = max(chunk_start, entry_start)
                    overlap_end = min(chunk_end, entry_start + slot_numel)
                    if overlap_start >= overlap_end:
                        continue
                    proxy = bundle["proxy_by_key"][(group_index, expert_id, slot_index)]
                    proxy_offset = overlap_start - entry_start
                    chunk_offset = overlap_start - chunk_start
                    numel = overlap_end - overlap_start
                    proxy.main_grad.view(-1)[proxy_offset : proxy_offset + numel].copy_(
                        context["chunk"][chunk_offset : chunk_offset + numel]
                    )

            copy_done = torch.cuda.Event()
            copy_done.record(torch.cuda.current_stream())
            state = group._get_nep_nccl_shared_buffer_state()
            state["buffer_slot_events"].setdefault(context["buffer_slot_key"], []).append(copy_done)

    def _get_nep_nccl_native_edp_bucket_group(self, contexts):
        """Wrap one logical owner group in Megatron's native DDP lifecycle."""
        if isinstance(contexts, dict):
            contexts = [contexts]
        if not contexts:
            raise RuntimeError("NEP native DDP requires at least one owner context")
        contexts = sorted(
            contexts,
            key=lambda context: (context["group"]._nep_nccl_group_index, context["chunk_index"]),
        )

        cfg = self._nep_runtime_config
        owner_ep_rank = contexts[0]["owner_ep_rank"]
        if any(context["owner_ep_rank"] != owner_ep_rank for context in contexts):
            raise RuntimeError("NEP native DDP cannot combine different owner ranks")
        if cfg["ep_rank"] != owner_ep_rank:
            return None

        if self.ddp_config.use_distributed_optimizer:
            bundle = contexts[0]["group"]._nep_distopt_owner_bundle
            if any(
                context["group"]._nep_distopt_owner_bundle is not bundle for context in contexts
            ):
                raise RuntimeError("NEP DistOpt EDP batch spans different owner buffers")
            native_group = bundle.get("native_group")
            if native_group is None:
                raise RuntimeError("NEP DistOpt owner rank is missing its native bucket group")
            if native_group.grad_reduce_handle is not None:
                raise RuntimeError("NEP DistOpt owner group still has an outstanding reduction")
            return native_group

        edp_group = cfg.get("edp_group")
        if edp_group is None:
            raise RuntimeError(
                "Nonuniform EP NCCL owner rank requires runtime_config['edp_group']."
            )

        scaling_factors = {
            entry["bucket"].gradient_scaling_factor
            for context in contexts
            for entry in context["group"]._nep_nccl_entries
        }
        if len(scaling_factors) != 1:
            raise RuntimeError(
                "NEP owner-layout DDP requires one gradient scaling factor per expert group; "
                f"got {sorted(scaling_factors)}"
            )
        gradient_scaling_factor = next(iter(scaling_factors))
        cache_key = (
            id(edp_group),
            tuple((context["chunk"].data_ptr(), context["chunk"].numel()) for context in contexts),
        )
        native_group = self._nep_nccl_native_edp_bucket_groups.get(cache_key)
        if native_group is None:
            native_buckets = [
                _ParamAndGradBucket(
                    params=[],
                    param_data=None,
                    grad_data=context["chunk"],
                    offset=0,
                    numel_unpadded=context["chunk"].numel(),
                    gradient_scaling_factor=gradient_scaling_factor,
                    bucket_id=bucket_index,
                    param_index_map={},
                    params_with_extra_main_grads=[],
                )
                for bucket_index, context in enumerate(contexts)
            ]
            native_ddp_config = _nep_owner_ddp_config(self.ddp_config)
            native_group = _ParamAndGradBucketGroup(
                native_buckets, native_ddp_config, edp_group, dist.get_world_size(group=edp_group)
            )
            # NEP owns readiness and explicitly starts DDP after Gather.
            native_group.is_first_batch = False
            self._nep_nccl_native_edp_bucket_groups[cache_key] = native_group
        else:
            if len(native_group.buckets) != len(contexts):
                raise RuntimeError("Cached NEP DDP group changed its chunk count")
            for native_bucket, context in zip(native_group.buckets, contexts):
                if native_bucket.grad_data is not context["chunk"]:
                    raise RuntimeError("Cached NEP DDP bucket no longer owns the task tensor")
                native_bucket.gradient_scaling_factor = gradient_scaling_factor

        if native_group.grad_reduce_handle is not None:
            raise RuntimeError("NEP owner-layout DDP group still has an outstanding reduction")
        return native_group

    def _start_nep_nccl_owner_edp_reduce_contexts(self, contexts: List[dict]) -> None:
        """Launch one native EDP group after all of its Gather chunks are ready."""
        if not contexts:
            return
        contexts = sorted(
            contexts,
            key=lambda context: (context["group"]._nep_nccl_group_index, context["chunk_index"]),
        )
        cfg = self._nep_runtime_config
        owner_ep_rank = contexts[0]["owner_ep_rank"]
        if any(context["owner_ep_rank"] != owner_ep_rank for context in contexts):
            raise RuntimeError("NEP native EDP batch contains different owner ranks")

        if cfg["ep_rank"] != owner_ep_rank:
            return

        edp_group = cfg.get("edp_group")
        if edp_group is None:
            raise RuntimeError(
                "Nonuniform EP NCCL owner rank requires runtime_config['edp_group']."
            )
        native_group = self._get_nep_nccl_native_edp_bucket_group(contexts)
        if self.ddp_config.use_distributed_optimizer:
            with torch.profiler.record_function("nep_distopt_stage_owner_grads"):
                self._copy_nep_nccl_contexts_to_distopt_grads(contexts)
        with torch.profiler.record_function("nep_native_ddp_start_grad_sync"):
            native_group.start_grad_sync()
        native_state = {
            "group": native_group,
            "contexts": contexts,
            "started": True,
            "finished": False,
            "scatter_dependency_ordered": False,
        }
        active_native_states = getattr(self, "_nep_nccl_active_native_edp_states", None)
        if active_native_states is None:
            active_native_states = []
            self._nep_nccl_active_native_edp_states = active_native_states
        active_native_states.append(native_state)
        for context in contexts:
            context["native_edp_bucket_group"] = native_group
            context["native_edp_state"] = native_state
            context["native_edp_started"] = True

    def _start_nep_nccl_owner_edp_reduce(self, context: dict) -> None:
        """Launch native EDP for a single-context path."""
        self._start_nep_nccl_owner_edp_reduce_contexts([context])

    def _order_nep_nccl_owner_edp_before_scatter(self, context: dict) -> None:
        """Order Scatter after native EDP without consuming the final DDP handle."""
        if self._nep_runtime_config["ep_rank"] != context["owner_ep_rank"]:
            return
        native_state = context.get("native_edp_state")
        if native_state is None:
            raise RuntimeError("NEP Scatter reached an owner before native DDP started")
        if native_state["scatter_dependency_ordered"]:
            return
        if not context.get("native_edp_started", False):
            raise RuntimeError("NEP Scatter reached an owner before native DDP started")

        native_group = native_state["group"]
        if self.ddp_config.overlap_grad_reduce:
            if native_group.grad_reduce_handle is None:
                raise RuntimeError("NEP Scatter reached an owner without an EDP Work handle")
            with torch.profiler.record_function("nep_native_ddp_order_scatter_stream"):
                _nep_block_current_stream(native_group.grad_reduce_handle)
        elif native_group.grad_reduce_handle is not None:
            raise RuntimeError("Synchronous NEP native DDP left an outstanding reduction")
        native_state["scatter_dependency_ordered"] = True

    def _finish_nep_nccl_native_edp_reductions(self) -> None:
        """Finish every native owner EDP Work once, at the final DDP drain."""
        native_states = getattr(self, "_nep_nccl_active_native_edp_states", [])
        for native_state in native_states:
            if native_state["finished"]:
                continue
            if (
                not self.ddp_config.use_distributed_optimizer
                and not native_state["scatter_dependency_ordered"]
            ):
                raise RuntimeError("NEP final DDP drain reached an unordered EDP Scatter")

            native_group = native_state["group"]
            if self.ddp_config.overlap_grad_reduce:
                with torch.profiler.record_function("nep_native_ddp_finish_grad_sync_final"):
                    native_group.finish_grad_sync()
            elif native_group.grad_reduce_handle is not None:
                raise RuntimeError("Synchronous NEP native DDP left an outstanding reduction")
            native_state["finished"] = True
            native_state["started"] = False
            for grouped_context in native_state["contexts"]:
                grouped_context["native_edp_started"] = False
        native_states.clear()

    def _stage_nep_nccl_owner_edp_contexts(self, contexts: List[dict]) -> List[List[dict]]:
        """Return complete owner/EDP buckets while retaining partial Gather buckets."""
        state = getattr(self, "_nep_nccl_scheduler_state", None)
        if state is None:
            raise RuntimeError("Two-level NEP Gather requires the shared task scheduler")
        pending = state["pending_edp_contexts"]
        expected = state["expected_edp_contexts"]
        touched_keys = []
        for context in contexts:
            group = context["group"]
            edp_bucket_index = group._nep_nccl_edp_bucket_index
            key = (edp_bucket_index, context["owner_ep_rank"])
            context_key = (group._nep_nccl_group_index, context["chunk_index"])
            bucket_contexts = pending.setdefault(key, {})
            if context_key in bucket_contexts:
                raise RuntimeError(
                    "NEP Gather context was staged twice for "
                    f"EDP bucket {edp_bucket_index}, owner {context['owner_ep_rank']}, "
                    f"context {context_key}"
                )
            bucket_contexts[context_key] = context
            if key not in touched_keys:
                touched_keys.append(key)

        complete_batches = []
        for key in touched_keys:
            bucket_contexts = pending[key]
            expected_count = expected[key]
            if len(bucket_contexts) > expected_count:
                raise RuntimeError(
                    f"NEP EDP bucket {key} received {len(bucket_contexts)} Gather contexts; "
                    f"expected {expected_count}"
                )
            if len(bucket_contexts) == expected_count:
                complete_batches.append(
                    [bucket_contexts[context_key] for context_key in sorted(bucket_contexts)]
                )
                del pending[key]
        return complete_batches

    def _start_nep_nccl_owner_edp_reduce_batch(
        self, contexts: List[dict]
    ) -> List[List[dict]]:
        """Launch one native EDP group per original expert bucket and owner."""
        context_batches = self._stage_nep_nccl_owner_edp_contexts(contexts)
        for context_batch in context_batches:
            group = context_batch[0]["group"]
            if group._nep_runtime_config["ep_rank"] == context_batch[0]["owner_ep_rank"]:
                group._start_nep_nccl_owner_edp_reduce_contexts(context_batch)
        return context_batches

    def _coalesce_nep_nccl_scatter_contexts(self, contexts: List[dict]) -> dict:
        """Represent one original EDP bucket as one Scatter scheduling unit."""
        if not contexts:
            raise RuntimeError("Cannot build a two-level NEP Scatter from no contexts")
        if len(contexts) == 1:
            return contexts[0]
        contexts = sorted(
            contexts,
            key=lambda context: (
                getattr(context["group"], "_nep_nccl_group_index", -1),
                context["chunk_index"],
            ),
        )
        scatter_context = dict(contexts[0])
        scatter_context["scatter_contexts"] = tuple(contexts)
        return scatter_context

    def _prepare_nep_nccl_owner_task_context(
        self, owner_ep_rank: int, chunk_index: int, chunk_start: int, chunk_end: int, async_op: bool
    ) -> Optional[dict]:
        """Allocate and prepare one owner task without launching collectives."""
        chunk_size = chunk_end - chunk_start
        if chunk_size <= 0:
            self._mark_nep_nccl_task_started(owner_ep_rank, chunk_index)
            return None

        buffer_slot = self._get_nep_nccl_task_buffer_slot(owner_ep_rank, chunk_index)
        chunk_dtype = self.buckets[0].grad_data.dtype
        chunk_device = self.buckets[0].grad_data.device
        buffer_slot_key = (buffer_slot, chunk_size, chunk_dtype, chunk_device)
        self._order_nep_nccl_buffer_slot(buffer_slot_key)

        gather_buf_cache = self._get_nep_nccl_shared_buffer_state()["gather_buf_cache"]
        if (
            self.ddp_config.use_distributed_optimizer
            and self._nep_runtime_config["ep_rank"] != owner_ep_rank
        ):
            # DistOpt consumes the common owner layout only on its owner rank. Source ranks
            # contribute packed Gather inputs, but never read the full owner-layout scratch.
            chunk = self._get_nep_nccl_cached_tensor(
                gather_buf_cache, ("empty", chunk_dtype, chunk_device), 0, chunk_dtype, chunk_device
            )
        else:
            chunk = self._get_nep_nccl_cached_tensor(
                gather_buf_cache,
                ("owner_layout_gather", buffer_slot, chunk_size, chunk_dtype, chunk_device),
                chunk_size,
                chunk_dtype,
                chunk_device,
            )
        if async_op:
            self._nep_nccl_async_tensors.append(chunk)

        self._prep_nep_nccl_owner_entries_for_sync(owner_ep_rank)
        return {
            "group": self,
            "owner_ep_rank": owner_ep_rank,
            "chunk_index": chunk_index,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "chunk": chunk,
            "buffer_slot": buffer_slot,
            "buffer_slot_key": buffer_slot_key,
            "async_op": async_op,
        }

    def _start_nep_nccl_owner_task(
        self,
        owner_ep_rank: int,
        chunk_index: int,
        chunk_start: int,
        chunk_end: int,
        async_op: bool,
    ) -> None:
        """Launch one ordered owner-layout gather/allreduce/scatter task."""
        context = self._prepare_nep_nccl_owner_task_context(
            owner_ep_rank, chunk_index, chunk_start, chunk_end, async_op
        )
        if context is None:
            return

        chunk = context["chunk"]
        buffer_slot_key = context["buffer_slot_key"]
        self._start_nep_nccl_owner_all_to_all_gather(
            owner_ep_rank,
            chunk_index,
            chunk_start,
            chunk_end,
            chunk,
            buffer_slot_key,
            async_op=async_op,
        )

        if not self.is_first_batch:
            raise RuntimeError("Steady-state two-level NEP Gather must launch from AccumulateGrad")
        complete_context_batches = self._start_nep_nccl_owner_edp_reduce_batch([context])
        for context_batch in complete_context_batches:
            scatter_context = self._coalesce_nep_nccl_scatter_contexts(context_batch)
            scatter_context["group"]._start_nep_nccl_owner_task_scatter(scatter_context)

    def _prepare_nep_nccl_owner_task_scatter_train(self, context: dict) -> dict:
        """Prepare every chunk in one Scatter train without submitting collectives."""
        scatter_contexts = context.get("scatter_contexts")
        ordered_native_states = set()
        for task_context in scatter_contexts or (context,):
            native_state = task_context.get("native_edp_state")
            if native_state is not None and id(native_state) in ordered_native_states:
                continue
            task_group = task_context.get("group", self)
            task_group._order_nep_nccl_owner_edp_before_scatter(task_context)
            if native_state is not None:
                ordered_native_states.add(id(native_state))

        scatter_chunks = _get_nep_nccl_scatter_chunks()
        if scatter_contexts is not None:
            if scatter_chunks != 1:
                raise RuntimeError(
                    "Two-level NEP Gather currently requires "
                    "MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=1"
                )
            descriptors = [
                self._prepare_nep_nccl_owner_all_to_all_scatter_batch(list(scatter_contexts))
            ]
        else:
            scatter_ranges = self._nep_nccl_scatter_chunk_ranges(
                context["owner_ep_rank"],
                context["chunk_start"],
                context["chunk_end"],
                scatter_chunks,
            )
            descriptors = []
            for scatter_chunk_index, (scatter_start, scatter_end) in enumerate(scatter_ranges):
                local_start = scatter_start - context["chunk_start"]
                local_end = scatter_end - context["chunk_start"]
                descriptors.append(
                    self._prepare_nep_nccl_owner_all_to_all_scatter(
                        context["owner_ep_rank"],
                        context["chunk_index"],
                        scatter_start,
                        scatter_end,
                        context["chunk"][local_start:local_end],
                        context["buffer_slot_key"],
                        async_op=context["async_op"],
                        scatter_chunk_index=scatter_chunk_index,
                    )
                )

        return {
            "group": self,
            "context": context,
            "descriptors": descriptors,
            "next_descriptor": 0,
            "task_marked": False,
        }


    def _finish_nep_nccl_scatter_train_submission(self, train: dict) -> None:
        """Finish bookkeeping after every descriptor in one train is submitted."""
        context = train["context"]
        if not train["task_marked"]:
            scatter_contexts = context.get("scatter_contexts")
            if scatter_contexts is None:
                self._mark_nep_nccl_task_started(context["owner_ep_rank"], context["chunk_index"])
            else:
                for task_context in scatter_contexts:
                    task_context["group"]._mark_nep_nccl_task_started(
                        task_context["owner_ep_rank"], task_context["chunk_index"]
                    )
            train["task_marked"] = True

    def _mark_nep_distopt_task_complete(self, context: dict) -> None:
        """Complete task bookkeeping without scattering reduced gradients."""
        scatter_contexts = context.get("scatter_contexts")
        if scatter_contexts is None:
            self._mark_nep_nccl_task_started(context["owner_ep_rank"], context["chunk_index"])
            return
        for task_context in scatter_contexts:
            task_context["group"]._mark_nep_nccl_task_started(
                task_context["owner_ep_rank"], task_context["chunk_index"]
            )

    @torch.no_grad()
    def _scatter_nep_distopt_owner_params_to_physical_holders(self) -> None:
        """Redistribute all-gathered logical owner parameters within each EP replica."""
        bundle = self._nep_distopt_owner_bundle
        ep_rank = self._nep_runtime_config["ep_rank"]
        ep_group = self._nep_runtime_config["ep_group"]
        for bucket_group in bundle["groups"]:
            layout = bucket_group._get_nep_nccl_owner_layout()
            group_index = bucket_group._nep_nccl_group_index
            for owner_ep_rank in range(layout["min_ep_size"]):
                transfer_group, _, transfer_size, transfer_ranks = (
                    bucket_group._get_nep_nccl_transfer_group_info(owner_ep_rank)
                )
                if ep_rank not in transfer_ranks:
                    continue
                local_entries = bucket_group._nep_nccl_owner_entries(owner_ep_rank)
                if not local_entries:
                    raise RuntimeError(
                        "NEP DistOpt parameter Scatter found no local physical entries"
                    )
                reference_param = local_entries[0]["bucket"].params_list[0]
                transfer_slot = bundle["param_transfer_slot"]
                cache_key = (transfer_slot, reference_param.dtype, reference_param.device)
                transfer_storage = bundle["param_transfer_buffers"].get(cache_key)
                if transfer_storage is None:
                    transfer_storage = torch.empty(
                        bundle["param_transfer_numel_by_slot"][transfer_slot],
                        dtype=reference_param.dtype,
                        device=reference_param.device,
                    )
                    bundle["param_transfer_buffers"][cache_key] = transfer_storage
                owner_params = transfer_storage[: layout["owner_numel"]]
                if ep_rank == owner_ep_rank:
                    for expert_id in bucket_group._nep_nccl_owner_expert_ids(owner_ep_rank):
                        _, owner_slot = bucket_group._nep_nccl_owner_slot_for_expert(expert_id)
                        for slot_index, (slot_offset, slot_numel) in enumerate(
                            zip(
                                bucket_group._nep_nccl_slot_offsets,
                                bucket_group._nep_nccl_slot_numels,
                            )
                        ):
                            proxy = bundle["proxy_by_key"][(group_index, expert_id, slot_index)]
                            start = owner_slot * layout["expert_stride"] + slot_offset
                            owner_params[start : start + slot_numel].copy_(proxy.detach().view(-1))

                if transfer_group is not None and transfer_size > 1:
                    dist.broadcast(
                        owner_params,
                        src=get_global_rank(ep_group, owner_ep_rank),
                        group=transfer_group,
                    )

                for entry in local_entries:
                    start = bucket_group._nep_nccl_entry_owner_start(entry, owner_ep_rank)
                    entry["bucket"].params_list[0].copy_(
                        owner_params[start : start + entry["numel"]].view_as(
                            entry["bucket"].params_list[0]
                        )
                    )

    def start_param_sync(self, force_sync: bool = False):
        """Start native owner parameter all-gather and redistribute when ready."""
        if not self.ddp_config.use_distributed_optimizer:
            return super().start_param_sync(force_sync=force_sync)
        if not self._nep_distopt_param_sync_representative:
            return

        bundle = self._nep_distopt_owner_bundle
        if self.param_gather_dispatched:
            if (
                force_sync
                and self.ddp_config.overlap_param_gather
                and not bundle["param_sync_completed"]
            ):
                self.finish_param_sync(skip_next_bucket_dispatch=True)
            return

        bundle["param_sync_completed"] = False
        native_group = bundle.get("native_group")
        if native_group is not None:
            with torch.profiler.record_function("nep_distopt_native_param_all_gather"):
                native_group.start_param_sync(force_sync=force_sync)
                if not self.ddp_config.overlap_param_gather:
                    native_group._post_param_sync()
        self.param_gather_dispatched = True
        if self.ddp_config.overlap_param_gather and not force_sync:
            return

        with torch.profiler.record_function("nep_distopt_param_scatter"):
            self._scatter_nep_distopt_owner_params_to_physical_holders()
        bundle["param_sync_completed"] = True

    def finish_param_sync(self, skip_next_bucket_dispatch: bool = False):
        """Finish owner all-gather, redistribute params, and launch the next bucket."""
        if not self.ddp_config.use_distributed_optimizer:
            return super().finish_param_sync(skip_next_bucket_dispatch=skip_next_bucket_dispatch)
        if not self._nep_distopt_param_sync_representative:
            return

        bundle = self._nep_distopt_owner_bundle
        if not self.param_gather_dispatched:
            self.start_param_sync()
        if bundle["param_sync_completed"]:
            return

        native_group = bundle.get("native_group")
        if native_group is not None:
            with torch.profiler.record_function("nep_distopt_finish_native_param_all_gather"):
                native_group.finish_param_sync(skip_next_bucket_dispatch=True)
        with torch.profiler.record_function("nep_distopt_param_scatter"):
            self._scatter_nep_distopt_owner_params_to_physical_holders()
        bundle["param_sync_completed"] = True

        next_group = self.next_param_gather_bucket_group
        if next_group is not None and not skip_next_bucket_dispatch:
            next_group.start_param_sync()

    def _start_nep_nccl_owner_task_scatter(self, context: dict) -> None:
        if self.ddp_config.use_distributed_optimizer:
            self._mark_nep_distopt_task_complete(context)
            return
        train = self._prepare_nep_nccl_owner_task_scatter_train(context)
        for descriptor in train["descriptors"]:
            self._submit_nep_nccl_owner_all_to_all_scatter(descriptor)
            self._order_nep_nccl_owner_all_to_all_scatter_completion(descriptor)
            self._finish_nep_nccl_owner_all_to_all_scatter(descriptor)
        self._finish_nep_nccl_scatter_train_submission(train)


    def _start_nep_nccl_split_host_phase_batch(
        self, task_batch: List[dict], dispatch_stream: torch.cuda.Stream, batch_index: int
    ):
        """Launch one Gather batch and its pair-scoped host rendezvous."""
        contexts = []
        with torch.cuda.stream(dispatch_stream):
            for task in task_batch:
                context = task["group"]._prepare_nep_nccl_owner_task_context(
                    task["owner_ep_rank"],
                    task["chunk_index"],
                    task["chunk_start"],
                    task["chunk_end"],
                    async_op=True,
                )
                if context is not None:
                    contexts.append(context)
            for context in contexts:
                group = context["group"]
                group._start_nep_nccl_owner_all_to_all_gather(
                    context["owner_ep_rank"],
                    context["chunk_index"],
                    context["chunk_start"],
                    context["chunk_end"],
                    context["chunk"],
                    context["buffer_slot_key"],
                    async_op=True,
                )
            gather_done_event = torch.cuda.Event()
            gather_done_event.record(dispatch_stream)
            for context in contexts:
                context["gather_done_event"] = gather_done_event

        edp_stream = self._get_nep_nccl_ordered_edp_stream()
        edp_stream.wait_event(gather_done_event)
        with torch.cuda.stream(edp_stream):
            complete_context_batches = self._start_nep_nccl_owner_edp_reduce_batch(
                contexts
            )

        complete_keys = {
            (
                context_batch[0]["group"]._nep_nccl_edp_bucket_index,
                context_batch[0]["owner_ep_rank"],
            )
            for context_batch in complete_context_batches
        }
        staged_contexts = [
            context
            for context in contexts
            if (context["group"]._nep_nccl_edp_bucket_index, context["owner_ep_rank"])
            not in complete_keys
        ]
        pending = []
        if staged_contexts:
            pending.append(
                {
                    "batch_index": batch_index,
                    "contexts": staged_contexts,
                    "dispatch_stream": dispatch_stream,
                    "gather_done_event": gather_done_event,
                    "phase": "gather_staged",
                }
            )
        complete_contexts = [
            self._coalesce_nep_nccl_scatter_contexts(context_batch)
            for context_batch in complete_context_batches
        ]
        if complete_contexts:
            pending.append(
                {
                    "batch_index": batch_index,
                    "contexts": complete_contexts,
                    "dispatch_stream": dispatch_stream,
                    "edp_stream": edp_stream,
                    "local_transfer_contexts": {},
                    "local_edp_contexts": {},
                    "gather_barrier_works": [],
                    "gather_done_event": gather_done_event,
                    "phase": "edp_launched",
                }
            )
        return pending

    def _start_nep_nccl_process_group_dispatch_batch(
        self,
        state: dict,
        force_ready: bool,
        async_op: bool,
        compute_ready_event: torch.cuda.Event,
    ) -> List[dict]:
        """Launch ready ProcessGroup tasks in pair-scoped ordered phases."""
        if not async_op:
            raise RuntimeError("NEP dispatch-boundary tasks must be asynchronous")

        ready_tasks = []
        tasks = state["task_sequence"]
        while state["task_next_index"] < len(tasks):
            task = tasks[state["task_next_index"]]
            if not force_ready and not task["group"]._nep_nccl_owner_task_ready(
                task["owner_ep_rank"]
            ):
                break
            ready_tasks.append(task)
            state["task_next_index"] += 1
        if not ready_tasks:
            return []

        owner_order = []
        owner_transfer_ranks = {}
        for task in ready_tasks:
            owner = task["owner_ep_rank"]
            transfer_ranks = frozenset(task["group"]._nep_nccl_owner_transfer_ranks(owner))
            previous_ranks = owner_transfer_ranks.setdefault(owner, transfer_ranks)
            if previous_ranks != transfer_ranks:
                raise RuntimeError(
                    f"NEP owner {owner} has inconsistent transfer ranks within one boundary"
                )
            if owner not in owner_order:
                owner_order.append(owner)

        owner_waves = []
        for owner in owner_order:
            transfer_ranks = owner_transfer_ranks[owner]
            for wave in owner_waves:
                if wave["transfer_ranks"].isdisjoint(transfer_ranks):
                    wave["owners"].add(owner)
                    wave["transfer_ranks"].update(transfer_ranks)
                    break
            else:
                owner_waves.append({"owners": {owner}, "transfer_ranks": set(transfer_ranks)})

        task_batches = []
        max_batch_size = _get_nep_nccl_async_chunk_window()
        for wave in owner_waves:
            current_batch = []
            current_slots = set()
            for task in ready_tasks:
                if task["owner_ep_rank"] not in wave["owners"]:
                    continue
                slot = task["group"]._get_nep_nccl_task_buffer_slot(
                    task["owner_ep_rank"], task["chunk_index"]
                )
                if current_batch and (
                    len(current_batch) >= max_batch_size
                    or slot in current_slots
                ):
                    task_batches.append(current_batch)
                    current_batch = []
                    current_slots = set()
                current_batch.append(task)
                current_slots.add(slot)
            if current_batch:
                task_batches.append(current_batch)
        submission_window = 1
        remaining_task_batches = list(
            enumerate(task_batches[submission_window:], start=submission_window)
        )
        pending_host_phases = []
        for batch_index, task_batch in enumerate(task_batches[:submission_window]):
            dispatch_stream = self._get_nep_nccl_comm_stream(batch_index)
            dispatch_stream.wait_event(compute_ready_event)
            pending = self._start_nep_nccl_split_host_phase_batch(
                task_batch, dispatch_stream, batch_index=batch_index
            )
            for pending_phase in pending:
                pending_phase["submission_slot"] = batch_index
            pending_host_phases.extend(pending)
        pending_host_phases[-1]["remaining_task_batches"] = remaining_task_batches
        return pending_host_phases

    def _finish_nep_nccl_process_group_dispatch_batches(
        self,
        pending_host_phases: List[dict],
        scatter_context_batches: Optional[List[List[dict]]] = None,
    ) -> bool:
        """Progress ordered EDP and Scatter phases after backward compute is queued."""
        for pending in pending_host_phases:
            contexts = pending["contexts"]
            dispatch_stream = pending["dispatch_stream"]
            phase = pending.get("phase", "gather_launched")
            if phase == "finished":
                continue
            if phase == "gather_staged":
                pending["phase"] = "finished"
                remaining_task_batches = pending.get("remaining_task_batches")
                if remaining_task_batches:
                    next_batch_index, next_task_batch = remaining_task_batches.pop(0)
                    next_pending = self._start_nep_nccl_split_host_phase_batch(
                        next_task_batch, dispatch_stream, next_batch_index
                    )
                    if not isinstance(next_pending, list) or not next_pending:
                        raise RuntimeError(
                            "Two-level NEP Gather produced no phase record for a queued batch"
                        )
                    for pending_phase in next_pending:
                        pending_phase["submission_slot"] = pending.get("submission_slot", 0)
                    next_pending[-1]["remaining_task_batches"] = remaining_task_batches
                    pending_index = pending_host_phases.index(pending)
                    pending_host_phases[pending_index : pending_index + 1] = next_pending
                    return self._finish_nep_nccl_process_group_dispatch_batches(
                        pending_host_phases,
                        scatter_context_batches=scatter_context_batches,
                    )
                continue


            if pending["phase"] == "scatter_ready":
                pass
            elif pending["phase"] == "edp_launched":
                pending["phase"] = "edp_complete"

                pending["phase"] = "scatter_ready"
            else:
                raise RuntimeError(f"Unknown split NEP host phase: {pending['phase']}")

            if scatter_context_batches is not None:
                scatter_context_batches.append(list(contexts))
            else:
                with torch.cuda.stream(dispatch_stream):
                    for context in contexts:
                        context["group"]._start_nep_nccl_owner_task_scatter(context)
            pending["phase"] = "finished"

            remaining_task_batches = pending.get("remaining_task_batches")
            if remaining_task_batches:
                next_batch_index, next_task_batch = remaining_task_batches.pop(0)
                next_pending = self._start_nep_nccl_split_host_phase_batch(
                    next_task_batch, dispatch_stream, next_batch_index
                )
                if isinstance(next_pending, list):
                    if not next_pending:
                        raise RuntimeError("Two-level NEP Gather produced no phase record")
                    for pending_phase in next_pending:
                        pending_phase["submission_slot"] = pending.get("submission_slot", 0)
                    next_pending[-1]["remaining_task_batches"] = remaining_task_batches
                    pending_index = pending_host_phases.index(pending)
                    pending_host_phases[pending_index : pending_index + 1] = next_pending
                else:
                    next_pending["submission_slot"] = pending.get("submission_slot", 0)
                    next_pending["remaining_task_batches"] = remaining_task_batches
                    pending.clear()
                    pending.update(next_pending)
                return self._finish_nep_nccl_process_group_dispatch_batches(
                    pending_host_phases,
                    scatter_context_batches=scatter_context_batches,
                )

        return all(pending.get("phase") == "finished" for pending in pending_host_phases)

    def _try_start_nep_nccl_ready_tasks(
        self,
        force_ready: bool = False,
        async_op_override: Optional[bool] = None,
        compute_ready_event: Optional[torch.cuda.Event] = None,
    ) -> Optional[List[dict]]:
        """Launch globally ordered owner tasks while local dependencies are ready."""
        state = getattr(self, "_nep_nccl_scheduler_state", None)
        if state is None or "task_sequence" not in state:
            self._start_nonuniform_ep_nccl_grad_sync(
                async_op=(
                    self.ddp_config.overlap_grad_reduce and not self.is_first_batch
                    if async_op_override is None
                    else async_op_override
                )
            )
            return

        async_op = (
            self.ddp_config.overlap_grad_reduce and not self.is_first_batch
            if async_op_override is None
            else async_op_override
        )

        if compute_ready_event is not None and async_op:
            return self._start_nep_nccl_process_group_dispatch_batch(
                state,
                force_ready,
                async_op,
                compute_ready_event,
            )

        def next_task_is_ready() -> bool:
            tasks = state["task_sequence"]
            if state["task_next_index"] >= len(tasks):
                return False
            task = tasks[state["task_next_index"]]
            return force_ready or task["group"]._nep_nccl_owner_task_ready(task["owner_ep_rank"])

        def launch_next_task() -> None:
            task = state["task_sequence"][state["task_next_index"]]
            group = task["group"]
            group._start_nep_nccl_owner_task(
                task["owner_ep_rank"],
                task["chunk_index"],
                task["chunk_start"],
                task["chunk_end"],
                async_op=async_op,
            )
            state["task_next_index"] += 1

        def launch_ready_tasks_on_current_stream() -> None:
            while next_task_is_ready():
                launch_next_task()

        if async_op:
            compute_stream = (
                None if compute_ready_event is not None else torch.cuda.current_stream()
            )
            while True:
                if not next_task_is_ready():
                    break
                task_index = state["task_next_index"]
                task = state["task_sequence"][task_index]
                group = task["group"]
                owner_ep_rank = task["owner_ep_rank"]
                stream_slot = group._get_nep_nccl_task_buffer_slot(
                    owner_ep_rank, task["chunk_index"]
                )
                nccl_stream = group._get_nep_nccl_comm_stream(stream_slot)
                if compute_ready_event is not None:
                    nccl_stream.wait_event(compute_ready_event)
                else:
                    nccl_stream.wait_stream(compute_stream)
                with torch.cuda.stream(nccl_stream):
                    group._flush_nep_nccl_pending_scatters(buffer_slot=stream_slot)
                    launch_next_task()
            if force_ready:
                self._flush_nep_nccl_pending_scatters(force_all=True)
        else:
            launch_ready_tasks_on_current_stream()

    def _start_nonuniform_ep_nccl_grad_sync(self, async_op: bool = False):
        layout = self._get_nep_nccl_owner_layout()
        min_ep_size = layout["min_ep_size"]
        owner_numel = layout["owner_numel"]
        if owner_numel == 0:
            self._nep_nccl_ready = True
            return
        for owner_ep_rank in range(min_ep_size):
            for chunk_index, (start, end) in enumerate(layout["chunk_ranges"]):
                self._start_nep_nccl_owner_task(
                    owner_ep_rank, chunk_index, start, end, async_op=async_op
                )

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Start synchronous NCCL nonuniform EP gradient synchronization."""
        if self._nep_nccl_ready:
            return
        if self.is_first_batch and self.grad_reduce_handle is not None:
            return
        assert (
            self.grad_reduce_handle is None
        ), "Should not have multiple communication calls outstanding at once"

        async_op = self.ddp_config.overlap_grad_reduce and not self.is_first_batch

        if self.ddp_config.check_for_nan_in_grad or self.ddp_config.check_for_large_grads:
            self.check_grads(
                check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
                check_for_large=self.ddp_config.check_for_large_grads,
            )

        self._try_start_nep_nccl_ready_tasks(force_ready=True, async_op_override=async_op)
        self.grad_reduce_handle = None

    def _finish_nonuniform_ep_nccl_grad_sync(self):
        self._drain_nep_nccl_async_window(force_all=True)
        for nccl_stream in self._nep_nccl_streams.values():
            torch.cuda.current_stream().wait_stream(nccl_stream)
        self._nep_nccl_async_handles = []
        self._nep_nccl_async_tensors = []
        # Drop large chunk temporaries after this bucket group completes; they
        # will be recreated as needed in the next overlap window.
        self._nep_nccl_send_chunk_cache.clear()
        self._nep_nccl_gather_buf_cache.clear()
        self._nep_nccl_gather_list_cache.clear()

    def finish_nep_pre_sync(self, force_all_reduce: Optional[bool] = False):
        """Drain NEP NCCL work before generic DDP waits on dense bucket groups."""
        if not self.ddp_config.overlap_grad_reduce:
            return

        if not self.is_first_batch and not self._nep_nccl_ready:
            assert self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts, (
                f"Communication call has not been issued for this bucket "
                f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
                "params have grad available)"
            )
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        if self.is_first_batch:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        if not self._nep_nccl_ready:
            return
        self._finish_nonuniform_ep_nccl_grad_sync()

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        self.param_gather_dispatched = False
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            self._finish_nonuniform_ep_nccl_grad_sync()
            self._finish_nep_nccl_native_edp_reductions()
            return

        if not self.is_first_batch and not self._nep_nccl_ready:
            assert self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts, (
                f"Communication call has not been issued for this bucket "
                f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
                "params have grad available)"
            )
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        if self.is_first_batch:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        assert self._nep_nccl_ready, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
            "params have grad available)"
        )
        self._finish_nonuniform_ep_nccl_grad_sync()
        self._finish_nep_nccl_native_edp_reductions()

    def register_grad_ready(
        self, param: torch.nn.Parameter, force_all_reduce: Optional[bool] = False
    ):
        """Track ready grads and launch NCCL collectives in bucket-group order."""
        assert (
            self.ddp_config.overlap_grad_reduce
        ), "register_grad_ready() should only be called when overlap_grad_reduce is True"
        if self.is_last_microbatch:
            assert param in self.param_to_bucket, "Param is not in the bucket group"
            if param not in self.per_param_grad_ready_counts:
                self.per_param_grad_ready_counts[param] = 0
            self.per_param_grad_ready_counts[param] += 1
            bucket_ready = (
                self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts
            )
            if not self.is_first_batch:
                if self._nep_dispatch_boundary_launch:
                    if bucket_ready:
                        self._nep_dispatch_boundary_ready = True
                    callback = self._nep_dispatch_boundary_callback
                    if self._nep_dispatch_boundary_ready and callback is not None:
                        callback(
                            self._nep_dispatch_boundary_groups,
                            self._nep_dispatch_boundary_module_label,
                        )
                else:
                    self._try_start_nep_nccl_ready_tasks(force_ready=False, async_op_override=True)
                if bucket_ready:
                    assert len(self.per_param_grad_ready_counts) == len(self.params)

    def reset(self):
        for native_group in getattr(self, "_nep_nccl_native_edp_bucket_groups", {}).values():
            if native_group.grad_reduce_handle is not None:
                raise RuntimeError("NEP reset found an unfinished native DDP reduction")
        if getattr(self, "_nep_nccl_active_native_edp_states", []):
            raise RuntimeError("NEP reset found undrained native DDP state")
        super().reset()
        self._nep_nccl_grad_sync_started = False
        self._nep_nccl_ready = self._nep_nccl_task_count == 0
        self.grad_reduce_handle = None
        self._nep_nccl_async_handles = []
        self._nep_nccl_async_tensors = []
        self._nep_nccl_started_tasks = set()
        self._nep_nccl_prepped_experts = set()
        self._nep_dispatch_boundary_ready = False
        self._nep_dispatch_boundary_launched = False
        self._nep_dispatch_boundary_launching = False
        self._nep_dispatch_boundary_wait_logged = False
        reset_ordered_bucket_group_scheduler(
            self, "_nep_nccl_scheduler_state", "_nep_nccl_group_index"
        )
        state = getattr(self, "_nep_nccl_scheduler_state", None)
        if state is not None and getattr(self, "_nep_nccl_group_index", -1) == 0:
            if state.get("pending_scatters"):
                raise RuntimeError("NEP reset found deferred scatters that were not flushed")
            if state.get("pending_edp_contexts"):
                raise RuntimeError("NEP reset found incomplete two-level Gather buckets")
            state["task_next_index"] = 0
            state["pending_scatters"] = []
            state["pending_edp_contexts"] = {}


def _coalesce_nep_nccl_bucket_groups_for_edp_order(
    bucket_groups: List[NonuniformEPNCCLParamAndGradBucketGroup],
    runtime_config: dict,
    nonuniform_ep_config: NonuniformEPConfig,
) -> List[NonuniformEPNCCLParamAndGradBucketGroup]:
    """Merge local bucket groups so every rank in an EDP group has the same count."""
    if not bucket_groups:
        return bucket_groups

    edp_group = runtime_config.get("edp_group")
    if edp_group is None:
        return bucket_groups

    local_count = torch.tensor(
        [len(bucket_groups)], dtype=torch.int32, device=torch.cuda.current_device()
    )
    dist.all_reduce(local_count, op=dist.ReduceOp.MIN, group=edp_group)
    target_count = int(local_count.item())
    if target_count <= 0 or target_count >= len(bucket_groups):
        return bucket_groups

    merged_groups = []
    local_count = len(bucket_groups)
    for group_index in range(target_count):
        start = group_index * local_count // target_count
        end = (group_index + 1) * local_count // target_count
        group_slice = bucket_groups[start:end]
        if not group_slice:
            continue
        buckets = []
        for bucket_group in group_slice:
            buckets.extend(bucket_group.buckets)
        first_group = group_slice[0]
        if (
            first_group.ddp_config.use_distributed_optimizer
            or first_group.ddp_config.overlap_param_gather
        ):
            collective_group = first_group.intra_distributed_optimizer_instance_group
        else:
            collective_group = first_group.data_parallel_group
        base_group = _ParamAndGradBucketGroup(
            buckets,
            first_group.ddp_config,
            collective_group,
            dist.get_world_size(group=collective_group),
        )
        merged_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
            NonuniformEPNCCLParamAndGradBucketGroup
        )
        merged_group.__dict__ = base_group.__dict__.copy()
        merged_group.configure_nonuniform_ep_nccl(runtime_config, nonuniform_ep_config)
        merged_groups.append(merged_group)

    return merged_groups


def _iter_buffer_bucket_params(buffer):
    source_buckets = getattr(buffer, "buckets", None)
    if not isinstance(source_buckets, (list, tuple)):
        yield 0, sorted(buffer.param_index_map, key=lambda param: buffer.param_index_map[param][0])
        return

    for bucket_index, bucket in enumerate(source_buckets):
        params = getattr(bucket, "params_list", None)
        if params is None:
            params = list(getattr(bucket, "params", []))
        yield bucket_index, sorted(params, key=lambda param: buffer.param_index_map[param][0])


def _build_expert_param_bucket_specs(buffers, runtime_config, config, param_to_name):
    """Build one NCCL bucket spec per logical expert parameter slot."""
    local_expert_indices = runtime_config.get("local_expert_indices")
    local_expert_id_set = set(local_expert_indices) if local_expert_indices is not None else set()
    specs = []

    for buffer in buffers:
        for source_bucket_index, bucket_params in _iter_buffer_bucket_params(buffer):
            for param in bucket_params:
                if param not in buffer.param_index_map:
                    continue
                name = param_to_name.get(param, "")
                expert_id = _local_expert_id_from_name(
                    name, config.expert_name_pattern, local_expert_indices
                )
                if expert_id is None:
                    continue

                local_expert_id_set.add(expert_id)
                start, end = buffer.param_index_map[param][:2]
                specs.append(
                    _ExpertBucketSpec(
                        buffer=buffer,
                        source_bucket_index=source_bucket_index,
                        expert_id=expert_id,
                        params=[param],
                        start=start,
                        end=end,
                        slot_key=(_expert_slot_key_from_name(name, config.expert_name_pattern),),
                    )
                )

    runtime_config["_local_expert_id_set"] = local_expert_id_set
    return specs


def _group_expert_bucket_specs_in_backward_order(
    specs: List[_ExpertBucketSpec],
) -> List[Tuple[Tuple[str, ...], List[_ExpertBucketSpec]]]:
    """Group expert slots while preserving their first grad-buffer occurrence."""
    grouped_specs: Dict[Tuple[str, ...], List[_ExpertBucketSpec]] = {}
    for spec in specs:
        grouped_specs.setdefault(spec.slot_key, []).append(spec)
    return list(grouped_specs.items())


def _expert_slot_module_key(slot_key: Tuple[str, ...]) -> Tuple[str, ...]:
    """Return the MoE module path shared by an expert parameter slot."""
    module_names = []
    for name in slot_key:
        match = re.split(r"\.(?:experts|local_experts)\.", name, maxsplit=1)
        if len(match) != 2:
            raise RuntimeError(f"Cannot identify the MoE module for expert slot {name!r}")
        module_names.append(match[0])
    return tuple(module_names)


def _partition_expert_bucket_specs(
    grouped_specs: List[Tuple[Tuple[str, ...], List[_ExpertBucketSpec]]], target_group_count: int
) -> List[List[Tuple[Tuple[str, ...], List[_ExpertBucketSpec]]]]:
    """Partition backward-ordered slots, splitting modules only to reach the target."""
    if not grouped_specs:
        return []

    module_blocks = []
    seen_module_keys = set()
    for grouped_spec in grouped_specs:
        module_key = _expert_slot_module_key(grouped_spec[0])
        if not module_blocks or module_blocks[-1][0] != module_key:
            if module_key in seen_module_keys:
                raise RuntimeError(f"Expert slots for MoE module {module_key} are not contiguous")
            seen_module_keys.add(module_key)
            module_blocks.append((module_key, []))
        module_blocks[-1][1].append(grouped_spec)

    split_units = module_blocks
    if target_group_count > len(module_blocks):
        split_units = [(None, [grouped_spec]) for grouped_spec in grouped_specs]

    group_count = min(target_group_count, len(split_units))
    base_size, remainder = divmod(len(split_units), group_count)
    partitions = []
    start = 0
    for group_index in range(group_count):
        block_count = base_size + (1 if group_index < remainder else 0)
        partition = []
        for _, module_specs in split_units[start : start + block_count]:
            partition.extend(module_specs)
        partitions.append(partition)
        start += block_count
    return partitions


def wrap_nonuniform_ep_nccl_bucket_groups(
    bucket_groups: List[_ParamAndGradBucketGroup],
    runtime_config: dict,
    nonuniform_ep_config: NonuniformEPConfig,
    param_to_bucket_group: Optional[Dict[torch.nn.Parameter, _ParamAndGradBucketGroup]] = None,
) -> List[NonuniformEPNCCLParamAndGradBucketGroup]:
    """Replace generic expert bucket groups with Approach-A NCCL bucket groups."""
    wrapped_bucket_groups = []
    old_to_new = {}

    for bucket_group in bucket_groups:
        if isinstance(bucket_group, NonuniformEPNCCLParamAndGradBucketGroup):
            wrapped_bucket_group = bucket_group
        else:
            wrapped_bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
                NonuniformEPNCCLParamAndGradBucketGroup
            )
            wrapped_bucket_group.__dict__ = bucket_group.__dict__.copy()

        wrapped_bucket_group.configure_nonuniform_ep_nccl(runtime_config, nonuniform_ep_config)
        old_to_new[bucket_group] = wrapped_bucket_group
        wrapped_bucket_groups.append(wrapped_bucket_group)

    wrapped_bucket_groups = _coalesce_nep_nccl_bucket_groups_for_edp_order(
        wrapped_bucket_groups, runtime_config, nonuniform_ep_config
    )
    for wrapped_bucket_group in wrapped_bucket_groups:
        wrapped_bucket_group.next_param_gather_bucket_group = None

    if param_to_bucket_group is not None:
        for wrapped_bucket_group in wrapped_bucket_groups:
            for bucket in wrapped_bucket_group.buckets:
                for param in bucket.params_list:
                    param_to_bucket_group[param] = wrapped_bucket_group

    configure_ordered_bucket_group_scheduler(
        wrapped_bucket_groups,
        "_nep_nccl_scheduler_state",
        "_nep_nccl_group_index",
        "_nep_nccl_ready",
    )
    return wrapped_bucket_groups


def _configure_nep_nccl_task_scheduler(
    bucket_groups: List[NonuniformEPNCCLParamAndGradBucketGroup],
) -> None:
    """Attach a deterministic owner/chunk task order shared by NCCL bucket groups."""
    if not bucket_groups:
        return

    state = getattr(bucket_groups[0], "_nep_nccl_scheduler_state", None)
    if state is None:
        state = {"groups": bucket_groups, "next_index": 0}
        for index, bucket_group in enumerate(bucket_groups):
            bucket_group._nep_nccl_scheduler_state = state
            bucket_group._nep_nccl_group_index = index

    edp_groups = {}
    for bucket_group in bucket_groups:
        edp_bucket_index = getattr(
            bucket_group, "_nep_nccl_edp_bucket_index", bucket_group._nep_nccl_group_index
        )
        edp_groups.setdefault(edp_bucket_index, []).append(bucket_group)

    shared_native_groups = {}
    shared_native_states = []
    expected_edp_contexts = {}
    task_sequence = []
    for bucket_group in bucket_groups:
        bucket_group._nep_nccl_native_edp_bucket_groups = shared_native_groups
        bucket_group._nep_nccl_active_native_edp_states = shared_native_states
        layout = bucket_group._get_nep_nccl_owner_layout()
        bucket_group._nep_nccl_task_count = layout["min_ep_size"] * layout["num_chunks"]
        bucket_group._nep_nccl_ready = bucket_group._nep_nccl_task_count == 0
        edp_bucket_index = getattr(
            bucket_group, "_nep_nccl_edp_bucket_index", bucket_group._nep_nccl_group_index
        )
        for owner_ep_rank in range(layout["min_ep_size"]):
            expected_edp_contexts[(edp_bucket_index, owner_ep_rank)] = (
                expected_edp_contexts.get((edp_bucket_index, owner_ep_rank), 0)
                + layout["num_chunks"]
            )
            for chunk_index, (chunk_start, chunk_end) in enumerate(layout["chunk_ranges"]):
                task_sequence.append(
                    {
                        "group": bucket_group,
                        "owner_ep_rank": owner_ep_rank,
                        "chunk_index": chunk_index,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    }
                )

    group_slot_offsets = []
    next_group_slot = 0
    for bucket_group in bucket_groups:
        group_slot_offsets.append(next_group_slot)
        layout = bucket_group._get_nep_nccl_owner_layout()
        next_group_slot += layout["min_ep_size"] * max(1, layout["num_chunks"])

    state["group_slot_offsets"] = tuple(group_slot_offsets)
    state["task_sequence"] = task_sequence
    state["task_next_index"] = 0
    state["pending_scatters"] = []
    state["edp_groups"] = {
        edp_bucket_index: tuple(
            sorted(
                groups,
                key=lambda group: getattr(
                    group, "_nep_nccl_gather_bucket_index", group._nep_nccl_group_index
                ),
            )
        )
        for edp_bucket_index, groups in edp_groups.items()
    }
    state["expected_edp_contexts"] = expected_edp_contexts
    state["pending_edp_contexts"] = {}


def _configure_nep_end_iteration_scatter_buffers(
    bucket_groups: List[NonuniformEPNCCLParamAndGradBucketGroup],
) -> None:
    """Preallocate one persistent Gather/Scatter buffer set per iteration task."""
    if not bucket_groups:
        return
    if bucket_groups[0].ddp_config.use_distributed_optimizer:
        # DistOpt updates logical owner shards and redistributes parameters;
        # it never scatters reduced gradients back to physical expert holders.
        return

    state = bucket_groups[0]._get_nep_nccl_shared_buffer_state()
    cache = state["gather_buf_cache"]
    slots = set()
    scatter_chunks = _get_nep_nccl_scatter_chunks()
    two_level_scatter_tasks = {}
    for task in state["task_sequence"]:
        group = task["group"]
        owner_ep_rank = task["owner_ep_rank"]
        chunk_index = task["chunk_index"]
        chunk_start = task["chunk_start"]
        chunk_end = task["chunk_end"]
        chunk_size = chunk_end - chunk_start
        slot = group._get_nep_nccl_task_buffer_slot(owner_ep_rank, chunk_index)
        if slot in slots:
            raise RuntimeError(f"End-of-iteration NEP buffer slot {slot} is not unique")
        slots.add(slot)

        dtype = group.buckets[0].grad_data.dtype
        device = group.buckets[0].grad_data.device
        group._get_nep_nccl_cached_tensor(
            cache,
            ("owner_layout_gather", slot, chunk_size, dtype, device),
            chunk_size,
            dtype,
            device,
        )

        ep_rank = group._nep_runtime_config["ep_rank"]
        source_ranks = group._nep_nccl_owner_source_ranks(owner_ep_rank)
        transfer_ranks = group._nep_nccl_owner_transfer_ranks(owner_ep_rank)
        remote_source_ranks = [rank for rank in source_ranks if rank != owner_ep_rank]
        if ep_rank not in transfer_ranks or not remote_source_ranks:
            continue

        if ep_rank in remote_source_ranks:
            gather_input_numel = group._nep_nccl_owner_source_payload_numel(
                owner_ep_rank, ep_rank, chunk_start, chunk_end
            )
            group._get_nep_nccl_cached_tensor(
                cache,
                ("owner_layout_a2a_gather_input", slot, gather_input_numel, dtype, device),
                gather_input_numel,
                dtype,
                device,
            )
        if ep_rank == owner_ep_rank:
            gather_output_numel = sum(
                group._nep_nccl_owner_source_payload_numel(
                    owner_ep_rank, source_ep_rank, chunk_start, chunk_end
                )
                for source_ep_rank in remote_source_ranks
            )
            group._get_nep_nccl_cached_tensor(
                cache,
                ("owner_layout_a2a_gather_output", slot, gather_output_numel, dtype, device),
                gather_output_numel,
                dtype,
                device,
            )

        edp_bucket_index = group._nep_nccl_edp_bucket_index
        two_level_scatter_tasks.setdefault((edp_bucket_index, owner_ep_rank), []).append(task)
        continue

    if two_level_scatter_tasks:
        if scatter_chunks != 1:
            raise RuntimeError(
                "Two-level NEP Gather currently requires "
                "MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=1"
            )
        for (edp_bucket_index, owner_ep_rank), tasks in two_level_scatter_tasks.items():
            representative = tasks[0]["group"]
            dtype = representative.buckets[0].grad_data.dtype
            device = representative.buckets[0].grad_data.device
            ep_rank = representative._nep_runtime_config["ep_rank"]
            source_ranks = representative._nep_nccl_owner_source_ranks(owner_ep_rank)
            transfer_ranks = representative._nep_nccl_owner_transfer_ranks(owner_ep_rank)
            if ep_rank not in transfer_ranks:
                continue
            remote_source_ranks = [rank for rank in source_ranks if rank != owner_ep_rank]
            cache_prefix = (edp_bucket_index, owner_ep_rank, dtype, device)
            if ep_rank == owner_ep_rank:
                scatter_input_numel = sum(
                    task["group"]._nep_nccl_owner_source_payload_numel(
                        owner_ep_rank, destination_ep_rank, task["chunk_start"], task["chunk_end"]
                    )
                    for destination_ep_rank in remote_source_ranks
                    for task in tasks
                )
                representative._get_nep_nccl_cached_tensor(
                    cache,
                    ("owner_layout_a2a_scatter_input_edp",) + cache_prefix,
                    scatter_input_numel,
                    dtype,
                    device,
                )
            elif ep_rank in remote_source_ranks:
                scatter_output_numel = sum(
                    task["group"]._nep_nccl_owner_source_payload_numel(
                        owner_ep_rank, ep_rank, task["chunk_start"], task["chunk_end"]
                    )
                    for task in tasks
                )
                representative._get_nep_nccl_cached_tensor(
                    cache,
                    ("owner_layout_a2a_scatter_output_edp",) + cache_prefix,
                    scatter_output_numel,
                    dtype,
                    device,
                )

    state["end_iteration_scatter_buffer_slots"] = len(slots)






def build_nonuniform_ep_nccl_bucket_groups(
    buffers,
    ddp_config: DistributedDataParallelConfig,
    runtime_config: dict,
    nonuniform_ep_config: NonuniformEPConfig,
    param_to_bucket_group: Dict[torch.nn.Parameter, _ParamAndGradBucketGroup],
    param_to_name: Dict[torch.nn.Parameter, str],
) -> List[NonuniformEPNCCLParamAndGradBucketGroup]:
    """Build common-layout NCCL Approach-A expert groups in backward order."""
    ep_group = runtime_config["ep_group"]
    specs = _build_expert_param_bucket_specs(
        buffers, runtime_config, nonuniform_ep_config, param_to_name
    )
    if buffers and not specs:
        raise RuntimeError(
            "Cannot configure NEP NCCL: expert buffers exist, but no params matched "
            "NonuniformEPConfig.expert_name_pattern."
        )

    bucket_groups = []
    # Grad-buffer offsets already follow backprop order. Preserve first occurrence,
    # then combine adjacent slots so one native DDP readiness unit spans useful
    # backward compute from several MoE layers.
    ordered_grouped_specs = _group_expert_bucket_specs_in_backward_order(specs)
    edp_partitions = _partition_expert_bucket_specs(
        ordered_grouped_specs, _get_nep_nccl_expert_bucket_group_count()
    )
    grouped_partitions = []
    for edp_bucket_index, edp_partition in enumerate(edp_partitions):
        gather_partitions = [edp_partition]
        for gather_bucket_index, gather_partition in enumerate(gather_partitions):
            grouped_partitions.append(
                (edp_bucket_index, gather_bucket_index, len(gather_partitions), gather_partition)
            )

    for group_index, (
        edp_bucket_index,
        gather_bucket_index,
        gather_bucket_count,
        grouped_partition,
    ) in enumerate(grouped_partitions):
        buckets = []
        entries = []
        slot_keys = []
        slot_numels = []
        slot_offset = 0
        for slot_index, (slot_key, unordered_group_specs) in enumerate(grouped_partition):
            group_specs = sorted(unordered_group_specs, key=lambda spec: spec.expert_id)
            candidate_slot_numels = {spec.end - spec.start for spec in group_specs}
            if len(candidate_slot_numels) != 1:
                raise RuntimeError(
                    "NEP NCCL requires equal parameter-slot sizes across experts for "
                    f"slot {slot_key}; got {sorted(candidate_slot_numels)}"
                )
            slot_numel = next(iter(candidate_slot_numels))
            slot_keys.append(slot_key)
            slot_numels.append(slot_numel)
            seen_experts = set()
            for spec in group_specs:
                if spec.expert_id in seen_experts:
                    raise RuntimeError(
                        "NEP NCCL requires one local grad slice per expert and slot; "
                        f"expert {spec.expert_id} appears multiple times for slot {slot_key}"
                    )
                seen_experts.add(spec.expert_id)
                param_data = (
                    spec.buffer.param_data[spec.start : spec.end]
                    if spec.buffer.param_data is not None
                    else None
                )
                grad_data = spec.buffer.grad_data[spec.start : spec.end]
                bucket = _ParamAndGradBucket(
                    params=spec.params,
                    param_data=param_data,
                    grad_data=grad_data,
                    offset=spec.start,
                    numel_unpadded=spec.end - spec.start,
                    gradient_scaling_factor=spec.buffer.gradient_scaling_factor,
                    bucket_id=group_index,
                    param_index_map=spec.buffer.param_index_map,
                    params_with_extra_main_grads=[],
                )
                buckets.append(bucket)
                entries.append(
                    {
                        "expert_id": spec.expert_id,
                        "slot_index": slot_index,
                        "slot_offset": slot_offset,
                        "entry_key": (spec.expert_id, slot_index),
                        "bucket": bucket,
                        "numel": spec.end - spec.start,
                    }
                )
            slot_offset += slot_numel

        bucket_group = NonuniformEPNCCLParamAndGradBucketGroup(
            buckets, ddp_config, ep_group, dist.get_world_size(group=ep_group)
        )
        bucket_group.configure_nonuniform_ep_nccl(
            runtime_config,
            nonuniform_ep_config,
            entries=entries,
            slot_keys=tuple(slot_keys),
            slot_numels=tuple(slot_numels),
        )
        bucket_group._nep_nccl_edp_bucket_index = edp_bucket_index
        bucket_group._nep_nccl_gather_bucket_index = gather_bucket_index
        bucket_group._nep_nccl_gather_bucket_count = gather_bucket_count
        bucket_groups.append(bucket_group)

    for buffer in buffers:
        for param in buffer.param_index_map:
            param_to_bucket_group.pop(param, None)
    for bucket_group in bucket_groups:
        for bucket in bucket_group.buckets:
            for param in bucket.params_list:
                param_to_bucket_group[param] = bucket_group

    configure_ordered_bucket_group_scheduler(
        bucket_groups, "_nep_nccl_scheduler_state", "_nep_nccl_group_index", "_nep_nccl_ready"
    )
    _configure_nep_nccl_task_scheduler(bucket_groups)
    _configure_nep_end_iteration_scatter_buffers(bucket_groups)
    return bucket_groups


class NonuniformEPDistributedDataParallel(DistributedDataParallel):
    """DDP wrapper that opts expert params into nonuniform EP ownership transfer."""

    @staticmethod
    def _synchronize_bucket_size(ddp_config: DistributedDataParallelConfig) -> None:
        """Use the healthy/full-replica native bucket size on every NEP rank."""
        if ddp_config.num_buckets is None or ddp_config.bucket_size is None:
            return

        local_bucket_size = ddp_config.bucket_size
        bucket_size = torch.tensor(
            local_bucket_size, dtype=torch.int64, device=torch.cuda.current_device()
        )
        # Reduced-EP ranks hold more local experts. MIN selects the full-replica
        # value, preserving the healthy run's native dense bucket boundaries while
        # keeping collective shapes identical across DP participants.
        dist.all_reduce(
            bucket_size,
            op=dist.ReduceOp.MIN,
            group=parallel_state.get_data_parallel_group(with_context_parallel=True),
        )
        ddp_config.bucket_size = int(bucket_size.item())

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        nonuniform_ep_config: Optional[NonuniformEPConfig] = None,
        disable_bucketing: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        full_param_layout: Optional[FullParamLayout] = None,
    ):
        self.nonuniform_ep_config = nonuniform_ep_config or NonuniformEPConfig()
        runtime_config = _get_runtime_config(self.nonuniform_ep_config)
        if ddp_config.use_distributed_optimizer:
            if ddp_config.num_distributed_optimizer_instances != 1:
                raise RuntimeError("NEP distributed optimizer initially supports one instance")
            if ddp_config.use_megatron_fsdp:
                raise RuntimeError("NEP distributed optimizer does not support Megatron FSDP")
            if ddp_config.nccl_ub:
                raise RuntimeError("NEP distributed optimizer does not support NCCL UB yet")
            if ddp_config.fp8_param_gather or ddp_config.fp4_param_gather:
                raise RuntimeError(
                    "NEP distributed optimizer initially supports BF16 owner parameters only"
                )

        self._synchronize_bucket_size(ddp_config)
        if ddp_config.use_distributed_optimizer:
            # The caller computes this layout before NEP can synchronize the native
            # bucket threshold. Reduced replicas have different physical expert
            # parameter counts, so that stale threshold can produce different dense
            # reduce-scatter bucket boundaries across DP participants. Recompute with
            # the synchronized threshold using the native DistOpt layout machinery.
            from ..optimizer.distrib_optimizer import DistributedOptimizer

            effective_bucket_size = (
                None
                if disable_bucketing or parallel_state.get_pipeline_model_parallel_rank() > 0
                else ddp_config.bucket_size
            )
            full_param_layout = DistributedOptimizer.compute_full_param_layout(
                [param for param in module.parameters() if param.requires_grad],
                effective_bucket_size,
                parallel_state.get_data_parallel_world_size(with_context_parallel=True),
                ddp_config,
                expert_data_parallel_world_size=(
                    parallel_state.get_expert_data_parallel_world_size()
                ),
            )

        parent_kwargs = {
            "config": config,
            "ddp_config": ddp_config,
            "module": module,
            "disable_bucketing": disable_bucketing,
            "pg_collection": pg_collection,
            "full_param_layout": full_param_layout,
        }
        super().__init__(
            **filter_kwargs_for_callable(DistributedDataParallel.__init__, parent_kwargs)
        )

        self._nonuniform_ep_runtime_config = runtime_config
        self._param_to_name = {param: name for name, param in self.module.named_parameters()}
        self.expert_parallel_bucket_groups = build_nonuniform_ep_nccl_bucket_groups(
            self.expert_parallel_buffers,
            self.ddp_config,
            runtime_config,
            self.nonuniform_ep_config,
            self.param_to_bucket_group,
            self._param_to_name,
        )
        if self.ddp_config.overlap_param_gather:
            bucket_groups = self.expert_parallel_bucket_groups
            for index in range(1, len(bucket_groups)):
                bucket_groups[len(bucket_groups) - index].next_param_gather_bucket_group = (
                    bucket_groups[len(bucket_groups) - index - 1]
                )
        self._configure_nep_distributed_optimizer_buffers()
        self._nep_dispatch_pending_completion_event = None
        self._nep_dispatch_pending_host_phases = None
        self._nep_dispatch_inflight_completion_events = []
        self._nep_dispatch_waiting_groups = None
        self._nep_dispatch_waiting_module_label = None
        self._nep_scatter_batches = []
        self._nep_end_iteration_scatter_context_batches = []
        self._nep_scatter_inflight_event = None
        self._nep_scatter_stream = None
        self._nep_scatter_backward_complete = False
        self._nep_scatter_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self._configure_nep_dispatch_boundary_hooks()
        self._configure_expert_gradient_scaling(config, runtime_config)
        if self.ddp_config.use_distributed_optimizer or runtime_config.get(
            "has_nondivisible_expert_placement", False
        ):
            self._synchronize_nondivisible_expert_parameters()

    def _configure_nep_distributed_optimizer_buffers(self) -> None:
        """Create persistent logical-owner buffers consumed by native DistOpt."""
        self.nonuniform_ep_distributed_optimizer_params_with_names = []
        self.nonuniform_ep_distributed_optimizer_buffers = []
        if not self.ddp_config.use_distributed_optimizer:
            return

        runtime_config = self._nonuniform_ep_runtime_config
        ep_rank = runtime_config["ep_rank"]
        min_ep_size = runtime_config["min_ep_size"]
        edp_group = runtime_config.get("edp_group")
        if ep_rank < min_ep_size and edp_group is None:
            raise RuntimeError("NEP DistOpt owner ranks require an expert-data-parallel group")

        groups_by_edp_bucket = {}
        for bucket_group in self.expert_parallel_bucket_groups:
            edp_bucket_index = bucket_group._nep_nccl_edp_bucket_index
            groups_by_edp_bucket.setdefault(edp_bucket_index, []).append(bucket_group)

        # Preserve adjacent parameter-bucket concurrency while bounding temporary
        # physical-layout staging to two alternating slots across this DDP instance.
        param_transfer_buffers = {}
        param_transfer_slot_count = min(2, len(groups_by_edp_bucket))
        param_transfer_numel_by_slot = {
            slot: max(
                group._get_nep_nccl_owner_layout()["owner_numel"]
                for index, groups in groups_by_edp_bucket.items()
                if index % param_transfer_slot_count == slot
                for group in groups
            )
            for slot in range(param_transfer_slot_count)
        }
        for edp_bucket_index in sorted(groups_by_edp_bucket):
            bucket_groups = tuple(
                sorted(
                    groups_by_edp_bucket[edp_bucket_index],
                    key=lambda group: group._nep_nccl_group_index,
                )
            )
            bundle = {
                "edp_bucket_index": edp_bucket_index,
                "groups": bucket_groups,
                "buffer": None,
                "native_group": None,
                "proxy_by_key": {},
                "params_with_names": [],
                "param_transfer_buffers": param_transfer_buffers,
                "param_transfer_slot": edp_bucket_index % param_transfer_slot_count,
                "param_transfer_numel_by_slot": param_transfer_numel_by_slot,
                "param_sync_completed": False,
            }

            if ep_rank < min_ep_size:
                params_with_names = []
                proxy_by_key = {}
                for bucket_group in bucket_groups:
                    layout = bucket_group._get_nep_nccl_owner_layout()
                    owner_expert_ids = [
                        expert_id
                        for expert_id in layout["owner_expert_slots"][ep_rank]
                        if expert_id is not None
                    ]
                    templates = {}
                    for entry in bucket_group._nep_nccl_entries:
                        params = entry["bucket"].params_list
                        if len(params) != 1:
                            raise RuntimeError(
                                "NEP DistOpt requires exactly one physical parameter per slot"
                            )
                        templates.setdefault(entry["slot_index"], params[0])
                    if len(templates) != len(bucket_group._nep_nccl_slot_keys):
                        raise RuntimeError(
                            "NEP DistOpt could not find a physical template for every owner slot"
                        )

                    for expert_id in owner_expert_ids:
                        for slot_index, slot_key in enumerate(bucket_group._nep_nccl_slot_keys):
                            template = templates[slot_index]
                            proxy = torch.nn.Parameter(
                                torch.empty_like(template.detach()), requires_grad=True
                            )
                            _copy_nep_optimizer_parameter_attributes(template, proxy)
                            proxy.nonuniform_ep_logical_expert_id = expert_id
                            proxy.nonuniform_ep_owner_rank = ep_rank
                            proxy.nonuniform_ep_group_index = bucket_group._nep_nccl_group_index
                            proxy.nonuniform_ep_slot_index = slot_index
                            name = _nep_distopt_proxy_name(slot_key, expert_id)
                            proxy.nonuniform_ep_logical_name = name
                            key = (bucket_group._nep_nccl_group_index, expert_id, slot_index)
                            proxy_by_key[key] = proxy
                            params_with_names.append((proxy, name))

                if not params_with_names:
                    raise RuntimeError("NEP DistOpt owner rank has no logical expert parameters")
                param_dtypes = {param.dtype for param, _ in params_with_names}
                grad_dtypes = {
                    group.buckets[0].grad_data.dtype for group in bucket_groups if group.buckets
                }
                scaling_factors = {
                    bucket.gradient_scaling_factor
                    for group in bucket_groups
                    for bucket in group.buckets
                }
                if len(param_dtypes) != 1 or len(grad_dtypes) != 1 or len(scaling_factors) != 1:
                    raise RuntimeError(
                        "NEP DistOpt owner buckets require one parameter dtype, gradient dtype, "
                        "and scaling factor"
                    )

                params = [param for param, _ in params_with_names]
                owner_layout = _compute_nep_distopt_owner_layout(
                    params, edp_group.size(), self.ddp_config
                )
                owner_buffer = _ParamAndGradBuffer(
                    self.ddp_config,
                    next(iter(param_dtypes)),
                    next(iter(grad_dtypes)),
                    params_with_names,
                    edp_group,
                    None,
                    dict(params_with_names),
                    next(iter(scaling_factors)),
                    list(range(len(params))),
                    False,
                    ProcessGroupCollection(tp=self.tp_group, dp_cp=self.dp_cp_group),
                    param_layout=owner_layout,
                )
                owner_buffer.nonuniform_ep_owner_layout = True
                native_group = _ParamAndGradBucketGroup(
                    owner_buffer.buckets,
                    _nep_owner_ddp_config(self.ddp_config),
                    edp_group,
                    edp_group.size(),
                )
                # Gather readiness is owned explicitly by the NEP scheduler.
                native_group.is_first_batch = False
                bundle.update(
                    {
                        "buffer": owner_buffer,
                        "native_group": native_group,
                        "proxy_by_key": proxy_by_key,
                        "params_with_names": params_with_names,
                    }
                )
                self.nonuniform_ep_distributed_optimizer_params_with_names.extend(params_with_names)
                self.nonuniform_ep_distributed_optimizer_buffers.append(owner_buffer)

            for group_index, bucket_group in enumerate(bucket_groups):
                bucket_group._nep_distopt_owner_bundle = bundle
                bucket_group._nep_distopt_param_sync_representative = group_index == 0

    def get_nonuniform_ep_distributed_optimizer_state(self) -> tuple:
        """Return owner proxy parameters and buffers for the optimizer factory."""
        return (
            tuple(
                (name, param)
                for param, name in self.nonuniform_ep_distributed_optimizer_params_with_names
            ),
            tuple(self.nonuniform_ep_distributed_optimizer_buffers),
        )

    @torch.no_grad()
    def _synchronize_nondivisible_expert_parameters(self) -> None:
        """Give every physical holder the same logical expert parameters.

        Uniform EP initializes equal-shaped expert tensors deterministically on
        every EDP replica. Non-divisible EP does not: for example, EP8 creates
        two-expert tensors while EP6 creates a mix of two- and three-expert
        tensors, so their RNG streams differ. Reuse the Approach-A owner layout
        and existing process groups once at DDP construction time:

        1. gather each replica's physical expert shards into owner layouts;
        2. broadcast the first EDP owner's layout to the other replicas;
        3. broadcast that layout back to every physical holder in each replica.

        This is startup-only synchronization; it adds no training-iteration work.
        """
        runtime_config = self._nonuniform_ep_runtime_config
        ep_rank = runtime_config["ep_rank"]
        ep_group = runtime_config["ep_group"]
        edp_group = runtime_config.get("edp_group")

        for bucket_group in self.expert_parallel_bucket_groups:
            layout = bucket_group._get_nep_nccl_owner_layout()
            for owner_ep_rank in range(layout["min_ep_size"]):
                transfer_group, _, transfer_size, transfer_ranks = (
                    bucket_group._get_nep_nccl_transfer_group_info(owner_ep_rank)
                )
                if ep_rank not in transfer_ranks:
                    continue

                local_entries = bucket_group._nep_nccl_owner_entries(owner_ep_rank)
                if not local_entries:
                    raise RuntimeError(
                        "NEP parameter synchronization found no local expert entries for "
                        f"owner={owner_ep_rank}, ep_rank={ep_rank}"
                    )
                reference_param = local_entries[0]["bucket"].params_list[0]
                owner_params = torch.zeros(
                    layout["owner_numel"],
                    dtype=reference_param.dtype,
                    device=reference_param.device,
                )
                for entry in local_entries:
                    params = entry["bucket"].params_list
                    if len(params) != 1 or params[0].numel() != entry["numel"]:
                        raise RuntimeError(
                            "NEP parameter synchronization requires one exact parameter "
                            f"per expert entry; got params={len(params)}, "
                            f"param_numel={sum(param.numel() for param in params)}, "
                            f"entry_numel={entry['numel']}"
                        )
                    start = bucket_group._nep_nccl_entry_owner_start(entry, owner_ep_rank)
                    owner_params[start : start + entry["numel"]].copy_(params[0].detach().view(-1))

                if transfer_group is not None and transfer_size > 1:
                    dist.all_reduce(owner_params, group=transfer_group)

                if ep_rank == owner_ep_rank and edp_group is not None and edp_group.size() > 1:
                    dist.broadcast(owner_params, src=get_global_rank(edp_group, 0), group=edp_group)

                if self.ddp_config.use_distributed_optimizer and ep_rank == owner_ep_rank:
                    bundle = bucket_group._nep_distopt_owner_bundle
                    group_index = bucket_group._nep_nccl_group_index
                    for expert_id in bucket_group._nep_nccl_owner_expert_ids(owner_ep_rank):
                        _, owner_slot = bucket_group._nep_nccl_owner_slot_for_expert(expert_id)
                        for slot_index, (slot_offset, slot_numel) in enumerate(
                            zip(
                                bucket_group._nep_nccl_slot_offsets,
                                bucket_group._nep_nccl_slot_numels,
                            )
                        ):
                            proxy = bundle["proxy_by_key"][(group_index, expert_id, slot_index)]
                            start = owner_slot * layout["expert_stride"] + slot_offset
                            proxy.copy_(owner_params[start : start + slot_numel].view_as(proxy))

                if transfer_group is not None and transfer_size > 1:
                    dist.broadcast(
                        owner_params,
                        src=get_global_rank(ep_group, owner_ep_rank),
                        group=transfer_group,
                    )

                for entry in local_entries:
                    start = bucket_group._nep_nccl_entry_owner_start(entry, owner_ep_rank)
                    entry["bucket"].params_list[0].copy_(
                        owner_params[start : start + entry["numel"]].view_as(
                            entry["bucket"].params_list[0]
                        )
                    )






    def _retire_nep_scatter_chunk(self, force: bool = False) -> bool:
        """Retire the last Scatter chunk without launching NCCL from another thread."""
        completion_event = self._nep_scatter_inflight_event
        if completion_event is None:
            return True
        if force:
            with torch.profiler.record_function("nep_wait_scatter_chunk_completion"):
                completion_event.synchronize()
        elif not completion_event.query():
            return False
        self._nep_scatter_inflight_event = None
        return True





    def _submit_nep_scatter_chunk(self) -> bool:
        """Submit one end-of-iteration Scatter chunk in stream order."""
        self._retire_nep_scatter_chunk()
        if not self._nep_scatter_batches:
            return False

        batch = self._nep_scatter_batches[0]
        while batch.get("submission_complete", False):
            self._nep_scatter_batches.pop(0)
            if not self._nep_scatter_batches:
                return True
            batch = self._nep_scatter_batches[0]

        scatter_stream = self._nep_scatter_stream
        if scatter_stream is None:
            raise RuntimeError("End-of-iteration NEP Scatter stream was not initialized")
        train = batch["trains"][batch["next_train"]]
        descriptor_index = train["next_descriptor"]
        descriptor = train["descriptors"][descriptor_index]
        context = train["context"]
        group_index = getattr(train["group"], "_nep_nccl_group_index", -1)
        with torch.cuda.stream(scatter_stream):
            with torch.profiler.record_function(
                "nep_scheduled_scatter_chunk_"
                f"g{group_index}_o{context['owner_ep_rank']}_"
                f"c{context['chunk_index']}_s{descriptor_index}"
            ):
                train["group"]._submit_nep_nccl_owner_all_to_all_scatter(descriptor)
                train["group"]._order_nep_nccl_owner_all_to_all_scatter_completion(descriptor)
                train["group"]._finish_nep_nccl_owner_all_to_all_scatter(descriptor)
            chunk_done_event = torch.cuda.Event()
            chunk_done_event.record(scatter_stream)

            train["next_descriptor"] += 1
            if train["next_descriptor"] == len(train["descriptors"]):
                train["group"]._finish_nep_nccl_scatter_train_submission(train)
                batch["next_train"] += 1
            if batch["next_train"] == len(batch["trains"]):
                batch["completion_event"].record(scatter_stream)
                batch["submission_complete"] = True

        self._nep_scatter_inflight_event = chunk_done_event
        return True


    def _queue_nep_scatter_context_batches(
        self,
        context_batches: List[List[dict]],
        completion_event: torch.cuda.Event,
        module_label: str,
    ) -> None:
        """Prepare a layer's end-of-iteration Scatter trains in owner order."""
        scatter_stream = self._nep_scatter_stream
        if scatter_stream is None:
            raise RuntimeError("End-of-iteration NEP Scatter state was not initialized")

        scheduled_context_batches = context_batches or [[]]
        with torch.cuda.stream(scatter_stream):
            for batch_index, contexts in enumerate(scheduled_context_batches):
                trains = []
                for context in contexts:
                    train = context["group"]._prepare_nep_nccl_owner_task_scatter_train(context)
                    train["task_marked"] = True
                    trains.append(train)

                is_last_batch = batch_index == len(scheduled_context_batches) - 1
                batch_completion_event = completion_event if is_last_batch else torch.cuda.Event()
                batch_label = (
                    module_label
                    if len(scheduled_context_batches) == 1
                    else f"{module_label}:owner_batch_{batch_index}"
                )
                batch = {
                    "trains": trains,
                    "next_train": 0,
                    "completion_event": batch_completion_event,
                    "module_label": batch_label,
                    "submission_complete": not trains,
                }
                if not trains:
                    batch_completion_event.record(scatter_stream)
                self._nep_scatter_batches.append(batch)

    def _defer_nep_scatter_context_batches_to_iteration_end(
        self,
        context_batches: List[List[dict]],
        completion_event: torch.cuda.Event,
        module_label: str,
    ) -> None:
        """Retain persistent task buffers until the end-of-iteration Scatter drain."""
        if self.ddp_config.use_distributed_optimizer:
            for contexts in context_batches:
                for context in contexts:
                    context["group"]._mark_nep_distopt_task_complete(context)
            completion_event.record(torch.cuda.current_stream())
            return

        for contexts in context_batches:
            for context in contexts:
                scatter_contexts = context.get("scatter_contexts")
                if scatter_contexts is None:
                    context["group"]._mark_nep_nccl_task_started(
                        context["owner_ep_rank"], context["chunk_index"]
                    )
                else:
                    for task_context in scatter_contexts:
                        task_context["group"]._mark_nep_nccl_task_started(
                            task_context["owner_ep_rank"], task_context["chunk_index"]
                        )
        self._nep_end_iteration_scatter_context_batches.append(
            (context_batches, completion_event, module_label)
        )

    def _materialize_next_nep_end_iteration_scatter_batch(self) -> bool:
        """Pack every deferred EDP bucket into one Scatter per owner."""
        if not self._nep_end_iteration_scatter_context_batches:
            return False

        deferred_batches = self._nep_end_iteration_scatter_context_batches
        self._nep_end_iteration_scatter_context_batches = []
        contexts_by_owner = {}
        completion_events = []
        for context_batches, completion_event, _ in deferred_batches:
            completion_events.append(completion_event)
            for context_batch in context_batches:
                for context in context_batch:
                    task_contexts = context.get("scatter_contexts") or (context,)
                    for task_context in task_contexts:
                        contexts_by_owner.setdefault(task_context["owner_ep_rank"], []).append(
                            task_context
                        )

        packed_contexts = []
        for owner_ep_rank in sorted(contexts_by_owner):
            owner_contexts = contexts_by_owner[owner_ep_rank]
            owner_contexts.sort(
                key=lambda context: (
                    context["group"]._nep_nccl_edp_bucket_index,
                    context["group"]._nep_nccl_group_index,
                    context["chunk_index"],
                )
            )
            packed_contexts.append(
                owner_contexts[0]["group"]._coalesce_nep_nccl_scatter_contexts(owner_contexts)
            )
        if not packed_contexts:
            raise RuntimeError("Deferred NEP Scatter drain has no contexts")

        self._nep_end_iteration_scatter_completion_events = completion_events[:-1]
        self._queue_nep_scatter_context_batches(
            [packed_contexts], completion_events[-1], "end_iteration_scatter"
        )
        return True

    def _drain_nep_scatter_scheduler(self) -> None:
        """Queue all deferred Scatter chunks, then wait once before final gradient sync."""
        if self.ddp_config.use_distributed_optimizer:
            if self._nep_end_iteration_scatter_context_batches or self._nep_scatter_batches:
                raise RuntimeError("NEP DistOpt unexpectedly queued a gradient Scatter")
            return
        if not self._nep_scatter_backward_complete:
            raise RuntimeError("NEP Scatter drain requires a completed backward pass")
        with torch.profiler.record_function("nep_end_iteration_scatter_global_fence"):
            torch.cuda.synchronize()
        while self._materialize_next_nep_end_iteration_scatter_batch():
            pass
        while self._nep_scatter_batches:
            if not self._submit_nep_scatter_chunk():
                raise RuntimeError("End-of-iteration NEP Scatter drain made no progress")
        for completion_event in getattr(
            self, "_nep_end_iteration_scatter_completion_events", ()
        ):
            completion_event.record(self._nep_scatter_stream)
        self._nep_end_iteration_scatter_completion_events = []
        self._retire_nep_scatter_chunk(force=True)



    def _configure_nep_dispatch_boundary_hooks(self) -> None:
        """Map expert buckets to their AccumulateGrad launch callbacks."""
        from ..transformer.moe.moe_layer import BaseMoELayer

        expert_groups = set(self.expert_parallel_bucket_groups)
        assigned_groups = set()
        for module_name, module in self.module.named_modules():
            if not isinstance(module, BaseMoELayer):
                continue

            module_groups = []
            for param in module.parameters():
                bucket_group = self.param_to_bucket_group.get(param)
                if bucket_group in expert_groups and bucket_group not in module_groups:
                    module_groups.append(bucket_group)
            module_groups.sort(key=lambda group: group._nep_nccl_group_index)
            for bucket_group in module_groups:
                assigned_groups.add(bucket_group)
                bucket_group._nep_dispatch_boundary_launch = True
                bucket_group._nep_dispatch_boundary_callback = (
                    self._launch_and_release_nep_two_level_gather
                )
                bucket_group._nep_dispatch_boundary_groups = (bucket_group,)
                bucket_group._nep_dispatch_boundary_module_label = module_name

        missing_groups = expert_groups - assigned_groups
        if missing_groups:
            missing_indices = sorted(group._nep_nccl_group_index for group in missing_groups)
            raise RuntimeError(
                "NCCL NEP could not map expert bucket groups to MoE modules: "
                f"groups={missing_indices}"
            )



    def _launch_and_release_nep_two_level_gather(self, groups: tuple, module_label: str) -> None:
        """Submit one ready Gather group and release its host boundary immediately."""
        if len(groups) != 1:
            raise RuntimeError("Two-level NEP Gather callbacks must contain exactly one group")
        state = groups[0]._nep_nccl_scheduler_state
        task_index = state["task_next_index"]
        if task_index >= len(state["task_sequence"]):
            return
        next_group = state["task_sequence"][task_index]["group"]
        if not next_group._nep_dispatch_boundary_ready:
            return
        next_label = next_group._nep_dispatch_boundary_module_label or module_label
        if not self._launch_nep_dispatch_boundary_tasks((next_group,), next_label):
            return
        self._wait_for_nep_dispatch_launch()

    def _launch_nep_dispatch_boundary_tasks(self, groups: tuple, module_label: str) -> bool:
        """Submit a layer batch after dispatch and all local source grads are ready."""
        waiting_groups = self._nep_dispatch_waiting_groups
        if waiting_groups is None:
            self._nep_dispatch_waiting_groups = groups
            self._nep_dispatch_waiting_module_label = module_label
        elif waiting_groups != groups:
            raise RuntimeError(
                "NEP dispatch boundary became ready before the prior boundary was consumed: "
                f"waiting={self._nep_dispatch_waiting_module_label} ready={module_label}"
            )
        if any(group._nep_dispatch_boundary_launched for group in groups):
            return True
        if any(group._nep_dispatch_boundary_launching for group in groups):
            return False
        if not all(group._nep_dispatch_boundary_ready for group in groups):
            return False
        if not all(group._nep_dispatch_boundary_inputs_ready() for group in groups):
            if not all(group._nep_dispatch_boundary_wait_logged for group in groups):
                for group in groups:
                    group._nep_dispatch_boundary_wait_logged = True
            return False

        if (
            self._nep_dispatch_pending_completion_event is not None
            or getattr(self, "_nep_dispatch_pending_host_phases", None) is not None
        ):
            raise RuntimeError("Prior NEP dispatch completion has not been consumed")
        compute_ready_event = torch.cuda.Event()
        compute_ready_event.record(torch.cuda.current_stream())
        completion_event = torch.cuda.Event()
        for group in groups:
            group._nep_dispatch_boundary_launching = True
        self._nep_dispatch_pending_completion_event = completion_event
        self._run_nep_dispatch_boundary_tasks(
            groups, module_label, compute_ready_event, completion_event
        )
        return True

    def _run_nep_dispatch_boundary_tasks(
        self,
        groups: tuple,
        module_label: str,
        compute_ready_event: torch.cuda.Event,
        completion_event: Optional[torch.cuda.Event] = None,
    ) -> torch.cuda.Event:
        """Enqueue one layer batch and return its completion event."""
        try:
            state = groups[0]._nep_nccl_scheduler_state
            first_task_index = state["task_next_index"]
            pending_host_phases = groups[0]._try_start_nep_nccl_ready_tasks(
                force_ready=False, async_op_override=True, compute_ready_event=compute_ready_event
            )
            last_task_index = state["task_next_index"]

            if completion_event is None:
                completion_event = torch.cuda.Event()
            if pending_host_phases:
                if getattr(self, "_nep_dispatch_pending_host_phases", None) is not None:
                    raise RuntimeError("Prior split NEP host phases have not been consumed")
                self._nep_dispatch_pending_host_phases = (groups[0], pending_host_phases)
            else:
                not_started = [
                    group._nep_nccl_group_index for group in groups if not group._nep_nccl_ready
                ]
                if not_started:
                    raise RuntimeError(
                        "NEP dispatch boundary did not launch every layer bucket group: "
                        f"module={module_label}, groups={not_started}"
                    )
                completion_event.record(groups[0]._get_nep_nccl_comm_stream(0))
            launched_groups = tuple(
                dict.fromkeys(
                    task["group"]
                    for task in state["task_sequence"][first_task_index:last_task_index]
                )
            )
            if not launched_groups:
                raise RuntimeError(
                    "Two-level NEP Gather callback did not advance the canonical task prefix"
                )
            for group in launched_groups:
                group._nep_dispatch_boundary_launched = True
            return completion_event
        finally:
            for group in groups:
                group._nep_dispatch_boundary_launching = False




    def _finish_pending_nep_dispatch_host_phases(self) -> bool:
        """Finish a launched Gather/EDP pipeline and defer its packed Scatter."""
        pending_host_phases = getattr(self, "_nep_dispatch_pending_host_phases", None)
        if pending_host_phases is None:
            return False

        groups = self._nep_dispatch_waiting_groups
        module_label = self._nep_dispatch_waiting_module_label
        completion_event = self._nep_dispatch_pending_completion_event
        if groups is None:
            raise RuntimeError("Split NEP host phases are missing their waiting groups")
        if completion_event is None:
            raise RuntimeError("Split NEP host phases are missing their completion event")

        scatter_context_batches = []
        boundary_group, host_phases = pending_host_phases
        phases_finished = boundary_group._finish_nep_nccl_process_group_dispatch_batches(
            host_phases, scatter_context_batches=scatter_context_batches
        )
        if phases_finished is False:
            return False

        if scatter_context_batches:
            self._defer_nep_scatter_context_batches_to_iteration_end(
                scatter_context_batches, completion_event, module_label
            )
        else:
            completion_event.record(boundary_group._get_nep_nccl_comm_stream(0))
        not_ready = [group._nep_nccl_group_index for group in groups if not group._nep_nccl_ready]
        if not_ready and scatter_context_batches:
            raise RuntimeError(
                "Split NEP host phases did not schedule every layer bucket group: "
                f"module={module_label}, groups={not_ready}"
            )
        self._nep_dispatch_pending_host_phases = None
        return True


    def _wait_for_nep_dispatch_launch(self, final: bool = False) -> None:
        """Retire one host launch and optionally fence all device-inflight reshards."""
        groups = self._nep_dispatch_waiting_groups
        module_label = self._nep_dispatch_waiting_module_label
        completion_event = self._nep_dispatch_pending_completion_event
        pending_host_phases = getattr(self, "_nep_dispatch_pending_host_phases", None)

        if pending_host_phases is not None:
            if not self._finish_pending_nep_dispatch_host_phases():
                raise RuntimeError("The next model-EP boundary did not finish split NEP phases")

        if completion_event is not None:
            inflight_events = getattr(self, "_nep_dispatch_inflight_completion_events", None)
            if inflight_events is None:
                inflight_events = []
                self._nep_dispatch_inflight_completion_events = inflight_events
            inflight_events.append((module_label, completion_event))
            self._nep_dispatch_pending_completion_event = None
        if groups is not None:
            if not all(group._nep_dispatch_boundary_launched for group in groups):
                raise RuntimeError(f"NEP dispatch launch did not finish: module={module_label}")
            self._nep_dispatch_waiting_groups = None
            self._nep_dispatch_waiting_module_label = None

        if final:
            self._nep_scatter_backward_complete = True
            self._drain_nep_scatter_scheduler()

            inflight_events = getattr(self, "_nep_dispatch_inflight_completion_events", None)
            if inflight_events:
                current_stream = torch.cuda.current_stream()
                for _, deferred_completion_event in inflight_events:
                    current_stream.wait_event(deferred_completion_event)
                inflight_events.clear()

    def _configure_expert_gradient_scaling(self, config: TransformerConfig, runtime_config: dict):
        if config.calculate_per_token_loss:
            expert_gradient_scaling_factor = 1.0
        elif self.ddp_config.average_in_collective:
            expert_gradient_scaling_factor = runtime_config.get("num_replicas", 1) / max(
                1, runtime_config.get("dp_size", self.dp_cp_group.size())
            )
        else:
            expert_gradient_scaling_factor = 1.0 / self.dp_cp_group.size()

        for bucket_group in self.expert_parallel_bucket_groups:
            for bucket in bucket_group.buckets:
                bucket.gradient_scaling_factor = expert_gradient_scaling_factor
        for buffer in self.nonuniform_ep_distributed_optimizer_buffers:
            buffer.gradient_scaling_factor = expert_gradient_scaling_factor
            for bucket in buffer.buckets:
                bucket.gradient_scaling_factor = expert_gradient_scaling_factor

    def zero_grad_buffer(self):
        """Reset physical gradients and persistent logical-owner gradients."""
        super().zero_grad_buffer()
        for buffer in self.nonuniform_ep_distributed_optimizer_buffers:
            buffer.reset()

    def scale_gradients(self, scaling_factor: float):
        """Scale physical gradients and persistent logical-owner gradients."""
        super().scale_gradients(scaling_factor)
        for buffer in self.nonuniform_ep_distributed_optimizer_buffers:
            buffer.scale_gradients(scaling_factor)

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        self._wait_for_nep_dispatch_launch(final=True)
        result = super().finish_grad_sync(force_all_reduce=force_all_reduce)
        return result
