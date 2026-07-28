# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in nonuniform expert-parallel gradient ownership transfer.
This module keeps nonuniform EP out of generic Megatron DDP.  Expert params
are wrapped into expert-level bucket groups.  Non-owner ranks transfer whole
expert gradients to an owner rank with all-to-all ops; owner ranks accumulate
those incoming gradients into their normal contiguous ``main_grad`` storage before
running the ordinary expert-data-parallel grad sync.  The synced gradients are
then scattered back to every source rank so non-distributed optimizers can step
the local expert params on every rank.
"""
import os
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .. import parallel_state
from ..process_groups_config import ProcessGroupCollection
from ..transformer.cuda_graphs import is_graph_capturing
from ..transformer.transformer_config import TransformerConfig
from .distributed_data_parallel import DistributedDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .nonuniform_common import (
    NonuniformEPRankGenerator,
    compute_nonuniform_ep_expert_placement,
    configure_ordered_bucket_group_scheduler,
    filter_kwargs_for_callable,
    get_nonuniform_ep_runtime_config,
    reset_ordered_bucket_group_scheduler,
    set_nonuniform_ep_runtime_config,
)
from .param_and_grad_buffer import _ParamAndGradBucket, _ParamAndGradBucketGroup

_NEP_TAG_SLOT_STRIDE = 256
_NEP_NCCL_DEFAULT_MAX_GATHER_BYTES = 8 * 1024 * 1024 * 1024
_NEP_NCCL_DEFAULT_ASYNC_CHUNK_WINDOW = 2
_NEP_NCCL_DEFAULT_EXPERT_BUCKET_GROUPS = 3


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
    if work is None:
        return
    block_current_stream = getattr(work, "block_current_stream", None)
    if block_current_stream is not None:
        block_current_stream()
    else:
        work.wait()


@dataclass
class NonuniformEPConfig:
    """Configuration for the single supported NCCL nonuniform-EP path."""

    approach: str = "nccl"
    runtime_config: Optional[dict] = None
    expert_owner: Optional[Dict[int, int]] = None
    expert_name_pattern: Union[str, re.Pattern] = field(
        default_factory=_default_expert_name_pattern
    )
    require_owner_local_expert: bool = True

    def __post_init__(self):
        if self.approach != "nccl":
            raise ValueError("Only the NCCL nonuniform-EP approach is supported")
        if isinstance(self.expert_name_pattern, str):
            self.expert_name_pattern = re.compile(self.expert_name_pattern)


def _source_ep_ranks_for_owner(
    expert_placement: List[List[int]], owner_ep_rank: int, num_experts: int, min_ep_size: int
) -> List[int]:
    experts_per_owner = num_experts // min_ep_size
    owner_first_expert = owner_ep_rank * experts_per_owner
    owner_last_expert = owner_first_expert + experts_per_owner
    return [
        source_ep_rank
        for source_ep_rank, expert_ids in enumerate(expert_placement)
        if any(owner_first_expert <= expert_id < owner_last_expert for expert_id in expert_ids)
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
        "ep_group_gloo": None,
        "nep_transfer_group": ep_group if ep_group is not None else dist.group.WORLD,
        "nep_owner_gather_groups": {},
        "nep_owner_transfer_groups": {},
        "nep_owner_transfer_groups_gloo": {},
        "nep_owner_scatter_launch_groups_gloo": {},
        "nep_owner_scatter_ready_groups_gloo": {},
        "nep_owner_transfer_group_ranks": {},
        "nep_owner_source_ranks": {},
        "edp_group": None,
        "edp_group_gloo": None,
        "ep_rank": ep_rank,
        "local_expert_indices": None,
        "expert_placement": None,
    }


def _get_runtime_config(config: NonuniformEPConfig) -> dict:
    if config.runtime_config is not None:
        return dict(config.runtime_config)
    return _runtime_config_from_parallel_state()


def _get_nep_owner_scatter_launch_group(runtime_config: dict, owner_ep_rank: int):
    return runtime_config["nep_owner_scatter_launch_groups_gloo"].get(owner_ep_rank)


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
    return parallel_state.create_group(
        ranks, timeout=timeout, backend=backend, pg_options=pg_options, group_desc=desc
    )


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
        if num_moe_experts % local_ep_size != 0:
            raise RuntimeError(
                f"num_moe_experts ({num_moe_experts}) must be divisible by local EP "
                f"size {local_ep_size} for replica {replica_index}"
            )
    timeout = timedelta(minutes=distributed_timeout_minutes)
    nccl_comm_cfgs = _get_nccl_communicator_configs(nccl_communicator_config_path)
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
    min_ep_size = generator.min_k * tp * cp // etp
    nep_transfer_group = None
    ep_phase_group_gloo = None
    nep_owner_gather_groups = {}
    nep_owner_transfer_groups = {}
    nep_owner_transfer_groups_gloo = {}
    nep_owner_scatter_launch_groups_gloo = {}
    nep_owner_scatter_ready_groups_gloo = {}
    nep_owner_transfer_group_ranks = {}
    nep_owner_source_ranks = {}
    for ranks in generator.get_ranks("ep"):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "ep")
        group_gloo = (
            _create_group(ranks, timeout, nccl_comm_cfgs, "NEP_EP_PHASE_GLOO", "gloo")
            if create_gloo_process_groups
            else None
        )
        transfer_group = _create_group(ranks, timeout, nccl_comm_cfgs, "nep_grad_transfer")
        group_expert_placement, _ = compute_nonuniform_ep_expert_placement(
            num_moe_experts, len(ranks), min_ep_size, preferred_follower_fanout=1
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
            owner_transfer_group_gloo = None
            owner_scatter_launch_group_gloo = None
            if len(transfer_global_ranks) > 1:
                owner_transfer_group = _create_group(
                    transfer_global_ranks, timeout, nccl_comm_cfgs, "nep_owner_transfer"
                )
                owner_gather_group = _create_group(
                    transfer_global_ranks, timeout, nccl_comm_cfgs, "nep_owner_gather"
                )
                if create_gloo_process_groups:
                    owner_transfer_group_gloo = _create_group(
                        transfer_global_ranks,
                        timeout,
                        nccl_comm_cfgs,
                        "nep_owner_transfer_gloo",
                        "gloo",
                    )
                    owner_scatter_launch_group_gloo = _create_group(
                        transfer_global_ranks,
                        timeout,
                        nccl_comm_cfgs,
                        "NEP_OWNER_SCATTER_LAUNCH_GLOO",
                        "gloo",
                    )
            if rank in ranks:
                nep_owner_source_ranks[owner_ep_rank] = source_ep_ranks
                nep_owner_transfer_group_ranks[owner_ep_rank] = transfer_ep_ranks
                if rank in transfer_global_ranks:
                    if owner_gather_group is not None:
                        nep_owner_gather_groups[owner_ep_rank] = owner_gather_group
                    nep_owner_transfer_groups[owner_ep_rank] = owner_transfer_group
                    nep_owner_transfer_groups_gloo[owner_ep_rank] = owner_transfer_group_gloo
                    nep_owner_scatter_launch_groups_gloo[owner_ep_rank] = (
                        owner_scatter_launch_group_gloo
                    )
        if rank in ranks:
            _set_parallel_state_attr("_EXPERT_MODEL_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_EXPERT_MODEL_PARALLEL_RANKS", ranks)
            nep_transfer_group = transfer_group
            ep_phase_group_gloo = group_gloo
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
        num_moe_experts, local_ep_size, min_ep_size, preferred_follower_fanout=1
    )
    runtime_config = {
        "needs_reshard": local_ep_size > min_ep_size,
        "local_ep_size": local_ep_size,
        "min_ep_size": min_ep_size,
        "num_replicas": generator.num_replicas,
        "dp_size": sum(num_tp_cp_per_replica),
        "ep_group": ep_group,
        "ep_group_gloo": ep_phase_group_gloo,
        "nep_transfer_group": nep_transfer_group,
        "nep_owner_gather_groups": nep_owner_gather_groups,
        "nep_owner_transfer_groups": nep_owner_transfer_groups,
        "nep_owner_transfer_groups_gloo": nep_owner_transfer_groups_gloo,
        "nep_owner_scatter_launch_groups_gloo": nep_owner_scatter_launch_groups_gloo,
        "nep_owner_scatter_ready_groups_gloo": nep_owner_scatter_ready_groups_gloo,
        "nep_owner_transfer_group_ranks": nep_owner_transfer_group_ranks,
        "nep_owner_source_ranks": nep_owner_source_ranks,
        "dp_cp_group_gloo": (
            parallel_state.get_data_parallel_group_gloo(with_context_parallel=True)
            if create_gloo_process_groups
            else None
        ),
        "edp_group": parallel_state.get_expert_data_parallel_group(),
        "edp_group_gloo": (
            parallel_state.get_expert_data_parallel_group_gloo()
            if create_gloo_process_groups
            else None
        ),
        "ep_rank": ep_rank,
        "is_edp_eligible": ep_rank < min_ep_size,
        "is_b_leader": ep_rank < min_ep_size,
        "local_expert_indices": expert_placement[ep_rank],
        "expert_placement": expert_placement,
        "expert_gather_map": expert_gather_map,
    }
    set_nonuniform_ep_runtime_config(runtime_config)
    return runtime_config


def _owner_for_expert(
    expert_id: int, runtime_config: dict, explicit_owner: Optional[Dict[int, int]]
) -> int:
    if explicit_owner is not None and expert_id in explicit_owner:
        return explicit_owner[expert_id]
    min_ep_size = runtime_config.get("min_ep_size")
    placement = runtime_config.get("expert_placement")
    if min_ep_size is not None and placement is not None:
        num_experts = sum(len(experts) for experts in placement)
        experts_per_owner = num_experts // min_ep_size
        return min(expert_id // experts_per_owner, min_ep_size - 1)
    ep_rank = runtime_config["ep_rank"]
    return ep_rank


def _source_ep_ranks_for_expert(expert_id: int, runtime_config: dict) -> List[int]:
    placement = runtime_config.get("expert_placement")
    if placement is None:
        return [runtime_config["ep_rank"]]
    return [ep_rank for ep_rank, experts in enumerate(placement) if expert_id in experts]


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
    """Expert bucket group that transfers grads through owner layout."""

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
        self._nep_nccl_overlap_debug_events = []
        self._nep_dispatch_boundary_launch = False
        self._nep_dispatch_boundary_ready = False
        self._nep_dispatch_boundary_graph_replay_ready = False
        self._nep_dispatch_boundary_launched = False
        self._nep_dispatch_boundary_launching = False
        self._nep_dispatch_boundary_wait_logged = False
        self._nep_dispatch_boundary_callback = None
        self._nep_dispatch_boundary_groups = ()
        self._nep_dispatch_boundary_module_label = None
        self._nep_dispatch_boundary_required_modules = set()
        self._nep_dispatch_boundary_ready_modules = set()

    def _get_nep_nccl_shared_buffer_state(self) -> dict:
        state = getattr(self, "_nep_nccl_scheduler_state", None)
        if state is None:
            state = self._nep_nccl_buffer_state
        state.setdefault("gather_buf_cache", {})
        state.setdefault("buffer_slot_handles", {})
        return state

    def _order_nep_nccl_buffer_slot(self, slot_key: tuple) -> None:
        state = self._get_nep_nccl_shared_buffer_state()
        slot_handles = state["buffer_slot_handles"]
        handles = slot_handles.pop(slot_key, [])
        for work in handles:
            _nep_block_current_stream(work)

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
            stream_key = stream_slot
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
        buffer_slots = _get_nep_nccl_async_chunk_window()
        group_index = max(0, getattr(self, "_nep_nccl_group_index", 0))
        task_ordinal = (
            group_index * layout["min_ep_size"] * max(1, layout["num_chunks"])
            + owner_ep_rank * max(1, layout["num_chunks"])
            + chunk_index
        )
        return task_ordinal % buffer_slots

    def _get_nep_nccl_owner_layout(self) -> dict:
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
        if num_experts % min_ep_size != 0:
            raise RuntimeError(
                f"NEP NCCL owner layout requires num_experts ({num_experts}) to be "
                f"divisible by min_ep_size ({min_ep_size})"
            )
        expert_stride = getattr(self, "_nep_nccl_expert_stride", None)
        if expert_stride is None:
            expert_stride = self._nep_nccl_slot_numel
        if expert_stride is None:
            raise RuntimeError("NEP NCCL bucket group is missing slot-size metadata")
        experts_per_owner = num_experts // min_ep_size
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

    def _nep_nccl_owner_entries(self, owner_ep_rank: int) -> List[dict]:
        layout = self._get_nep_nccl_owner_layout()
        experts_per_owner = layout["experts_per_owner"]
        owner_first_expert = owner_ep_rank * experts_per_owner
        owner_last_expert = owner_first_expert + experts_per_owner
        return [
            entry
            for entry in self._nep_nccl_entries
            if owner_first_expert <= entry["expert_id"] < owner_last_expert
        ]

    @staticmethod
    def _nep_nccl_entry_key(entry: dict) -> tuple:
        return entry.get("entry_key", (entry["expert_id"], entry.get("slot_index", 0)))

    def _nep_nccl_entry_owner_start(self, entry: dict, owner_first_expert: int) -> int:
        expert_stride = getattr(self, "_nep_nccl_expert_stride", self._nep_nccl_slot_numel)
        return (entry["expert_id"] - owner_first_expert) * expert_stride + entry.get(
            "slot_offset", 0
        )

    def _nep_nccl_owner_task_ready(
        self, owner_ep_rank: int, respect_dispatch_boundary: bool = True
    ) -> bool:
        if self.is_first_batch:
            return False
        if (
            respect_dispatch_boundary
            and self._nep_dispatch_boundary_launch
            and not self._nep_dispatch_boundary_ready
        ):
            return False
        if self._nep_dispatch_boundary_graph_replay_ready:
            return True
        for entry in self._nep_nccl_owner_entries(owner_ep_rank):
            for param in entry["bucket"].params_list:
                ready_count = self.per_param_grad_ready_counts.get(param, 0)
                expected_count = self.golden_per_param_grad_ready_counts.get(param)
                if expected_count is None or ready_count < expected_count:
                    return False
        return True

    def _nep_dispatch_boundary_inputs_ready(self) -> bool:
        layout = self._get_nep_nccl_owner_layout()
        return all(
            self._nep_nccl_owner_task_ready(owner_ep_rank, respect_dispatch_boundary=False)
            for owner_ep_rank in range(layout["min_ep_size"])
        )

    def _prep_nep_nccl_owner_entries_for_sync(self, owner_ep_rank: int) -> None:
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
        experts_per_owner = self._nep_nccl_experts_per_owner
        owner_first_expert = owner_ep_rank * experts_per_owner

        def build_views():
            destinations = []
            sources = []
            for entry in self._nep_nccl_entries:
                expert_id = entry["expert_id"]
                owner_local_expert_index = expert_id - owner_first_expert
                if owner_local_expert_index < 0 or owner_local_expert_index >= experts_per_owner:
                    continue
                entry_start = self._nep_nccl_entry_owner_start(entry, owner_first_expert)
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
        experts_per_owner = self._nep_nccl_experts_per_owner
        owner_first_expert = owner_ep_rank * experts_per_owner
        owner_last_expert = owner_first_expert + experts_per_owner
        segments = []
        for expert_id in source_expert_ids:
            if expert_id < owner_first_expert or expert_id >= owner_last_expert:
                continue
            expert_start = (expert_id - owner_first_expert) * expert_stride
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

    def _nep_nccl_owner_source_ranks(self, owner_ep_rank: int) -> List[int]:
        source_ranks_by_owner = self._nep_runtime_config.get("nep_owner_source_ranks")
        if source_ranks_by_owner is not None and owner_ep_rank in source_ranks_by_owner:
            return list(source_ranks_by_owner[owner_ep_rank])
        legacy_source_ranks = self._nep_runtime_config.get("nep_owner_transfer_group_ranks")
        if legacy_source_ranks is not None and owner_ep_rank in legacy_source_ranks:
            return list(legacy_source_ranks[owner_ep_rank])
        placement = self._nep_runtime_config.get("expert_placement")
        if placement is None:
            return [owner_ep_rank]
        experts_per_owner = self._nep_nccl_experts_per_owner
        owner_first_expert = owner_ep_rank * experts_per_owner
        owner_last_expert = owner_first_expert + experts_per_owner
        source_ranks = []
        for source_ep_rank, expert_ids in enumerate(placement):
            if any(owner_first_expert <= expert_id < owner_last_expert for expert_id in expert_ids):
                source_ranks.append(source_ep_rank)
        return source_ranks

    def _nep_nccl_owner_transfer_ranks(self, owner_ep_rank: int) -> List[int]:
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
        cfg = self._nep_runtime_config
        ep_rank = cfg["ep_rank"]
        group_index = getattr(self, "_nep_nccl_group_index", -1)
        chunk_size = chunk_end - chunk_start
        source_ranks = self._nep_nccl_owner_source_ranks(owner_ep_rank)
        transfer_ranks = self._nep_nccl_owner_transfer_ranks(owner_ep_rank)
        if ep_rank not in transfer_ranks:
            return
        remote_source_ranks = [rank for rank in source_ranks if rank != owner_ep_rank]
        if ep_rank == owner_ep_rank:
            self._pack_nep_nccl_owner_chunk(owner_ep_rank, chunk_start, chunk_end, chunk)
        if not remote_source_ranks:
            return
        transfer_group, _, transfer_size, transfer_source_ranks = (
            self._get_nep_nccl_transfer_group_info(owner_ep_rank, "nep_owner_gather_groups")
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

    def _submit_nep_nccl_owner_all_to_all_scatter(self, descriptor: Optional[dict]) -> None:
        if descriptor is None:
            return
        kind = descriptor["kind"]
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
        if descriptor is None:
            return
        if not descriptor.get("submitted", False):
            raise RuntimeError("NEP Scatter completion was ordered before submission")
        if descriptor["kind"] == "all_to_all":
            _nep_block_current_stream(descriptor["work"])
        descriptor["completion_ordered"] = True

    def _finish_nep_nccl_owner_all_to_all_scatter(self, descriptor: Optional[dict]) -> None:
        if descriptor is None:
            return
        if not descriptor.get("completion_ordered", False):
            raise RuntimeError("NEP Scatter copyback was queued before collective completion")
        kind = descriptor["kind"]
        if kind == "local":
            self._copy_nep_nccl_owner_chunk_to_local_grads(
                descriptor["owner_ep_rank"],
                descriptor["chunk_start"],
                descriptor["chunk_end"],
                descriptor["chunk"],
            )
            return
        if kind == "native":
            return
        if kind != "all_to_all":
            raise RuntimeError(f"Unknown NEP Scatter descriptor kind: {kind}")
        if descriptor["ep_rank"] == descriptor["owner_ep_rank"]:
            self._copy_nep_nccl_owner_chunk_to_local_grads(
                descriptor["owner_ep_rank"],
                descriptor["chunk_start"],
                descriptor["chunk_end"],
                descriptor["chunk"],
            )
        elif descriptor["ep_rank"] in descriptor["remote_source_ranks"]:
            self._copy_nep_nccl_scatter_payload_to_local_grads(
                descriptor["owner_ep_rank"],
                descriptor["ep_rank"],
                descriptor["chunk_start"],
                descriptor["chunk_end"],
                descriptor["scatter_output"],
            )

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

    def _get_nep_nccl_native_edp_bucket_group(self, contexts):
        if isinstance(contexts, dict):
            contexts = [contexts]
        if not contexts:
            raise RuntimeError("NEP native DDP requires at least one owner context")
        contexts = sorted(contexts, key=lambda context: context["chunk_index"])
        cfg = self._nep_runtime_config
        owner_ep_rank = contexts[0]["owner_ep_rank"]
        if any(context["owner_ep_rank"] != owner_ep_rank for context in contexts):
            raise RuntimeError("NEP native DDP cannot combine different owner ranks")
        if cfg["ep_rank"] != owner_ep_rank:
            return None
        if self.ddp_config.use_distributed_optimizer:
            raise RuntimeError("NEP owner-layout DDP does not support distributed optimizer")
        edp_group = cfg.get("edp_group")
        if edp_group is None:
            raise RuntimeError(
                "Nonuniform EP NCCL owner rank requires runtime_config['edp_group']."
            )
        scaling_factors = {
            entry["bucket"].gradient_scaling_factor for entry in self._nep_nccl_entries
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
                    bucket_id=context["chunk_index"],
                    param_index_map={},
                    params_with_extra_main_grads=[],
                )
                for context in contexts
            ]
            native_ddp_config = _nep_owner_ddp_config(self.ddp_config)
            native_group = _ParamAndGradBucketGroup(
                native_buckets, native_ddp_config, edp_group, dist.get_world_size(group=edp_group)
            )
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

    def _start_nep_nccl_owner_edp_reduce_contexts(
        self, contexts: List[dict], use_device_readiness: bool = True
    ) -> None:
        if not contexts:
            return
        contexts = sorted(contexts, key=lambda context: context["chunk_index"])
        cfg = self._nep_runtime_config
        owner_ep_rank = contexts[0]["owner_ep_rank"]
        if any(context["owner_ep_rank"] != owner_ep_rank for context in contexts):
            raise RuntimeError("NEP native EDP batch contains different owner ranks")
        if cfg["ep_rank"] != owner_ep_rank:
            if first_batch_task_group is not None:
                dist.barrier(group=first_batch_task_group)
            return
        edp_group = cfg.get("edp_group")
        if edp_group is None:
            raise RuntimeError(
                "Nonuniform EP NCCL owner rank requires runtime_config['edp_group']."
            )
        group_index = getattr(self, "_nep_nccl_group_index", -1)
        chunk_indices = [context["chunk_index"] for context in contexts]
        native_group = self._get_nep_nccl_native_edp_bucket_group(contexts)
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

    def _start_nep_nccl_owner_edp_reduce(
        self, context: dict, use_device_readiness: bool = True
    ) -> None:
        self._start_nep_nccl_owner_edp_reduce_contexts(
            [context], use_device_readiness=use_device_readiness
        )

    def _order_nep_nccl_owner_edp_before_scatter(self, context: dict) -> None:
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
        native_states = getattr(self, "_nep_nccl_active_native_edp_states", [])
        for native_state in native_states:
            if native_state["finished"]:
                continue
            if not native_state["scatter_dependency_ordered"]:
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

    def _start_nep_nccl_owner_edp_reduce_batch(
        self, contexts: List[dict], use_device_readiness: bool = True
    ) -> None:
        context_batches = {}
        for context in contexts:
            group = context["group"]
            if group._nep_runtime_config["ep_rank"] != context["owner_ep_rank"]:
                continue
            key = (id(group), context["owner_ep_rank"])
            context_batches.setdefault(key, []).append(context)
        for context_batch in context_batches.values():
            context_batch[0]["group"]._start_nep_nccl_owner_edp_reduce_contexts(
                context_batch, use_device_readiness=use_device_readiness
            )

    def _prepare_nep_nccl_owner_task_context(
        self, owner_ep_rank: int, chunk_index: int, chunk_start: int, chunk_end: int, async_op: bool
    ) -> Optional[dict]:
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
        defer_scatter: bool = False,
    ) -> None:
        context = self._prepare_nep_nccl_owner_task_context(
            owner_ep_rank, chunk_index, chunk_start, chunk_end, async_op
        )
        if context is None:
            return
        self._start_nep_nccl_owner_all_to_all_gather(
            owner_ep_rank,
            chunk_index,
            chunk_start,
            chunk_end,
            context["chunk"],
            context["buffer_slot_key"],
            async_op=async_op,
        )
        self._start_nep_nccl_owner_edp_reduce(context)
        if defer_scatter:
            self._get_nep_nccl_shared_buffer_state().setdefault("pending_scatters", []).append(
                context
            )
        else:
            self._start_nep_nccl_owner_task_scatter(context)

    def _prepare_nep_nccl_owner_task_scatter_train(self, context: dict) -> dict:
        self._order_nep_nccl_owner_edp_before_scatter(context)
        scatter_chunks = _get_nep_nccl_scatter_chunks()
        scatter_ranges = self._nep_nccl_scatter_chunk_ranges(
            context["owner_ep_rank"], context["chunk_start"], context["chunk_end"], scatter_chunks
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

    def _mark_nep_nccl_scatter_train_scheduled(self, train: dict) -> None:
        if train["task_marked"]:
            return
        context = train["context"]
        self._mark_nep_nccl_task_started(context["owner_ep_rank"], context["chunk_index"])
        train["task_marked"] = True

    def _finish_nep_nccl_scatter_train_submission(self, train: dict) -> None:
        context = train["context"]
        if not train["task_marked"]:
            self._mark_nep_nccl_task_started(context["owner_ep_rank"], context["chunk_index"])
            train["task_marked"] = True

    def _start_nep_nccl_owner_task_scatter(self, context: dict) -> None:
        train = self._prepare_nep_nccl_owner_task_scatter_train(context)
        for descriptor in train["descriptors"]:
            self._submit_nep_nccl_owner_all_to_all_scatter(descriptor)
            self._order_nep_nccl_owner_all_to_all_scatter_completion(descriptor)
            self._finish_nep_nccl_owner_all_to_all_scatter(descriptor)
        self._finish_nep_nccl_scatter_train_submission(train)

    def _start_nep_nccl_split_host_phase_batch(
        self, task_batch: List[dict], dispatch_stream: torch.cuda.Stream, batch_index: int
    ) -> dict:
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
        local_transfer_contexts = {}
        for context in contexts:
            cfg = context["group"]._nep_runtime_config
            owner = context["owner_ep_rank"]
            source_group_gloo = cfg.get("nep_owner_transfer_groups_gloo", {}).get(owner)
            if source_group_gloo is not None:
                scatter_launch_group_gloo = _get_nep_owner_scatter_launch_group(cfg, owner)
                if scatter_launch_group_gloo is None:
                    raise RuntimeError(
                        f"NEP owner {owner} is missing its Scatter-launch Gloo group"
                    )
                local_transfer_contexts.setdefault(
                    owner, (context, source_group_gloo, scatter_launch_group_gloo)
                )
        local_edp_contexts = {}
        for context in contexts:
            cfg = context["group"]._nep_runtime_config
            owner = context["owner_ep_rank"]
            if cfg["ep_rank"] != owner:
                continue
            edp_group_gloo = cfg.get("edp_group_gloo")
            if edp_group_gloo is not None and edp_group_gloo.size() > 1:
                local_edp_contexts.setdefault(owner, (context, edp_group_gloo))
        gather_barrier_works = []
        for owner, (_, source_group_gloo, _) in local_transfer_contexts.items():
            gather_barrier_works.append(
                (owner, dist.barrier(group=source_group_gloo, async_op=True))
            )
        return {
            "batch_index": batch_index,
            "contexts": contexts,
            "dispatch_stream": dispatch_stream,
            "local_transfer_contexts": local_transfer_contexts,
            "local_edp_contexts": local_edp_contexts,
            "gather_barrier_works": gather_barrier_works,
            "gather_done_event": gather_done_event,
            "phase": "gather_launched",
        }

    def _start_nep_nccl_process_group_dispatch_batch(
        self, state: dict, force_ready: bool, compute_ready_event: torch.cuda.Event
    ) -> List[dict]:
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
        dispatch_stream = self._get_nep_nccl_comm_stream(0)
        dispatch_stream.wait_event(compute_ready_event)
        owner_waves = []
        for task in ready_tasks:
            owner = task["owner_ep_rank"]
            transfer_ranks = frozenset(task["group"]._nep_nccl_owner_transfer_ranks(owner))
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
            current_batch, current_slots = [], set()
            for task in ready_tasks:
                if task["owner_ep_rank"] not in wave["owners"]:
                    continue
                slot = task["group"]._get_nep_nccl_task_buffer_slot(
                    task["owner_ep_rank"], task["chunk_index"]
                )
                if current_batch and (
                    len(current_batch) >= max_batch_size or slot in current_slots
                ):
                    task_batches.append(current_batch)
                    current_batch, current_slots = [], set()
                current_batch.append(task)
                current_slots.add(slot)
            if current_batch:
                task_batches.append(current_batch)
        pending = [self._start_nep_nccl_split_host_phase_batch(task_batches[0], dispatch_stream, 0)]
        pending[0]["remaining_task_batches"] = list(enumerate(task_batches[1:], start=1))
        return pending

    def _finish_nep_nccl_process_group_dispatch_batches(
        self,
        pending_host_phases: List[dict],
        device_align_phases: bool = False,
        finish_all_phases: bool = True,
        defer_scatter_submission: bool = False,
        scatter_context_batches: Optional[List[List[dict]]] = None,
    ) -> bool:
        for pending in pending_host_phases:
            batch_index = pending["batch_index"]
            contexts = pending["contexts"]
            dispatch_stream = pending["dispatch_stream"]
            phase = pending.get("phase", "gather_launched")
            if phase == "finished":
                continue
            if phase == "gather_launched":
                for owner, work in pending["gather_barrier_works"]:
                    with torch.profiler.record_function("nep_split_wait_gather_launch"):
                        work.wait()
                if device_align_phases:
                    gather_done_event = pending.get("gather_done_event")
                    if gather_done_event is None:
                        raise RuntimeError("Post-graph NEP phases are missing the Gather event")
                    with torch.profiler.record_function("nep_post_graph_wait_gather_device"):
                        gather_done_event.synchronize()
                edp_launch_barrier_works = []
                for owner, (_, edp_group_gloo) in pending["local_edp_contexts"].items():
                    edp_launch_barrier_works.append(
                        (owner, dist.barrier(group=edp_group_gloo, async_op=True))
                    )
                for owner, work in edp_launch_barrier_works:
                    with torch.profiler.record_function("nep_split_wait_edp_launch"):
                        work.wait()
                with torch.cuda.stream(dispatch_stream):
                    self._start_nep_nccl_owner_edp_reduce_batch(
                        contexts, use_device_readiness=False
                    )
                    if device_align_phases:
                        edp_done_event = torch.cuda.Event()
                        edp_done_event.record(dispatch_stream)
                        pending["edp_done_event"] = edp_done_event
                pending["phase"] = "edp_launched"
                if device_align_phases and not finish_all_phases:
                    continue
            if pending["phase"] == "scatter_ready":
                pass
            elif pending["phase"] == "edp_launched":
                if device_align_phases:
                    edp_done_event = pending.get("edp_done_event")
                    if edp_done_event is None:
                        raise RuntimeError("Post-graph NEP phases are missing the EDP event")
                    with torch.profiler.record_function("nep_post_graph_wait_edp_device"):
                        edp_done_event.synchronize()
                pending["phase"] = "edp_complete"
                scatter_barrier_works = []
                for owner, (_, _, scatter_launch_group_gloo) in pending[
                    "local_transfer_contexts"
                ].items():
                    scatter_barrier_works.append(
                        (owner, dist.barrier(group=scatter_launch_group_gloo, async_op=True))
                    )
                for _, work in scatter_barrier_works:
                    with torch.profiler.record_function("nep_split_wait_scatter_launch"):
                        work.wait()
                pending["phase"] = "scatter_ready"
                if defer_scatter_submission:
                    continue
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
                next_pending["remaining_task_batches"] = remaining_task_batches
                pending.clear()
                pending.update(next_pending)
                return self._finish_nep_nccl_process_group_dispatch_batches(
                    pending_host_phases,
                    device_align_phases=device_align_phases,
                    finish_all_phases=finish_all_phases,
                    defer_scatter_submission=defer_scatter_submission,
                    scatter_context_batches=scatter_context_batches,
                )
        return all(pending.get("phase") == "finished" for pending in pending_host_phases)

    def _try_start_nep_nccl_ready_tasks(
        self,
        force_ready: bool = False,
        async_op_override: Optional[bool] = None,
        compute_ready_event: Optional[torch.cuda.Event] = None,
    ) -> Optional[List[dict]]:
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
                state, force_ready, compute_ready_event
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
                defer_scatter=async_op,
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
                    launch_next_task()
        else:
            launch_ready_tasks_on_current_stream()

    def _start_nonuniform_ep_nccl_grad_sync(self, async_op: bool = False):
        cfg = self._nep_runtime_config
        layout = self._get_nep_nccl_owner_layout()
        local_ep_size = layout["local_ep_size"]
        ep_rank = layout["ep_rank"]
        min_ep_size = layout["min_ep_size"]
        is_edp_eligible = cfg.get("is_edp_eligible", ep_rank < min_ep_size)
        group_index = getattr(self, "_nep_nccl_group_index", -1)
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
        group_index = getattr(self, "_nep_nccl_group_index", -1)
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
        self._nep_nccl_send_chunk_cache.clear()
        self._nep_nccl_gather_buf_cache.clear()
        self._nep_nccl_gather_list_cache.clear()

    def finish_nep_pre_sync(self, force_all_reduce: Optional[bool] = False):
        if not self.ddp_config.overlap_grad_reduce:
            return
        group_index = getattr(self, "_nep_nccl_group_index", -1)
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
        group_index = getattr(self, "_nep_nccl_group_index", -1)
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
        assert (
            self.ddp_config.overlap_grad_reduce
        ), "register_grad_ready() should only be called when overlap_grad_reduce is True"
        if self.is_last_microbatch:
            assert param in self.param_to_bucket, "Param is not in the bucket group"
            if param not in self.per_param_grad_ready_counts:
                self.per_param_grad_ready_counts[param] = 0
            self.per_param_grad_ready_counts[param] += 1
            if not self.is_first_batch:
                if self._nep_dispatch_boundary_launch:
                    callback = self._nep_dispatch_boundary_callback
                    if self._nep_dispatch_boundary_ready and callback is not None:
                        callback(
                            self._nep_dispatch_boundary_groups,
                            self._nep_dispatch_boundary_module_label,
                        )
                else:
                    self._try_start_nep_nccl_ready_tasks(force_ready=False, async_op_override=True)
                if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
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
        self._nep_dispatch_boundary_graph_replay_ready = False
        self._nep_dispatch_boundary_launched = False
        self._nep_dispatch_boundary_launching = False
        self._nep_dispatch_boundary_wait_logged = False
        self._nep_dispatch_boundary_ready_modules = set()
        reset_ordered_bucket_group_scheduler(
            self, "_nep_nccl_scheduler_state", "_nep_nccl_group_index"
        )
        state = getattr(self, "_nep_nccl_scheduler_state", None)
        if state is not None and getattr(self, "_nep_nccl_group_index", -1) == 0:
            if state.get("pending_scatters"):
                raise RuntimeError("NEP reset found deferred scatters that were not flushed")
            state["task_next_index"] = 0
            state["pending_scatters"] = []


def _coalesce_nep_nccl_bucket_groups_for_edp_order(
    bucket_groups: List[NonuniformEPNCCLParamAndGradBucketGroup],
    runtime_config: dict,
    nonuniform_ep_config: NonuniformEPConfig,
) -> List[NonuniformEPNCCLParamAndGradBucketGroup]:
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


def _build_expert_bucket_specs(buffers, runtime_config, config, param_to_name):
    local_expert_indices = runtime_config.get("local_expert_indices")
    local_expert_id_set = set(local_expert_indices) if local_expert_indices is not None else set()
    specs = []
    for buffer in buffers:
        for source_bucket_index, bucket_params in _iter_buffer_bucket_params(buffer):
            current_expert_id = None
            current_params = []

            def flush_current():
                nonlocal current_expert_id, current_params
                if current_expert_id is None or not current_params:
                    current_expert_id = None
                    current_params = []
                    return
                params = current_params
                starts_ends = [buffer.param_index_map[param][:2] for param in params]
                start = min(start for start, _ in starts_ends)
                end = max(end for _, end in starts_ends)
                total = sum(param_end - param_start for param_start, param_end in starts_ends)
                if total != end - start:
                    raise RuntimeError(
                        f"Expert {current_expert_id} params are not contiguous in the grad buffer"
                    )
                slot_key = tuple(
                    _expert_slot_key_from_name(
                        param_to_name.get(param, ""), config.expert_name_pattern
                    )
                    for param in params
                )
                specs.append(
                    _ExpertBucketSpec(
                        buffer=buffer,
                        source_bucket_index=source_bucket_index,
                        expert_id=current_expert_id,
                        params=params,
                        start=start,
                        end=end,
                        slot_key=slot_key,
                    )
                )
                current_expert_id = None
                current_params = []

            for param in bucket_params:
                if param not in buffer.param_index_map:
                    continue
                name = param_to_name.get(param, "")
                expert_id = _local_expert_id_from_name(
                    name, config.expert_name_pattern, local_expert_indices
                )
                if expert_id is None:
                    flush_current()
                    continue
                local_expert_id_set.add(expert_id)
                if current_expert_id is not None and expert_id != current_expert_id:
                    flush_current()
                current_expert_id = expert_id
                current_params.append(param)
            flush_current()
    runtime_config["_local_expert_id_set"] = local_expert_id_set
    return specs


def _build_expert_param_bucket_specs(buffers, runtime_config, config, param_to_name):
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
    grouped_specs: Dict[Tuple[str, ...], List[_ExpertBucketSpec]] = {}
    for spec in specs:
        grouped_specs.setdefault(spec.slot_key, []).append(spec)
    return list(grouped_specs.items())


def _expert_slot_module_key(slot_key: Tuple[str, ...]) -> Tuple[str, ...]:
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
    group_count = min(target_group_count, len(module_blocks))
    base_size, remainder = divmod(len(module_blocks), group_count)
    partitions = []
    start = 0
    for group_index in range(group_count):
        block_count = base_size + (1 if group_index < remainder else 0)
        partition = []
        for _, module_specs in module_blocks[start : start + block_count]:
            partition.extend(module_specs)
        partitions.append(partition)
        start += block_count
    return partitions


def _build_synthetic_owner_bucket_specs(buffers, local_specs, runtime_config, config):
    placement = runtime_config.get("expert_placement")
    if placement is None:
        return []
    local_ep_rank = runtime_config["ep_rank"]
    local_expert_ids = {spec.expert_id for spec in local_specs}
    template_by_slot_key = {}
    for spec in local_specs:
        numel = spec.end - spec.start
        previous = template_by_slot_key.setdefault(
            spec.slot_key, (spec.buffer, spec.source_bucket_index, numel)
        )
        if previous[2] != numel:
            raise RuntimeError(
                "NEP synthetic owner buckets assume equal grad sizes for matching "
                "expert parameter slots."
            )
    if not template_by_slot_key:
        return []
    num_experts = sum(len(experts) for experts in placement)
    synthetic_expert_ids = []
    for expert_id in range(num_experts):
        if expert_id in local_expert_ids:
            continue
        owner_ep_rank = _owner_for_expert(expert_id, runtime_config, config.expert_owner)
        if owner_ep_rank != local_ep_rank:
            continue
        if local_ep_rank in _source_ep_ranks_for_expert(expert_id, runtime_config):
            continue
        synthetic_expert_ids.append(expert_id)
    specs = []
    for expert_id in synthetic_expert_ids:
        for slot_key, (buffer, source_bucket_index, numel) in sorted(template_by_slot_key.items()):
            specs.append(
                _ExpertBucketSpec(
                    buffer=buffer,
                    source_bucket_index=source_bucket_index,
                    expert_id=expert_id,
                    params=[],
                    start=0,
                    end=numel,
                    slot_key=slot_key,
                    synthetic_owner=True,
                )
            )
    return specs


def wrap_nonuniform_ep_nccl_bucket_groups(
    bucket_groups: List[_ParamAndGradBucketGroup],
    runtime_config: dict,
    nonuniform_ep_config: NonuniformEPConfig,
    param_to_bucket_group: Optional[Dict[torch.nn.Parameter, _ParamAndGradBucketGroup]] = None,
) -> List[NonuniformEPNCCLParamAndGradBucketGroup]:
    """Replace generic expert bucket groups with nonuniform-EP groups."""
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
    if not bucket_groups:
        return
    state = getattr(bucket_groups[0], "_nep_nccl_scheduler_state", None)
    if state is None:
        state = {"groups": bucket_groups, "next_index": 0}
        for index, bucket_group in enumerate(bucket_groups):
            bucket_group._nep_nccl_scheduler_state = state
            bucket_group._nep_nccl_group_index = index
    task_sequence = []
    for bucket_group in bucket_groups:
        layout = bucket_group._get_nep_nccl_owner_layout()
        bucket_group._nep_nccl_task_count = layout["min_ep_size"] * layout["num_chunks"]
        bucket_group._nep_nccl_ready = bucket_group._nep_nccl_task_count == 0
        for owner_ep_rank in range(layout["min_ep_size"]):
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
    state["task_sequence"] = task_sequence
    state["task_next_index"] = 0
    state["pending_scatters"] = []


def build_nonuniform_ep_nccl_bucket_groups(
    buffers,
    ddp_config: DistributedDataParallelConfig,
    runtime_config: dict,
    nonuniform_ep_config: NonuniformEPConfig,
    param_to_bucket_group: Dict[torch.nn.Parameter, _ParamAndGradBucketGroup],
    param_to_name: Dict[torch.nn.Parameter, str],
) -> List[NonuniformEPNCCLParamAndGradBucketGroup]:
    """Build backward-ordered nonuniform-EP bucket groups."""
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
    ordered_grouped_specs = _group_expert_bucket_specs_in_backward_order(specs)
    grouped_partitions = _partition_expert_bucket_specs(
        ordered_grouped_specs, _get_nep_nccl_expert_bucket_group_count()
    )
    for group_index, grouped_partition in enumerate(grouped_partitions):
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
    return bucket_groups


class NonuniformEPDistributedDataParallel(DistributedDataParallel):
    """DDP wrapper for NCCL nonuniform expert parallelism."""

    @staticmethod
    def _synchronize_bucket_size(ddp_config: DistributedDataParallelConfig) -> None:
        if ddp_config.num_buckets is None or ddp_config.bucket_size is None:
            return
        local_bucket_size = ddp_config.bucket_size
        bucket_size = torch.tensor(
            local_bucket_size, dtype=torch.int64, device=torch.cuda.current_device()
        )
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
    ):
        self.nonuniform_ep_config = nonuniform_ep_config or NonuniformEPConfig()
        runtime_config = _get_runtime_config(self.nonuniform_ep_config)
        if ddp_config.use_distributed_optimizer:
            raise RuntimeError(
                "NonuniformEPDistributedDataParallel currently supports only "
                "non-distributed optimizers. Synced expert grads are scattered back "
                "so every rank can step its local expert params."
            )
        self._synchronize_bucket_size(ddp_config)
        parent_kwargs = {
            "config": config,
            "ddp_config": ddp_config,
            "module": module,
            "disable_bucketing": disable_bucketing,
            "pg_collection": pg_collection,
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
        self._nep_dispatch_boundary_hook_handles = []
        self._nep_dispatch_boundary_pre_hook_handles = []
        self._nep_dispatch_pending_completion_event = None
        self._nep_dispatch_pending_host_phases = None
        self._nep_dispatch_inflight_completion_events = []
        self._nep_dispatch_waiting_groups = None
        self._nep_dispatch_waiting_module_label = None
        self._nep_model_ep_a2a_burst_depth = 0
        self._nep_model_ep_a2a_burst_count = 0
        self._nep_scatter_batches = []
        self._nep_scatter_inflight_event = None
        self._nep_scatter_next_batch_ordinal = 0
        self._nep_scatter_next_layer_ordinal = 0
        self._nep_scatter_alignment_tensor = None
        self._nep_scatter_alignment_work = None
        self._nep_scatter_stream = None
        self._nep_scatter_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self._configure_nep_dispatch_boundary_hooks()
        self._configure_expert_gradient_scaling(config, runtime_config)

    @staticmethod
    def _find_nep_local_cuda_graph_manager(module_name: str, named_modules: dict):
        path = module_name.split(".") if module_name else []
        for prefix_length in range(len(path), -1, -1):
            parent_name = ".".join(path[:prefix_length])
            parent = named_modules.get(parent_name)
            if getattr(parent, "use_partial_cudagraphs", False):
                return None
            graph_manager = getattr(parent, "cudagraph_manager", None)
            if graph_manager is not None:
                return graph_manager
        return None

    @staticmethod
    def _nep_module_uses_partial_cuda_graphs(module_name: str, named_modules: dict) -> bool:
        path = module_name.split(".") if module_name else []
        for prefix_length in range(len(path), -1, -1):
            parent_name = ".".join(path[:prefix_length])
            parent = named_modules.get(parent_name)
            if getattr(parent, "use_partial_cudagraphs", False):
                return True
        return False

    @staticmethod
    def _coalesce_nep_cuda_graph_boundary(module_entries: list) -> tuple:
        groups = []
        seen_group_ids = set()
        for _, module_groups in module_entries:
            for group in module_groups:
                if id(group) in seen_group_ids:
                    continue
                seen_group_ids.add(id(group))
                groups.append(group)
        ordered_groups = tuple(sorted(groups, key=lambda group: group._nep_nccl_group_index))
        module_label = (
            "cuda_graph[" + ",".join(module_name for module_name, _ in module_entries) + "]"
        )
        return ordered_groups, module_label

    def _retire_nep_scatter_chunk(self, force: bool = False) -> bool:
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

    def _peek_nep_scatter_ticket_window(self) -> tuple:
        first_ticket = None
        last_ticket = None
        descriptor_count = 0
        window_key = None
        for batch in self._nep_scatter_batches:
            if batch.get("submission_complete", False):
                continue
            for train_index in range(batch["next_train"], len(batch["trains"])):
                train = batch["trains"][train_index]
                train_group_index = getattr(train["group"], "_nep_nccl_group_index", -1)
                train_window_key = (
                    batch.get("layer_ordinal", batch["schedule_ordinal"]),
                    train_group_index,
                )
                if window_key is None:
                    window_key = train_window_key
                elif train_window_key != window_key:
                    return "ready", descriptor_count, first_ticket, last_ticket
                descriptor_start = (
                    train["next_descriptor"] if train_index == batch["next_train"] else 0
                )
                context = train["context"]
                for descriptor_index in range(descriptor_start, len(train["descriptors"])):
                    ticket = (
                        batch["schedule_ordinal"],
                        train_index,
                        train_group_index,
                        context["owner_ep_rank"],
                        context["chunk_index"],
                        descriptor_index,
                    )
                    if first_ticket is None:
                        first_ticket = ticket
                    last_ticket = ticket
                    descriptor_count += 1
        if first_ticket is None:
            return "empty", 0, None, None
        return "ready", descriptor_count, first_ticket, last_ticket

    def _consume_model_ep_aligned_nep_scatter_ticket(
        self, after_event: Optional[torch.cuda.Event] = None
    ) -> int:
        alignment_work = self._nep_scatter_alignment_work
        if alignment_work is None:
            return 0
        with torch.profiler.record_function("nep_model_ep_scatter_ticket_wait"):
            alignment_work.wait()
        self._nep_scatter_alignment_work = None
        alignment_tensor = self._nep_scatter_alignment_tensor
        if alignment_tensor is None:
            raise RuntimeError("NEP Scatter ticket completed without its alignment tensor")
        value_count = alignment_tensor.numel() // 2
        minimum = alignment_tensor[:value_count].tolist()
        maximum = (-alignment_tensor[value_count:]).tolist()
        if minimum[0] != 2 or maximum[0] != 2:
            return 0
        if minimum[1:] != maximum[1:]:
            raise RuntimeError(
                "NEP Scatter ranks reached an ordered boundary with different windows: "
                f"minimum={tuple(minimum[1:])}, maximum={tuple(maximum[1:])}"
            )
        descriptor_count = minimum[1]
        first_ticket = tuple(minimum[2:8])
        last_ticket = tuple(minimum[8:14])
        status, local_count, local_first, local_last = self._peek_nep_scatter_ticket_window()
        if (
            status != "ready"
            or local_count != descriptor_count
            or local_first != first_ticket
            or local_last != last_ticket
        ):
            raise RuntimeError(
                "NEP Scatter queue changed after its window agreement: "
                f"agreed={(descriptor_count, first_ticket, last_ticket)} "
                f"local={(status, local_count, local_first, local_last)}"
            )
        with torch.profiler.record_function(
            f"nep_submit_scatter_window_g{first_ticket[2]}_n{descriptor_count}"
        ):
            for descriptor_index in range(descriptor_count):
                submitted = self._submit_nep_scatter_chunk(
                    after_event=after_event if descriptor_index == 0 else None,
                    queue_behind_inflight=True,
                )
                if not submitted:
                    raise RuntimeError(
                        "NEP Scatter agreed window could not submit descriptor "
                        f"{descriptor_index + 1}/{descriptor_count}"
                    )
        return descriptor_count

    def _launch_model_ep_aligned_nep_scatter_ticket(self) -> None:
        runtime_config = self._nonuniform_ep_runtime_config
        ep_group_gloo = runtime_config.get("ep_group_gloo")
        if ep_group_gloo is None:
            raise RuntimeError("A2A-gated NEP Scatter requires a model-EP Gloo group")
        if self._nep_scatter_alignment_work is not None:
            raise RuntimeError("NEP Scatter launched a ticket before consuming its predecessor")
        status, descriptor_count, first_ticket, last_ticket = self._peek_nep_scatter_ticket_window()
        status_value = {"empty": 0, "busy": 1, "ready": 2}[status]
        local_values = [status_value, descriptor_count]
        local_values.extend(first_ticket if first_ticket is not None else (-1,) * 6)
        local_values.extend(last_ticket if last_ticket is not None else (-1,) * 6)
        alignment_tensor = self._nep_scatter_alignment_tensor
        if alignment_tensor is None or alignment_tensor.numel() != 2 * len(local_values):
            alignment_tensor = torch.empty(2 * len(local_values), dtype=torch.int64)
            self._nep_scatter_alignment_tensor = alignment_tensor
        positive = alignment_tensor[: len(local_values)]
        negative = alignment_tensor[len(local_values) :]
        positive.copy_(torch.tensor(local_values, dtype=torch.int64))
        negative.copy_(-positive)
        with torch.profiler.record_function("nep_model_ep_scatter_ticket_launch"):
            self._nep_scatter_alignment_work = dist.all_reduce(
                alignment_tensor, op=dist.ReduceOp.MIN, group=ep_group_gloo, async_op=True
            )
        if self._nep_scatter_alignment_work is None:
            raise RuntimeError("Nonblocking NEP Scatter ticket launch returned no work handle")

    def _submit_model_ep_aligned_nep_scatter_chunk(
        self, after_event: Optional[torch.cuda.Event] = None
    ) -> int:
        runtime_config = self._nonuniform_ep_runtime_config
        if not runtime_config.get("needs_reshard", False):
            return int(self._submit_nep_scatter_chunk(after_event=after_event))
        submitted = self._consume_model_ep_aligned_nep_scatter_ticket(after_event=after_event)
        self._launch_model_ep_aligned_nep_scatter_ticket()
        return submitted

    def _submit_nep_scatter_chunk(
        self,
        after_event: Optional[torch.cuda.Event] = None,
        force: bool = False,
        queue_behind_inflight: bool = False,
    ) -> bool:
        if self._nep_model_ep_a2a_burst_depth != 0:
            return False
        predecessor_complete = self._retire_nep_scatter_chunk(force=force)
        if not self._nep_scatter_batches:
            return False
        batch = self._nep_scatter_batches[0]
        while batch.get("submission_complete", False):
            if not predecessor_complete and not queue_behind_inflight:
                return False
            self._nep_scatter_batches.pop(0)
            if not self._nep_scatter_batches:
                return True
            batch = self._nep_scatter_batches[0]
        if not predecessor_complete and not queue_behind_inflight:
            return False
        scatter_stream = self._nep_scatter_stream
        if scatter_stream is None:
            raise RuntimeError("A2A-gated NEP Scatter stream was not initialized")
        train = batch["trains"][batch["next_train"]]
        descriptor_index = train["next_descriptor"]
        descriptor = train["descriptors"][descriptor_index]
        context = train["context"]
        group_index = getattr(train["group"], "_nep_nccl_group_index", -1)
        with torch.cuda.stream(scatter_stream):
            if after_event is not None:
                scatter_stream.wait_event(after_event)
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
        a2a_completion_event: Optional[torch.cuda.Event],
    ) -> None:
        scatter_stream = self._nep_scatter_stream
        if scatter_stream is None:
            raise RuntimeError("A2A-gated NEP Scatter state was not initialized")
        scheduled_context_batches = context_batches or [[]]
        layer_ordinal = self._nep_scatter_next_layer_ordinal
        self._nep_scatter_next_layer_ordinal += 1
        with torch.cuda.stream(scatter_stream):
            if a2a_completion_event is not None:
                scatter_stream.wait_event(a2a_completion_event)
            for batch_index, contexts in enumerate(scheduled_context_batches):
                trains = []
                for context in contexts:
                    train = context["group"]._prepare_nep_nccl_owner_task_scatter_train(context)
                    context["group"]._mark_nep_nccl_scatter_train_scheduled(train)
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
                    "schedule_ordinal": self._nep_scatter_next_batch_ordinal,
                    "layer_ordinal": layer_ordinal,
                }
                self._nep_scatter_next_batch_ordinal += 1
                if not trains:
                    batch_completion_event.record(scatter_stream)
                self._nep_scatter_batches.append(batch)

    def _drain_nep_scatter_scheduler(self) -> None:
        if self._nep_model_ep_a2a_burst_depth != 0:
            raise RuntimeError("Cannot drain NEP Scatter during a model-EP A2A burst")
        if self._nonuniform_ep_runtime_config.get("needs_reshard", False):
            self._consume_model_ep_aligned_nep_scatter_ticket()
        while self._nep_scatter_batches:
            if not self._submit_nep_scatter_chunk(force=True):
                raise RuntimeError("A2A-gated NEP Scatter drain made no progress")
        self._retire_nep_scatter_chunk(force=True)

    def model_ep_a2a_burst_begin(self) -> None:
        if self._nep_model_ep_a2a_burst_depth != 0:
            raise RuntimeError("Nested model-EP A2A bursts are not supported")
        self._nep_model_ep_a2a_burst_depth = 1

    def model_ep_a2a_burst_end(self) -> None:
        completion_event = torch.cuda.Event()
        completion_event.record(torch.cuda.current_stream())
        if self._nep_model_ep_a2a_burst_depth != 1:
            raise RuntimeError("Model-EP A2A burst ended without a matching begin")
        self._nep_model_ep_a2a_burst_depth = 0
        self._nep_model_ep_a2a_burst_count += 1
        if self._nep_dispatch_pending_host_phases is not None:
            with torch.profiler.record_function("nep_a2a_burst_end_queue_scatter"):
                phases_finished = self._finish_pending_nep_dispatch_host_phases(
                    defer_scatter_submission=False, scatter_after_event=completion_event
                )
            if not phases_finished:
                raise RuntimeError("Model-EP A2A burst end did not queue all staged Scatter work")
            self._wait_for_nep_dispatch_launch()
        self._submit_model_ep_aligned_nep_scatter_chunk(after_event=completion_event)

    def _configure_nep_dispatch_boundary_hooks(self) -> None:
        from ..transformer.moe.moe_layer import BaseMoELayer

        expert_groups = set(self.expert_parallel_bucket_groups)
        assigned_groups = {}
        named_modules = dict(self.module.named_modules())
        graph_manager_entries = {}
        for module_name, module in named_modules.items():
            if not isinstance(module, BaseMoELayer):
                continue
            module_groups = []
            for param in module.parameters():
                bucket_group = self.param_to_bucket_group.get(param)
                if bucket_group in expert_groups and bucket_group not in module_groups:
                    module_groups.append(bucket_group)
            if not module_groups:
                continue
            token_dispatcher = getattr(module, "token_dispatcher", None)
            attach_scheduler = getattr(token_dispatcher, "set_model_ep_a2a_burst_scheduler", None)
            if attach_scheduler is None:
                raise RuntimeError("NEP requires an A2A-aware token dispatcher")
            attach_scheduler(self)
            module_groups.sort(key=lambda group: group._nep_nccl_group_index)
            for bucket_group in module_groups:
                assigned_groups.setdefault(bucket_group, set()).add(module_name)
                bucket_group._nep_dispatch_boundary_launch = True
                bucket_group._nep_dispatch_boundary_callback = (
                    self._launch_nep_dispatch_boundary_tasks
                )
                bucket_group._nep_dispatch_boundary_groups = (bucket_group,)
                bucket_group._nep_dispatch_boundary_module_label = module_name
                bucket_group._nep_dispatch_boundary_required_modules.add(module_name)
            uses_partial_cuda_graphs = self._nep_module_uses_partial_cuda_graphs(
                module_name, named_modules
            )
            graph_manager = self._find_nep_local_cuda_graph_manager(module_name, named_modules)
            if graph_manager is not None:
                entry = graph_manager_entries.setdefault(
                    id(graph_manager), {"manager": graph_manager, "modules": []}
                )
                entry["modules"].append((module_name, tuple(module_groups)))

            def dispatch_boundary_hook(
                unused_module,
                unused_grad_input,
                unused_grad_output,
                groups=tuple(module_groups),
                module_label=module_name,
            ):
                self._mark_nep_dispatch_boundary_ready(groups, module_label)

            def dispatch_boundary_pre_hook(unused_module, unused_grad_output):
                if not is_graph_capturing():
                    self._wait_for_nep_dispatch_launch()

            if uses_partial_cuda_graphs:

                def dispatch_input_grad_callback(
                    groups=tuple(module_groups), module_label=module_name
                ):
                    self._mark_nep_dispatch_boundary_ready(groups, module_label)

                module.register_expert_compute_input_grad_callback(dispatch_input_grad_callback)
                module.register_expert_compute_dgrad_callback(self._wait_for_nep_dispatch_launch)
                self._nep_dispatch_boundary_pre_hook_handles.append(
                    module.register_full_backward_pre_hook(dispatch_boundary_pre_hook)
                )
            else:
                self._nep_dispatch_boundary_hook_handles.append(
                    module.register_full_backward_hook(dispatch_boundary_hook)
                )
                self._nep_dispatch_boundary_pre_hook_handles.append(
                    module.register_full_backward_pre_hook(dispatch_boundary_pre_hook)
                )
        for entry in graph_manager_entries.values():
            graph_groups, graph_label = self._coalesce_nep_cuda_graph_boundary(entry["modules"])
            ready_modules_by_group = {group: set() for group in graph_groups}
            for module_name, module_groups in entry["modules"]:
                for group in module_groups:
                    ready_modules_by_group[group].add(module_name)

            def graph_post_replay_hook(
                groups=graph_groups, module_label=graph_label, ready_modules=ready_modules_by_group
            ):
                self._mark_nep_dispatch_boundary_ready(
                    groups, module_label, graph_replay=True, ready_modules_by_group=ready_modules
                )

            entry["manager"].register_backward_replay_hooks(
                pre_hook=self._wait_for_nep_dispatch_launch, post_hook=graph_post_replay_hook
            )
        missing_groups = expert_groups - set(assigned_groups)
        if missing_groups:
            missing_indices = sorted(group._nep_nccl_group_index for group in missing_groups)
            raise RuntimeError(
                "NCCL NEP could not map expert bucket groups to MoE dispatch boundaries: "
                f"groups={missing_indices}"
            )

    def _mark_nep_dispatch_boundary_ready(
        self,
        groups: tuple,
        module_label: str,
        graph_replay: bool = False,
        defer_launch: bool = False,
        ready_modules_by_group: Optional[dict] = None,
    ) -> None:
        if (
            is_graph_capturing()
            or any(group.is_first_batch for group in groups)
            or not all(group.is_last_microbatch for group in groups)
        ):
            return
        for group in groups:
            ready_modules = getattr(group, "_nep_dispatch_boundary_ready_modules", None)
            if ready_modules is None:
                ready_modules = set()
                group._nep_dispatch_boundary_ready_modules = ready_modules
            if ready_modules_by_group is None:
                ready_modules.add(module_label)
            else:
                ready_modules.update(ready_modules_by_group.get(group, ()))
        incomplete_groups = []
        for group in groups:
            required_modules = getattr(
                group, "_nep_dispatch_boundary_required_modules", {module_label}
            )
            if not required_modules.issubset(group._nep_dispatch_boundary_ready_modules):
                incomplete_groups.append(group)
        if incomplete_groups:
            return
        for group in groups:
            group._nep_dispatch_boundary_ready = True
            if graph_replay:
                group._nep_dispatch_boundary_graph_replay_ready = True
        self._nep_dispatch_waiting_groups = groups
        self._nep_dispatch_waiting_module_label = module_label
        if not defer_launch:
            self._launch_nep_dispatch_boundary_tasks(groups, module_label)

    def _launch_waiting_nep_dispatch_boundary_tasks(self) -> None:
        groups = self._nep_dispatch_waiting_groups
        if (
            groups is None
            or self._nep_dispatch_pending_completion_event is not None
            or getattr(self, "_nep_dispatch_pending_host_phases", None) is not None
        ):
            return
        module_label = self._nep_dispatch_waiting_module_label
        if not self._launch_nep_dispatch_boundary_tasks(groups, module_label):
            raise RuntimeError(
                "NEP dispatch boundary reached its launch point before gradients were ready: "
                f"module={module_label}"
            )

    def _launch_nep_dispatch_boundary_tasks(self, groups: tuple, module_label: str) -> bool:
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
        try:
            pending_host_phases = groups[0]._try_start_nep_nccl_ready_tasks(
                force_ready=False, async_op_override=True, compute_ready_event=compute_ready_event
            )
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
            for group in groups:
                group._nep_dispatch_boundary_launched = True
            return completion_event
        finally:
            for group in groups:
                group._nep_dispatch_boundary_launching = False

    def _finish_pending_nep_dispatch_host_phases(
        self,
        device_align_phases: bool = False,
        finish_all_phases: bool = True,
        defer_scatter_submission: bool = False,
        scatter_after_event: Optional[torch.cuda.Event] = None,
    ) -> bool:
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
        scatter_context_batches = [] if not defer_scatter_submission else None
        boundary_group, host_phases = pending_host_phases
        finish_kwargs = {
            "device_align_phases": device_align_phases,
            "finish_all_phases": finish_all_phases,
            "defer_scatter_submission": defer_scatter_submission,
        }
        if scatter_context_batches is not None:
            finish_kwargs["scatter_context_batches"] = scatter_context_batches
        phases_finished = boundary_group._finish_nep_nccl_process_group_dispatch_batches(
            host_phases, **finish_kwargs
        )
        if phases_finished is False:
            return False
        if scatter_context_batches is None:
            completion_event.record(boundary_group._get_nep_nccl_comm_stream(0))
        else:
            self._queue_nep_scatter_context_batches(
                scatter_context_batches, completion_event, module_label, scatter_after_event
            )
        not_ready = [group._nep_nccl_group_index for group in groups if not group._nep_nccl_ready]
        if not_ready:
            raise RuntimeError(
                "Split NEP host phases did not schedule every layer bucket group: "
                f"module={module_label}, groups={not_ready}"
            )
        self._nep_dispatch_pending_host_phases = None
        return True

    def _wait_for_nep_dispatch_launch(self, final: bool = False) -> None:
        groups = self._nep_dispatch_waiting_groups
        module_label = self._nep_dispatch_waiting_module_label
        completion_event = self._nep_dispatch_pending_completion_event
        pending_host_phases = self._nep_dispatch_pending_host_phases
        if groups is not None and completion_event is None:
            self._launch_waiting_nep_dispatch_boundary_tasks()
            completion_event = self._nep_dispatch_pending_completion_event
            pending_host_phases = self._nep_dispatch_pending_host_phases
        if pending_host_phases is not None:
            if not self._finish_pending_nep_dispatch_host_phases(
                defer_scatter_submission=not final
            ):
                if not final:
                    return
                raise RuntimeError("The final NEP host phase did not complete")
        if completion_event is not None:
            self._nep_dispatch_inflight_completion_events.append((module_label, completion_event))
            self._nep_dispatch_pending_completion_event = None
        if groups is not None:
            if not all(group._nep_dispatch_boundary_launched for group in groups):
                raise RuntimeError(f"NEP dispatch launch did not finish: module={module_label}")
            self._nep_dispatch_waiting_groups = None
            self._nep_dispatch_waiting_module_label = None
        if final:
            self._drain_nep_scatter_scheduler()
            current_stream = torch.cuda.current_stream()
            for _, event in self._nep_dispatch_inflight_completion_events:
                current_stream.wait_event(event)
            self._nep_dispatch_inflight_completion_events.clear()

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

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        self._wait_for_nep_dispatch_launch(final=True)
        result = super().finish_grad_sync(force_all_reduce=force_all_reduce)
        return result
