# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in nonuniform expert-parallel gradient ownership transfer.

This module keeps nonuniform EP out of generic Megatron DDP.  Expert params
are wrapped into expert-level bucket groups.  Non-owner ranks transfer whole
expert gradients to an owner rank with point-to-point ops; owner ranks accumulate
those incoming gradients into their normal contiguous ``main_grad`` storage before
running the ordinary expert-data-parallel grad sync.  The synced gradients are
then scattered back to every source rank so non-distributed optimizers can step
the local expert params on every rank.
"""

import copy
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .. import parallel_state
from ..process_groups_config import ProcessGroupCollection
from ..transformer.cuda_graphs import is_graph_capturing
from ..transformer.transformer_config import TransformerConfig
from ._cuda_stream_ops import get_cuda_stream_memory_ops
from ._native_nccl import get_native_nccl
from .distributed_data_parallel import DistributedDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .nonuniform_common import (
    NonuniformEPRankGenerator,
    compute_nonuniform_ep_expert_placement,
    configure_ordered_bucket_group_scheduler,
    filter_kwargs_for_callable,
    get_global_rank,
    get_nonuniform_ep_runtime_config,
    reset_ordered_bucket_group_scheduler,
    set_nonuniform_ep_runtime_config,
    try_start_ordered_bucket_groups,
)
from .param_and_grad_buffer import _ParamAndGradBucket, _ParamAndGradBucketGroup

logger = logging.getLogger(__name__)
_NEP_TAG_SLOT_STRIDE = 256
_NEP_NCCL_DEFAULT_MAX_GATHER_BYTES = 8 * 1024 * 1024 * 1024
_NEP_NCCL_DEFAULT_ASYNC_CHUNK_WINDOW = 2
_NEP_NCCL_DEFAULT_EXPERT_BUCKET_GROUPS = 3
_NCCL_CTA_POLICY_ZERO = 2


def _nep_debug_print(message: str) -> None:
    """Print NEP debug messages when explicitly enabled."""
    if os.getenv("MEGATRON_NONUNIFORM_EP_DEBUG", "0").lower() not in ("1", "true", "yes", "on"):
        return
    try:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    except Exception:
        rank = -1
    rank_filter = os.getenv("MEGATRON_NONUNIFORM_EP_DEBUG_RANKS")
    if rank_filter:
        selected_ranks = {
            int(part) for part in rank_filter.replace(",", " ").split() if part.strip()
        }
        if rank not in selected_ranks:
            return
    print(f"[NEP_DEBUG rank={rank}] {message}", flush=True)


def _nep_debug_chunk_checksum(label: str, contexts: List[dict]) -> None:
    """Synchronously log owner-layout chunks for targeted correctness debugging."""
    if os.getenv("MEGATRON_NONUNIFORM_EP_DEBUG_CHUNK_CHECKSUM", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    rank = dist.get_rank()
    rank_filter = os.getenv("MEGATRON_NONUNIFORM_EP_DEBUG_RANKS")
    if rank_filter and rank not in {
        int(part) for part in rank_filter.replace(",", " ").split() if part.strip()
    }:
        return
    values = [float(context["chunk"].float().sum().item()) for context in contexts]
    print(f"[NEP_CHUNK_CHECKSUM rank={rank}] {label} values={values}", flush=True)


def _nep_overlap_debug_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_zero_sm_reshard_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_edp_ready_gate_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_EDP_READY_GATE", "1").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_bucket_ready_gather_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_BUCKET_READY_GATHER", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_device_ordered_edp_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_DEVICE_ORDERED_EDP", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_host_edp_ready_gate_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_same_communicator_ready_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_SAME_COMM_READY", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_defer_dispatch_host_launch_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_defer_model_ep_fence_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_a2a_scatter_scheduler_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_end_iteration_scatter_enabled() -> bool:
    """Return whether every Scatter is deferred until backward has completed."""
    return os.getenv("MEGATRON_NONUNIFORM_EP_END_ITERATION_SCATTER", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_pipeline_host_phases_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_split_host_phases_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_post_graph_phases_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_post_graph_host_phases_enabled() -> bool:
    return os.getenv("MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_benchmark_skip_scatter_enabled() -> bool:
    """Return whether the intentionally incorrect no-Scatter benchmark is enabled."""
    return os.getenv("MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_benchmark_skip_owner_grad_check_enabled() -> bool:
    """Return whether the synthetic owner-DDP gradient check is disabled for profiling."""
    return os.getenv("MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _nep_owner_ddp_config(
    ddp_config: DistributedDataParallelConfig,
) -> DistributedDataParallelConfig:
    """Return the config used by synthetic owner DDP groups."""
    if not _nep_benchmark_skip_owner_grad_check_enabled():
        return ddp_config

    # The live config may have both num_buckets and its resolved bucket_size.
    # Do not invoke __post_init__ again while changing profiling-only flags.
    native_ddp_config = copy.copy(ddp_config)
    native_ddp_config.check_for_nan_in_grad = False
    native_ddp_config.check_for_large_grads = False
    return native_ddp_config


class NonuniformEPApproach(str, Enum):
    """Gradient synchronization approach for nonuniform EP expert params."""

    P2P = "p2p"
    NCCL = "nccl"


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


def _get_nep_parallel_gather_submission_window() -> int:
    value = os.getenv("MEGATRON_NONUNIFORM_EP_PARALLEL_GATHER_WINDOW")
    if value is None:
        return 1
    submission_window = int(value)
    if submission_window <= 0:
        raise RuntimeError("MEGATRON_NONUNIFORM_EP_PARALLEL_GATHER_WINDOW must be positive")
    return submission_window


def _get_nep_nccl_expert_bucket_group_count() -> int:
    value = os.getenv("MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS")
    if value is None:
        return _NEP_NCCL_DEFAULT_EXPERT_BUCKET_GROUPS
    group_count = int(value)
    if group_count <= 0:
        raise RuntimeError("MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS must be positive")
    return group_count


def _get_nep_nccl_gather_buckets_per_edp() -> Optional[int]:
    value = os.getenv("MEGATRON_NONUNIFORM_EP_NCCL_GATHER_BUCKETS_PER_EDP")
    if value in (None, ""):
        return None
    gather_bucket_count = int(value)
    if gather_bucket_count <= 0:
        raise RuntimeError("MEGATRON_NONUNIFORM_EP_NCCL_GATHER_BUCKETS_PER_EDP must be positive")
    return gather_bucket_count


def _nep_two_level_gather_enabled() -> bool:
    return _get_nep_nccl_gather_buckets_per_edp() is not None


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


def _signal_nep_edp_ready(
    ready_group,
    device_ready_event: Optional[torch.cuda.Event],
    device_index: int,
    stream_ptr: int,
    address: int,
    generation: int,
    group_index: int,
    buffer_slot: int,
    gate_name: str,
) -> None:
    """Release a GPU stream wait after device readiness and a host rendezvous."""
    torch.cuda.set_device(device_index)
    if device_ready_event is not None:
        _nep_debug_print(
            "before stream_ready_device_wait "
            f"gate={gate_name} group={group_index} slot={buffer_slot} "
            f"generation={generation}"
        )
        device_ready_event.synchronize()
        _nep_debug_print(
            "after stream_ready_device_wait "
            f"gate={gate_name} group={group_index} slot={buffer_slot} "
            f"generation={generation}"
        )

    ready_work = dist.barrier(group=ready_group, async_op=True)
    _nep_debug_print(
        "before stream_ready_host_wait "
        f"gate={gate_name} group={group_index} slot={buffer_slot} generation={generation}"
    )
    ready_work.wait()
    _nep_debug_print(
        "after stream_ready_host_wait "
        f"gate={gate_name} group={group_index} slot={buffer_slot} generation={generation}"
    )
    get_cuda_stream_memory_ops().write_value32(stream_ptr, address, generation)
    _nep_debug_print(
        "after stream_ready_stream_write "
        f"gate={gate_name} group={group_index} slot={buffer_slot} generation={generation}"
    )


@dataclass
class NonuniformEPConfig:
    """User-facing opt-in config for nonuniform EP gradient ownership transfer."""

    approach: Union[NonuniformEPApproach, str] = NonuniformEPApproach.P2P
    runtime_config: Optional[dict] = None
    expert_owner: Optional[Dict[int, int]] = None
    expert_name_pattern: Union[str, re.Pattern] = field(
        default_factory=_default_expert_name_pattern
    )
    require_owner_local_expert: bool = True
    grad_transfer_tag_base: int = 711_000
    grad_scatter_tag_base: int = 811_000

    def __post_init__(self):
        self.approach = NonuniformEPApproach(self.approach)
        if isinstance(self.expert_name_pattern, str):
            self.expert_name_pattern = re.compile(self.expert_name_pattern)


@dataclass
class _ExpertBucketPlan:
    expert_id: int
    tag_slot: int
    owner_ep_rank: int
    owner_global_rank: int
    source_ep_ranks: List[int]
    source_global_ranks: List[int]
    bucket_slices: List[Tuple[int, int]]
    bucket_group_index: int
    synthetic_owner: bool = False

    @property
    def numel(self) -> int:
        return sum(end - start for start, end in self.bucket_slices)


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


class _P2PGradTransferHandle:
    """Wait handle that drains p2p works and applies grad receive buffers."""

    def __init__(self, works, recv_accumulations=None, recv_copies=None, keepalive_buffers=None):
        self.works = works
        self.recv_accumulations = recv_accumulations or []
        self.recv_copies = recv_copies or []
        self.keepalive_buffers = keepalive_buffers or []

    def is_completed(self):
        """Return whether all p2p work has completed without draining buffers."""
        for work in self.works:
            is_completed = getattr(work, "is_completed", None)
            if is_completed is None or not is_completed():
                return False
        return True

    def wait(self):
        for work in self.works:
            work.wait()
        for bucket, slices, flat_buffer in self.recv_accumulations:
            _accumulate_flat_into_bucket(bucket, slices, flat_buffer)
        for bucket, slices, flat_buffer in self.recv_copies:
            _copy_flat_into_bucket(bucket, slices, flat_buffer)
        self.works = []
        self.recv_accumulations = []
        self.recv_copies = []
        self.keepalive_buffers = []


def _pack_bucket_slices(bucket, slices: List[Tuple[int, int]]) -> torch.Tensor:
    total = sum(end - start for start, end in slices)
    flat = torch.empty(total, dtype=bucket.grad_data.dtype, device=bucket.grad_data.device)
    _pack_bucket_slices_into(bucket, slices, flat)
    return flat


def _pack_bucket_slices_into(bucket, slices: List[Tuple[int, int]], flat: torch.Tensor) -> None:
    offset = 0
    for start, end in slices:
        next_offset = offset + (end - start)
        flat[offset:next_offset].copy_(bucket.grad_data[start:end])
        offset = next_offset


def _accumulate_flat_into_bucket(bucket, slices: List[Tuple[int, int]], flat: torch.Tensor) -> None:
    offset = 0
    for start, end in slices:
        next_offset = offset + (end - start)
        bucket.grad_data[start:end].add_(flat[offset:next_offset])
        offset = next_offset


def _copy_flat_into_bucket(bucket, slices: List[Tuple[int, int]], flat: torch.Tensor) -> None:
    offset = 0
    for start, end in slices:
        next_offset = offset + (end - start)
        bucket.grad_data[start:end].copy_(flat[offset:next_offset])
        offset = next_offset


def _source_ep_ranks_for_owner(
    expert_placement: List[List[int]], owner_ep_rank: int, num_experts: int, min_ep_size: int
) -> List[int]:
    """Return EP ranks that physically hold an owner's expert range."""
    experts_per_owner = num_experts // min_ep_size
    owner_first_expert = owner_ep_rank * experts_per_owner
    owner_last_expert = owner_first_expert + experts_per_owner
    return [
        source_ep_rank
        for source_ep_rank, expert_ids in enumerate(expert_placement)
        if any(owner_first_expert <= expert_id < owner_last_expert for expert_id in expert_ids)
    ]


def _zero_sm_transfer_ranks_by_owner(
    source_ranks_by_owner: Dict[int, List[int]], min_ep_size: int
) -> Dict[int, List[int]]:
    """Avoid two-rank cross-host CE groups with a helper from the same follower lane."""
    transfer_ranks_by_owner = {
        owner_ep_rank: [owner_ep_rank]
        + [source_rank for source_rank in source_ranks if source_rank != owner_ep_rank]
        for owner_ep_rank, source_ranks in source_ranks_by_owner.items()
    }
    available_ranks = sorted(
        set(range(min_ep_size)).union(
            source_rank
            for source_ranks in source_ranks_by_owner.values()
            for source_rank in source_ranks
        )
    )
    owners_by_follower = {}
    for owner_ep_rank, source_ranks in source_ranks_by_owner.items():
        if len(source_ranks) != 2 or owner_ep_rank not in source_ranks:
            continue
        follower_ep_rank = next(rank for rank in source_ranks if rank != owner_ep_rank)
        owners_by_follower.setdefault(follower_ep_rank, []).append(owner_ep_rank)

    for follower_ep_rank, owner_ep_ranks in owners_by_follower.items():
        owner_ep_ranks = sorted(owner_ep_ranks)
        for owner_index, owner_ep_rank in enumerate(owner_ep_ranks):
            if len(owner_ep_ranks) > 1:
                helper_ep_rank = owner_ep_ranks[(owner_index + 1) % len(owner_ep_ranks)]
            else:
                rotated_owner_ranks = [
                    (owner_ep_rank + offset) % min_ep_size for offset in range(1, min_ep_size)
                ]
                helper_candidates = rotated_owner_ranks + [
                    rank for rank in available_ranks if rank not in rotated_owner_ranks
                ]
                try:
                    helper_ep_rank = next(
                        rank
                        for rank in helper_candidates
                        if rank not in (owner_ep_rank, follower_ep_rank)
                    )
                except StopIteration as exc:
                    raise RuntimeError(
                        "A two-rank cross-host NEP transfer group requires a third helper rank"
                    ) from exc
            transfer_ranks_by_owner[owner_ep_rank] = [
                owner_ep_rank,
                helper_ep_rank,
                follower_ep_rank,
            ]

    return transfer_ranks_by_owner


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
    """Return the phase-specific Scatter launch rendezvous group."""
    if "nep_owner_scatter_launch_groups_gloo" in runtime_config:
        return runtime_config["nep_owner_scatter_launch_groups_gloo"].get(owner_ep_rank)
    # Preserve compatibility with focused tests and external runtime configs
    # created before Scatter launch received a dedicated communicator.
    return runtime_config.get("nep_owner_transfer_groups_gloo", {}).get(owner_ep_rank)


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


def _create_group(
    ranks, timeout, nccl_comm_cfgs, desc, backend=None, cta_policy=None, max_ctas=None
):
    _nep_debug_print(f"before_create_group desc={desc} backend={backend} ranks={ranks}")
    pg_options = (
        None if backend == "gloo" else parallel_state.get_nccl_options(desc, nccl_comm_cfgs)
    )
    if cta_policy is not None or max_ctas is not None:
        if pg_options is None:
            pg_options = dist.ProcessGroupNCCL.Options()
    if cta_policy is not None:
        if not hasattr(pg_options.config, "cta_policy"):
            raise RuntimeError(
                "MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD requires a PyTorch build "
                "that exposes ProcessGroupNCCL.Options.config.cta_policy"
            )
        pg_options.config.cta_policy = cta_policy
    if max_ctas is not None:
        if not hasattr(pg_options.config, "max_ctas") or not hasattr(pg_options.config, "min_ctas"):
            raise RuntimeError("NEP EDP readiness gating requires ProcessGroupNCCL CTA limits")
        pg_options.config.max_ctas = max_ctas
        pg_options.config.min_ctas = max_ctas
    group = parallel_state.create_group(
        ranks, timeout=timeout, backend=backend, pg_options=pg_options, group_desc=desc
    )
    _nep_debug_print(f"after_create_group desc={desc} backend={backend} ranks={ranks}")
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
    enable_edp_ready_gate: bool = False,
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
    _nep_debug_print(
        "initialize_nonuniform_ep_process_groups enter "
        f"world_size={world_size} topology={num_tp_cp_per_replica} tp={tp} cp={cp} "
        f"etp={etp} num_moe_experts={num_moe_experts}"
    )

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
    zero_sm_reshard = _nep_zero_sm_reshard_enabled()
    use_separate_owner_gather_groups = not zero_sm_reshard and (
        "nep_owner_gather" in nccl_comm_cfgs
        or _nep_a2a_scatter_scheduler_enabled()
        or _nep_end_iteration_scatter_enabled()
    )
    edp_ready_gate_enabled = (
        enable_edp_ready_gate
        and _nep_edp_ready_gate_enabled()
        and len(set(num_tp_cp_per_replica)) > 1
    )

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
            num_moe_experts,
            len(ranks),
            min_ep_size,
            preferred_follower_fanout=2 if zero_sm_reshard else 1,
        )
        source_ranks_by_owner = {
            owner_ep_rank: _source_ep_ranks_for_owner(
                group_expert_placement, owner_ep_rank, num_moe_experts, min_ep_size
            )
            for owner_ep_rank in range(min_ep_size)
        }
        transfer_ranks_by_owner = (
            _zero_sm_transfer_ranks_by_owner(source_ranks_by_owner, min_ep_size)
            if zero_sm_reshard
            else {
                owner_ep_rank: [owner_ep_rank]
                + [source_rank for source_rank in source_ranks if source_rank != owner_ep_rank]
                for owner_ep_rank, source_ranks in source_ranks_by_owner.items()
            }
        )
        for owner_ep_rank in range(min_ep_size):
            source_ep_ranks = source_ranks_by_owner[owner_ep_rank]
            transfer_ep_ranks = transfer_ranks_by_owner[owner_ep_rank]
            transfer_global_ranks = [ranks[ep_rank] for ep_rank in transfer_ep_ranks]
            owner_transfer_group = None
            owner_gather_group = None
            owner_transfer_group_gloo = None
            owner_scatter_launch_group_gloo = None
            owner_scatter_ready_group_gloo = None
            if len(transfer_global_ranks) > 1:
                owner_transfer_group = _create_group(
                    transfer_global_ranks,
                    timeout,
                    nccl_comm_cfgs,
                    "nep_owner_transfer",
                    cta_policy=_NCCL_CTA_POLICY_ZERO if zero_sm_reshard else None,
                )
                if use_separate_owner_gather_groups:
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
                    if not _nep_end_iteration_scatter_enabled():
                        owner_scatter_launch_group_gloo = _create_group(
                            transfer_global_ranks,
                            timeout,
                            nccl_comm_cfgs,
                            "NEP_OWNER_SCATTER_LAUNCH_GLOO",
                            "gloo",
                        )
                        owner_scatter_ready_group_gloo = _create_group(
                            transfer_global_ranks,
                            timeout,
                            nccl_comm_cfgs,
                            "NEP_OWNER_SCATTER_READY_GLOO",
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
                    nep_owner_scatter_ready_groups_gloo[owner_ep_rank] = (
                        owner_scatter_ready_group_gloo
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
        num_moe_experts,
        local_ep_size,
        min_ep_size,
        preferred_follower_fanout=2 if zero_sm_reshard else 1,
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
        "nep_edp_ready_group": None,
        "edp_ready_gate_enabled": edp_ready_gate_enabled,
        "ep_rank": ep_rank,
        "is_edp_eligible": ep_rank < min_ep_size,
        "is_b_leader": ep_rank < min_ep_size,
        "local_expert_indices": expert_placement[ep_rank],
        "expert_placement": expert_placement,
        "expert_gather_map": expert_gather_map,
        "zero_sm_reshard": zero_sm_reshard,
    }
    set_nonuniform_ep_runtime_config(runtime_config)
    _nep_debug_print(
        "initialize_nonuniform_ep_process_groups exit "
        f"rank={rank} local_ep_size={local_ep_size} ep_rank={ep_rank} "
        f"local_expert_indices={runtime_config['local_expert_indices']}"
    )
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


class NonuniformEPParamAndGradBucketGroup(_ParamAndGradBucketGroup):
    """Expert-level bucket group that transfers grads to owner ranks before DP sync."""

    def configure_nonuniform_ep(
        self,
        runtime_config: dict,
        nonuniform_ep_config: NonuniformEPConfig,
        plan: Union[_ExpertBucketPlan, List[_ExpertBucketPlan]],
    ) -> None:
        self._nep_runtime_config = runtime_config
        self._nep_config = nonuniform_ep_config
        self._nep_plans = plan if isinstance(plan, list) else [plan]
        self._nep_plan = self._nep_plans[0]
        if len(self._nep_plans) != len(self.buckets):
            raise RuntimeError(
                "NEP bucket groups require one transfer plan per bucket: "
                f"got {len(self._nep_plans)} plans for {len(self.buckets)} buckets"
            )
        self._nep_started = False
        self._nep_gather_done = False
        self._nep_owner_dp_sync_started = False
        self._nep_scatter_started = False
        self._nep_ready = False
        self._nep_gather_handle = None
        self._nep_scatter_handle = None
        self._nep_gather_recv_buffers = []
        self._nep_scatter_send_buffers = []
        self._nep_gather_send_buffer = None
        self._nep_scatter_recv_buffer = None

        ep_rank = runtime_config["ep_rank"]
        self._nep_entries = []
        owner_flags = []
        for bucket, entry_plan in zip(self.buckets, self._nep_plans):
            is_owner = ep_rank == entry_plan.owner_ep_rank
            owner_flags.append(is_owner)
            if (
                nonuniform_ep_config.require_owner_local_expert
                and is_owner
                and not entry_plan.synthetic_owner
                and entry_plan.expert_id not in runtime_config.get("_local_expert_id_set", set())
            ):
                raise RuntimeError(
                    "NEP owner mode requires the owner rank to hold optimizer-visible params "
                    f"for expert {entry_plan.expert_id}; owner ep_rank={entry_plan.owner_ep_rank}"
                )
            self._nep_entries.append(
                {
                    "bucket": bucket,
                    "plan": entry_plan,
                    "is_owner": is_owner,
                    "gather_recv_buffers": [],
                    "scatter_send_buffers": [],
                    "gather_send_buffer": None,
                    "scatter_recv_buffer": None,
                }
            )
        if any(owner_flags) and not all(owner_flags):
            raise RuntimeError("NEP grouped bucket contains mixed owner and non-owner plans")
        self._nep_is_owner = any(owner_flags)
        self._allocate_nep_persistent_grad_buffers()

    def _allocate_nep_persistent_grad_buffers(self):
        """Allocate persistent p2p staging buffers for this expert bucket."""
        ep_rank = self._nep_runtime_config["ep_rank"]

        if not hasattr(self, "_nep_entries"):
            self._nep_entries = [
                {
                    "bucket": self.buckets[0],
                    "plan": self._nep_plan,
                    "is_owner": self._nep_is_owner,
                    "gather_recv_buffers": [],
                    "scatter_send_buffers": [],
                    "gather_send_buffer": None,
                    "scatter_recv_buffer": None,
                }
            ]

        for entry in self._nep_entries:
            plan = entry["plan"]
            bucket = entry["bucket"]
            if plan.numel == 0:
                continue

            if entry["is_owner"]:
                for source_ep_rank, source_global_rank in zip(
                    plan.source_ep_ranks, plan.source_global_ranks
                ):
                    if source_ep_rank == ep_rank:
                        continue
                    entry["gather_recv_buffers"].append(
                        (
                            source_ep_rank,
                            source_global_rank,
                            torch.empty(
                                plan.numel,
                                dtype=bucket.grad_data.dtype,
                                device=bucket.grad_data.device,
                            ),
                        )
                    )
                entry["scatter_send_buffers"] = entry["gather_recv_buffers"]
                self._nep_gather_recv_buffers.extend(entry["gather_recv_buffers"])
                self._nep_scatter_send_buffers = self._nep_gather_recv_buffers
            else:
                entry["gather_send_buffer"] = torch.empty(
                    plan.numel, dtype=bucket.grad_data.dtype, device=bucket.grad_data.device
                )
                entry["scatter_recv_buffer"] = torch.empty(
                    plan.numel, dtype=bucket.grad_data.dtype, device=bucket.grad_data.device
                )
                self._nep_gather_send_buffer = entry["gather_send_buffer"]
                self._nep_scatter_recv_buffer = entry["scatter_recv_buffer"]

    def _grad_transfer_tag(self, plan: Optional[_ExpertBucketPlan] = None) -> int:
        plan = plan or self._nep_plan
        return (
            self._nep_config.grad_transfer_tag_base
            + plan.expert_id * _NEP_TAG_SLOT_STRIDE
            + plan.tag_slot
        )

    def _grad_scatter_tag(self, plan: Optional[_ExpertBucketPlan] = None) -> int:
        plan = plan or self._nep_plan
        return (
            self._nep_config.grad_scatter_tag_base
            + plan.expert_id * _NEP_TAG_SLOT_STRIDE
            + plan.tag_slot
        )

    def _copy_extra_main_grads_to_grad_buffer(self):
        for bucket in self.buckets:
            for param in bucket.params_with_extra_main_grads:
                if getattr(param, "main_grad_copy_in_grad_buffer", None) is not None:
                    param.main_grad_copy_in_grad_buffer.copy_(param.main_grad)

    def _start_owner_dp_sync_after_gather(self, force_all_reduce: Optional[bool] = False):
        saved_extra_main_grads = []
        for bucket in self.buckets:
            saved_extra_main_grads.append((bucket, bucket.params_with_extra_main_grads))
            bucket.params_with_extra_main_grads = []
        try:
            return super().start_grad_sync(force_all_reduce=force_all_reduce)
        finally:
            for bucket, params_with_extra_main_grads in saved_extra_main_grads:
                bucket.params_with_extra_main_grads = params_with_extra_main_grads

    def _start_nep_gather_to_owner(self):
        works = []
        recv_accumulations = []
        keepalive_buffers = []
        transfer_group = self._nep_runtime_config.get(
            "nep_transfer_group", self._nep_runtime_config["ep_group"]
        )

        for entry in self._nep_entries:
            plan = entry["plan"]
            bucket = entry["bucket"]
            if entry["is_owner"]:
                for source_ep_rank, _, recv_buffer in entry["gather_recv_buffers"]:
                    works.append(
                        dist.irecv(
                            recv_buffer,
                            group=transfer_group,
                            group_src=source_ep_rank,
                            tag=self._grad_transfer_tag(plan),
                        )
                    )
                    recv_accumulations.append((bucket, plan.bucket_slices, recv_buffer))
            else:
                send_buffer = entry["gather_send_buffer"]
                _pack_bucket_slices_into(bucket, plan.bucket_slices, send_buffer)
                keepalive_buffers.append(send_buffer)
                works.append(
                    dist.isend(
                        send_buffer,
                        group=transfer_group,
                        group_dst=plan.owner_ep_rank,
                        tag=self._grad_transfer_tag(plan),
                    )
                )

        self._nep_gather_handle = _P2PGradTransferHandle(
            works, recv_accumulations=recv_accumulations, keepalive_buffers=keepalive_buffers
        )

    def _wait_nep_gather_to_owner(self):
        self._complete_nep_gather_to_owner(nonblocking=False)

    def _complete_nep_gather_to_owner(self, nonblocking: bool = False) -> bool:
        if self._nep_gather_done:
            return True
        if self._nep_gather_handle is not None:
            if nonblocking and not self._nep_gather_handle.is_completed():
                return False
            self._nep_gather_handle.wait()
            self._nep_gather_handle = None
        self._nep_gather_done = True
        return True

    def _try_start_ready_owner_dp_syncs(
        self, nonblocking: bool, force_all_reduce: Optional[bool] = False
    ) -> None:
        state = getattr(self, "_nep_owner_dp_sync_scheduler_state", None)
        if state is None:
            if (
                self._nep_is_owner
                and self._nep_started
                and not self._nep_owner_dp_sync_started
                and self._complete_nep_gather_to_owner(nonblocking=nonblocking)
            ):
                self._start_owner_dp_sync_after_gather(force_all_reduce=force_all_reduce)
                self._nep_owner_dp_sync_started = True
            return

        groups = state["groups"]
        while state["next_index"] < len(groups):
            group = groups[state["next_index"]]
            if not group._nep_started:
                break
            if not group._complete_nep_gather_to_owner(nonblocking=nonblocking):
                break
            if not group._nep_owner_dp_sync_started:
                group._start_owner_dp_sync_after_gather(force_all_reduce=force_all_reduce)
                group._nep_owner_dp_sync_started = True
            state["next_index"] += 1

    def _start_nep_scatter_from_owner(self):
        if self._nep_scatter_started:
            return
        works = []
        recv_copies = []
        keepalive_buffers = []
        transfer_group = self._nep_runtime_config.get(
            "nep_transfer_group", self._nep_runtime_config["ep_group"]
        )

        for entry in self._nep_entries:
            plan = entry["plan"]
            bucket = entry["bucket"]
            if entry["is_owner"]:
                for source_ep_rank, _, send_buffer in entry["scatter_send_buffers"]:
                    _pack_bucket_slices_into(bucket, plan.bucket_slices, send_buffer)
                    keepalive_buffers.append(send_buffer)
                    works.append(
                        dist.isend(
                            send_buffer,
                            group=transfer_group,
                            group_dst=source_ep_rank,
                            tag=self._grad_scatter_tag(plan),
                        )
                    )
            else:
                recv_buffer = entry["scatter_recv_buffer"]
                works.append(
                    dist.irecv(
                        recv_buffer,
                        group=transfer_group,
                        group_src=plan.owner_ep_rank,
                        tag=self._grad_scatter_tag(plan),
                    )
                )
                recv_copies.append((bucket, plan.bucket_slices, recv_buffer))

        self._nep_scatter_handle = _P2PGradTransferHandle(
            works, recv_copies=recv_copies, keepalive_buffers=keepalive_buffers
        )
        self._nep_scatter_started = True

    def _record_nep_scatter_wait(self, copy_back_after_wait: bool = False):
        handle = self._nep_scatter_handle
        self._nep_scatter_handle = None
        state = getattr(self, "_nep_post_sync_state", None)
        if state is None:
            if handle is not None:
                handle.wait()
            if copy_back_after_wait:
                self._copy_back_extra_main_grads()
            return

        state["entries"].append((self, handle, copy_back_after_wait))
        if self is state["last_bucket_group"]:
            try:
                for group, pending_handle, pending_copy_back in state["entries"]:
                    if pending_handle is not None:
                        pending_handle.wait()
                    if pending_copy_back:
                        group._copy_back_extra_main_grads()
            finally:
                state["entries"] = []

    def _wait_nep_scatter_from_owner(self):
        if self._nep_scatter_handle is not None:
            self._nep_scatter_handle.wait()
            self._nep_scatter_handle = None

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Transfer expert grads to owner, then run normal DP sync on owner ranks."""
        if self._nep_started:
            return
        self._nep_started = True

        self._copy_extra_main_grads_to_grad_buffer()
        self._start_nep_gather_to_owner()
        if not self._nep_is_owner:
            self.grad_reduce_handle = None
            return
        self._try_start_ready_owner_dp_syncs(nonblocking=True, force_all_reduce=force_all_reduce)

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Finish owner DP sync and scatter synced grads back to source ranks."""
        self.param_gather_dispatched = False
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            if self._nep_is_owner and self.grad_reduce_handle is not None:
                self.grad_reduce_handle.wait()
                self.grad_reduce_handle = None
            if not self._nep_is_owner:
                self._wait_nep_gather_to_owner()
            self._start_nep_scatter_from_owner()
            self._wait_nep_scatter_from_owner()
            self._copy_back_extra_main_grads()
            return

        if self.is_first_batch:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        elif not self._nep_started and len(self.params) == 0:
            self.start_grad_sync(force_all_reduce=force_all_reduce)

        if not self._nep_is_owner:
            self._wait_nep_gather_to_owner()
            self._start_nep_scatter_from_owner()
            self._record_nep_scatter_wait(copy_back_after_wait=True)
            return
        self._try_start_ready_owner_dp_syncs(nonblocking=False, force_all_reduce=force_all_reduce)
        result = super().finish_grad_sync(force_all_reduce=force_all_reduce)
        self._start_nep_scatter_from_owner()
        self._record_nep_scatter_wait(copy_back_after_wait=False)
        return result

    def finish_nep_pre_sync(self, force_all_reduce: Optional[bool] = False):
        """Drain pre-sync p2p and start owner allreduces before generic DDP waits."""
        if not self.ddp_config.overlap_grad_reduce:
            return

        if self.is_first_batch:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        elif not self._nep_started and len(self.params) == 0:
            self.start_grad_sync(force_all_reduce=force_all_reduce)

        if not self._nep_started:
            return
        if not self._nep_is_owner:
            self._wait_nep_gather_to_owner()
            return
        self._try_start_ready_owner_dp_syncs(nonblocking=False, force_all_reduce=force_all_reduce)

    def register_grad_ready(
        self, param: torch.nn.Parameter, force_all_reduce: Optional[bool] = False
    ):
        """Track ready grads and launch expert transfers in deterministic bucket order."""
        assert (
            self.ddp_config.overlap_grad_reduce
        ), "register_grad_ready() should only be called when overlap_grad_reduce is True"
        if self.is_last_microbatch:
            assert param in self.param_to_bucket, "Param is not in the bucket group"
            if param not in self.per_param_grad_ready_counts:
                self.per_param_grad_ready_counts[param] = 0
            self.per_param_grad_ready_counts[param] += 1
            if not self.is_first_batch:
                if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
                    assert len(self.per_param_grad_ready_counts) == len(self.params)
                    self._nep_ready = True
                    try_start_ordered_bucket_groups(
                        self,
                        "_nep_scheduler_state",
                        "_nep_ready",
                        "start_grad_sync",
                        force_all_reduce=force_all_reduce,
                    )

    def reset(self):
        """Reset per-iteration metadata."""
        super().reset()
        self._nep_started = False
        self._nep_gather_done = False
        self._nep_owner_dp_sync_started = False
        self._nep_scatter_started = False
        self._nep_ready = len(self.params) == 0
        self._nep_gather_handle = None
        self._nep_scatter_handle = None
        reset_ordered_bucket_group_scheduler(self, "_nep_scheduler_state", "_nep_group_index")
        state = getattr(self, "_nep_owner_dp_sync_scheduler_state", None)
        if state is not None and getattr(self, "_nep_owner_dp_sync_group_index", -1) == 0:
            state["next_index"] = 0


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
        self._nep_edp_ready_futures = []
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

    def _start_nep_nccl_same_communicator_ready(self, group, token_key: tuple) -> None:
        """Rendezvous on the communicator that owns the following data operation."""
        if group is None or group.size() <= 1:
            return

        state = self._get_nep_nccl_shared_buffer_state()
        buffers = state.setdefault("same_communicator_ready_buffers", {})
        token = buffers.get(token_key)
        if token is None:
            token = torch.zeros(1, dtype=torch.float32, device=torch.cuda.current_device())
            buffers[token_key] = token
        slot_key = ("same_communicator_ready",) + token_key
        self._order_nep_nccl_buffer_slot(slot_key)
        _nep_debug_print(
            f"before same_communicator_ready token={token_key} group_size={group.size()}"
        )
        work = dist.all_reduce(token, group=group, async_op=True)
        self._record_nep_nccl_work(work, slot_key)
        _nep_debug_print(f"after same_communicator_ready token={token_key}")

    def _prepare_nep_nccl_stream_ready_gate(
        self, buffer_slot: int, ready_group, gate_name: str
    ) -> dict:
        """Queue a stream wait and return the token used to release it."""
        state = self._get_nep_nccl_shared_buffer_state()
        flags = state.get("edp_ready_flags")
        generations = state.get("edp_ready_generations")
        executor = state.get("edp_ready_executor")
        signal_stream = state.get("edp_ready_signal_stream")
        if (
            flags is None
            or generations is None
            or executor is None
            or signal_stream is None
            or buffer_slot not in flags
        ):
            raise RuntimeError("NEP stream-readiness gate was not initialized")

        generation = generations[buffer_slot] + 1
        generations[buffer_slot] = generation
        flag = flags[buffer_slot]
        group_index = getattr(self, "_nep_nccl_group_index", -1)
        with torch.profiler.record_function(f"nep_stream_{gate_name}_ready"):
            get_cuda_stream_memory_ops().wait_value32(
                torch.cuda.current_stream().cuda_stream, flag.data_ptr(), generation
            )
        _nep_debug_print(
            "enqueued stream_ready_stream_wait "
            f"gate={gate_name} group={group_index} slot={buffer_slot} "
            f"generation={generation}"
        )
        return {
            "ready_group": ready_group,
            "device_index": torch.cuda.current_device(),
            "stream_ptr": signal_stream.cuda_stream,
            "address": flag.data_ptr(),
            "generation": generation,
            "group_index": group_index,
            "buffer_slot": buffer_slot,
            "gate_name": gate_name,
            "executor": executor,
        }

    def _release_nep_nccl_stream_ready_gate(
        self, gate: dict, device_ready_event: Optional[torch.cuda.Event] = None
    ) -> None:
        """Release a prepared stream wait after every participant reaches its rendezvous."""
        future = gate["executor"].submit(
            _signal_nep_edp_ready,
            gate["ready_group"],
            device_ready_event,
            gate["device_index"],
            gate["stream_ptr"],
            gate["address"],
            gate["generation"],
            gate["group_index"],
            gate["buffer_slot"],
            gate["gate_name"],
        )
        self._nep_edp_ready_futures.append(future)
        _nep_debug_print(
            "submitted stream_ready_host_wait "
            f"gate={gate['gate_name']} group={gate['group_index']} "
            f"slot={gate['buffer_slot']} generation={gate['generation']}"
        )

    def _start_nep_nccl_stream_ready_gate(
        self,
        buffer_slot: int,
        ready_group,
        device_ready_event: Optional[torch.cuda.Event],
        gate_name: str,
    ) -> None:
        """Gate the current stream on device completion plus a host rendezvous."""
        gate = self._prepare_nep_nccl_stream_ready_gate(buffer_slot, ready_group, gate_name)
        self._release_nep_nccl_stream_ready_gate(gate, device_ready_event)

    def _start_nep_nccl_edp_ready_gate(
        self, buffer_slot: int, device_ready_event: Optional[torch.cuda.Event] = None
    ) -> None:
        """Gate EDP until both owner replicas have device-ready Gather payloads."""
        cfg = self._nep_runtime_config
        ready_group = cfg.get("edp_group_gloo")
        if not cfg.get("edp_ready_gate_enabled", False) or ready_group is None:
            return
        self._start_nep_nccl_stream_ready_gate(buffer_slot, ready_group, device_ready_event, "edp")

    def _start_nep_nccl_scatter_ready_gate(
        self, owner_ep_rank: int, buffer_slot: int, device_ready_event: torch.cuda.Event
    ) -> None:
        """Gate Scatter until the owner finishes its EDP reduction."""
        cfg = self._nep_runtime_config
        ready_group = cfg.get("nep_owner_scatter_ready_groups_gloo", {}).get(owner_ep_rank)
        if not cfg.get("edp_ready_gate_enabled", False) or ready_group is None:
            return
        self._start_nep_nccl_stream_ready_gate(
            buffer_slot, ready_group, device_ready_event, f"scatter_owner_{owner_ep_rank}"
        )

    def _prepare_nep_nccl_scatter_descriptor_ready_gate(
        self, descriptor: Optional[dict], buffer_slot: int
    ) -> Optional[dict]:
        """Keep a Scatter NCCL kernel dormant until every communicator rank has queued it."""
        if (
            descriptor is None
            or descriptor["kind"] != "all_to_all"
            or _nep_end_iteration_scatter_enabled()
        ):
            return None
        ready_group = self._nep_runtime_config.get("nep_owner_scatter_ready_groups_gloo", {}).get(
            descriptor["owner_ep_rank"]
        )
        if ready_group is None:
            raise RuntimeError(
                "Scheduled NEP Scatter requires a dedicated owner Scatter-readiness Gloo group"
            )
        return self._prepare_nep_nccl_stream_ready_gate(
            buffer_slot,
            ready_group,
            (
                f"scatter_descriptor_owner_{descriptor['owner_ep_rank']}_"
                f"chunk_{descriptor['chunk_index']}"
            ),
        )

    def _release_nep_nccl_scatter_descriptor_ready_gate(self, gate: Optional[dict]) -> None:
        """Release a descriptor only after its NCCL operation is queued behind the wait."""
        if gate is not None:
            self._release_nep_nccl_stream_ready_gate(gate)

    def _start_nep_nccl_edp_readiness(
        self, buffer_slot: int, device_ready_event: Optional[torch.cuda.Event] = None
    ) -> None:
        """Rendezvous EDP owners without queueing NCCL ahead of warmup reshards."""
        cfg = self._nep_runtime_config
        if self.is_first_batch and cfg.get("edp_ready_gate_enabled", False):
            # First-batch task barriers establish ordering while NCCL and CUDA paths warm up.
            return

        self._start_nep_nccl_edp_ready_gate(buffer_slot, device_ready_event)

    def _drain_nep_edp_ready_futures(self) -> None:
        """Surface readiness-worker failures before waiting on dependent NCCL work."""
        for future in self._nep_edp_ready_futures:
            future.result()
        self._nep_edp_ready_futures.clear()

    def _synchronize_first_batch_zero_sm_phase(self, owner_ep_rank: int, phase: str) -> None:
        """Finish zero-SM phases before crossing communicators on warmup."""
        if not self.is_first_batch or not self._nep_runtime_config.get("zero_sm_reshard", False):
            return
        transfer_ranks = self._nep_nccl_owner_transfer_ranks(owner_ep_rank)
        ep_rank = self._nep_runtime_config["ep_rank"]
        if phase != "edp" and (len(transfer_ranks) <= 1 or ep_rank not in transfer_ranks):
            return
        _nep_debug_print(f"before first_batch_zero_sm_sync phase={phase} owner={owner_ep_rank}")
        torch.cuda.current_stream().synchronize()
        _nep_debug_print(f"after first_batch_zero_sm_sync phase={phase} owner={owner_ep_rank}")

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
        if self._nep_runtime_config.get("zero_sm_reshard", False) and not self.is_first_batch:
            stream_key = "zero_sm"
        elif self._nep_dispatch_boundary_launch and not self.is_first_batch:
            submission_window = _get_nep_parallel_gather_submission_window()
            stream_key = (
                "dispatch"
                if submission_window == 1
                else ("dispatch", stream_slot % submission_window)
            )
        elif _nep_end_iteration_scatter_enabled():
            stream_key = ("end_iteration", stream_slot % _get_nep_nccl_async_chunk_window())
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
        if _nep_end_iteration_scatter_enabled():
            return task_ordinal
        return task_ordinal % buffer_slots

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

    def _nep_nccl_owner_entries(self, owner_ep_rank: int) -> List[dict]:
        """Return local expert-slot entries that contribute to an owner-layout chunk."""
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
        """Return True when this rank's local inputs for an owner task are ready."""
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
        foreach_copy = (
            getattr(torch, "_foreach_copy_", None)
            if self._nep_runtime_config.get("zero_sm_reshard", False)
            else None
        )
        if foreach_copy is None:
            for destination, source in zip(destinations, sources):
                destination.copy_(source)
            return
        foreach_copy(destinations, sources)

    def _foreach_add_(self, destinations: List[torch.Tensor], sources: List[torch.Tensor]) -> None:
        if not destinations:
            return
        foreach_add = (
            getattr(torch, "_foreach_add_", None)
            if self._nep_runtime_config.get("zero_sm_reshard", False)
            else None
        )
        if foreach_add is None:
            for destination, source in zip(destinations, sources):
                destination.add_(source)
            return
        foreach_add(destinations, sources)

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

        experts_per_owner = self._nep_nccl_experts_per_owner
        owner_first_expert = owner_ep_rank * experts_per_owner
        owner_last_expert = owner_first_expert + experts_per_owner
        source_ranks = []
        for source_ep_rank, expert_ids in enumerate(placement):
            if any(owner_first_expert <= expert_id < owner_last_expert for expert_id in expert_ids):
                source_ranks.append(source_ep_rank)
        return source_ranks

    def _nep_nccl_owner_transfer_ranks(self, owner_ep_rank: int) -> List[int]:
        """Return all ranks that participate in an owner's reshard communicator."""
        transfer_ranks = self._nep_runtime_config.get("nep_owner_transfer_group_ranks")
        if transfer_ranks is not None and owner_ep_rank in transfer_ranks:
            return list(transfer_ranks[owner_ep_rank])
        return self._nep_nccl_owner_source_ranks(owner_ep_rank)

    def _get_nep_zero_sm_buffers(
        self, buffer_slot: int, dtype: torch.dtype, transfer_size: int, payload_numel: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self._get_nep_nccl_shared_buffer_state()
        small_buffers = state.get("zero_sm_small_buffers")
        large_buffers = state.get("zero_sm_large_buffers")
        if small_buffers is None or large_buffers is None:
            raise RuntimeError("NEP zero-SM staging buffers were not initialized")
        small = small_buffers[(buffer_slot, dtype)]
        large = large_buffers[(buffer_slot, dtype)]
        large_numel = transfer_size * payload_numel
        if small.numel() < payload_numel or large.numel() < large_numel:
            raise RuntimeError(
                "NEP zero-SM staging buffer is too small: "
                f"small={small.numel()}/{payload_numel}, "
                f"large={large.numel()}/{large_numel}"
            )
        return small[:payload_numel], large[:large_numel]

    def _get_nep_zero_sm_comm_ptr(self, transfer_group) -> int:
        state = self._get_nep_nccl_shared_buffer_state()
        comm_ptr = state.get("zero_sm_comm_ptrs", {}).get(id(transfer_group))
        if comm_ptr is None:
            raise RuntimeError("NEP zero-SM communicator pointer was not initialized")
        return comm_ptr

    def _start_nep_nccl_owner_native_gather(
        self,
        owner_ep_rank: int,
        chunk_start: int,
        chunk_end: int,
        chunk: torch.Tensor,
        buffer_slot_key: tuple,
    ) -> None:
        """Gather dense follower payloads with NCCL's copy-engine collective."""
        ep_rank = self._nep_runtime_config["ep_rank"]
        source_ranks = self._nep_nccl_owner_source_ranks(owner_ep_rank)
        transfer_ranks = self._nep_nccl_owner_transfer_ranks(owner_ep_rank)
        if ep_rank not in transfer_ranks:
            return
        remote_source_ranks = [rank for rank in source_ranks if rank != owner_ep_rank]
        if not remote_source_ranks:
            return

        transfer_group, _, transfer_size, transfer_source_ranks = (
            self._get_nep_nccl_transfer_group_info(owner_ep_rank)
        )
        owner_transfer_rank = transfer_source_ranks.index(owner_ep_rank)
        payload_numel = max(
            self._nep_nccl_owner_source_payload_numel(
                owner_ep_rank, source_ep_rank, chunk_start, chunk_end
            )
            for source_ep_rank in remote_source_ranks
        )
        if payload_numel == 0:
            return
        buffer_slot = buffer_slot_key[0]
        send, recv = self._get_nep_zero_sm_buffers(
            buffer_slot, chunk.dtype, transfer_size, payload_numel
        )
        send.zero_()
        if ep_rank in remote_source_ranks:
            local_payload_numel = self._nep_nccl_owner_source_payload_numel(
                owner_ep_rank, ep_rank, chunk_start, chunk_end
            )
            self._pack_nep_nccl_source_payload(
                owner_ep_rank, ep_rank, chunk_start, chunk_end, send[:local_payload_numel]
            )

        role = (
            "owner"
            if ep_rank == owner_ep_rank
            else "source" if ep_rank in remote_source_ranks else "helper"
        )
        _nep_debug_print(
            "before native_zero_sm_gather "
            f"group={getattr(self, '_nep_nccl_group_index', -1)} "
            f"owner={owner_ep_rank} ep_rank={ep_rank} role={role} "
            f"transfer_ranks={transfer_ranks} payload_numel={payload_numel}"
        )
        host_start = time.perf_counter()
        with torch.profiler.record_function("nep_zero_sm_gather"):
            get_native_nccl().gather(
                send,
                recv,
                payload_numel,
                owner_transfer_rank,
                self._get_nep_zero_sm_comm_ptr(transfer_group),
            )
        _nep_debug_print(
            "after native_zero_sm_gather "
            f"group={getattr(self, '_nep_nccl_group_index', -1)} "
            f"owner={owner_ep_rank} ep_rank={ep_rank} role={role} "
            f"host_ms={(time.perf_counter() - host_start) * 1000.0:.3f}"
        )

        if ep_rank == owner_ep_rank:
            for source_ep_rank in remote_source_ranks:
                source_transfer_rank = transfer_source_ranks.index(source_ep_rank)
                source_numel = self._nep_nccl_owner_source_payload_numel(
                    owner_ep_rank, source_ep_rank, chunk_start, chunk_end
                )
                source_start = source_transfer_rank * payload_numel
                self._accumulate_nep_nccl_source_payload(
                    owner_ep_rank,
                    source_ep_rank,
                    chunk_start,
                    chunk_end,
                    recv[source_start : source_start + source_numel],
                    chunk,
                )

    def _start_nep_nccl_owner_native_scatter(
        self,
        owner_ep_rank: int,
        chunk_start: int,
        chunk_end: int,
        chunk: torch.Tensor,
        buffer_slot_key: tuple,
    ) -> None:
        """Scatter reduced follower payloads with NCCL's copy-engine collective."""
        ep_rank = self._nep_runtime_config["ep_rank"]
        source_ranks = self._nep_nccl_owner_source_ranks(owner_ep_rank)
        transfer_ranks = self._nep_nccl_owner_transfer_ranks(owner_ep_rank)
        if ep_rank not in transfer_ranks:
            return
        remote_source_ranks = [rank for rank in source_ranks if rank != owner_ep_rank]
        if not remote_source_ranks:
            if ep_rank == owner_ep_rank:
                self._copy_nep_nccl_owner_chunk_to_local_grads(
                    owner_ep_rank, chunk_start, chunk_end, chunk
                )
            return

        transfer_group, _, transfer_size, transfer_source_ranks = (
            self._get_nep_nccl_transfer_group_info(owner_ep_rank)
        )
        owner_transfer_rank = transfer_source_ranks.index(owner_ep_rank)
        payload_numel = max(
            self._nep_nccl_owner_source_payload_numel(
                owner_ep_rank, source_ep_rank, chunk_start, chunk_end
            )
            for source_ep_rank in remote_source_ranks
        )
        if payload_numel == 0:
            if ep_rank == owner_ep_rank:
                self._copy_nep_nccl_owner_chunk_to_local_grads(
                    owner_ep_rank, chunk_start, chunk_end, chunk
                )
            return
        buffer_slot = buffer_slot_key[0]
        recv, send = self._get_nep_zero_sm_buffers(
            buffer_slot, chunk.dtype, transfer_size, payload_numel
        )
        recv.zero_()
        send.zero_()
        if ep_rank == owner_ep_rank:
            for destination_ep_rank in remote_source_ranks:
                destination_transfer_rank = transfer_source_ranks.index(destination_ep_rank)
                destination_numel = self._nep_nccl_owner_source_payload_numel(
                    owner_ep_rank, destination_ep_rank, chunk_start, chunk_end
                )
                destination_start = destination_transfer_rank * payload_numel
                self._pack_nep_nccl_scatter_payload(
                    owner_ep_rank,
                    destination_ep_rank,
                    chunk_start,
                    chunk_end,
                    chunk,
                    send[destination_start : destination_start + destination_numel],
                )

        role = (
            "owner"
            if ep_rank == owner_ep_rank
            else "source" if ep_rank in remote_source_ranks else "helper"
        )
        _nep_debug_print(
            "before native_zero_sm_scatter "
            f"group={getattr(self, '_nep_nccl_group_index', -1)} "
            f"owner={owner_ep_rank} ep_rank={ep_rank} role={role} "
            f"transfer_ranks={transfer_ranks} payload_numel={payload_numel}"
        )
        host_start = time.perf_counter()
        with torch.profiler.record_function("nep_zero_sm_scatter"):
            get_native_nccl().scatter(
                send,
                recv,
                payload_numel,
                owner_transfer_rank,
                self._get_nep_zero_sm_comm_ptr(transfer_group),
            )
        _nep_debug_print(
            "after native_zero_sm_scatter "
            f"group={getattr(self, '_nep_nccl_group_index', -1)} "
            f"owner={owner_ep_rank} ep_rank={ep_rank} role={role} "
            f"host_ms={(time.perf_counter() - host_start) * 1000.0:.3f}"
        )

        if ep_rank == owner_ep_rank:
            self._copy_nep_nccl_owner_chunk_to_local_grads(
                owner_ep_rank, chunk_start, chunk_end, chunk
            )
        elif ep_rank in remote_source_ranks:
            local_payload_numel = self._nep_nccl_owner_source_payload_numel(
                owner_ep_rank, ep_rank, chunk_start, chunk_end
            )
            self._copy_nep_nccl_scatter_payload_to_local_grads(
                owner_ep_rank, ep_rank, chunk_start, chunk_end, recv[:local_payload_numel]
            )

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

        if cfg.get("zero_sm_reshard", False):
            self._start_nep_nccl_owner_native_gather(
                owner_ep_rank, chunk_start, chunk_end, chunk, buffer_slot_key
            )
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

        _nep_debug_print(
            "before ep_all_to_all_owner_gather "
            f"group={group_index} owner={owner_ep_rank} chunk={chunk_index} "
            f"chunk_size={chunk_size} ep_rank={ep_rank} sources={source_ranks} "
            f"transfer_sources={transfer_source_ranks}"
        )
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
        _nep_debug_print(
            "after ep_all_to_all_owner_gather "
            f"group={group_index} owner={owner_ep_rank} chunk={chunk_index}"
        )

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

        if cfg.get("zero_sm_reshard", False):
            return {
                "kind": "native",
                "owner_ep_rank": owner_ep_rank,
                "chunk_index": chunk_index,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "chunk": chunk,
                "buffer_slot_key": buffer_slot_key,
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
        """Pack one Scatter collective for all Gather buckets in an EDP bucket."""
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
        if cfg.get("zero_sm_reshard", False):
            raise RuntimeError("Two-level NEP Scatter does not support zero-SM reshard")
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

        edp_bucket_index = representative._nep_nccl_edp_bucket_index
        cache = representative._get_nep_nccl_shared_buffer_state()["gather_buf_cache"]
        empty = representative._get_nep_nccl_cached_tensor(
            cache, ("empty", dtype, device), 0, dtype, device
        )
        cache_prefix = (edp_bucket_index, owner_ep_rank, dtype, device)

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
                edp_bucket_index,
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
        if kind == "native":
            self._start_nep_nccl_owner_native_scatter(
                descriptor["owner_ep_rank"],
                descriptor["chunk_start"],
                descriptor["chunk_end"],
                descriptor["chunk"],
                descriptor["buffer_slot_key"],
            )
            descriptor["submitted"] = True
            descriptor["completion_ordered"] = True
            return
        if kind != "all_to_all":
            raise RuntimeError(f"Unknown NEP Scatter descriptor kind: {kind}")

        _nep_debug_print(
            "before ep_all_to_all_owner_scatter "
            f"group={descriptor['group_index']} owner={descriptor['owner_ep_rank']} "
            f"chunk={descriptor['chunk_index']} chunk_size={descriptor['chunk_size']} "
            f"ep_rank={descriptor['ep_rank']} destinations={descriptor['source_ranks']} "
            f"transfer_sources={descriptor['transfer_source_ranks']}"
        )
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
        if kind == "native":
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
        _nep_debug_print(
            "after ep_all_to_all_owner_scatter "
            f"group={descriptor['group_index']} owner={descriptor['owner_ep_rank']} "
            f"chunk={descriptor['chunk_index']}"
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
            raise RuntimeError("NEP owner-layout DDP does not support distributed optimizer")
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

    def _start_nep_nccl_owner_edp_reduce_contexts(
        self, contexts: List[dict], use_device_readiness: bool = True
    ) -> None:
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

        first_batch_task_group = None
        if self.is_first_batch and cfg.get("edp_ready_gate_enabled", False):
            first_batch_task_group = cfg.get("dp_cp_group_gloo")
            if first_batch_task_group is None:
                raise RuntimeError("First-batch NEP readiness requires a DP-CP Gloo group")
            torch.cuda.current_stream().synchronize()
            dist.barrier(group=first_batch_task_group)
        if cfg["ep_rank"] != owner_ep_rank:
            if first_batch_task_group is not None:
                dist.barrier(group=first_batch_task_group)
            return

        edp_group = cfg.get("edp_group")
        if edp_group is None:
            raise RuntimeError(
                "Nonuniform EP NCCL owner rank requires runtime_config['edp_group']."
            )
        if use_device_readiness and cfg.get("edp_ready_gate_enabled", False):
            self._start_nep_nccl_edp_readiness(
                contexts[0]["buffer_slot"], contexts[-1].get("gather_done_event")
            )

        group_index = getattr(self, "_nep_nccl_group_index", -1)
        chunk_indices = [context["chunk_index"] for context in contexts]
        _nep_debug_print(
            "before native_ddp_owner_group "
            f"group={group_index} owner={owner_ep_rank} chunks={chunk_indices} "
            f"numel={sum(context['chunk'].numel() for context in contexts)} "
            f"edp_rank={edp_group.rank()}"
        )
        native_group = self._get_nep_nccl_native_edp_bucket_group(contexts)
        _nep_debug_chunk_checksum(f"before_edp group={group_index} owner={owner_ep_rank}", contexts)
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
        self._synchronize_first_batch_zero_sm_phase(owner_ep_rank, "edp")
        if first_batch_task_group is not None:
            torch.cuda.current_stream().synchronize()
            dist.barrier(group=first_batch_task_group)
        _nep_debug_print(
            "after native_ddp_owner_group "
            f"group={group_index} owner={owner_ep_rank} chunks={chunk_indices}"
        )

    def _start_nep_nccl_owner_edp_reduce(
        self, context: dict, use_device_readiness: bool = True
    ) -> None:
        """Launch native EDP for a single-context path."""
        self._start_nep_nccl_owner_edp_reduce_contexts(
            [context], use_device_readiness=use_device_readiness
        )

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
            _nep_debug_chunk_checksum(
                "after_edp "
                f"group={getattr(self, '_nep_nccl_group_index', -1)} "
                f"owner={context['owner_ep_rank']}",
                native_state["contexts"],
            )
        elif native_group.grad_reduce_handle is not None:
            raise RuntimeError("Synchronous NEP native DDP left an outstanding reduction")
        native_state["scatter_dependency_ordered"] = True

    def _finish_nep_nccl_native_edp_reductions(self) -> None:
        """Finish every native owner EDP Work once, at the final DDP drain."""
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

    def _stage_nep_nccl_owner_edp_contexts(self, contexts: List[dict]) -> List[List[dict]]:
        """Return complete owner/EDP buckets while retaining partial Gather buckets."""
        if not _nep_two_level_gather_enabled():
            context_batches = {}
            for context in contexts:
                group = context["group"]
                key = (id(group), context["owner_ep_rank"])
                context_batches.setdefault(key, []).append(context)
            return list(context_batches.values())

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
        self, contexts: List[dict], use_device_readiness: bool = True
    ) -> List[List[dict]]:
        """Launch one native EDP group per original expert bucket and owner."""
        context_batches = self._stage_nep_nccl_owner_edp_contexts(contexts)
        for context_batch in context_batches:
            group = context_batch[0]["group"]
            if group._nep_runtime_config["ep_rank"] == context_batch[0]["owner_ep_rank"]:
                group._start_nep_nccl_owner_edp_reduce_contexts(
                    context_batch, use_device_readiness=use_device_readiness
                )
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

    def _start_nep_nccl_dispatch_boundary_task(self, context: dict) -> None:
        """Enqueue one ordered zero-SM task at a post-dispatch backward boundary."""
        owner_ep_rank = context["owner_ep_rank"]
        self._start_nep_nccl_owner_all_to_all_gather(
            owner_ep_rank,
            context["chunk_index"],
            context["chunk_start"],
            context["chunk_end"],
            context["chunk"],
            context["buffer_slot_key"],
            async_op=True,
        )
        self._start_nep_nccl_owner_edp_reduce(context, use_device_readiness=True)
        self._start_nep_nccl_owner_task_scatter(context)

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
        """Launch one ordered owner-layout gather/allreduce/scatter task."""
        context = self._prepare_nep_nccl_owner_task_context(
            owner_ep_rank, chunk_index, chunk_start, chunk_end, async_op
        )
        if context is None:
            return

        cfg = self._nep_runtime_config
        ep_rank = cfg["ep_rank"]
        chunk = context["chunk"]
        buffer_slot_key = context["buffer_slot_key"]
        host_phase_gating = (
            async_op
            and _nep_host_edp_ready_gate_enabled()
            and not cfg.get("zero_sm_reshard", False)
        )
        if host_phase_gating:
            transfer_ranks = self._nep_nccl_owner_transfer_ranks(owner_ep_rank)
            if ep_rank not in transfer_ranks:
                self._mark_nep_nccl_task_started(owner_ep_rank, chunk_index)
                return
            self._start_nep_nccl_owner_all_to_all_gather(
                owner_ep_rank,
                chunk_index,
                chunk_start,
                chunk_end,
                chunk,
                buffer_slot_key,
                async_op=True,
            )
            gather_done_event = torch.cuda.Event()
            gather_done_event.record(torch.cuda.current_stream())
            context["gather_done_event"] = gather_done_event
            context["stage"] = "gather"
            state = getattr(self, "_nep_nccl_scheduler_state", None)
            if state is None:
                raise RuntimeError("Host-gated NEP phases require a shared task scheduler")
            state.setdefault("pending_owner_tasks", []).append(context)
            return

        if (
            getattr(self, "_nep_dispatch_boundary_launch", False)
            and async_op
            and cfg.get("zero_sm_reshard", False)
        ):
            transfer_ranks = self._nep_nccl_owner_transfer_ranks(owner_ep_rank)
            if ep_rank not in transfer_ranks:
                self._mark_nep_nccl_task_started(owner_ep_rank, chunk_index)
                return
            self._start_nep_nccl_dispatch_boundary_task(context)
            return

        self._start_nep_nccl_owner_all_to_all_gather(
            owner_ep_rank,
            chunk_index,
            chunk_start,
            chunk_end,
            chunk,
            buffer_slot_key,
            async_op=async_op,
        )
        self._synchronize_first_batch_zero_sm_phase(owner_ep_rank, "gather")

        if _nep_two_level_gather_enabled():
            if not self.is_first_batch:
                raise RuntimeError(
                    "Steady-state two-level NEP Gather must launch from AccumulateGrad"
                )
            complete_context_batches = self._start_nep_nccl_owner_edp_reduce_batch([context])
            for context_batch in complete_context_batches:
                scatter_context = self._coalesce_nep_nccl_scatter_contexts(context_batch)
                scatter_context["group"]._start_nep_nccl_owner_task_scatter(scatter_context)
            return

        self._start_nep_nccl_owner_edp_reduce(context)
        if defer_scatter:
            state = getattr(self, "_nep_nccl_scheduler_state", None)
            if state is None:
                raise RuntimeError("Deferred NEP scatter requires a shared task scheduler")
            state.setdefault("pending_scatters", []).append(context)
        else:
            self._start_nep_nccl_owner_task_scatter(context)

    def _prepare_nep_nccl_owner_task_scatter_train(self, context: dict) -> dict:
        """Prepare every chunk in one Scatter train without submitting collectives."""
        if _nep_benchmark_skip_scatter_enabled():
            raise RuntimeError("Skipped NEP Scatter cannot be prepared as a chunk train")

        self._order_nep_nccl_owner_edp_before_scatter(context)
        cfg = self._nep_runtime_config
        scatter_chunks = _get_nep_nccl_scatter_chunks()
        if cfg.get("zero_sm_reshard", False) and scatter_chunks != 1:
            raise RuntimeError(
                "MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS only supports "
                "the all-to-all reshard path"
            )

        scatter_contexts = context.get("scatter_contexts")
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

        for descriptor in descriptors:
            if (
                _nep_a2a_scatter_scheduler_enabled()
                and descriptor is not None
                and descriptor["kind"] == "local"
            ):
                self._submit_nep_nccl_owner_all_to_all_scatter(descriptor)
                self._order_nep_nccl_owner_all_to_all_scatter_completion(descriptor)
                self._finish_nep_nccl_owner_all_to_all_scatter(descriptor)
                copy_done = torch.cuda.Event()
                copy_done.record(torch.cuda.current_stream())
                state = self._get_nep_nccl_shared_buffer_state()
                state["buffer_slot_events"].setdefault(context["buffer_slot_key"], []).append(
                    copy_done
                )
                descriptor["kind"] = "completed_local"
        return {
            "group": self,
            "context": context,
            "descriptors": descriptors,
            "next_descriptor": 0,
            "task_marked": False,
        }

    def _mark_nep_nccl_scatter_train_scheduled(self, train: dict) -> None:
        """Mark a background Scatter train as issued to the scheduler."""
        if train["task_marked"]:
            return
        if self.is_first_batch:
            raise RuntimeError("A2A-gated Scatter scheduling is not supported on the first batch")
        context = train["context"]
        scatter_contexts = context.get("scatter_contexts")
        if scatter_contexts is None:
            self._mark_nep_nccl_task_started(context["owner_ep_rank"], context["chunk_index"])
        else:
            for task_context in scatter_contexts:
                task_context["group"]._mark_nep_nccl_task_started(
                    task_context["owner_ep_rank"], task_context["chunk_index"]
                )
        train["task_marked"] = True

    def _finish_nep_nccl_scatter_train_submission(self, train: dict) -> None:
        """Finish bookkeeping after every descriptor in one train is submitted."""
        context = train["context"]
        cfg = self._nep_runtime_config
        self._synchronize_first_batch_zero_sm_phase(context["owner_ep_rank"], "scatter")
        if self.is_first_batch and cfg.get("edp_ready_gate_enabled", False):
            task_group_gloo = cfg.get("dp_cp_group_gloo")
            if task_group_gloo is None:
                raise RuntimeError("First-batch NEP readiness requires a DP-CP Gloo group")
            torch.cuda.current_stream().synchronize()
            dist.barrier(group=task_group_gloo)
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

    def _start_nep_nccl_owner_task_scatter(self, context: dict) -> None:
        if _nep_benchmark_skip_scatter_enabled():
            owner_ep_rank = context["owner_ep_rank"]
            if self._nep_runtime_config["ep_rank"] == owner_ep_rank:
                self._order_nep_nccl_owner_edp_before_scatter(context)
                with torch.profiler.record_function("nep_benchmark_skip_scatter_owner_copy"):
                    scatter_contexts = context.get("scatter_contexts")
                    if scatter_contexts is None:
                        self._copy_nep_nccl_owner_chunk_to_local_grads(
                            owner_ep_rank,
                            context["chunk_start"],
                            context["chunk_end"],
                            context["chunk"],
                        )
                    else:
                        for task_context in scatter_contexts:
                            task_context["group"]._copy_nep_nccl_owner_chunk_to_local_grads(
                                owner_ep_rank,
                                task_context["chunk_start"],
                                task_context["chunk_end"],
                                task_context["chunk"],
                            )
            _nep_debug_print(
                "benchmark skip ep_all_to_all_owner_scatter "
                f"group={getattr(self, '_nep_nccl_group_index', -1)} "
                f"owner={owner_ep_rank} chunk={context['chunk_index']}"
            )
            scatter_contexts = context.get("scatter_contexts")
            if scatter_contexts is None:
                self._mark_nep_nccl_task_started(owner_ep_rank, context["chunk_index"])
            else:
                for task_context in scatter_contexts:
                    task_context["group"]._mark_nep_nccl_task_started(
                        owner_ep_rank, task_context["chunk_index"]
                    )
            return

        train = self._prepare_nep_nccl_owner_task_scatter_train(context)
        for descriptor in train["descriptors"]:
            self._submit_nep_nccl_owner_all_to_all_scatter(descriptor)
            self._order_nep_nccl_owner_all_to_all_scatter_completion(descriptor)
            self._finish_nep_nccl_owner_all_to_all_scatter(descriptor)
        self._finish_nep_nccl_scatter_train_submission(train)

    def _progress_nep_nccl_pending_owner_tasks(self, force_all: bool = False) -> bool:
        """Advance host-gated tasks without queueing a GPU readiness collective."""
        state = getattr(self, "_nep_nccl_scheduler_state", None)
        if state is None:
            return True

        def event_is_complete(event: torch.cuda.Event) -> bool:
            if force_all:
                event.synchronize()
                return True
            return event.query()

        def work_is_complete(work) -> bool:
            if force_all:
                work.wait()
                return True
            return work.is_completed()

        pending_tasks = state.setdefault("pending_owner_tasks", [])
        while pending_tasks:
            context = pending_tasks[0]
            group = context["group"]
            cfg = group._nep_runtime_config
            owner_ep_rank = context["owner_ep_rank"]
            is_owner = cfg["ep_rank"] == owner_ep_rank
            source_group_gloo = cfg.get("nep_owner_transfer_groups_gloo", {}).get(owner_ep_rank)
            scatter_launch_group_gloo = _get_nep_owner_scatter_launch_group(cfg, owner_ep_rank)

            if context["stage"] == "gather":
                if not event_is_complete(context["gather_done_event"]):
                    return False
                if source_group_gloo is not None and not context.get("gather_launch_gated"):
                    _nep_debug_print(
                        "before host_source_gather_barrier "
                        f"group={getattr(group, '_nep_nccl_group_index', -1)} "
                        f"owner={owner_ep_rank} chunk={context['chunk_index']}"
                    )
                    context["source_gather_ready_work"] = dist.barrier(
                        group=source_group_gloo, async_op=True
                    )
                    context["stage"] = "source_gather_ready"
                else:
                    context["stage"] = "source_gather_ready_done"

            if context["stage"] == "source_gather_ready":
                if not work_is_complete(context["source_gather_ready_work"]):
                    return False
                _nep_debug_print(
                    "after host_source_gather_barrier "
                    f"group={getattr(group, '_nep_nccl_group_index', -1)} "
                    f"owner={owner_ep_rank} chunk={context['chunk_index']}"
                )
                context["stage"] = "source_gather_ready_done"

            if context["stage"] == "source_gather_ready_done":
                if is_owner:
                    edp_group_gloo = cfg.get("edp_group_gloo")
                    if edp_group_gloo is None:
                        raise RuntimeError("Host-gated NEP EDP readiness requires EDP Gloo groups")
                    _nep_debug_print(
                        "before host_edp_ready_barrier "
                        f"group={getattr(group, '_nep_nccl_group_index', -1)} "
                        f"owner={owner_ep_rank} chunk={context['chunk_index']}"
                    )
                    context["edp_ready_work"] = dist.barrier(group=edp_group_gloo, async_op=True)
                    context["stage"] = "edp_ready"
                else:
                    if scatter_launch_group_gloo is None:
                        raise RuntimeError(
                            "NEP follower rank is missing its owner Scatter-launch Gloo group"
                        )
                    _nep_debug_print(
                        "before host_source_scatter_barrier "
                        f"group={getattr(group, '_nep_nccl_group_index', -1)} "
                        f"owner={owner_ep_rank} chunk={context['chunk_index']}"
                    )
                    context["source_scatter_ready_work"] = dist.barrier(
                        group=scatter_launch_group_gloo, async_op=True
                    )
                    context["stage"] = "source_scatter_ready"

            if context["stage"] == "edp_ready":
                if not work_is_complete(context["edp_ready_work"]):
                    return False
                _nep_debug_print(
                    "after host_edp_ready_barrier "
                    f"group={getattr(group, '_nep_nccl_group_index', -1)} "
                    f"owner={owner_ep_rank} chunk={context['chunk_index']}"
                )
                nccl_stream = group._get_nep_nccl_comm_stream(context["buffer_slot"])
                with torch.cuda.stream(nccl_stream):
                    group._start_nep_nccl_owner_edp_reduce(context, use_device_readiness=False)
                    edp_done_event = torch.cuda.Event()
                    edp_done_event.record(nccl_stream)
                context["edp_done_event"] = edp_done_event
                context["stage"] = "edp"

            if context["stage"] == "edp":
                if not event_is_complete(context["edp_done_event"]):
                    return False
                if scatter_launch_group_gloo is not None:
                    _nep_debug_print(
                        "before host_source_scatter_barrier "
                        f"group={getattr(group, '_nep_nccl_group_index', -1)} "
                        f"owner={owner_ep_rank} chunk={context['chunk_index']}"
                    )
                    context["source_scatter_ready_work"] = dist.barrier(
                        group=scatter_launch_group_gloo, async_op=True
                    )
                    context["stage"] = "source_scatter_ready"
                else:
                    nccl_stream = group._get_nep_nccl_comm_stream(context["buffer_slot"])
                    with torch.cuda.stream(nccl_stream):
                        group._start_nep_nccl_owner_task_scatter(context)
                    pending_tasks.pop(0)
                    continue

            if context["stage"] == "source_scatter_ready":
                if not work_is_complete(context["source_scatter_ready_work"]):
                    return False
                _nep_debug_print(
                    "after host_source_scatter_barrier "
                    f"group={getattr(group, '_nep_nccl_group_index', -1)} "
                    f"owner={owner_ep_rank} chunk={context['chunk_index']}"
                )
                nccl_stream = group._get_nep_nccl_comm_stream(context["buffer_slot"])
                with torch.cuda.stream(nccl_stream):
                    group._start_nep_nccl_owner_task_scatter(context)
                pending_tasks.pop(0)
                continue

            raise RuntimeError(f"Unknown pending NEP task stage: {context['stage']}")
        return True

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

        if _nep_two_level_gather_enabled():
            if not _nep_device_ordered_edp_enabled():
                raise RuntimeError("Two-level NEP Gather requires device-ordered EDP")
            if _get_nep_parallel_gather_submission_window() != 1:
                raise RuntimeError("Two-level NEP Gather requires one Gather submission stream")
            with torch.cuda.stream(dispatch_stream):
                complete_context_batches = self._start_nep_nccl_owner_edp_reduce_batch(
                    contexts, use_device_readiness=False
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
                        "local_transfer_contexts": {},
                        "local_edp_contexts": {},
                        "gather_barrier_works": [],
                        "gather_done_event": gather_done_event,
                        "phase": "edp_launched",
                    }
                )
            return pending

        local_transfer_contexts = {}
        for context in contexts:
            cfg = context["group"]._nep_runtime_config
            owner = context["owner_ep_rank"]
            source_group_gloo = cfg.get("nep_owner_transfer_groups_gloo", {}).get(owner)
            if source_group_gloo is not None:
                scatter_launch_group_gloo = None
                if not _nep_end_iteration_scatter_enabled():
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

        if _nep_same_communicator_ready_enabled():
            raise RuntimeError("Split NEP host phases do not support same-communicator readiness")
        if any(
            context["group"]._nep_runtime_config.get("edp_ready_gate_enabled", False)
            for context in contexts
        ):
            raise RuntimeError(
                "Split NEP host phases require the device EDP readiness gate to be disabled"
            )
        if _nep_device_ordered_edp_enabled():
            with torch.cuda.stream(dispatch_stream):
                self._start_nep_nccl_owner_edp_reduce_batch(contexts, use_device_readiness=False)
            return {
                "batch_index": batch_index,
                "contexts": contexts,
                "dispatch_stream": dispatch_stream,
                "local_transfer_contexts": local_transfer_contexts,
                "local_edp_contexts": local_edp_contexts,
                "gather_barrier_works": [],
                "gather_done_event": gather_done_event,
                "phase": "edp_launched",
            }
        gather_barrier_works = []
        for owner, (_, source_group_gloo, _) in local_transfer_contexts.items():
            _nep_debug_print(
                f"submit split_dispatch_gather_owner_barrier " f"batch={batch_index} owner={owner}"
            )
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
        self,
        state: dict,
        force_ready: bool,
        async_op: bool,
        compute_ready_event: torch.cuda.Event,
        split_host_phases: bool = False,
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

        pipeline_host_phases = split_host_phases and _nep_pipeline_host_phases_enabled()
        task_batches = []
        max_batch_size = _get_nep_nccl_async_chunk_window()
        for wave in owner_waves:
            current_batch = []
            current_owners = set()
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
                    or (pipeline_host_phases and task["owner_ep_rank"] in current_owners)
                ):
                    task_batches.append(current_batch)
                    current_batch = []
                    current_owners = set()
                    current_slots = set()
                current_batch.append(task)
                current_owners.add(task["owner_ep_rank"])
                current_slots.add(slot)
            if current_batch:
                task_batches.append(current_batch)
        continue_host_phase_batches = pipeline_host_phases or (
            split_host_phases
            and (_nep_a2a_scatter_scheduler_enabled() or _nep_end_iteration_scatter_enabled())
        )
        if split_host_phases and len(task_batches) != 1 and not continue_host_phase_batches:
            _nep_debug_print(
                "split_host_phases_fallback "
                f"task_batches={len(task_batches)} ready_tasks={len(ready_tasks)}"
            )
            split_host_phases = False

        if split_host_phases:
            submission_window = min(_get_nep_parallel_gather_submission_window(), len(task_batches))
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
                if isinstance(pending, list):
                    for pending_phase in pending:
                        pending_phase["submission_slot"] = batch_index
                    pending_host_phases.extend(pending)
                else:
                    pending["submission_slot"] = batch_index
                    pending_host_phases.append(pending)
            if continue_host_phase_batches:
                if _nep_two_level_gather_enabled():
                    pending_host_phases[-1]["remaining_task_batches"] = remaining_task_batches
                else:
                    for pending in pending_host_phases:
                        pending["remaining_task_batches"] = remaining_task_batches
            return pending_host_phases

        dispatch_stream = self._get_nep_nccl_comm_stream(0)
        dispatch_stream.wait_event(compute_ready_event)
        pending_host_phases = []
        for batch_index, task_batch in enumerate(task_batches):
            contexts = []
            same_communicator_ready = _nep_same_communicator_ready_enabled()
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
                    scatter_launch_group_gloo = None
                    if not _nep_end_iteration_scatter_enabled():
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

            if not same_communicator_ready:
                for owner, (_, source_group_gloo, _) in local_transfer_contexts.items():
                    _nep_debug_print(
                        f"before dispatch_gather_owner_barrier batch={batch_index} owner={owner}"
                    )
                    dist.barrier(group=source_group_gloo)
                    _nep_debug_print(
                        f"after dispatch_gather_owner_barrier batch={batch_index} owner={owner}"
                    )

            if not same_communicator_ready:
                for owner, (_, edp_group_gloo) in local_edp_contexts.items():
                    _nep_debug_print(
                        f"before dispatch_edp_owner_launch_barrier "
                        f"batch={batch_index} owner={owner}"
                    )
                    dist.barrier(group=edp_group_gloo)
                    _nep_debug_print(
                        f"after dispatch_edp_owner_launch_barrier "
                        f"batch={batch_index} owner={owner}"
                    )

            with torch.cuda.stream(dispatch_stream):
                if same_communicator_ready:
                    for owner, (context, _) in local_edp_contexts.items():
                        edp_group = context["group"]._nep_runtime_config.get("edp_group")
                        context["group"]._start_nep_nccl_same_communicator_ready(
                            edp_group, ("edp", owner)
                        )
                self._start_nep_nccl_owner_edp_reduce_batch(contexts, use_device_readiness=True)
                edp_done_event = torch.cuda.Event()
                edp_done_event.record(dispatch_stream)

            if not same_communicator_ready:
                for owner, (_, _, scatter_launch_group_gloo) in local_transfer_contexts.items():
                    _nep_debug_print(
                        f"before dispatch_edp_owner_barrier batch={batch_index} owner={owner}"
                    )
                    dist.barrier(group=scatter_launch_group_gloo)
                    _nep_debug_print(
                        f"after dispatch_edp_owner_barrier batch={batch_index} owner={owner}"
                    )

            with torch.cuda.stream(dispatch_stream):
                for owner, (context, _, _) in local_transfer_contexts.items():
                    if same_communicator_ready:
                        transfer_group = (
                            context["group"]
                            ._nep_runtime_config.get("nep_owner_transfer_groups", {})
                            .get(owner)
                        )
                        context["group"]._start_nep_nccl_same_communicator_ready(
                            transfer_group, ("scatter", owner)
                        )
                    else:
                        context["group"]._start_nep_nccl_scatter_ready_gate(
                            owner, context["buffer_slot"], edp_done_event
                        )
                for context in contexts:
                    context["group"]._start_nep_nccl_owner_task_scatter(context)

        return pending_host_phases

    def _finish_nep_nccl_process_group_dispatch_batches(
        self,
        pending_host_phases: List[dict],
        device_align_phases: bool = False,
        finish_all_phases: bool = True,
        defer_scatter_submission: bool = False,
        scatter_context_batches: Optional[List[List[dict]]] = None,
    ) -> bool:
        """Progress ordered EDP and Scatter phases after backward compute is queued."""
        for pending in pending_host_phases:
            batch_index = pending["batch_index"]
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
                        device_align_phases=device_align_phases,
                        finish_all_phases=finish_all_phases,
                        defer_scatter_submission=defer_scatter_submission,
                        scatter_context_batches=scatter_context_batches,
                    )
                continue

            if phase == "gather_launched":
                for owner, work in pending["gather_barrier_works"]:
                    with torch.profiler.record_function("nep_split_wait_gather_launch"):
                        work.wait()
                    _nep_debug_print(
                        f"finish split_dispatch_gather_owner_barrier "
                        f"batch={batch_index} owner={owner}"
                    )
                if device_align_phases:
                    gather_done_event = pending.get("gather_done_event")
                    if gather_done_event is None:
                        raise RuntimeError("Post-graph NEP phases are missing the Gather event")
                    with torch.profiler.record_function("nep_post_graph_wait_gather_device"):
                        gather_done_event.synchronize()

                edp_launch_barrier_works = []
                for owner, (_, edp_group_gloo) in pending["local_edp_contexts"].items():
                    _nep_debug_print(
                        f"submit split_dispatch_edp_owner_launch_barrier "
                        f"batch={batch_index} owner={owner}"
                    )
                    edp_launch_barrier_works.append(
                        (owner, dist.barrier(group=edp_group_gloo, async_op=True))
                    )
                for owner, work in edp_launch_barrier_works:
                    with torch.profiler.record_function("nep_split_wait_edp_launch"):
                        work.wait()
                    _nep_debug_print(
                        f"finish split_dispatch_edp_owner_launch_barrier "
                        f"batch={batch_index} owner={owner}"
                    )

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

                if (
                    not _nep_benchmark_skip_scatter_enabled()
                    and not _nep_end_iteration_scatter_enabled()
                ):
                    scatter_barrier_works = []
                    for owner, (_, _, scatter_launch_group_gloo) in pending[
                        "local_transfer_contexts"
                    ].items():
                        _nep_debug_print(
                            f"submit split_dispatch_edp_owner_barrier "
                            f"batch={batch_index} owner={owner}"
                        )
                        scatter_barrier_works.append(
                            (owner, dist.barrier(group=scatter_launch_group_gloo, async_op=True))
                        )
                    for owner, work in scatter_barrier_works:
                        with torch.profiler.record_function("nep_split_wait_scatter_launch"):
                            work.wait()
                        _nep_debug_print(
                            f"finish split_dispatch_edp_owner_barrier "
                            f"batch={batch_index} owner={owner}"
                        )

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
            if _get_nep_parallel_gather_submission_window() > 1:
                completion_event = torch.cuda.Event()
                completion_event.record(dispatch_stream)
                pending["completion_event"] = completion_event
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

        if (
            compute_ready_event is not None
            and async_op
            and not self._nep_runtime_config.get("zero_sm_reshard", False)
        ):
            return self._start_nep_nccl_process_group_dispatch_batch(
                state,
                force_ready,
                async_op,
                compute_ready_event,
                split_host_phases=_nep_split_host_phases_enabled(),
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
                if state.setdefault("pending_owner_tasks", []):
                    self._progress_nep_nccl_pending_owner_tasks(force_all=force_ready)
                    if state["pending_owner_tasks"]:
                        break
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
                debug_start_event = None
                debug_done_event = None
                with torch.cuda.stream(nccl_stream):
                    if _nep_overlap_debug_enabled():
                        debug_start_event = torch.cuda.Event(enable_timing=True)
                        debug_start_event.record(nccl_stream)
                    group._flush_nep_nccl_pending_scatters(buffer_slot=stream_slot)
                    launch_next_task()
                    if _nep_overlap_debug_enabled():
                        debug_done_event = torch.cuda.Event(enable_timing=True)
                        debug_done_event.record(nccl_stream)
                if debug_start_event is not None and debug_done_event is not None:
                    group._nep_nccl_overlap_debug_events.append(
                        (task_index, task_index + 1, debug_start_event, debug_done_event)
                    )
                if state["pending_owner_tasks"] and not force_ready:
                    break
            if force_ready:
                self._progress_nep_nccl_pending_owner_tasks(force_all=True)
                self._flush_nep_nccl_pending_scatters(force_all=True)
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
        _nep_debug_print(
            "nccl_sync enter "
            f"group={group_index} buckets={len(self.buckets)} ep_rank={ep_rank} "
            f"local_ep_size={local_ep_size} min_ep_size={min_ep_size} "
            f"is_edp_eligible={is_edp_eligible} "
            f"edp_group_present={cfg.get('edp_group') is not None} "
            f"owner_numel={owner_numel} "
            f"max_chunk_numel={layout['max_chunk_numel']} slot_key={self._nep_nccl_slot_key}"
        )
        for owner_ep_rank in range(min_ep_size):
            for chunk_index, (start, end) in enumerate(layout["chunk_ranges"]):
                self._start_nep_nccl_owner_task(
                    owner_ep_rank, chunk_index, start, end, async_op=async_op
                )
        _nep_debug_print(f"nccl_sync exit group={group_index}")

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Start synchronous NCCL nonuniform EP gradient synchronization."""
        group_index = getattr(self, "_nep_nccl_group_index", -1)
        _nep_debug_print(
            "start_grad_sync enter "
            f"group={group_index} started={self._nep_nccl_grad_sync_started} "
            f"ready={self._nep_nccl_ready} "
            f"first_batch={self.is_first_batch} grad_reduce_handle={self.grad_reduce_handle is not None} "
            f"force_all_reduce={force_all_reduce} params={len(self.params)}"
        )
        if self._nep_nccl_ready:
            _nep_debug_print(f"start_grad_sync skip_ready group={group_index}")
            return
        if self.is_first_batch and self.grad_reduce_handle is not None:
            _nep_debug_print(f"start_grad_sync skip_first_batch_handle group={group_index}")
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

        _nep_debug_print(
            f"start_grad_sync before_nccl_task_scheduler group={group_index} async_op={async_op}"
        )
        self._try_start_nep_nccl_ready_tasks(force_ready=True, async_op_override=async_op)
        self.grad_reduce_handle = None
        _nep_debug_print(f"start_grad_sync exit group={group_index}")

    def _finish_nonuniform_ep_nccl_grad_sync(self):
        overlap_debug = _nep_overlap_debug_enabled()
        compute_ready_event = None
        if overlap_debug and self._nep_nccl_streams:
            compute_ready_event = torch.cuda.Event(enable_timing=True)
            compute_ready_event.record(torch.cuda.current_stream())
        drain_start = time.perf_counter() if overlap_debug else None
        self._drain_nep_edp_ready_futures()
        self._drain_nep_nccl_async_window(force_all=True)
        drain_ms = (time.perf_counter() - drain_start) * 1000.0 if overlap_debug else None
        for nccl_stream in self._nep_nccl_streams.values():
            torch.cuda.current_stream().wait_stream(nccl_stream)
        if overlap_debug and self._nep_nccl_overlap_debug_events:
            after_wait_event = torch.cuda.Event(enable_timing=True)
            after_wait_event.record(torch.cuda.current_stream())
            after_wait_event.synchronize()
            group_index = getattr(self, "_nep_nccl_group_index", -1)
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
            for start_idx, end_idx, start_event, done_event in self._nep_nccl_overlap_debug_events:
                comm_ms = start_event.elapsed_time(done_event)
                ready_since_start_ms = (
                    start_event.elapsed_time(compute_ready_event)
                    if compute_ready_event is not None
                    else float("nan")
                )
                finish_wait_ms = (
                    compute_ready_event.elapsed_time(after_wait_event)
                    if compute_ready_event is not None
                    else float("nan")
                )
                print(
                    "[NEP_OVERLAP_DEBUG "
                    f"rank={rank} group={group_index} tasks={start_idx}:{end_idx} "
                    f"comm_ms={comm_ms:.3f} ready_since_comm_start_ms={ready_since_start_ms:.3f} "
                    f"finish_wait_ms={finish_wait_ms:.3f} cpu_drain_ms={drain_ms:.3f}]",
                    flush=True,
                )
            self._nep_nccl_overlap_debug_events = []
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

        group_index = getattr(self, "_nep_nccl_group_index", -1)
        _nep_debug_print(
            "finish_nep_pre_sync enter "
            f"group={group_index} first_batch={self.is_first_batch} "
            f"started={self._nep_nccl_grad_sync_started} "
            f"ready={self._nep_nccl_ready} ready_count={len(self.per_param_grad_ready_counts)} "
            f"params={len(self.params)}"
        )
        if not self.is_first_batch and not self._nep_nccl_ready:
            assert self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts, (
                f"Communication call has not been issued for this bucket "
                f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
                "params have grad available)"
            )
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            _nep_debug_print(f"finish_nep_pre_sync after_force_start group={group_index}")
        if self.is_first_batch:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        if not self._nep_nccl_ready:
            _nep_debug_print(f"finish_nep_pre_sync skip_not_ready group={group_index}")
            return
        self._finish_nonuniform_ep_nccl_grad_sync()
        _nep_debug_print(f"finish_nep_pre_sync exit group={group_index}")

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        group_index = getattr(self, "_nep_nccl_group_index", -1)
        _nep_debug_print(
            "finish_grad_sync enter "
            f"group={group_index} overlap={self.ddp_config.overlap_grad_reduce} "
            f"first_batch={self.is_first_batch} started={self._nep_nccl_grad_sync_started} "
            f"ready={self._nep_nccl_ready} ready_count={len(self.per_param_grad_ready_counts)} "
            f"params={len(self.params)}"
        )
        self.param_gather_dispatched = False
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            self._finish_nonuniform_ep_nccl_grad_sync()
            self._finish_nep_nccl_native_edp_reductions()
            _nep_debug_print(f"finish_grad_sync exit_nonoverlap group={group_index}")
            return

        if not self.is_first_batch and not self._nep_nccl_ready:
            assert self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts, (
                f"Communication call has not been issued for this bucket "
                f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
                "params have grad available)"
            )
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            _nep_debug_print(f"finish_grad_sync after_force_start group={group_index}")
        if self.is_first_batch:
            _nep_debug_print(f"finish_grad_sync first_batch_start group={group_index}")
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        assert self._nep_nccl_ready, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
            "params have grad available)"
        )
        self._finish_nonuniform_ep_nccl_grad_sync()
        self._finish_nep_nccl_native_edp_reductions()
        _nep_debug_print(f"finish_grad_sync exit group={group_index}")

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
                    if bucket_ready and _nep_bucket_ready_gather_enabled():
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
                    _nep_debug_print(
                        "register_grad_ready bucket_ready "
                        f"group={getattr(self, '_nep_nccl_group_index', -1)} "
                        f"params={len(self.params)} force_all_reduce={force_all_reduce}"
                    )

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
            if state.get("pending_owner_tasks"):
                raise RuntimeError("NEP reset found host-gated owner tasks that were not flushed")
            if state.get("pending_edp_contexts"):
                raise RuntimeError("NEP reset found incomplete two-level Gather buckets")
            state["task_next_index"] = 0
            state["pending_scatters"] = []
            state["pending_owner_tasks"] = []
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

    _nep_debug_print(
        "coalesced nccl bucket groups "
        f"local_count={local_count} target_count={target_count} "
        f"merged_count={len(merged_groups)}"
    )
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


def _build_synthetic_owner_bucket_specs(buffers, local_specs, runtime_config, config):
    """Build owner-side buckets for experts physically held by extra EP ranks."""
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
        if _nep_two_level_gather_enabled():
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

    state["task_sequence"] = task_sequence
    state["task_next_index"] = 0
    state["pending_scatters"] = []
    state["pending_owner_tasks"] = []
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
    if not bucket_groups or not _nep_end_iteration_scatter_enabled():
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

        if _nep_two_level_gather_enabled():
            edp_bucket_index = group._nep_nccl_edp_bucket_index
            two_level_scatter_tasks.setdefault((edp_bucket_index, owner_ep_rank), []).append(task)
            continue

        for scatter_chunk_index, (scatter_start, scatter_end) in enumerate(
            group._nep_nccl_scatter_chunk_ranges(
                owner_ep_rank, chunk_start, chunk_end, scatter_chunks
            )
        ):
            if ep_rank == owner_ep_rank:
                scatter_input_numel = sum(
                    group._nep_nccl_owner_source_payload_numel(
                        owner_ep_rank, destination_ep_rank, scatter_start, scatter_end
                    )
                    for destination_ep_rank in remote_source_ranks
                )
                group._get_nep_nccl_cached_tensor(
                    cache,
                    (
                        "owner_layout_a2a_scatter_input",
                        slot,
                        chunk_index,
                        scatter_chunk_index,
                        scatter_input_numel,
                        dtype,
                        device,
                    ),
                    scatter_input_numel,
                    dtype,
                    device,
                )
            elif ep_rank in remote_source_ranks:
                scatter_output_numel = group._nep_nccl_owner_source_payload_numel(
                    owner_ep_rank, ep_rank, scatter_start, scatter_end
                )
                group._get_nep_nccl_cached_tensor(
                    cache,
                    (
                        "owner_layout_a2a_scatter_output",
                        slot,
                        chunk_index,
                        scatter_chunk_index,
                        scatter_output_numel,
                        dtype,
                        device,
                    ),
                    scatter_output_numel,
                    dtype,
                    device,
                )

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
    _nep_debug_print(f"preallocated end-of-iteration Scatter buffers slots={len(slots)}")


def _configure_nep_edp_ready_gate(
    bucket_groups: List[NonuniformEPNCCLParamAndGradBucketGroup],
) -> None:
    """Configure host rendezvous plus stream-ordered readiness signaling."""
    if not bucket_groups:
        return

    state = getattr(bucket_groups[0], "_nep_nccl_scheduler_state", None)
    if state is None:
        raise RuntimeError("NEP EDP readiness gating requires the shared task scheduler")
    if "edp_ready_flags" in state:
        return

    runtime_config = bucket_groups[0]._nep_runtime_config
    ready_group = runtime_config.get("edp_group_gloo")
    transfer_groups = runtime_config.get("nep_owner_scatter_ready_groups_gloo", {})
    has_edp_peer = (
        runtime_config.get("is_edp_eligible", False)
        and ready_group is not None
        and ready_group.size() > 1
    )
    has_transfer_peer = any(group is not None for group in transfer_groups.values())
    edp_ready_enabled = runtime_config.get("edp_ready_gate_enabled", False) and has_edp_peer
    transfer_ready_enabled = has_transfer_peer and (
        runtime_config.get("edp_ready_gate_enabled", False) or _nep_a2a_scatter_scheduler_enabled()
    )
    enabled = edp_ready_enabled or transfer_ready_enabled
    if not enabled:
        state["edp_ready_flags"] = {}
        state["edp_ready_generations"] = {}
        state["edp_ready_executor"] = None
        state["edp_ready_signal_stream"] = None
        return

    device = torch.device("cuda", torch.cuda.current_device())
    buffer_slots = _get_nep_nccl_async_chunk_window()
    flags = {
        buffer_slot: torch.zeros(1, dtype=torch.int32, device=device)
        for buffer_slot in range(buffer_slots)
    }
    torch.cuda.synchronize(device)
    get_cuda_stream_memory_ops()

    state["edp_ready_flags"] = flags
    state["edp_ready_generations"] = {buffer_slot: 0 for buffer_slot in flags}
    state["edp_ready_executor"] = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="nep-edp-ready"
    )
    state["edp_ready_signal_stream"] = torch.cuda.Stream(device=device)
    if edp_ready_enabled:
        dist.barrier(group=ready_group)
    if transfer_ready_enabled:
        for owner_ep_rank in sorted(transfer_groups):
            transfer_group = transfer_groups[owner_ep_rank]
            if transfer_group is not None:
                dist.barrier(group=transfer_group)
    _nep_debug_print("configured stream-ordered readiness gate " f"slots={len(flags)}")


def _configure_nep_zero_sm_buffers(
    bucket_groups: List[NonuniformEPNCCLParamAndGradBucketGroup],
) -> None:
    """Preallocate and register persistent buffers for copy-engine reshard collectives."""
    if not bucket_groups or not bucket_groups[0]._nep_runtime_config.get("zero_sm_reshard", False):
        return

    state = getattr(bucket_groups[0], "_nep_nccl_scheduler_state", None)
    if state is None:
        raise RuntimeError("NEP zero-SM reshard requires the shared task scheduler")
    if state.get("zero_sm_mem_pool") is not None:
        return

    runtime_config = bucket_groups[0]._nep_runtime_config
    device = torch.device("cuda", torch.cuda.current_device())
    group_entries = []
    seen_groups = set()
    for owner_ep_rank in range(runtime_config["min_ep_size"]):
        group = runtime_config.get("nep_owner_transfer_groups", {}).get(owner_ep_rank)
        if group is None or id(group) in seen_groups:
            continue
        backend = group._get_backend(device)
        if not hasattr(backend, "mem_allocator"):
            raise RuntimeError(
                "MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD requires " "ProcessGroupNCCL.mem_allocator"
            )
        if not hasattr(backend, "_comm_ptr"):
            raise RuntimeError(
                "MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD requires " "ProcessGroupNCCL._comm_ptr()"
            )
        seen_groups.add(id(group))
        group_entries.append((group, backend))

    for group, _ in group_entries:
        warmup = torch.ones(1, dtype=torch.float32, device=device)
        dist.all_reduce(warmup, group=group)
    if group_entries:
        torch.cuda.synchronize(device)
    dist.barrier()

    # Reduced replicas have no owner transfers and need no registered staging buffers.
    if not group_entries:
        state["zero_sm_mem_pool"] = False
        state["zero_sm_small_buffers"] = {}
        state["zero_sm_large_buffers"] = {}
        state["zero_sm_comm_ptrs"] = {}
        dist.barrier()
        return

    buffer_specs = {}
    for task in state["task_sequence"]:
        group = task["group"]
        owner_ep_rank = task["owner_ep_rank"]
        source_ranks = group._nep_nccl_owner_source_ranks(owner_ep_rank)
        transfer_ranks = group._nep_nccl_owner_transfer_ranks(owner_ep_rank)
        remote_source_ranks = [rank for rank in source_ranks if rank != owner_ep_rank]
        if not remote_source_ranks:
            continue
        payload_numel = max(
            group._nep_nccl_owner_source_payload_numel(
                owner_ep_rank, source_ep_rank, task["chunk_start"], task["chunk_end"]
            )
            for source_ep_rank in remote_source_ranks
        )
        dtype = group.buckets[0].grad_data.dtype
        spec = buffer_specs.setdefault(dtype, {"small_numel": 0, "large_numel": 0})
        spec["small_numel"] = max(spec["small_numel"], payload_numel)
        spec["large_numel"] = max(spec["large_numel"], len(transfer_ranks) * payload_numel)

    pool = torch.cuda.MemPool(group_entries[0][1].mem_allocator)
    small_buffers = {}
    large_buffers = {}
    buffer_slots = _get_nep_nccl_async_chunk_window()
    with torch.cuda.use_mem_pool(pool):
        for buffer_slot in range(buffer_slots):
            for dtype in sorted(buffer_specs, key=str):
                spec = buffer_specs[dtype]
                small_buffers[(buffer_slot, dtype)] = torch.empty(
                    spec["small_numel"], dtype=dtype, device=device
                )
                large_buffers[(buffer_slot, dtype)] = torch.empty(
                    spec["large_numel"], dtype=dtype, device=device
                )

    comm_ptrs = {}
    for group, backend in group_entries:
        try:
            backend.register_mem_pool(pool, symm=True)
        except TypeError:
            backend.register_mem_pool(pool)
        comm_ptrs[id(group)] = int(backend._comm_ptr())

    state["zero_sm_mem_pool"] = pool
    state["zero_sm_small_buffers"] = small_buffers
    state["zero_sm_large_buffers"] = large_buffers
    state["zero_sm_comm_ptrs"] = comm_ptrs
    state["zero_sm_buffer_specs"] = buffer_specs
    _nep_debug_print(
        "configured zero-SM buffers "
        f"slots={buffer_slots} specs={buffer_specs} groups={len(group_entries)}"
    )
    dist.barrier()


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
    gather_buckets_per_edp = _get_nep_nccl_gather_buckets_per_edp()
    grouped_partitions = []
    for edp_bucket_index, edp_partition in enumerate(edp_partitions):
        gather_partitions = (
            [edp_partition]
            if gather_buckets_per_edp is None
            else _partition_expert_bucket_specs(edp_partition, gather_buckets_per_edp)
        )
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
    _configure_nep_edp_ready_gate(bucket_groups)
    _configure_nep_zero_sm_buffers(bucket_groups)
    return bucket_groups


def build_nonuniform_ep_expert_bucket_groups(
    buffers,
    ddp_config: DistributedDataParallelConfig,
    runtime_config: dict,
    nonuniform_ep_config: NonuniformEPConfig,
    param_to_bucket_group: Dict[torch.nn.Parameter, _ParamAndGradBucketGroup],
    param_to_name: Dict[torch.nn.Parameter, str],
) -> List[NonuniformEPParamAndGradBucketGroup]:
    """Build one opt-in bucket group per local expert."""
    ep_group = runtime_config["ep_group"]
    edp_group = runtime_config.get("edp_group")
    bucket_groups = []
    specs = _build_expert_bucket_specs(buffers, runtime_config, nonuniform_ep_config, param_to_name)
    if buffers and not specs:
        raise RuntimeError(
            "Cannot configure NEP: expert buffers exist, but no params matched "
            "NonuniformEPConfig.expert_name_pattern."
        )

    all_specs = list(specs)
    all_specs.extend(
        _build_synthetic_owner_bucket_specs(buffers, specs, runtime_config, nonuniform_ep_config)
    )
    buffer_order = {buffer: index for index, buffer in enumerate(buffers)}
    slot_keys = sorted({spec.slot_key for spec in all_specs})
    if len(slot_keys) > _NEP_TAG_SLOT_STRIDE:
        raise RuntimeError(
            f"NEP p2p tags support at most {_NEP_TAG_SLOT_STRIDE} expert parameter slots; "
            f"got {len(slot_keys)}"
        )
    max_tag_offset = max((spec.expert_id for spec in all_specs), default=0) * _NEP_TAG_SLOT_STRIDE
    max_tag_offset += max(0, len(slot_keys) - 1)
    if (
        nonuniform_ep_config.grad_transfer_tag_base
        <= nonuniform_ep_config.grad_scatter_tag_base
        <= nonuniform_ep_config.grad_transfer_tag_base + max_tag_offset
    ):
        raise RuntimeError(
            "NEP grad transfer and scatter p2p tag ranges overlap; increase "
            "grad_scatter_tag_base or decrease grad_transfer_tag_base."
        )
    slot_index_by_key = {slot_key: index for index, slot_key in enumerate(slot_keys)}
    all_specs.sort(
        key=lambda spec: (
            buffer_order[spec.buffer],
            slot_index_by_key[spec.slot_key],
            spec.expert_id,
            spec.synthetic_owner,
        )
    )

    grouped_specs = []
    current_key = None
    current_specs = []
    for spec in all_specs:
        owner_ep_rank = _owner_for_expert(
            spec.expert_id, runtime_config, nonuniform_ep_config.expert_owner
        )
        owner_role = "owner" if runtime_config["ep_rank"] == owner_ep_rank else "source"
        key = (spec.buffer, slot_index_by_key[spec.slot_key], owner_role)
        if current_key is not None and key != current_key:
            grouped_specs.append(current_specs)
            current_specs = []
        current_key = key
        current_specs.append(spec)
    if current_specs:
        grouped_specs.append(current_specs)

    for group_index, group_specs in enumerate(grouped_specs):
        buckets = []
        plans = []
        is_owner_rank = False
        for spec in group_specs:
            owner_ep_rank = _owner_for_expert(
                spec.expert_id, runtime_config, nonuniform_ep_config.expert_owner
            )
            source_ep_ranks = _source_ep_ranks_for_expert(spec.expert_id, runtime_config)
            owner_global_rank = get_global_rank(ep_group, owner_ep_rank)
            source_global_ranks = [get_global_rank(ep_group, rank) for rank in source_ep_ranks]
            if edp_group is None and any(rank != owner_ep_rank for rank in source_ep_ranks):
                raise RuntimeError(
                    "NEP p2p ownership transfer requires an owner-only expert-data-parallel "
                    "group in runtime_config['edp_group'] when any local expert transfers to "
                    "a different owner rank."
                )
            plan = _ExpertBucketPlan(
                expert_id=spec.expert_id,
                tag_slot=slot_index_by_key[spec.slot_key],
                owner_ep_rank=owner_ep_rank,
                owner_global_rank=owner_global_rank,
                source_ep_ranks=source_ep_ranks,
                source_global_ranks=source_global_ranks,
                bucket_slices=[(0, spec.end - spec.start)],
                bucket_group_index=group_index,
                synthetic_owner=spec.synthetic_owner,
            )
            plans.append(plan)

            if spec.synthetic_owner:
                param_data = None
                grad_data = torch.empty(
                    spec.end - spec.start,
                    dtype=spec.buffer.grad_data.dtype,
                    device=spec.buffer.grad_data.device,
                )
            else:
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
            is_owner_rank = is_owner_rank or runtime_config["ep_rank"] == owner_ep_rank
            for param in spec.params:
                param.nonuniform_ep_expert_id = spec.expert_id
                param.nonuniform_ep_owner_rank = owner_ep_rank

        buffer = group_specs[0].buffer
        collective_group = (
            edp_group if (is_owner_rank and edp_group is not None) else buffer.data_parallel_group
        )
        bucket_group = NonuniformEPParamAndGradBucketGroup(
            buckets, ddp_config, collective_group, collective_group.size()
        )
        bucket_group.configure_nonuniform_ep(runtime_config, nonuniform_ep_config, plans)
        bucket_groups.append(bucket_group)

    for buffer in buffers:
        for param in buffer.param_index_map:
            param_to_bucket_group.pop(param, None)
    for bucket_group in bucket_groups:
        for bucket in bucket_group.buckets:
            for param in bucket.params_list:
                param_to_bucket_group[param] = bucket_group

    configure_ordered_bucket_group_scheduler(
        bucket_groups, "_nep_scheduler_state", "_nep_group_index", "_nep_ready"
    )
    owner_bucket_groups = [
        bucket_group for bucket_group in bucket_groups if bucket_group._nep_is_owner
    ]
    owner_dp_sync_state = {"groups": owner_bucket_groups, "next_index": 0}
    for index, bucket_group in enumerate(owner_bucket_groups):
        bucket_group._nep_owner_dp_sync_scheduler_state = owner_dp_sync_state
        bucket_group._nep_owner_dp_sync_group_index = index

    post_sync_state = (
        {"entries": [], "last_bucket_group": bucket_groups[-1]} if bucket_groups else None
    )
    for bucket_group in bucket_groups:
        bucket_group._nep_post_sync_state = post_sync_state
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
        _nep_debug_print(
            "synchronized native DDP bucket size "
            f"local={local_bucket_size} full_replica={ddp_config.bucket_size}"
        )

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
        if self.nonuniform_ep_config.approach == NonuniformEPApproach.NCCL:
            self.expert_parallel_bucket_groups = build_nonuniform_ep_nccl_bucket_groups(
                self.expert_parallel_buffers,
                self.ddp_config,
                runtime_config,
                self.nonuniform_ep_config,
                self.param_to_bucket_group,
                self._param_to_name,
            )
        else:
            self.expert_parallel_bucket_groups = build_nonuniform_ep_expert_bucket_groups(
                self.expert_parallel_buffers,
                self.ddp_config,
                runtime_config,
                self.nonuniform_ep_config,
                self.param_to_bucket_group,
                self._param_to_name,
            )
        self._nep_dispatch_boundary_hook_handles = []
        self._nep_dispatch_boundary_pre_hook_handles = []
        self._nep_dispatch_completion_executor = None
        self._nep_dispatch_pending_completion_event = None
        self._nep_dispatch_pending_completion_future = None
        self._nep_dispatch_pending_host_phases = None
        self._nep_dispatch_inflight_completion_events = []
        self._nep_dispatch_deferred_compute_ready_event = None
        self._nep_dispatch_waiting_groups = None
        self._nep_dispatch_waiting_module_label = None
        self._nep_model_ep_a2a_burst_depth = 0
        self._nep_model_ep_a2a_burst_count = 0
        self._nep_scatter_batches = []
        self._nep_end_iteration_scatter_context_batches = []
        self._nep_scatter_inflight_event = None
        self._nep_scatter_next_batch_ordinal = 0
        self._nep_scatter_next_layer_ordinal = 0
        self._nep_scatter_alignment_tensor = None
        self._nep_scatter_alignment_work = None
        self._nep_scatter_stream = None
        self._nep_scatter_backward_complete = False
        if self.nonuniform_ep_config.approach == NonuniformEPApproach.NCCL:
            if _nep_two_level_gather_enabled():
                required_modes = {
                    "MEGATRON_NONUNIFORM_EP_BUCKET_READY_GATHER": (
                        _nep_bucket_ready_gather_enabled()
                    ),
                    "MEGATRON_NONUNIFORM_EP_DEVICE_ORDERED_EDP": (
                        _nep_device_ordered_edp_enabled()
                    ),
                    "MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES": (_nep_split_host_phases_enabled()),
                    "MEGATRON_NONUNIFORM_EP_END_ITERATION_SCATTER": (
                        _nep_end_iteration_scatter_enabled()
                    ),
                }
                missing_modes = [name for name, enabled in required_modes.items() if not enabled]
                if missing_modes:
                    raise RuntimeError("Two-level NEP Gather requires " + ", ".join(missing_modes))
            if _nep_a2a_scatter_scheduler_enabled() and _nep_end_iteration_scatter_enabled():
                raise RuntimeError(
                    "A2A-gated and end-of-iteration NEP Scatter modes are mutually exclusive"
                )
            if _nep_a2a_scatter_scheduler_enabled() or _nep_end_iteration_scatter_enabled():
                if runtime_config.get("zero_sm_reshard", False):
                    raise RuntimeError("Deferred NEP Scatter does not support zero-SM reshard")
                if not _nep_split_host_phases_enabled():
                    raise RuntimeError(
                        "Deferred NEP Scatter requires MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES=1"
                    )
                if not _nep_defer_model_ep_fence_enabled():
                    raise RuntimeError(
                        "Deferred NEP Scatter requires MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE=1"
                    )
                self._nep_scatter_stream = torch.cuda.Stream(device=torch.cuda.current_device())
            if runtime_config.get("zero_sm_reshard", False):
                self._nep_dispatch_completion_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="nep-completion"
                )
            self._configure_nep_dispatch_boundary_hooks()
        self._configure_expert_gradient_scaling(config, runtime_config)

    @staticmethod
    def _find_nep_local_cuda_graph_manager(module_name: str, named_modules: dict):
        """Find the nearest full-layer local graph manager containing an MoE module."""
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
        """Return whether an MoE module is owned by a partial-graph transformer layer."""
        path = module_name.split(".") if module_name else []
        for prefix_length in range(len(path), -1, -1):
            parent_name = ".".join(path[:prefix_length])
            parent = named_modules.get(parent_name)
            if getattr(parent, "use_partial_cudagraphs", False):
                return True
        return False

    @staticmethod
    def _coalesce_nep_cuda_graph_boundary(module_entries: list) -> tuple:
        """Combine all MoE groups covered by one local CUDA graph replay."""
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

    @staticmethod
    def _find_nep_non_moe_cuda_graph_managers(named_modules: dict, moe_type: type) -> tuple:
        """Return unique local graph managers whose module subtree contains no MoE layer."""
        manager_entries = {}
        for module in named_modules.values():
            graph_manager = getattr(module, "cudagraph_manager", None)
            if graph_manager is None:
                continue
            entry = manager_entries.setdefault(
                id(graph_manager), {"manager": graph_manager, "contains_moe": False}
            )
            child_modules = getattr(module, "modules", None)
            if child_modules is not None:
                entry["contains_moe"] |= any(
                    isinstance(child, moe_type) for child in child_modules()
                )
        return tuple(
            entry["manager"] for entry in manager_entries.values() if not entry["contains_moe"]
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

    def _peek_nep_scatter_ticket_window(self) -> tuple:
        """Return the contiguous descriptor window for the next expert bucket group."""
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
        """Consume one aligned ticket window and submit its descriptors in order."""
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
        """Launch a nonblocking agreement for the next bucket-group window."""
        if getattr(self, "_nep_scatter_backward_complete", False):
            raise RuntimeError("NEP Scatter cannot poll for a new window after backward completes")
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
        """Advance ordered Scatter windows at a model-EP boundary."""
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
        """Submit one Scatter chunk on the caller thread with explicit device ordering."""
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
            _nep_debug_print(f"retired scheduled Scatter batch module={batch['module_label']}")
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
                descriptor_ready_gate = train[
                    "group"
                ]._prepare_nep_nccl_scatter_descriptor_ready_gate(
                    descriptor, context["buffer_slot"]
                )
                train["group"]._submit_nep_nccl_owner_all_to_all_scatter(descriptor)
                train["group"]._release_nep_nccl_scatter_descriptor_ready_gate(
                    descriptor_ready_gate
                )
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
                _nep_debug_print(
                    f"submitted scheduled Scatter batch module={batch['module_label']}"
                )

        self._nep_scatter_inflight_event = chunk_done_event
        _nep_debug_print(
            "submitted scheduled Scatter chunk "
            f"group={group_index} owner={context['owner_ep_rank']} "
            f"chunk={context['chunk_index']} scatter={descriptor_index}"
        )
        return True

    def _progress_nep_scatter_after_compute_launch(self) -> None:
        """Retire Scatter work without launching rank-dependent collectives."""
        if (
            not _nep_a2a_scatter_scheduler_enabled()
            or not self._nep_scatter_batches
            or self._nep_model_ep_a2a_burst_depth != 0
        ):
            return

        self._retire_nep_scatter_chunk()

    def _queue_nep_scatter_context_batches(
        self,
        context_batches: List[List[dict]],
        completion_event: torch.cuda.Event,
        module_label: str,
        a2a_completion_event: Optional[torch.cuda.Event],
        tasks_already_marked: bool = False,
    ) -> None:
        """Prepare a layer's Scatter trains while preserving ordered owner batches."""
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
                    if tasks_already_marked:
                        train["task_marked"] = True
                    else:
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
                _nep_debug_print(
                    f"queued scheduled Scatter batch module={batch_label} trains={len(trains)}"
                )

    def _defer_nep_scatter_context_batches_to_iteration_end(
        self,
        context_batches: List[List[dict]],
        completion_event: torch.cuda.Event,
        module_label: str,
    ) -> None:
        """Retain persistent task buffers until the end-of-iteration Scatter drain."""
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
        """Prepare one deferred layer after all backward GPU work completes."""
        if not self._nep_end_iteration_scatter_context_batches:
            return False
        context_batches, completion_event, module_label = (
            self._nep_end_iteration_scatter_context_batches.pop(0)
        )
        contexts = [context for batch in context_batches for context in batch]
        contexts.sort(
            key=lambda context: (
                context["group"]._nep_nccl_group_index,
                context["owner_ep_rank"],
                context["chunk_index"],
            )
        )
        self._queue_nep_scatter_context_batches(
            [contexts],
            completion_event,
            module_label,
            a2a_completion_event=None,
            tasks_already_marked=True,
        )
        return True

    def _drain_nep_scatter_scheduler(self) -> None:
        """Queue all remaining Scatter chunks, then wait once before final gradient sync."""
        a2a_scheduler = _nep_a2a_scatter_scheduler_enabled()
        end_iteration_scatter = _nep_end_iteration_scatter_enabled()
        if not a2a_scheduler and not end_iteration_scatter:
            return
        if not self._nep_scatter_backward_complete:
            raise RuntimeError("NEP Scatter drain requires a completed backward pass")
        if self._nep_model_ep_a2a_burst_depth != 0:
            raise RuntimeError("Cannot drain NEP Scatter during a model-EP A2A burst")
        if end_iteration_scatter:
            with torch.profiler.record_function("nep_end_iteration_scatter_global_fence"):
                torch.cuda.synchronize()
            while self._materialize_next_nep_end_iteration_scatter_batch():
                pass
            while self._nep_scatter_batches:
                if not self._submit_nep_scatter_chunk(queue_behind_inflight=True):
                    raise RuntimeError("End-of-iteration NEP Scatter drain made no progress")
            self._retire_nep_scatter_chunk(force=True)
            return
        if self._nonuniform_ep_runtime_config.get("needs_reshard", False):
            self._consume_model_ep_aligned_nep_scatter_ticket()
        while self._nep_scatter_batches:
            if not self._submit_nep_scatter_chunk(queue_behind_inflight=True):
                raise RuntimeError("A2A-gated NEP Scatter drain made no progress")
        self._retire_nep_scatter_chunk(force=True)

    def model_ep_a2a_burst_begin(self) -> None:
        """Mark a native model-EP A2A burst before its collective is enqueued."""
        if not _nep_a2a_scatter_scheduler_enabled():
            return
        if getattr(self, "_nep_scatter_backward_complete", False):
            if (
                self._nep_scatter_batches
                or self._nep_scatter_inflight_event is not None
                or self._nep_scatter_alignment_work is not None
            ):
                raise RuntimeError("A new backward pass started before NEP Scatter drained")
            self._nep_scatter_backward_complete = False
        if self._nep_model_ep_a2a_burst_depth != 0:
            raise RuntimeError("Nested model-EP A2A bursts are not supported")
        self._nep_model_ep_a2a_burst_depth = 1
        _nep_debug_print("model_ep_a2a_burst_begin")

    def model_ep_a2a_burst_end(self) -> None:
        """Device-order one Scatter chunk after this model-EP A2A burst."""
        if not _nep_a2a_scatter_scheduler_enabled():
            return

        completion_event = torch.cuda.Event()
        completion_event.record(torch.cuda.current_stream())
        if self._nep_model_ep_a2a_burst_depth != 1:
            raise RuntimeError("Model-EP A2A burst ended without a matching begin")
        self._nep_model_ep_a2a_burst_depth = 0
        self._nep_model_ep_a2a_burst_count += 1
        _nep_debug_print(f"model_ep_a2a_burst_end count={self._nep_model_ep_a2a_burst_count}")
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
        """Launch NCCL reshard pipelines after each MoE dispatch backward."""
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

            if _nep_a2a_scatter_scheduler_enabled():
                token_dispatcher = getattr(module, "token_dispatcher", None)
                attach_scheduler = getattr(
                    token_dispatcher, "set_model_ep_a2a_burst_scheduler", None
                )
                if attach_scheduler is None:
                    raise RuntimeError("A2A-gated NEP Scatter requires an A2A token dispatcher")
                attach_scheduler(self)

            module_groups.sort(key=lambda group: group._nep_nccl_group_index)
            dispatch_groups = tuple(module_groups)
            for bucket_group in module_groups:
                assigned_groups.setdefault(bucket_group, set()).add(module_name)
                bucket_group._nep_dispatch_boundary_launch = True
                bucket_group._nep_dispatch_boundary_callback = (
                    self._launch_and_release_nep_two_level_gather
                    if _nep_two_level_gather_enabled()
                    else self._launch_nep_dispatch_boundary_tasks
                )
                bucket_group._nep_dispatch_boundary_groups = (
                    (bucket_group,) if _nep_two_level_gather_enabled() else dispatch_groups
                )
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

        post_graph_phases = _nep_post_graph_phases_enabled()
        post_graph_host_phases = _nep_post_graph_host_phases_enabled()
        if post_graph_phases and post_graph_host_phases:
            raise RuntimeError(
                "Device-aligned and host-only post-graph NEP phases are mutually exclusive"
            )
        non_moe_graph_managers = ()
        if _nep_a2a_scatter_scheduler_enabled() or post_graph_phases or post_graph_host_phases:
            non_moe_graph_managers = self._find_nep_non_moe_cuda_graph_managers(
                named_modules, BaseMoELayer
            )
        if _nep_a2a_scatter_scheduler_enabled():
            for graph_manager in non_moe_graph_managers:
                graph_manager.register_backward_replay_hooks(
                    post_hook=self._progress_nep_scatter_after_compute_launch
                )
        if _nep_pipeline_host_phases_enabled():
            if not post_graph_host_phases:
                raise RuntimeError(
                    "Pipelined NEP host phases require "
                    "MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES=1"
                )
            if _nep_defer_model_ep_fence_enabled():
                raise RuntimeError("Pipelined NEP host phases require the per-layer model-EP fence")
        if post_graph_phases or post_graph_host_phases:
            if not _nep_split_host_phases_enabled():
                raise RuntimeError(
                    "Post-graph NEP phases require MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES=1"
                )
            for graph_manager in non_moe_graph_managers:
                graph_manager.register_backward_replay_hooks(
                    post_hook=self._progress_nep_dispatch_after_graph_launch
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
        """Launch a combined group only after every constituent MoE module is ready."""
        if (
            is_graph_capturing()
            or any(group.is_first_batch for group in groups)
            or not all(group.is_last_microbatch for group in groups)
        ):
            return
        if _nep_two_level_gather_enabled():
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
            _nep_debug_print(
                f"dispatch_backward_boundary_waiting_for_modules module={module_label} "
                f"groups={[group._nep_nccl_group_index for group in incomplete_groups]}"
            )
            return

        for group in groups:
            group._nep_dispatch_boundary_ready = True
            if graph_replay:
                group._nep_dispatch_boundary_graph_replay_ready = True
        self._nep_dispatch_waiting_groups = groups
        self._nep_dispatch_waiting_module_label = module_label
        host_launch_deferred = defer_launch or _nep_defer_dispatch_host_launch_enabled()
        if host_launch_deferred and _nep_post_graph_host_phases_enabled():
            if getattr(self, "_nep_dispatch_deferred_compute_ready_event", None) is not None:
                raise RuntimeError("Prior deferred NEP compute-ready event was not consumed")
            compute_ready_event = torch.cuda.Event()
            compute_ready_event.record(torch.cuda.current_stream())
            self._nep_dispatch_deferred_compute_ready_event = compute_ready_event
        if not host_launch_deferred:
            self._launch_nep_dispatch_boundary_tasks(groups, module_label)

    def _launch_waiting_nep_dispatch_boundary_tasks(self) -> None:
        """Launch the prior ready layer after the next combine backward is enqueued."""
        groups = self._nep_dispatch_waiting_groups
        if (
            groups is None
            or self._nep_dispatch_pending_completion_event is not None
            or getattr(self, "_nep_dispatch_pending_completion_future", None) is not None
            or getattr(self, "_nep_dispatch_pending_host_phases", None) is not None
        ):
            return
        module_label = self._nep_dispatch_waiting_module_label
        if not self._launch_nep_dispatch_boundary_tasks(groups, module_label):
            raise RuntimeError(
                "NEP dispatch boundary reached its launch point before gradients were ready: "
                f"module={module_label}"
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
                _nep_debug_print(
                    f"dispatch_backward_boundary_waiting_for_grads module={module_label} "
                    f"groups={[group._nep_nccl_group_index for group in groups]}"
                )
                for group in groups:
                    group._nep_dispatch_boundary_wait_logged = True
            return False

        if (
            self._nep_dispatch_pending_completion_event is not None
            or getattr(self, "_nep_dispatch_pending_completion_future", None) is not None
            or getattr(self, "_nep_dispatch_pending_host_phases", None) is not None
        ):
            raise RuntimeError("Prior NEP dispatch completion has not been consumed")
        compute_ready_event = getattr(self, "_nep_dispatch_deferred_compute_ready_event", None)
        if compute_ready_event is None:
            compute_ready_event = torch.cuda.Event()
            compute_ready_event.record(torch.cuda.current_stream())
        else:
            self._nep_dispatch_deferred_compute_ready_event = None
        completion_event = torch.cuda.Event()
        device_index = torch.cuda.current_device()
        for group in groups:
            group._nep_dispatch_boundary_launching = True
        self._nep_dispatch_pending_completion_event = completion_event
        if self._nonuniform_ep_runtime_config.get("zero_sm_reshard", False):
            self._nep_dispatch_pending_completion_future = (
                self._submit_nep_dispatch_launch_and_completion(
                    groups, module_label, compute_ready_event, completion_event, device_index
                )
            )
        else:
            self._run_nep_dispatch_boundary_tasks(
                groups, module_label, compute_ready_event, completion_event
            )
        _nep_debug_print(
            f"launched dispatch_backward_boundary module={module_label} "
            f"groups={[group._nep_nccl_group_index for group in groups]}"
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
        group_indices = [group._nep_nccl_group_index for group in groups]
        _nep_debug_print(
            f"before dispatch_backward_boundary module={module_label} groups={group_indices}"
        )
        try:
            launch_start = time.perf_counter()
            state = None
            first_task_index = None
            if _nep_two_level_gather_enabled():
                state = groups[0]._nep_nccl_scheduler_state
                first_task_index = state["task_next_index"]
            pending_host_phases = groups[0]._try_start_nep_nccl_ready_tasks(
                force_ready=False, async_op_override=True, compute_ready_event=compute_ready_event
            )
            last_task_index = state["task_next_index"] if state is not None else None
            launch_ms = (time.perf_counter() - launch_start) * 1000.0

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
            launched_groups = groups
            if _nep_two_level_gather_enabled():
                assert state is not None
                assert first_task_index is not None
                assert last_task_index is not None
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
            _nep_debug_print(
                f"after dispatch_backward_boundary_launched module={module_label} "
                f"groups={[group._nep_nccl_group_index for group in launched_groups]} "
                f"split={bool(pending_host_phases)} "
                f"launch_ms={launch_ms:.3f}"
            )
            return completion_event
        finally:
            for group in groups:
                group._nep_dispatch_boundary_launching = False

    @staticmethod
    def _complete_nep_dispatch_boundary(
        completion_event: torch.cuda.Event, boundary_group
    ) -> tuple:
        """Complete local reshard and rendezvous without blocking the autograd thread."""
        completion_wait_start = time.perf_counter()
        completion_event.synchronize()
        completion_wait_ms = (time.perf_counter() - completion_wait_start) * 1000.0
        completion_barrier_start = time.perf_counter()
        dist.barrier(group=boundary_group)
        completion_barrier_ms = (time.perf_counter() - completion_barrier_start) * 1000.0
        return completion_wait_ms, completion_barrier_ms

    def _launch_and_complete_nep_dispatch_boundary(
        self,
        groups: tuple,
        module_label: str,
        compute_ready_event: torch.cuda.Event,
        completion_event: torch.cuda.Event,
        boundary_group,
        device_index: int,
    ) -> tuple:
        """Launch reshard work off the autograd thread, then complete its rendezvous."""
        torch.cuda.set_device(device_index)
        self._run_nep_dispatch_boundary_tasks(
            groups, module_label, compute_ready_event, completion_event
        )
        return self._complete_nep_dispatch_boundary(completion_event, boundary_group)

    def _submit_nep_dispatch_launch_and_completion(
        self,
        groups: tuple,
        module_label: str,
        compute_ready_event: torch.cuda.Event,
        completion_event: torch.cuda.Event,
        device_index: int,
    ):
        executor = self._nep_dispatch_completion_executor
        if executor is None:
            raise RuntimeError("Zero-SM NEP completion executor was not initialized")
        boundary_group = self._nonuniform_ep_runtime_config.get("dp_cp_group_gloo")
        if boundary_group is None:
            raise RuntimeError("Zero-SM NEP completion requires a DP-CP Gloo group")
        return executor.submit(
            self._launch_and_complete_nep_dispatch_boundary,
            groups,
            module_label,
            compute_ready_event,
            completion_event,
            boundary_group,
            device_index,
        )

    def _finish_pending_nep_dispatch_host_phases(
        self,
        device_align_phases: bool = False,
        finish_all_phases: bool = True,
        defer_scatter_submission: bool = False,
        scatter_after_event: Optional[torch.cuda.Event] = None,
    ) -> bool:
        """Progress a split NEP pipeline without consuming its model-EP completion fence."""
        pending_host_phases = getattr(self, "_nep_dispatch_pending_host_phases", None)
        if pending_host_phases is None:
            return False

        groups = self._nep_dispatch_waiting_groups
        module_label = self._nep_dispatch_waiting_module_label
        completion_event = self._nep_dispatch_pending_completion_event
        completion_future = getattr(self, "_nep_dispatch_pending_completion_future", None)
        if groups is None:
            raise RuntimeError("Split NEP host phases are missing their waiting groups")
        if completion_event is None:
            raise RuntimeError("Split NEP host phases are missing their completion event")
        if completion_future is not None:
            raise RuntimeError("Split ProcessGroup NEP dispatch unexpectedly used a worker")

        end_iteration_scatter = _nep_end_iteration_scatter_enabled()
        scatter_context_batches = (
            []
            if end_iteration_scatter
            or (_nep_a2a_scatter_scheduler_enabled() and not defer_scatter_submission)
            else None
        )
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
            _nep_debug_print(f"progressed split dispatch host phases module={module_label}")
            return False

        if scatter_context_batches is None:
            completion_stream = boundary_group._get_nep_nccl_comm_stream(0)
            if _get_nep_parallel_gather_submission_window() > 1:
                for pending in host_phases:
                    phase_completion_event = pending.get("completion_event")
                    if phase_completion_event is None:
                        raise RuntimeError(
                            "Parallel NEP Gather phase is missing its completion event"
                        )
                    completion_stream.wait_event(phase_completion_event)
            completion_event.record(completion_stream)
        elif end_iteration_scatter and scatter_context_batches:
            self._defer_nep_scatter_context_batches_to_iteration_end(
                scatter_context_batches, completion_event, module_label
            )
        elif end_iteration_scatter:
            completion_event.record(boundary_group._get_nep_nccl_comm_stream(0))
        else:
            self._queue_nep_scatter_context_batches(
                scatter_context_batches, completion_event, module_label, scatter_after_event
            )
        not_ready = [group._nep_nccl_group_index for group in groups if not group._nep_nccl_ready]
        if not_ready and not (_nep_two_level_gather_enabled() and not scatter_context_batches):
            raise RuntimeError(
                "Split NEP host phases did not schedule every layer bucket group: "
                f"module={module_label}, groups={not_ready}"
            )
        self._nep_dispatch_pending_host_phases = None
        _nep_debug_print(
            f"finished split dispatch host phases module={module_label} "
            f"device_aligned={device_align_phases}"
        )
        return True

    def _progress_nep_dispatch_after_graph_launch(self) -> None:
        """Progress split NEP phases after useful non-MoE graph work is queued."""
        if _nep_post_graph_host_phases_enabled():
            if _nep_defer_dispatch_host_launch_enabled():
                self._launch_waiting_nep_dispatch_boundary_tasks()
            self._finish_pending_nep_dispatch_host_phases()
        elif _nep_post_graph_phases_enabled():
            self._finish_pending_nep_dispatch_host_phases(
                device_align_phases=True, finish_all_phases=False
            )

    def _wait_for_nep_dispatch_launch(self, final: bool = False) -> None:
        """Retire one host launch and optionally fence all device-inflight reshards."""
        groups = self._nep_dispatch_waiting_groups
        module_label = self._nep_dispatch_waiting_module_label
        completion_event = self._nep_dispatch_pending_completion_event
        completion_future = getattr(self, "_nep_dispatch_pending_completion_future", None)
        pending_host_phases = getattr(self, "_nep_dispatch_pending_host_phases", None)
        if groups is not None and completion_event is None:
            self._launch_waiting_nep_dispatch_boundary_tasks()
            completion_event = self._nep_dispatch_pending_completion_event
            completion_future = getattr(self, "_nep_dispatch_pending_completion_future", None)
            pending_host_phases = getattr(self, "_nep_dispatch_pending_host_phases", None)

        if pending_host_phases is not None:
            defer_scatter_submission = _nep_a2a_scatter_scheduler_enabled() and not final
            phases_finished = self._finish_pending_nep_dispatch_host_phases(
                device_align_phases=_nep_post_graph_phases_enabled(),
                defer_scatter_submission=defer_scatter_submission,
            )
            if not phases_finished:
                if defer_scatter_submission:
                    _nep_debug_print("staged Scatter until the next model-EP A2A burst end")
                    return
                raise RuntimeError("The next model-EP boundary did not finish split NEP phases")
            pending_host_phases = None

        if completion_event is not None:
            if not self._nonuniform_ep_runtime_config.get("zero_sm_reshard", False):
                if completion_future is not None:
                    raise RuntimeError("ProcessGroup NEP dispatch unexpectedly used a worker")
                if _nep_defer_model_ep_fence_enabled():
                    inflight_events = getattr(
                        self, "_nep_dispatch_inflight_completion_events", None
                    )
                    if inflight_events is None:
                        inflight_events = []
                        self._nep_dispatch_inflight_completion_events = inflight_events
                    inflight_events.append((module_label, completion_event))
                    _nep_debug_print(
                        f"deferred dispatch_backward_boundary completion module={module_label}"
                    )
                else:
                    torch.cuda.current_stream().wait_event(completion_event)
                    _nep_debug_print(
                        f"ordered dispatch_backward_boundary completion module={module_label}"
                    )
            else:
                if completion_future is None:
                    raise RuntimeError("NEP dispatch completion event is missing its worker future")
                completion_join_start = time.perf_counter()
                completion_wait_ms, completion_barrier_ms = completion_future.result()
                completion_join_ms = (time.perf_counter() - completion_join_start) * 1000.0
                _nep_debug_print(
                    f"completed dispatch_backward_boundary module={module_label} "
                    f"completion_wait_ms={completion_wait_ms:.3f} "
                    f"completion_barrier_ms={completion_barrier_ms:.3f} "
                    f"completion_join_ms={completion_join_ms:.3f}"
                )
            self._nep_dispatch_pending_completion_event = None
            self._nep_dispatch_pending_completion_future = None
        elif completion_future is not None:
            raise RuntimeError("NEP dispatch completion future is missing its CUDA event")
        elif pending_host_phases is not None:
            raise RuntimeError("Split NEP host phases are missing their CUDA completion event")
        if groups is not None:
            if not all(group._nep_dispatch_boundary_launched for group in groups):
                raise RuntimeError(f"NEP dispatch launch did not finish: module={module_label}")
            self._nep_dispatch_waiting_groups = None
            self._nep_dispatch_waiting_module_label = None

        if final:
            self._nep_scatter_backward_complete = True
            _nep_debug_print("NEP Scatter scheduler received backward-complete signal")
            self._drain_nep_scatter_scheduler()

            inflight_events = getattr(self, "_nep_dispatch_inflight_completion_events", None)
            if inflight_events:
                current_stream = torch.cuda.current_stream()
                for deferred_module_label, deferred_completion_event in inflight_events:
                    current_stream.wait_event(deferred_completion_event)
                    _nep_debug_print(
                        "ordered deferred dispatch_backward_boundary completion "
                        f"module={deferred_module_label}"
                    )
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

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        _nep_debug_print(
            "NonuniformEPDDP finish_grad_sync enter "
            f"dense_groups={len(self.bucket_groups)} "
            f"expert_groups={len(self.expert_parallel_bucket_groups)} "
            f"overlap={self.ddp_config.overlap_grad_reduce}"
        )
        self._wait_for_nep_dispatch_launch(final=True)
        _nep_debug_print("NonuniformEPDDP finish_grad_sync before_super")
        result = super().finish_grad_sync(force_all_reduce=force_all_reduce)
        _nep_debug_print("NonuniformEPDDP finish_grad_sync after_super")
        return result
