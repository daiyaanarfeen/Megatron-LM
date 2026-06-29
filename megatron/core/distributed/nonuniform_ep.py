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

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
import logging
import os
import re
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


def _nep_debug_print(message: str) -> None:
    """Print NEP debug messages when explicitly enabled."""
    if os.getenv("MEGATRON_NONUNIFORM_EP_DEBUG", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    try:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    except Exception:
        rank = -1
    rank_filter = os.getenv("MEGATRON_NONUNIFORM_EP_DEBUG_RANKS")
    if rank_filter:
        selected_ranks = {
            int(part)
            for part in rank_filter.replace(",", " ").split()
            if part.strip()
        }
        if rank not in selected_ranks:
            return
    print(f"[NEP_DEBUG rank={rank}] {message}", flush=True)


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
        raise RuntimeError(
            "MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES must be positive"
        )
    return max_gather_bytes


def _get_nep_nccl_async_chunk_window() -> int:
    value = os.getenv("MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW")
    if value is None:
        return _NEP_NCCL_DEFAULT_ASYNC_CHUNK_WINDOW
    chunk_window = int(value)
    if chunk_window <= 0:
        raise RuntimeError(
            "MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW must be positive"
        )
    return chunk_window


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

    def __init__(
        self,
        works,
        recv_accumulations=None,
        recv_copies=None,
        keepalive_buffers=None,
    ):
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
    expert_placement: List[List[int]],
    owner_ep_rank: int,
    num_experts: int,
    min_ep_size: int,
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
        'needs_reshard': False,
        'local_ep_size': local_ep_size,
        'min_ep_size': local_ep_size,
        'num_replicas': 1,
        'dp_size': 1,
        'ep_group': ep_group if ep_group is not None else dist.group.WORLD,
        'nep_transfer_group': ep_group if ep_group is not None else dist.group.WORLD,
        'nep_owner_transfer_groups': {},
        'nep_owner_transfer_group_ranks': {},
        'edp_group': None,
        'ep_rank': ep_rank,
        'local_expert_indices': None,
        'expert_placement': None,
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
    _nep_debug_print(f"before_create_group desc={desc} backend={backend} ranks={ranks}")
    group = parallel_state.create_group(
        ranks,
        timeout=timeout,
        backend=backend,
        pg_options=(
            None if backend == "gloo" else parallel_state.get_nccl_options(desc, nccl_comm_cfgs)
        ),
        group_desc=desc,
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
        tp=tp,
        cp=cp,
        num_tp_cp_per_replica=num_tp_cp_per_replica,
        etp=etp,
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

    # Attention/data groups.
    for ranks in generator.get_ranks('dp-cp'):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "dp_cp")
        group_gloo = (
            _create_group(
                ranks,
                timeout,
                nccl_comm_cfgs,
                "DATA_PARALLEL_GROUP_WITH_CP_GLOO",
                "gloo",
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

    for ranks in generator.get_ranks('dp'):
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

    for ranks in generator.get_ranks('cp'):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "cp")
        if rank in ranks:
            _set_parallel_state_attr("_CONTEXT_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_CONTEXT_PARALLEL_GLOBAL_RANKS", ranks)

    for ranks in generator.get_ranks('tp'):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp")
        if rank in ranks:
            _set_parallel_state_attr("_TENSOR_MODEL_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS", ranks)

    for ranks in generator.get_ranks('tp'):
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

    for ranks in generator.get_ranks('tp-dp-cp'):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp_dp_cp")
        if rank in ranks:
            _set_parallel_state_attr("_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP", group)

    for ranks in generator.get_ranks('tp-dp'):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp_dp")
        if rank in ranks:
            _set_parallel_state_attr("_TENSOR_AND_DATA_PARALLEL_GROUP", group)

    for ranks in generator.get_ranks('tp-cp'):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp_cp")
        if rank in ranks:
            _set_parallel_state_attr("_TENSOR_AND_CONTEXT_PARALLEL_GROUP", group)

    # Expert groups.
    min_ep_size = generator.min_k * tp * cp // etp
    nep_transfer_group = None
    nep_owner_transfer_groups = {}
    nep_owner_transfer_group_ranks = {}
    for ranks in generator.get_ranks('ep'):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "ep")
        transfer_group = _create_group(ranks, timeout, nccl_comm_cfgs, "nep_grad_transfer")
        group_expert_placement, _ = compute_nonuniform_ep_expert_placement(
            num_moe_experts,
            len(ranks),
            min_ep_size,
        )
        for owner_ep_rank in range(min_ep_size):
            source_ep_ranks = _source_ep_ranks_for_owner(
                group_expert_placement,
                owner_ep_rank,
                num_moe_experts,
                min_ep_size,
            )
            source_global_ranks = [ranks[source_ep_rank] for source_ep_rank in source_ep_ranks]
            owner_transfer_group = None
            if len(source_global_ranks) > 1:
                owner_transfer_group = _create_group(
                    source_global_ranks,
                    timeout,
                    nccl_comm_cfgs,
                    "nep_owner_transfer",
                )
            if rank in ranks:
                nep_owner_transfer_group_ranks[owner_ep_rank] = source_ep_ranks
                if rank in source_global_ranks:
                    nep_owner_transfer_groups[owner_ep_rank] = owner_transfer_group
        if rank in ranks:
            _set_parallel_state_attr("_EXPERT_MODEL_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_EXPERT_MODEL_PARALLEL_RANKS", ranks)
            nep_transfer_group = transfer_group

    for ranks in generator.get_ranks('etp'):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "ep_tp")
        if rank in ranks:
            _set_parallel_state_attr("_EXPERT_TENSOR_PARALLEL_GROUP", group)

    for ranks in generator.get_ranks('etp-ep'):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp_ep_mp")
        if rank in ranks:
            _set_parallel_state_attr("_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP", group)

    for ranks in generator.get_ranks('etp-ep'):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "tp_ep_pp")
        if rank in ranks:
            _set_parallel_state_attr("_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP", group)

    edp_groups = generator.get_ranks('edp')
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
    )
    runtime_config = {
        'needs_reshard': local_ep_size > min_ep_size,
        'local_ep_size': local_ep_size,
        'min_ep_size': min_ep_size,
        'num_replicas': generator.num_replicas,
        'dp_size': sum(num_tp_cp_per_replica),
        'ep_group': ep_group,
        'nep_transfer_group': nep_transfer_group,
        'nep_owner_transfer_groups': nep_owner_transfer_groups,
        'nep_owner_transfer_group_ranks': nep_owner_transfer_group_ranks,
        'edp_group': parallel_state.get_expert_data_parallel_group(),
        'ep_rank': ep_rank,
        'is_edp_eligible': ep_rank < min_ep_size,
        'is_b_leader': ep_rank < min_ep_size,
        'local_expert_indices': expert_placement[ep_rank],
        'expert_placement': expert_placement,
        'expert_gather_map': expert_gather_map,
    }
    set_nonuniform_ep_runtime_config(runtime_config)
    _nep_debug_print(
        "initialize_nonuniform_ep_process_groups exit "
        f"rank={rank} local_ep_size={local_ep_size} ep_rank={ep_rank} "
        f"local_expert_indices={runtime_config['local_expert_indices']}"
    )
    return runtime_config


def _owner_for_expert(
    expert_id: int,
    runtime_config: dict,
    explicit_owner: Optional[Dict[int, int]],
) -> int:
    if explicit_owner is not None and expert_id in explicit_owner:
        return explicit_owner[expert_id]

    min_ep_size = runtime_config.get('min_ep_size')
    placement = runtime_config.get('expert_placement')
    if min_ep_size is not None and placement is not None:
        num_experts = sum(len(experts) for experts in placement)
        experts_per_owner = num_experts // min_ep_size
        return min(expert_id // experts_per_owner, min_ep_size - 1)

    ep_rank = runtime_config['ep_rank']
    return ep_rank


def _source_ep_ranks_for_expert(expert_id: int, runtime_config: dict) -> List[int]:
    placement = runtime_config.get('expert_placement')
    if placement is None:
        return [runtime_config['ep_rank']]
    return [ep_rank for ep_rank, experts in enumerate(placement) if expert_id in experts]


def _local_expert_id_from_name(
    name: str,
    pattern: re.Pattern,
    local_expert_indices: Optional[List[int]],
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

        ep_rank = runtime_config['ep_rank']
        self._nep_entries = []
        owner_flags = []
        for bucket, entry_plan in zip(self.buckets, self._nep_plans):
            is_owner = ep_rank == entry_plan.owner_ep_rank
            owner_flags.append(is_owner)
            if (
                nonuniform_ep_config.require_owner_local_expert
                and is_owner
                and not entry_plan.synthetic_owner
                and entry_plan.expert_id not in runtime_config.get('_local_expert_id_set', set())
            ):
                raise RuntimeError(
                    "NEP owner mode requires the owner rank to hold optimizer-visible params "
                    f"for expert {entry_plan.expert_id}; owner ep_rank={entry_plan.owner_ep_rank}"
                )
            self._nep_entries.append(
                {
                    'bucket': bucket,
                    'plan': entry_plan,
                    'is_owner': is_owner,
                    'gather_recv_buffers': [],
                    'scatter_send_buffers': [],
                    'gather_send_buffer': None,
                    'scatter_recv_buffer': None,
                }
            )
        if any(owner_flags) and not all(owner_flags):
            raise RuntimeError("NEP grouped bucket contains mixed owner and non-owner plans")
        self._nep_is_owner = any(owner_flags)
        self._allocate_nep_persistent_grad_buffers()

    def _allocate_nep_persistent_grad_buffers(self):
        """Allocate persistent p2p staging buffers for this expert bucket."""
        ep_rank = self._nep_runtime_config['ep_rank']

        if not hasattr(self, '_nep_entries'):
            self._nep_entries = [
                {
                    'bucket': self.buckets[0],
                    'plan': self._nep_plan,
                    'is_owner': self._nep_is_owner,
                    'gather_recv_buffers': [],
                    'scatter_send_buffers': [],
                    'gather_send_buffer': None,
                    'scatter_recv_buffer': None,
                }
            ]

        for entry in self._nep_entries:
            plan = entry['plan']
            bucket = entry['bucket']
            if plan.numel == 0:
                continue

            if entry['is_owner']:
                for source_ep_rank, source_global_rank in zip(
                    plan.source_ep_ranks, plan.source_global_ranks
                ):
                    if source_ep_rank == ep_rank:
                        continue
                    entry['gather_recv_buffers'].append(
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
                entry['scatter_send_buffers'] = entry['gather_recv_buffers']
                self._nep_gather_recv_buffers.extend(entry['gather_recv_buffers'])
                self._nep_scatter_send_buffers = self._nep_gather_recv_buffers
            else:
                entry['gather_send_buffer'] = torch.empty(
                    plan.numel,
                    dtype=bucket.grad_data.dtype,
                    device=bucket.grad_data.device,
                )
                entry['scatter_recv_buffer'] = torch.empty(
                    plan.numel,
                    dtype=bucket.grad_data.dtype,
                    device=bucket.grad_data.device,
                )
                self._nep_gather_send_buffer = entry['gather_send_buffer']
                self._nep_scatter_recv_buffer = entry['scatter_recv_buffer']

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
                if getattr(param, 'main_grad_copy_in_grad_buffer', None) is not None:
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
            'nep_transfer_group', self._nep_runtime_config['ep_group']
        )

        for entry in self._nep_entries:
            plan = entry['plan']
            bucket = entry['bucket']
            if entry['is_owner']:
                for source_ep_rank, _, recv_buffer in entry['gather_recv_buffers']:
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
                send_buffer = entry['gather_send_buffer']
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
            works,
            recv_accumulations=recv_accumulations,
            keepalive_buffers=keepalive_buffers,
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
        self,
        nonblocking: bool,
        force_all_reduce: Optional[bool] = False,
    ) -> None:
        state = getattr(self, '_nep_owner_dp_sync_scheduler_state', None)
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

        groups = state['groups']
        while state['next_index'] < len(groups):
            group = groups[state['next_index']]
            if not group._nep_started:
                break
            if not group._complete_nep_gather_to_owner(nonblocking=nonblocking):
                break
            if not group._nep_owner_dp_sync_started:
                group._start_owner_dp_sync_after_gather(force_all_reduce=force_all_reduce)
                group._nep_owner_dp_sync_started = True
            state['next_index'] += 1

    def _start_nep_scatter_from_owner(self):
        if self._nep_scatter_started:
            return
        works = []
        recv_copies = []
        keepalive_buffers = []
        transfer_group = self._nep_runtime_config.get(
            'nep_transfer_group', self._nep_runtime_config['ep_group']
        )

        for entry in self._nep_entries:
            plan = entry['plan']
            bucket = entry['bucket']
            if entry['is_owner']:
                for source_ep_rank, _, send_buffer in entry['scatter_send_buffers']:
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
                recv_buffer = entry['scatter_recv_buffer']
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
            works,
            recv_copies=recv_copies,
            keepalive_buffers=keepalive_buffers,
        )
        self._nep_scatter_started = True

    def _record_nep_scatter_wait(self, copy_back_after_wait: bool = False):
        handle = self._nep_scatter_handle
        self._nep_scatter_handle = None
        state = getattr(self, '_nep_post_sync_state', None)
        if state is None:
            if handle is not None:
                handle.wait()
            if copy_back_after_wait:
                self._copy_back_extra_main_grads()
            return

        state['entries'].append((self, handle, copy_back_after_wait))
        if self is state['last_bucket_group']:
            try:
                for group, pending_handle, pending_copy_back in state['entries']:
                    if pending_handle is not None:
                        pending_handle.wait()
                    if pending_copy_back:
                        group._copy_back_extra_main_grads()
            finally:
                state['entries'] = []

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
        self._try_start_ready_owner_dp_syncs(
            nonblocking=True,
            force_all_reduce=force_all_reduce,
        )

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
        self._try_start_ready_owner_dp_syncs(
            nonblocking=False,
            force_all_reduce=force_all_reduce,
        )
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
        self._try_start_ready_owner_dp_syncs(
            nonblocking=False,
            force_all_reduce=force_all_reduce,
        )

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
                        '_nep_scheduler_state',
                        '_nep_ready',
                        'start_grad_sync',
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
        reset_ordered_bucket_group_scheduler(
            self, '_nep_scheduler_state', '_nep_group_index'
        )
        state = getattr(self, '_nep_owner_dp_sync_scheduler_state', None)
        if state is not None and getattr(self, '_nep_owner_dp_sync_group_index', -1) == 0:
            state['next_index'] = 0


class NonuniformEPNCCLParamAndGradBucketGroup(_ParamAndGradBucketGroup):
    """Approach A: NCCL owner-layout reshard/allreduce/reshard for nonuniform EP."""

    def configure_nonuniform_ep_nccl(
        self,
        runtime_config: dict,
        nonuniform_ep_config: NonuniformEPConfig,
        entries: Optional[List[dict]] = None,
        slot_key: Optional[Tuple[str, ...]] = None,
        slot_numel: Optional[int] = None,
    ) -> None:
        self._nep_runtime_config = runtime_config
        self._nep_config = nonuniform_ep_config
        self._nep_nccl_entries = entries or []
        self._nep_nccl_slot_key = slot_key
        self._nep_nccl_slot_numel = slot_numel
        self._nep_nccl_grad_sync_started = False
        self._nep_nccl_ready = len(self.params) == 0
        self._nep_nccl_bucket_numels_cache = {}
        self._nep_nccl_async_handles = []
        self._nep_nccl_async_tensors = []
        self._nep_nccl_stream = None
        self._nep_nccl_logical_grad_data_cache = {}
        self._nep_nccl_send_chunk_cache = {}
        self._nep_nccl_gather_buf_cache = {}
        self._nep_nccl_gather_list_cache = {}
        self._nep_nccl_buffer_state = {}
        self._nep_nccl_owner_layout = None
        self._nep_nccl_started_tasks = set()
        self._nep_nccl_prepped_experts = set()
        self._nep_nccl_task_count = 0

    def _get_nep_nccl_shared_buffer_state(self) -> dict:
        state = getattr(self, '_nep_nccl_scheduler_state', None)
        if state is None:
            state = self._nep_nccl_buffer_state
        state.setdefault('gather_buf_cache', {})
        state.setdefault('buffer_slot_handles', {})
        return state

    def _wait_nep_nccl_buffer_slot(self, slot_key: tuple) -> None:
        state = self._get_nep_nccl_shared_buffer_state()
        slot_handles = state['buffer_slot_handles']
        handles = slot_handles.pop(slot_key, [])
        for work in handles:
            work.wait()

    def _record_nep_nccl_work(self, work, buffer_slot_key: Optional[tuple] = None) -> None:
        if work is None:
            return
        self._nep_nccl_async_handles.append(work)
        if buffer_slot_key is not None:
            state = self._get_nep_nccl_shared_buffer_state()
            state['buffer_slot_handles'].setdefault(buffer_slot_key, []).append(work)
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
        self,
        cache: dict,
        key: tuple,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
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

    def _get_nep_nccl_comm_stream(self) -> torch.cuda.Stream:
        state = getattr(self, '_nep_nccl_scheduler_state', None)
        if state is not None:
            stream = state.get('comm_stream')
            if stream is None:
                stream = torch.cuda.Stream(device=torch.cuda.current_device())
                state['comm_stream'] = stream
            self._nep_nccl_stream = stream
            return stream

        if self._nep_nccl_stream is None:
            self._nep_nccl_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        return self._nep_nccl_stream

    def _get_nep_nccl_owner_layout(self) -> dict:
        """Return cached owner-layout metadata for this expert slot bucket group."""
        if self._nep_nccl_owner_layout is not None:
            return self._nep_nccl_owner_layout

        cfg = self._nep_runtime_config
        local_ep_size = cfg['local_ep_size']
        ep_rank = cfg['ep_rank']
        min_ep_size = cfg.get('min_ep_size', local_ep_size)
        if min_ep_size < 2:
            raise RuntimeError(
                "NEP NCCL Approach A requires at least two owner EP ranks "
                f"(min_ep_size >= 2); got min_ep_size={min_ep_size}."
            )

        placement = cfg.get('expert_placement')
        if placement is None:
            num_experts = local_ep_size
        else:
            num_experts = sum(len(experts) for experts in placement)
        if num_experts % min_ep_size != 0:
            raise RuntimeError(
                f"NEP NCCL owner layout requires num_experts ({num_experts}) to be "
                f"divisible by min_ep_size ({min_ep_size})"
            )
        if self._nep_nccl_slot_numel is None:
            raise RuntimeError("NEP NCCL bucket group is missing slot-size metadata")

        experts_per_owner = num_experts // min_ep_size
        owner_numel = experts_per_owner * self._nep_nccl_slot_numel
        if owner_numel == 0:
            num_chunks = 0
            max_chunk_numel = 0
        else:
            max_gather_bytes = _get_nep_nccl_max_gather_bytes()
            max_chunk_numel = max(1, max_gather_bytes // self.buckets[0].grad_data.element_size())
            num_chunks = (owner_numel + max_chunk_numel - 1) // max_chunk_numel

        self._nep_nccl_experts_per_owner = experts_per_owner
        self._nep_nccl_owner_layout = {
            'ep_rank': ep_rank,
            'local_ep_size': local_ep_size,
            'min_ep_size': min_ep_size,
            'num_experts': num_experts,
            'experts_per_owner': experts_per_owner,
            'owner_numel': owner_numel,
            'max_chunk_numel': max_chunk_numel,
            'num_chunks': num_chunks,
        }
        return self._nep_nccl_owner_layout

    def _get_nep_nccl_transfer_group_info(self, owner_ep_rank: int) -> tuple:
        """Return the owner-source communicator used for NEP reshard all-to-alls."""
        cfg = self._nep_runtime_config
        source_ranks = self._nep_nccl_owner_source_ranks(owner_ep_rank)
        ep_rank = cfg['ep_rank']
        if ep_rank not in source_ranks:
            return None, -1, len(source_ranks), source_ranks
        if len(source_ranks) <= 1:
            return None, 0, len(source_ranks), source_ranks

        transfer_group = cfg.get('nep_owner_transfer_groups', {}).get(owner_ep_rank)
        if transfer_group is None:
            raise RuntimeError(
                "Missing NEP owner transfer group for owner "
                f"{owner_ep_rank} with source EP ranks {source_ranks}"
            )
        transfer_rank = dist.get_rank(group=transfer_group)
        transfer_size = dist.get_world_size(group=transfer_group)
        if transfer_size != len(source_ranks):
            raise RuntimeError(
                "NEP owner transfer group size must match owner source count; got "
                f"transfer_size={transfer_size}, source_ranks={source_ranks}"
            )
        return transfer_group, transfer_rank, transfer_size, source_ranks

    def _nep_nccl_owner_entries(self, owner_ep_rank: int) -> List[dict]:
        """Return local expert-slot entries that contribute to an owner-layout chunk."""
        layout = self._get_nep_nccl_owner_layout()
        experts_per_owner = layout['experts_per_owner']
        owner_first_expert = owner_ep_rank * experts_per_owner
        owner_last_expert = owner_first_expert + experts_per_owner
        return [
            entry
            for entry in self._nep_nccl_entries
            if owner_first_expert <= entry['expert_id'] < owner_last_expert
        ]

    def _nep_nccl_owner_task_ready(self, owner_ep_rank: int) -> bool:
        """Return True when this rank's local inputs for an owner task are ready."""
        if self.is_first_batch:
            return False
        for entry in self._nep_nccl_owner_entries(owner_ep_rank):
            for param in entry['bucket'].params_list:
                ready_count = self.per_param_grad_ready_counts.get(param, 0)
                expected_count = self.golden_per_param_grad_ready_counts.get(param)
                if expected_count is None or ready_count < expected_count:
                    return False
        return True

    def _prep_nep_nccl_owner_entries_for_sync(self, owner_ep_rank: int) -> None:
        """Apply one-time local grad prep for entries used by one owner task."""
        for entry in self._nep_nccl_owner_entries(owner_ep_rank):
            expert_id = entry['expert_id']
            if expert_id in self._nep_nccl_prepped_experts:
                continue
            bucket = entry['bucket']
            for param in bucket.params_with_extra_main_grads:
                if getattr(param, 'main_grad_copy_in_grad_buffer', None) is not None:
                    param.main_grad_copy_in_grad_buffer.copy_(param.main_grad)
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor
            self._nep_nccl_prepped_experts.add(expert_id)

    def _pack_nep_nccl_owner_chunk(
        self,
        owner_ep_rank: int,
        chunk_start: int,
        chunk_end: int,
        chunk: torch.Tensor,
    ) -> None:
        """Pack local source grads into a common owner-rank layout chunk."""
        slot_numel = self._nep_nccl_slot_numel
        experts_per_owner = self._nep_nccl_experts_per_owner
        owner_first_expert = owner_ep_rank * experts_per_owner

        chunk.zero_()
        for entry in self._nep_nccl_entries:
            expert_id = entry['expert_id']
            owner_local_expert_index = expert_id - owner_first_expert
            if owner_local_expert_index < 0 or owner_local_expert_index >= experts_per_owner:
                continue

            expert_start = owner_local_expert_index * slot_numel
            expert_end = expert_start + entry['numel']
            overlap_start = max(chunk_start, expert_start)
            overlap_end = min(chunk_end, expert_end)
            if overlap_start >= overlap_end:
                continue

            chunk_offset = overlap_start - chunk_start
            entry_offset = overlap_start - expert_start
            numel = overlap_end - overlap_start
            chunk[chunk_offset : chunk_offset + numel].copy_(
                entry['bucket'].grad_data[entry_offset : entry_offset + numel]
            )

    def _copy_nep_nccl_owner_chunk_to_local_grads(
        self,
        owner_ep_rank: int,
        chunk_start: int,
        chunk_end: int,
        chunk: torch.Tensor,
    ) -> None:
        """Copy a reduced common owner-rank layout chunk back to local source grads."""
        slot_numel = self._nep_nccl_slot_numel
        experts_per_owner = self._nep_nccl_experts_per_owner
        owner_first_expert = owner_ep_rank * experts_per_owner

        for entry in self._nep_nccl_entries:
            expert_id = entry['expert_id']
            owner_local_expert_index = expert_id - owner_first_expert
            if owner_local_expert_index < 0 or owner_local_expert_index >= experts_per_owner:
                continue

            expert_start = owner_local_expert_index * slot_numel
            expert_end = expert_start + entry['numel']
            overlap_start = max(chunk_start, expert_start)
            overlap_end = min(chunk_end, expert_end)
            if overlap_start >= overlap_end:
                continue

            chunk_offset = overlap_start - chunk_start
            entry_offset = overlap_start - expert_start
            numel = overlap_end - overlap_start
            entry['bucket'].grad_data[entry_offset : entry_offset + numel].copy_(
                chunk[chunk_offset : chunk_offset + numel]
            )

    def _nep_nccl_owner_source_ranks(self, owner_ep_rank: int) -> List[int]:
        """Return EP ranks that physically hold experts for an owner-layout chunk."""
        source_ranks_by_owner = self._nep_runtime_config.get('nep_owner_transfer_group_ranks')
        if source_ranks_by_owner is not None and owner_ep_rank in source_ranks_by_owner:
            return list(source_ranks_by_owner[owner_ep_rank])

        placement = self._nep_runtime_config.get('expert_placement')
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
        ep_rank = cfg['ep_rank']
        group_index = getattr(self, '_nep_nccl_group_index', -1)
        chunk_size = chunk_end - chunk_start
        source_ranks = self._nep_nccl_owner_source_ranks(owner_ep_rank)
        if owner_ep_rank not in source_ranks:
            raise RuntimeError(
                f"NEP owner {owner_ep_rank} is not in its source ranks {source_ranks}"
            )
        if ep_rank not in source_ranks:
            return

        remote_source_ranks = [rank for rank in source_ranks if rank != owner_ep_rank]
        transfer_group, _, transfer_size, transfer_source_ranks = (
            self._get_nep_nccl_transfer_group_info(owner_ep_rank)
        )
        owner_transfer_rank = transfer_source_ranks.index(owner_ep_rank)
        remote_transfer_ranks = [
            transfer_source_ranks.index(rank) for rank in remote_source_ranks
        ]

        self._pack_nep_nccl_owner_chunk(owner_ep_rank, chunk_start, chunk_end, chunk)

        if not remote_source_ranks:
            return

        cache = self._get_nep_nccl_shared_buffer_state()['gather_buf_cache']
        empty = self._get_nep_nccl_cached_tensor(
            cache,
            ("empty", chunk.dtype, chunk.device),
            0,
            chunk.dtype,
            chunk.device,
        )

        input_split_sizes = [0] * transfer_size
        if ep_rank in remote_source_ranks:
            input_split_sizes[owner_transfer_rank] = chunk_size
            gather_input = chunk
        else:
            gather_input = empty

        output_split_sizes = [0] * transfer_size
        if ep_rank == owner_ep_rank:
            for source_transfer_rank in remote_transfer_ranks:
                output_split_sizes[source_transfer_rank] = chunk_size
            gather_output_numel = len(remote_source_ranks) * chunk_size
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
        if async_op and gather_output.numel() > 0:
            self._nep_nccl_async_tensors.append(gather_output)

        if ep_rank == owner_ep_rank:
            if len(remote_source_ranks) == 1:
                chunk.add_(gather_output)
            else:
                chunk.add_(
                    gather_output.view(len(remote_source_ranks), chunk_size).sum(dim=0)
                )
        _nep_debug_print(
            "after ep_all_to_all_owner_gather "
            f"group={group_index} owner={owner_ep_rank} chunk={chunk_index}"
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
    ) -> None:
        """Reshard one reduced owner-layout chunk back to physical source ranks."""
        cfg = self._nep_runtime_config
        ep_rank = cfg['ep_rank']
        group_index = getattr(self, '_nep_nccl_group_index', -1)
        chunk_size = chunk_end - chunk_start
        source_ranks = self._nep_nccl_owner_source_ranks(owner_ep_rank)
        if owner_ep_rank not in source_ranks:
            raise RuntimeError(
                f"NEP owner {owner_ep_rank} is not in its source ranks {source_ranks}"
            )
        if ep_rank not in source_ranks:
            return

        remote_source_ranks = [rank for rank in source_ranks if rank != owner_ep_rank]
        transfer_group, _, transfer_size, transfer_source_ranks = (
            self._get_nep_nccl_transfer_group_info(owner_ep_rank)
        )
        owner_transfer_rank = transfer_source_ranks.index(owner_ep_rank)
        remote_transfer_ranks = [
            transfer_source_ranks.index(rank) for rank in remote_source_ranks
        ]

        if not remote_source_ranks:
            if ep_rank == owner_ep_rank:
                self._copy_nep_nccl_owner_chunk_to_local_grads(
                    owner_ep_rank, chunk_start, chunk_end, chunk
                )
            return

        cache = self._get_nep_nccl_shared_buffer_state()['gather_buf_cache']
        empty = self._get_nep_nccl_cached_tensor(
            cache,
            ("empty", chunk.dtype, chunk.device),
            0,
            chunk.dtype,
            chunk.device,
        )

        input_split_sizes = [0] * transfer_size
        if ep_rank == owner_ep_rank:
            for destination_transfer_rank in remote_transfer_ranks:
                input_split_sizes[destination_transfer_rank] = chunk_size
            if len(remote_source_ranks) == 1:
                scatter_input = chunk
            else:
                scatter_input = self._get_nep_nccl_cached_tensor(
                    cache,
                    (
                        "owner_layout_a2a_scatter_input",
                        buffer_slot_key[0],
                        len(remote_source_ranks) * chunk_size,
                        chunk.dtype,
                        chunk.device,
                    ),
                    len(remote_source_ranks) * chunk_size,
                    chunk.dtype,
                    chunk.device,
                )
                scatter_input_view = scatter_input.view(len(remote_source_ranks), chunk_size)
                scatter_input_view.copy_(
                    chunk.unsqueeze(0).expand(len(remote_source_ranks), chunk_size)
                )
        else:
            scatter_input = empty

        output_split_sizes = [0] * transfer_size
        if ep_rank in remote_source_ranks:
            output_split_sizes[owner_transfer_rank] = chunk_size
            scatter_output = self._get_nep_nccl_cached_tensor(
                cache,
                (
                    "owner_layout_a2a_scatter_output",
                    buffer_slot_key[0],
                    chunk_size,
                    chunk.dtype,
                    chunk.device,
                ),
                chunk_size,
                chunk.dtype,
                chunk.device,
            )
        else:
            scatter_output = empty

        _nep_debug_print(
            "before ep_all_to_all_owner_scatter "
            f"group={group_index} owner={owner_ep_rank} chunk={chunk_index} "
            f"chunk_size={chunk_size} ep_rank={ep_rank} destinations={source_ranks} "
            f"transfer_sources={transfer_source_ranks}"
        )
        work = dist.all_to_all_single(
            scatter_output,
            scatter_input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=transfer_group,
            async_op=async_op,
        )
        self._record_nep_nccl_work(work, buffer_slot_key)
        if async_op:
            if scatter_input is not empty and scatter_input is not chunk:
                self._nep_nccl_async_tensors.append(scatter_input)
            if scatter_output.numel() > 0:
                self._nep_nccl_async_tensors.append(scatter_output)

        if ep_rank == owner_ep_rank:
            self._copy_nep_nccl_owner_chunk_to_local_grads(
                owner_ep_rank, chunk_start, chunk_end, chunk
            )
        elif ep_rank in remote_source_ranks:
            self._copy_nep_nccl_owner_chunk_to_local_grads(
                owner_ep_rank, chunk_start, chunk_end, scatter_output
            )
        _nep_debug_print(
            "after ep_all_to_all_owner_scatter "
            f"group={group_index} owner={owner_ep_rank} chunk={chunk_index}"
        )

    def _mark_nep_nccl_task_started(self, owner_ep_rank: int, chunk_index: int) -> None:
        self._nep_nccl_grad_sync_started = True
        self._nep_nccl_started_tasks.add((owner_ep_rank, chunk_index))
        if len(self._nep_nccl_started_tasks) == self._nep_nccl_task_count:
            self._nep_nccl_ready = True

    def _start_nep_nccl_owner_task(
        self,
        owner_ep_rank: int,
        chunk_index: int,
        chunk_start: int,
        chunk_end: int,
        async_op: bool,
    ) -> None:
        """Launch one ordered owner-layout gather/allreduce/scatter task."""
        layout = self._get_nep_nccl_owner_layout()
        chunk_size = chunk_end - chunk_start
        if chunk_size <= 0:
            self._mark_nep_nccl_task_started(owner_ep_rank, chunk_index)
            return

        buffer_slots = _get_nep_nccl_async_chunk_window()
        buffer_slot = (
            owner_ep_rank * max(1, layout['num_chunks']) + chunk_index
        ) % buffer_slots
        chunk_dtype = self.buckets[0].grad_data.dtype
        chunk_device = self.buckets[0].grad_data.device
        buffer_slot_key = (
            buffer_slot,
            chunk_size,
            chunk_dtype,
            chunk_device,
        )
        self._wait_nep_nccl_buffer_slot(buffer_slot_key)

        gather_buf_cache = self._get_nep_nccl_shared_buffer_state()['gather_buf_cache']
        chunk = self._get_nep_nccl_cached_tensor(
            gather_buf_cache,
            (
                "owner_layout_gather",
                buffer_slot,
                chunk_size,
                chunk_dtype,
                chunk_device,
            ),
            chunk_size,
            chunk_dtype,
            chunk_device,
        )
        if async_op:
            self._nep_nccl_async_tensors.append(chunk)

        self._prep_nep_nccl_owner_entries_for_sync(owner_ep_rank)
        self._start_nep_nccl_owner_all_to_all_gather(
            owner_ep_rank,
            chunk_index,
            chunk_start,
            chunk_end,
            chunk,
            buffer_slot_key,
            async_op=async_op,
        )

        cfg = self._nep_runtime_config
        ep_rank = cfg['ep_rank']
        edp_group = cfg.get('edp_group')
        reduce_op = dist.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = dist.ReduceOp.AVG

        group_index = getattr(self, '_nep_nccl_group_index', -1)
        if ep_rank == owner_ep_rank:
            if edp_group is None:
                raise RuntimeError(
                    "Nonuniform EP NCCL owner rank requires runtime_config['edp_group']."
                )
            _nep_debug_print(
                "before edp_all_reduce_owner_chunk "
                f"group={group_index} owner={owner_ep_rank} chunk={chunk_index} "
                f"chunk_size={chunk_size} edp_rank={edp_group.rank()}"
            )
            work = dist.all_reduce(
                chunk,
                op=reduce_op,
                group=edp_group,
                async_op=async_op,
            )
            self._record_nep_nccl_work(work, buffer_slot_key)
            _nep_debug_print(
                "after edp_all_reduce_owner_chunk "
                f"group={group_index} owner={owner_ep_rank} chunk={chunk_index}"
            )

        self._start_nep_nccl_owner_all_to_all_scatter(
            owner_ep_rank,
            chunk_index,
            chunk_start,
            chunk_end,
            chunk,
            buffer_slot_key,
            async_op=async_op,
        )
        self._mark_nep_nccl_task_started(owner_ep_rank, chunk_index)

    def _try_start_nep_nccl_ready_tasks(
        self,
        force_ready: bool = False,
        async_op_override: Optional[bool] = None,
    ) -> None:
        """Launch globally ordered owner tasks while local dependencies are ready."""
        state = getattr(self, '_nep_nccl_scheduler_state', None)
        if state is None or 'task_sequence' not in state:
            self._start_nonuniform_ep_nccl_grad_sync(
                async_op=(
                    self.ddp_config.overlap_grad_reduce
                    and not self.is_first_batch
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

        def launch_ready_tasks_on_current_stream() -> None:
            tasks = state['task_sequence']
            while state['task_next_index'] < len(tasks):
                task = tasks[state['task_next_index']]
                group = task['group']
                owner_ep_rank = task['owner_ep_rank']
                if not force_ready and not group._nep_nccl_owner_task_ready(owner_ep_rank):
                    break
                group._start_nep_nccl_owner_task(
                    owner_ep_rank,
                    task['chunk_index'],
                    task['chunk_start'],
                    task['chunk_end'],
                    async_op=async_op,
                )
                state['task_next_index'] += 1

        if async_op:
            nccl_stream = self._get_nep_nccl_comm_stream()
            nccl_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(nccl_stream):
                launch_ready_tasks_on_current_stream()
        else:
            launch_ready_tasks_on_current_stream()

    def _start_nonuniform_ep_nccl_grad_sync(self, async_op: bool = False):
        cfg = self._nep_runtime_config
        layout = self._get_nep_nccl_owner_layout()
        local_ep_size = layout['local_ep_size']
        ep_rank = layout['ep_rank']
        min_ep_size = layout['min_ep_size']
        is_edp_eligible = cfg.get('is_edp_eligible', ep_rank < min_ep_size)
        group_index = getattr(self, '_nep_nccl_group_index', -1)
        owner_numel = layout['owner_numel']
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
            for chunk_index, start in enumerate(
                range(0, owner_numel, layout['max_chunk_numel'])
            ):
                end = min(start + layout['max_chunk_numel'], owner_numel)
                self._start_nep_nccl_owner_task(
                    owner_ep_rank,
                    chunk_index,
                    start,
                    end,
                    async_op=async_op,
                )
        _nep_debug_print(f"nccl_sync exit group={group_index}")

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Start synchronous NCCL nonuniform EP gradient synchronization."""
        group_index = getattr(self, '_nep_nccl_group_index', -1)
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
        self._try_start_nep_nccl_ready_tasks(
            force_ready=True,
            async_op_override=async_op,
        )
        self.grad_reduce_handle = None
        _nep_debug_print(f"start_grad_sync exit group={group_index}")

    def _finish_nonuniform_ep_nccl_grad_sync(self):
        self._drain_nep_nccl_async_window(force_all=True)
        if self._nep_nccl_stream is not None:
            torch.cuda.current_stream().wait_stream(self._nep_nccl_stream)
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

        group_index = getattr(self, '_nep_nccl_group_index', -1)
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
        group_index = getattr(self, '_nep_nccl_group_index', -1)
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
            if not self.is_first_batch:
                self._try_start_nep_nccl_ready_tasks(
                    force_ready=False,
                    async_op_override=True,
                )
                if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
                    assert len(self.per_param_grad_ready_counts) == len(self.params)
                    _nep_debug_print(
                        "register_grad_ready bucket_ready "
                        f"group={getattr(self, '_nep_nccl_group_index', -1)} "
                        f"params={len(self.params)} force_all_reduce={force_all_reduce}"
                    )

    def reset(self):
        super().reset()
        self._nep_nccl_grad_sync_started = False
        self._nep_nccl_ready = self._nep_nccl_task_count == 0
        self.grad_reduce_handle = None
        self._nep_nccl_async_handles = []
        self._nep_nccl_async_tensors = []
        self._nep_nccl_started_tasks = set()
        self._nep_nccl_prepped_experts = set()
        reset_ordered_bucket_group_scheduler(
            self, '_nep_nccl_scheduler_state', '_nep_nccl_group_index'
        )
        state = getattr(self, '_nep_nccl_scheduler_state', None)
        if state is not None and getattr(self, '_nep_nccl_group_index', -1) == 0:
            state['task_next_index'] = 0


def _coalesce_nep_nccl_bucket_groups_for_edp_order(
    bucket_groups: List[NonuniformEPNCCLParamAndGradBucketGroup],
    runtime_config: dict,
    nonuniform_ep_config: NonuniformEPConfig,
) -> List[NonuniformEPNCCLParamAndGradBucketGroup]:
    """Merge local bucket groups so every rank in an EDP group has the same count."""
    if not bucket_groups:
        return bucket_groups

    edp_group = runtime_config.get('edp_group')
    if edp_group is None:
        return bucket_groups

    local_count = torch.tensor(
        [len(bucket_groups)],
        dtype=torch.int32,
        device=torch.cuda.current_device(),
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
    source_buckets = getattr(buffer, 'buckets', None)
    if not isinstance(source_buckets, (list, tuple)):
        yield 0, sorted(buffer.param_index_map, key=lambda param: buffer.param_index_map[param][0])
        return

    for bucket_index, bucket in enumerate(source_buckets):
        params = getattr(bucket, 'params_list', None)
        if params is None:
            params = list(getattr(bucket, 'params', []))
        yield bucket_index, sorted(params, key=lambda param: buffer.param_index_map[param][0])


def _build_expert_bucket_specs(buffers, runtime_config, config, param_to_name):
    local_expert_indices = runtime_config.get('local_expert_indices')
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
                    name,
                    config.expert_name_pattern,
                    local_expert_indices,
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

    runtime_config['_local_expert_id_set'] = local_expert_id_set
    return specs


def _build_expert_param_bucket_specs(buffers, runtime_config, config, param_to_name):
    """Build one NCCL bucket spec per logical expert parameter slot."""
    local_expert_indices = runtime_config.get('local_expert_indices')
    local_expert_id_set = set(local_expert_indices) if local_expert_indices is not None else set()
    specs = []

    for buffer in buffers:
        for source_bucket_index, bucket_params in _iter_buffer_bucket_params(buffer):
            for param in bucket_params:
                if param not in buffer.param_index_map:
                    continue
                name = param_to_name.get(param, "")
                expert_id = _local_expert_id_from_name(
                    name,
                    config.expert_name_pattern,
                    local_expert_indices,
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
                        slot_key=(
                            _expert_slot_key_from_name(
                                name,
                                config.expert_name_pattern,
                            ),
                        ),
                    )
                )

    runtime_config['_local_expert_id_set'] = local_expert_id_set
    return specs


def _build_synthetic_owner_bucket_specs(buffers, local_specs, runtime_config, config):
    """Build owner-side buckets for experts physically held by extra EP ranks."""
    placement = runtime_config.get('expert_placement')
    if placement is None:
        return []

    local_ep_rank = runtime_config['ep_rank']
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
        owner_ep_rank = _owner_for_expert(
            expert_id,
            runtime_config,
            config.expert_owner,
        )
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
    param_to_bucket_group: Optional[
        Dict[torch.nn.Parameter, _ParamAndGradBucketGroup]
    ] = None,
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

        wrapped_bucket_group.configure_nonuniform_ep_nccl(
            runtime_config,
            nonuniform_ep_config,
        )
        old_to_new[bucket_group] = wrapped_bucket_group
        wrapped_bucket_groups.append(wrapped_bucket_group)

    wrapped_bucket_groups = _coalesce_nep_nccl_bucket_groups_for_edp_order(
        wrapped_bucket_groups,
        runtime_config,
        nonuniform_ep_config,
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
        '_nep_nccl_scheduler_state',
        '_nep_nccl_group_index',
        '_nep_nccl_ready',
    )
    return wrapped_bucket_groups


def _configure_nep_nccl_task_scheduler(
    bucket_groups: List[NonuniformEPNCCLParamAndGradBucketGroup],
) -> None:
    """Attach a deterministic owner/chunk task order shared by NCCL bucket groups."""
    if not bucket_groups:
        return

    state = getattr(bucket_groups[0], '_nep_nccl_scheduler_state', None)
    if state is None:
        state = {'groups': bucket_groups, 'next_index': 0}
        for index, bucket_group in enumerate(bucket_groups):
            bucket_group._nep_nccl_scheduler_state = state
            bucket_group._nep_nccl_group_index = index

    task_sequence = []
    for bucket_group in bucket_groups:
        layout = bucket_group._get_nep_nccl_owner_layout()
        bucket_group._nep_nccl_task_count = layout['min_ep_size'] * layout['num_chunks']
        bucket_group._nep_nccl_ready = bucket_group._nep_nccl_task_count == 0
        for owner_ep_rank in range(layout['min_ep_size']):
            for chunk_index, chunk_start in enumerate(
                range(0, layout['owner_numel'], layout['max_chunk_numel'])
            ):
                chunk_end = min(chunk_start + layout['max_chunk_numel'], layout['owner_numel'])
                task_sequence.append(
                    {
                        'group': bucket_group,
                        'owner_ep_rank': owner_ep_rank,
                        'chunk_index': chunk_index,
                        'chunk_start': chunk_start,
                        'chunk_end': chunk_end,
                    }
                )

    state['task_sequence'] = task_sequence
    state['task_next_index'] = 0


def build_nonuniform_ep_nccl_bucket_groups(
    buffers,
    ddp_config: DistributedDataParallelConfig,
    runtime_config: dict,
    nonuniform_ep_config: NonuniformEPConfig,
    param_to_bucket_group: Dict[torch.nn.Parameter, _ParamAndGradBucketGroup],
    param_to_name: Dict[torch.nn.Parameter, str],
) -> List[NonuniformEPNCCLParamAndGradBucketGroup]:
    """Build common-layout NCCL Approach-A bucket groups by expert parameter slot."""
    ep_group = runtime_config['ep_group']
    specs = _build_expert_param_bucket_specs(
        buffers,
        runtime_config,
        nonuniform_ep_config,
        param_to_name,
    )
    if buffers and not specs:
        raise RuntimeError(
            "Cannot configure NEP NCCL: expert buffers exist, but no params matched "
            "NonuniformEPConfig.expert_name_pattern."
        )

    grouped_specs: Dict[Tuple[str, ...], List[_ExpertBucketSpec]] = {}
    for spec in specs:
        grouped_specs.setdefault(spec.slot_key, []).append(spec)

    bucket_groups = []
    ordered_grouped_specs = sorted(grouped_specs.items(), key=lambda item: item[0], reverse=True)
    for group_index, (slot_key, unordered_group_specs) in enumerate(ordered_grouped_specs):
        group_specs = sorted(unordered_group_specs, key=lambda spec: spec.expert_id)
        slot_numels = {spec.end - spec.start for spec in group_specs}
        if len(slot_numels) != 1:
            raise RuntimeError(
                "NEP NCCL requires equal parameter-slot sizes across experts for "
                f"slot {slot_key}; got {sorted(slot_numels)}"
            )
        seen_experts = set()
        buckets = []
        entries = []
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
                    'expert_id': spec.expert_id,
                    'bucket': bucket,
                    'numel': spec.end - spec.start,
                }
            )

        bucket_group = NonuniformEPNCCLParamAndGradBucketGroup(
            buckets,
            ddp_config,
            ep_group,
            dist.get_world_size(group=ep_group),
        )
        bucket_group.configure_nonuniform_ep_nccl(
            runtime_config,
            nonuniform_ep_config,
            entries=entries,
            slot_key=slot_key,
            slot_numel=next(iter(slot_numels)),
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
        bucket_groups,
        '_nep_nccl_scheduler_state',
        '_nep_nccl_group_index',
        '_nep_nccl_ready',
    )
    _configure_nep_nccl_task_scheduler(bucket_groups)
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
    ep_group = runtime_config['ep_group']
    edp_group = runtime_config.get('edp_group')
    bucket_groups = []
    specs = _build_expert_bucket_specs(
        buffers,
        runtime_config,
        nonuniform_ep_config,
        param_to_name,
    )
    if buffers and not specs:
        raise RuntimeError(
            "Cannot configure NEP: expert buffers exist, but no params matched "
            "NonuniformEPConfig.expert_name_pattern."
        )

    all_specs = list(specs)
    all_specs.extend(
        _build_synthetic_owner_bucket_specs(
            buffers,
            specs,
            runtime_config,
            nonuniform_ep_config,
        )
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
        owner_role = 'owner' if runtime_config['ep_rank'] == owner_ep_rank else 'source'
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
                spec.expert_id,
                runtime_config,
                nonuniform_ep_config.expert_owner,
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
            is_owner_rank = is_owner_rank or runtime_config['ep_rank'] == owner_ep_rank
            for param in spec.params:
                param.nonuniform_ep_expert_id = spec.expert_id
                param.nonuniform_ep_owner_rank = owner_ep_rank

        buffer = group_specs[0].buffer
        collective_group = (
            edp_group if (is_owner_rank and edp_group is not None) else buffer.data_parallel_group
        )
        bucket_group = NonuniformEPParamAndGradBucketGroup(
            buckets,
            ddp_config,
            collective_group,
            collective_group.size(),
        )
        bucket_group.configure_nonuniform_ep(
            runtime_config,
            nonuniform_ep_config,
            plans,
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
        bucket_groups,
        '_nep_scheduler_state',
        '_nep_group_index',
        '_nep_ready',
    )
    owner_bucket_groups = [
        bucket_group for bucket_group in bucket_groups if bucket_group._nep_is_owner
    ]
    owner_dp_sync_state = {'groups': owner_bucket_groups, 'next_index': 0}
    for index, bucket_group in enumerate(owner_bucket_groups):
        bucket_group._nep_owner_dp_sync_scheduler_state = owner_dp_sync_state
        bucket_group._nep_owner_dp_sync_group_index = index

    post_sync_state = (
        {'entries': [], 'last_bucket_group': bucket_groups[-1]} if bucket_groups else None
    )
    for bucket_group in bucket_groups:
        bucket_group._nep_post_sync_state = post_sync_state
    return bucket_groups


class NonuniformEPDistributedDataParallel(DistributedDataParallel):
    """DDP wrapper that opts expert params into nonuniform EP ownership transfer."""

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

        parent_kwargs = {
            'config': config,
            'ddp_config': ddp_config,
            'module': module,
            'disable_bucketing': disable_bucketing,
            'pg_collection': pg_collection,
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
        self._configure_expert_gradient_scaling(config, runtime_config)

    def _configure_expert_gradient_scaling(self, config: TransformerConfig, runtime_config: dict):
        if config.calculate_per_token_loss:
            expert_gradient_scaling_factor = 1.0
        elif self.ddp_config.average_in_collective:
            expert_gradient_scaling_factor = runtime_config.get('num_replicas', 1) / max(
                1, runtime_config.get('dp_size', self.dp_cp_group.size())
            )
        else:
            expert_gradient_scaling_factor = 1.0 / self.dp_cp_group.size()

        for bucket_group in self.expert_parallel_bucket_groups:
            for bucket in bucket_group.buckets:
                bucket.gradient_scaling_factor = expert_gradient_scaling_factor

    def _delay_dense_grad_sync_for_nep(self) -> bool:
        return (
            self.ddp_config.overlap_grad_reduce
            and self.nonuniform_ep_config.approach == NonuniformEPApproach.NCCL
        )

    def _register_delayed_dense_grad_ready(
        self,
        bucket_group: _ParamAndGradBucketGroup,
        param: torch.nn.Parameter,
    ) -> None:
        """Track dense bucket readiness without launching dense DP collectives."""
        assert bucket_group.ddp_config.overlap_grad_reduce
        if bucket_group.is_last_microbatch:
            assert param in bucket_group.param_to_bucket, "Param is not in the bucket group"
            if param not in bucket_group.per_param_grad_ready_counts:
                bucket_group.per_param_grad_ready_counts[param] = 0
            bucket_group.per_param_grad_ready_counts[param] += 1

    def _start_delayed_dense_grad_syncs(
        self, force_all_reduce: Optional[bool] = False
    ) -> None:
        if not self._delay_dense_grad_sync_for_nep():
            return

        for bucket_group in self.bucket_groups:
            if bucket_group.is_first_batch:
                continue
            if bucket_group.grad_reduce_handle is not None:
                continue
            assert (
                bucket_group.per_param_grad_ready_counts
                == bucket_group.golden_per_param_grad_ready_counts
            ), (
                f"Communication call has not been issued for this dense bucket "
                f"({len(bucket_group.per_param_grad_ready_counts)}/{len(bucket_group.params)} "
                "params have grad available)"
            )
            bucket_group.start_grad_sync(force_all_reduce=force_all_reduce)

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        if not self._delay_dense_grad_sync_for_nep():
            return super()._make_backward_post_hook(param)

        def hook(*unused):
            if is_graph_capturing():
                return

            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.ddp_config.overlap_grad_reduce:
                    bucket_group = self.param_to_bucket_group[param]
                    if bucket_group in self.bucket_groups:
                        self._register_delayed_dense_grad_ready(bucket_group, param)
                    else:
                        bucket_group.register_grad_ready(param, self.force_all_reduce)

        return hook

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        _nep_debug_print(
            "NonuniformEPDDP finish_grad_sync enter "
            f"dense_groups={len(self.bucket_groups)} expert_groups={len(self.expert_parallel_bucket_groups)} "
            f"overlap={self.ddp_config.overlap_grad_reduce}"
        )
        if self.ddp_config.overlap_grad_reduce:
            for bucket_group in self.expert_parallel_bucket_groups:
                bucket_group.finish_nep_pre_sync(force_all_reduce=force_all_reduce)
            self._start_delayed_dense_grad_syncs(force_all_reduce=force_all_reduce)
        _nep_debug_print("NonuniformEPDDP finish_grad_sync before_super")
        result = super().finish_grad_sync(force_all_reduce=force_all_reduce)
        _nep_debug_print("NonuniformEPDDP finish_grad_sync after_super")
        return result
