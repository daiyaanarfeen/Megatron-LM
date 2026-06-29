# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Shared helpers for opt-in nonuniform distributed wrappers.

The helpers in this file intentionally avoid modifying the generic DDP, optimizer,
or param-buffer implementations.  NTP and NEP wrappers import this module to share
DDP subclass construction, bucket-group wrapping, handle tracking, and local buffer
layout utilities.
"""

import inspect
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from . import distributed_data_parallel as ddp_module


_NONUNIFORM_EP_RUNTIME_CONFIG: Optional[dict] = None


def set_nonuniform_ep_runtime_config(runtime_config: Optional[dict]) -> None:
    """Register opt-in NEP runtime metadata for forward token routing."""
    global _NONUNIFORM_EP_RUNTIME_CONFIG
    _NONUNIFORM_EP_RUNTIME_CONFIG = dict(runtime_config) if runtime_config is not None else None


def get_nonuniform_ep_runtime_config() -> Optional[dict]:
    """Return the opt-in NEP runtime metadata registered by the entrypoint."""
    return _NONUNIFORM_EP_RUNTIME_CONFIG


def clear_nonuniform_ep_runtime_config() -> None:
    """Clear opt-in NEP runtime metadata."""
    set_nonuniform_ep_runtime_config(None)


def get_nonuniform_ep_local_expert_indices() -> Optional[List[int]]:
    """Return this rank's placement-aware local expert IDs, if NEP is active."""
    runtime_config = get_nonuniform_ep_runtime_config()
    if runtime_config is None:
        return None
    local_expert_indices = runtime_config.get('local_expert_indices')
    if local_expert_indices is None:
        return None
    return [int(expert_id) for expert_id in local_expert_indices]


def build_expert_to_ep_rank_map(
    expert_placement: Optional[Sequence[Sequence[int]]],
    num_experts: Optional[int] = None,
) -> Optional[List[int]]:
    """Build ``expert_id -> ep_rank`` from a placement table.

    Token dispatch requires each global expert to have exactly one physical holder
    inside the local EP group.  Replicated experts need a separate routing policy and
    are intentionally rejected here.
    """
    if expert_placement is None:
        return None

    normalized_placement = [
        [int(expert_id) for expert_id in expert_ids] for expert_ids in expert_placement
    ]
    for ep_rank, expert_ids in enumerate(normalized_placement):
        if expert_ids != sorted(expert_ids):
            raise RuntimeError(
                "NEP expert placement entries must list local experts in ascending global "
                f"expert order for token dispatch; ep_rank {ep_rank} has {expert_ids}"
            )
    if num_experts is None:
        max_expert_id = max(
            (expert_id for expert_ids in normalized_placement for expert_id in expert_ids),
            default=-1,
        )
        num_experts = max_expert_id + 1

    expert_to_ep_rank = [-1] * num_experts
    for ep_rank, expert_ids in enumerate(normalized_placement):
        for expert_id in expert_ids:
            if expert_id < 0 or expert_id >= num_experts:
                raise RuntimeError(
                    f"NEP expert placement contains expert {expert_id}, but num_experts="
                    f"{num_experts}"
                )
            if expert_to_ep_rank[expert_id] != -1:
                raise RuntimeError(
                    "NEP token routing expects one physical holder per expert within an "
                    f"EP group; expert {expert_id} appears on both ep_rank "
                    f"{expert_to_ep_rank[expert_id]} and {ep_rank}"
                )
            expert_to_ep_rank[expert_id] = ep_rank

    missing_experts = [
        expert_id for expert_id, ep_rank in enumerate(expert_to_ep_rank) if ep_rank == -1
    ]
    if missing_experts:
        raise RuntimeError(
            "NEP expert placement must cover every global expert for token routing; "
            f"missing experts {missing_experts[:8]}"
        )
    return expert_to_ep_rank


def get_nonuniform_ep_expert_to_ep_rank_map(num_experts: int) -> Optional[List[int]]:
    """Return placement-aware token destination ranks, if NEP is active."""
    runtime_config = get_nonuniform_ep_runtime_config()
    if runtime_config is None:
        return None
    return build_expert_to_ep_rank_map(runtime_config.get('expert_placement'), num_experts)


def local_expert_indices_are_contiguous(local_expert_indices: Sequence[int]) -> bool:
    """Return whether local expert IDs form one contiguous ascending range."""
    return list(local_expert_indices) == list(
        range(local_expert_indices[0], local_expert_indices[0] + len(local_expert_indices))
    )


def compute_nonuniform_ep_expert_placement(
    num_experts: int, local_ep_size: int, min_ep_size: int
) -> Tuple[List[List[int]], Dict[int, List[Tuple[int, int, int]]]]:
    """Compute placement for replicas with ``local_ep_size >= min_ep_size``.

    Min-EP replicas use contiguous owner ranges. Wider replicas keep the first
    slice of each owner range on owner ranks and place the remaining experts
    round-robin on follower ranks. The returned gather map is shared by token
    routing tests and NEP gradient ownership transfer setup.
    """
    if num_experts % local_ep_size != 0 or num_experts % min_ep_size != 0:
        raise RuntimeError(
            "NEP expert placement requires num_experts to be divisible by both "
            f"local_ep_size ({local_ep_size}) and min_ep_size ({min_ep_size}); "
            f"got num_experts={num_experts}"
        )
    if local_ep_size < min_ep_size:
        raise RuntimeError(
            f"local_ep_size ({local_ep_size}) must be >= min_ep_size ({min_ep_size})"
        )

    experts_per_rank = num_experts // local_ep_size
    experts_per_owner = num_experts // min_ep_size
    experts_offloaded = experts_per_owner - experts_per_rank
    num_followers = local_ep_size - min_ep_size

    placement = [[] for _ in range(local_ep_size)]
    for owner_rank in range(min_ep_size):
        start = owner_rank * experts_per_owner
        placement[owner_rank] = list(range(start, start + experts_per_rank))

    if num_followers > 0 and experts_offloaded > 0:
        follower_index = 0
        for owner_rank in range(min_ep_size):
            offload_start = owner_rank * experts_per_owner + experts_per_rank
            for offset in range(experts_offloaded):
                expert_id = offload_start + offset
                follower_rank = min_ep_size + (follower_index % num_followers)
                placement[follower_rank].append(expert_id)
                follower_index += 1

    for expert_ids in placement:
        expert_ids.sort()

    gather_map = {}
    for follower_rank in range(min_ep_size, local_ep_size):
        gather_map[follower_rank] = []
        for local_index, expert_id in enumerate(placement[follower_rank]):
            owner_rank = expert_id // experts_per_owner
            owner_slot = expert_id - owner_rank * experts_per_owner
            gather_map[follower_rank].append((local_index, owner_rank, owner_slot))

    return placement, gather_map


class NonuniformEPRankGenerator:
    """Generate process groups for opt-in NEP replicas with different EP sizes."""

    _ATTENTION_KEYS = {'tp', 'cp', 'dp', 'dp-cp', 'tp-dp', 'tp-dp-cp', 'tp-cp'}
    _EXPERT_KEYS = {'etp', 'ep', 'etp-ep', 'edp'}

    def __init__(
        self,
        tp: int,
        cp: int,
        num_tp_cp_per_replica: Sequence[int],
        etp: Optional[int] = None,
        rank_offset: int = 0,
    ) -> None:
        self.tp = tp
        self.cp = cp
        self.tp_cp = tp * cp
        self.etp = etp if etp is not None else tp
        self.rank_offset = rank_offset
        self.num_tp_cp_per_replica = [int(value) for value in num_tp_cp_per_replica]
        self.num_replicas = len(self.num_tp_cp_per_replica)

        if self.etp > self.tp or self.tp % self.etp != 0:
            raise RuntimeError(f"expert TP ({self.etp}) must divide attention TP ({self.tp})")
        if any(value < 1 for value in self.num_tp_cp_per_replica):
            raise RuntimeError(
                "Every NEP replica must contain at least one TP*CP unit; got "
                f"{self.num_tp_cp_per_replica}"
            )

        self.world_size = sum(self.num_tp_cp_per_replica) * self.tp_cp
        self.dp = sum(self.num_tp_cp_per_replica)
        self.min_k = min(self.num_tp_cp_per_replica)

        self.replica_offsets = [0]
        for num_tp_cp in self.num_tp_cp_per_replica:
            self.replica_offsets.append(self.replica_offsets[-1] + num_tp_cp * self.tp_cp)

    def get_ranks(self, key: str) -> List[List[int]]:
        """Return rank groups for attention or expert dimensions."""
        if key in self._ATTENTION_KEYS:
            return self._get_attention_ranks(key)
        if key in self._EXPERT_KEYS:
            return self._get_expert_ranks(key)
        raise ValueError(
            f"Unknown nonuniform EP rank key {key}; expected "
            f"{sorted(self._ATTENTION_KEYS | self._EXPERT_KEYS)}"
        )

    def _get_attention_ranks(self, key: str) -> List[List[int]]:
        from .. import parallel_state

        name_to_size = {"tp": self.tp, "cp": self.cp, "dp": self.dp}
        ordered_tokens = "tp-cp-dp".split("-")
        ordered_size = [name_to_size[token] for token in ordered_tokens]
        token_list = key.split("-")
        mask = [token in token_list for token in ordered_tokens]
        ranks = parallel_state.generate_masked_orthogonal_rank_groups(
            self.world_size, ordered_size, mask
        )
        if self.rank_offset > 0:
            for rank_group in ranks:
                for index in range(len(rank_group)):
                    rank_group[index] += self.rank_offset
        return ranks

    def _get_expert_ranks(self, key: str) -> List[List[int]]:
        if key == 'etp-ep':
            return self._get_etp_ep_ranks()
        if key == 'etp':
            return self._get_etp_ranks()
        if key == 'ep':
            return self._get_ep_ranks()
        if key == 'edp':
            return self._get_edp_ranks()
        raise ValueError(f"Unknown expert key {key}")

    def _get_etp_ep_ranks(self) -> List[List[int]]:
        groups = []
        for replica_index, num_tp_cp in enumerate(self.num_tp_cp_per_replica):
            start = self.rank_offset + self.replica_offsets[replica_index]
            groups.append(list(range(start, start + num_tp_cp * self.tp_cp)))
        return groups

    def _get_etp_ranks(self) -> List[List[int]]:
        groups = []
        for replica_index, num_tp_cp in enumerate(self.num_tp_cp_per_replica):
            start = self.rank_offset + self.replica_offsets[replica_index]
            ep_size = num_tp_cp * self.tp_cp // self.etp
            for ep_index in range(ep_size):
                groups.append(
                    list(range(start + ep_index * self.etp, start + (ep_index + 1) * self.etp))
                )
        return groups

    def _get_ep_ranks(self) -> List[List[int]]:
        groups = []
        for replica_index, num_tp_cp in enumerate(self.num_tp_cp_per_replica):
            start = self.rank_offset + self.replica_offsets[replica_index]
            block = list(range(start, start + num_tp_cp * self.tp_cp))
            for etp_position in range(self.etp):
                groups.append(block[etp_position::self.etp])
        return groups

    def _get_edp_ranks(self) -> List[List[int]]:
        groups = []
        for position in range(self.min_k * self.tp_cp):
            groups.append(
                [
                    self.rank_offset + self.replica_offsets[replica_index] + position
                    for replica_index in range(self.num_replicas)
                ]
            )
        return groups


@dataclass
class NonuniformTPReplicaRanks:
    """Contiguous physical rank block assigned to one topology-aware NTP replica."""

    replica_index: int
    active_tp_size: int
    physical_tp_size: int
    ranks_by_cp: List[List[int]]
    inactive_ranks_by_cp: List[List[int]] = field(default_factory=list)

    @property
    def ranks(self) -> List[int]:
        """Return ranks in TP-fastest, then CP order."""
        return [rank for cp_ranks in self.ranks_by_cp for rank in cp_ranks]

    def inactive_local_tp_ranks(self, tp_base: int) -> List[int]:
        """Return local TP positions not present in this replica's active domain."""
        return list(range(self.active_tp_size, min(tp_base, self.physical_tp_size)))

    @property
    def inactive_ranks(self) -> List[int]:
        """Return physical ranks in this replica that do not join active NTP groups."""
        return [rank for cp_ranks in self.inactive_ranks_by_cp for rank in cp_ranks]


class NonuniformTPTopologyRankGenerator:
    """Generate process groups for NTP replicas placed in explicit TP domains.

    ``tp_domain_sizes`` lists the active TP size of each replica. ``physical_tp_size``
    can reserve a larger physical block for each replica, allowing NTP to keep
    Megatron's world-size divisibility while assigning reduced active TP groups to
    the front of each contiguous topology domain. TP remains the fastest-changing
    dimension inside each CP slice.
    """

    _SUPPORTED_KEYS = {'tp', 'cp', 'dp', 'dp-cp', 'tp-dp', 'tp-dp-cp', 'tp-cp'}

    def __init__(
        self,
        tp: int,
        cp: int,
        tp_domain_sizes: Sequence[int],
        rank_offset: int = 0,
        physical_tp_size: Optional[int] = None,
    ) -> None:
        self.tp = int(tp)
        self.cp = int(cp)
        self.tp_domain_sizes = [int(size) for size in tp_domain_sizes]
        self.rank_offset = int(rank_offset)
        self.physical_tp_size = (
            int(physical_tp_size) if physical_tp_size is not None else None
        )

        if self.tp < 1 or self.cp < 1:
            raise RuntimeError(f"TP and CP sizes must be positive; got TP={tp}, CP={cp}")
        if not self.tp_domain_sizes:
            raise RuntimeError("NTP topology requires at least one TP domain size")
        invalid_sizes = [
            size for size in self.tp_domain_sizes if size < 1 or size > self.tp
        ]
        if invalid_sizes:
            raise RuntimeError(
                "NTP TP domain sizes must be in [1, tp_base]; got "
                f"{invalid_sizes} for tp_base={self.tp}"
            )
        if self.physical_tp_size is not None and self.physical_tp_size < max(
            self.tp_domain_sizes
        ):
            raise RuntimeError(
                "NTP physical TP block size must be >= every active TP domain size; "
                f"got physical_tp_size={self.physical_tp_size}, "
                f"active sizes={self.tp_domain_sizes}"
            )

        self.num_replicas = len(self.tp_domain_sizes)
        self._physical_tp_sizes = [
            self.physical_tp_size if self.physical_tp_size is not None else size
            for size in self.tp_domain_sizes
        ]
        self.world_size = sum(self._physical_tp_sizes) * self.cp
        self.replica_offsets = [0]
        for physical_tp_size in self._physical_tp_sizes:
            self.replica_offsets.append(
                self.replica_offsets[-1] + physical_tp_size * self.cp
            )

        self.replicas: List[NonuniformTPReplicaRanks] = []
        self.rank_metadata: Dict[int, Dict[str, int]] = {}
        for replica_index, active_tp_size in enumerate(self.tp_domain_sizes):
            physical_tp_size = self._physical_tp_sizes[replica_index]
            start = self.rank_offset + self.replica_offsets[replica_index]
            ranks_by_cp = []
            inactive_ranks_by_cp = []
            for cp_rank in range(self.cp):
                cp_start = start + cp_rank * physical_tp_size
                cp_ranks = list(range(cp_start, cp_start + active_tp_size))
                inactive_cp_ranks = list(
                    range(cp_start + active_tp_size, cp_start + physical_tp_size)
                )
                ranks_by_cp.append(cp_ranks)
                inactive_ranks_by_cp.append(inactive_cp_ranks)
                for tp_rank, global_rank in enumerate(cp_ranks):
                    self.rank_metadata[global_rank] = {
                        'replica_index': replica_index,
                        'active_tp_size': active_tp_size,
                        'tp_rank': tp_rank,
                        'cp_rank': cp_rank,
                        'is_active': 1,
                    }
                for tp_rank in range(active_tp_size, physical_tp_size):
                    self.rank_metadata[cp_start + tp_rank] = {
                        'replica_index': replica_index,
                        'active_tp_size': active_tp_size,
                        'tp_rank': tp_rank,
                        'cp_rank': cp_rank,
                        'is_active': 0,
                    }
            self.replicas.append(
                NonuniformTPReplicaRanks(
                    replica_index=replica_index,
                    active_tp_size=active_tp_size,
                    physical_tp_size=physical_tp_size,
                    ranks_by_cp=ranks_by_cp,
                    inactive_ranks_by_cp=inactive_ranks_by_cp,
                )
            )

    def get_ranks(self, key: str) -> List[List[int]]:
        """Return topology-aware NTP rank groups for ``key``."""
        if key == 'tp':
            return self._get_tp_ranks()
        if key == 'cp':
            return self._get_cp_ranks()
        if key == 'dp':
            return self._get_dp_ranks()
        if key == 'dp-cp':
            return self._get_dp_cp_ranks()
        if key == 'tp-dp':
            return self._get_tp_dp_ranks()
        if key == 'tp-dp-cp':
            return self._get_tp_dp_cp_ranks()
        if key == 'tp-cp':
            return self._get_tp_cp_ranks()
        raise ValueError(
            f"Unknown nonuniform TP rank key {key}; expected {sorted(self._SUPPORTED_KEYS)}"
        )

    def get_rank_metadata(self, rank: int) -> Dict[str, int]:
        """Return topology coordinates for a global rank."""
        if rank not in self.rank_metadata:
            raise RuntimeError(f"Rank {rank} is not part of the NTP topology")
        return dict(self.rank_metadata[rank])

    def get_non_active_ranks_per_replica(
        self, pp_rank: int = 0
    ) -> Dict[Tuple[int, int, int], List[int]]:
        """Return legacy-shaped inactive local TP positions keyed by replica/CP/PP."""
        result = {}
        for replica in self.replicas:
            inactive = replica.inactive_local_tp_ranks(self.tp)
            if not inactive:
                continue
            for cp_rank in range(self.cp):
                result[(replica.replica_index, cp_rank, pp_rank)] = list(inactive)
        return result

    def _get_tp_ranks(self) -> List[List[int]]:
        return [
            list(cp_ranks) for replica in self.replicas for cp_ranks in replica.ranks_by_cp
        ]

    def _get_cp_ranks(self) -> List[List[int]]:
        groups = []
        for replica in self.replicas:
            for tp_rank in range(replica.active_tp_size):
                groups.append(
                    [replica.ranks_by_cp[cp_rank][tp_rank] for cp_rank in range(self.cp)]
                )
        return groups

    def _get_tp_cp_ranks(self) -> List[List[int]]:
        return [replica.ranks for replica in self.replicas]

    def _get_dp_ranks(self) -> List[List[int]]:
        groups = []
        for cp_rank in range(self.cp):
            for tp_rank in range(self.tp):
                ranks = [
                    replica.ranks_by_cp[cp_rank][tp_rank]
                    for replica in self.replicas
                    if tp_rank < replica.active_tp_size
                ]
                if ranks:
                    groups.append(ranks)
        return groups

    def _get_dp_cp_ranks(self) -> List[List[int]]:
        groups = []
        for tp_rank in range(self.tp):
            ranks = []
            for replica in self.replicas:
                if tp_rank >= replica.active_tp_size:
                    continue
                for cp_rank in range(self.cp):
                    ranks.append(replica.ranks_by_cp[cp_rank][tp_rank])
            if ranks:
                groups.append(ranks)
        return groups

    def _get_tp_dp_ranks(self) -> List[List[int]]:
        groups = []
        for cp_rank in range(self.cp):
            ranks = []
            for replica in self.replicas:
                ranks.extend(replica.ranks_by_cp[cp_rank])
            groups.append(ranks)
        return groups

    def _get_tp_dp_cp_ranks(self) -> List[List[int]]:
        return [[rank for replica in self.replicas for rank in replica.ranks]]


@dataclass
class PerBufferParamLayout:
    """Layout for parameters within one opt-in contiguous DDP buffer."""

    param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(default_factory=dict)
    side_grad_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(
        default_factory=dict
    )
    bucket_indices: List[Tuple[int, int]] = field(default_factory=list)
    per_bucket_numel_unpadded: List[int] = field(default_factory=list)
    param_indices: List[int] = field(default_factory=list)


@dataclass
class FullParamLayout:
    """Compatibility placeholder for wrappers that pass precomputed layouts."""

    layouts: Dict[object, PerBufferParamLayout] = field(default_factory=dict)


def pad_to_divisor(value: int, divisor: int) -> int:
    """Round up ``value`` to the nearest multiple of ``divisor``."""
    return int(math.ceil(value / divisor) * divisor)


def pad_param_start(param_start_index: int) -> int:
    """Align parameter start index to a 64-element boundary."""
    return pad_to_divisor(param_start_index, 64)


def pad_bucket_end(
    bucket_end_index: int, data_parallel_world_size: int, pad_for_high_nccl_busbw: bool
) -> int:
    """Pad bucket end for DP divisibility and optionally high NCCL bus bandwidth."""
    if pad_for_high_nccl_busbw:
        divisor = math.lcm(data_parallel_world_size, 128, 2**16)
    else:
        divisor = math.lcm(data_parallel_world_size, 128)
    return pad_to_divisor(bucket_end_index, divisor)


class ViewCopyHandle:
    """Wait handle that copies temporary contiguous receive buffers into views."""

    def __init__(self, handle, output_copies):
        self.handle = handle
        self.output_copies = output_copies

    def wait(self):
        self.handle.wait()
        for dst, src in self.output_copies:
            dst.copy_(src)
        self.output_copies = []


class CudaEventHandle:
    """Small wait handle compatible with Megatron bucket-group finish logic."""

    def __init__(self, events: List[torch.cuda.Event]):
        self.events = events

    def wait(self):
        current_stream = torch.cuda.current_stream()
        for event in self.events:
            current_stream.wait_event(event)
        self.events = []


def all_to_all_with_output_views(output_tensors, input_tensors, group, async_op: bool = False):
    """Run all_to_all, preserving non-contiguous output views via temporary buffers."""
    output_list = []
    output_copies = []
    for tensor in output_tensors:
        if tensor.is_contiguous():
            output_list.append(tensor)
        else:
            contiguous = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            output_list.append(contiguous)
            output_copies.append((tensor, contiguous))

    handle = dist.all_to_all(output_list, input_tensors, group=group, async_op=async_op)
    if async_op:
        return ViewCopyHandle(handle, output_copies)

    for dst, src in output_copies:
        dst.copy_(src)
    return None


def wait_handles(handles: Iterable[object]) -> None:
    """Wait every non-None handle in order."""
    for handle in handles:
        if handle is not None:
            handle.wait()


def record_post_sync_handles(bucket_group, state_attr: str, handles: List[object]) -> None:
    """Track post-sync handles and drain them when the last bucket group finishes."""
    state = getattr(bucket_group, state_attr, None)
    if state is None:
        wait_handles(handles)
        return

    state['handles'].extend(handles)
    if bucket_group is state['last_bucket_group']:
        try:
            wait_handles(state['handles'])
        finally:
            state['handles'] = []


def configure_post_sync_handle_tracker(bucket_groups: List[object], state_attr: str) -> None:
    """Attach a shared last-group handle tracker to ordered bucket groups."""
    if not bucket_groups:
        return
    state = {'handles': [], 'last_bucket_group': bucket_groups[-1]}
    for bucket_group in bucket_groups:
        setattr(bucket_group, state_attr, state)


def filter_kwargs_for_callable(fn: Callable, kwargs: Dict[str, object]) -> Dict[str, object]:
    """Return only kwargs accepted by ``fn``."""
    parameters = inspect.signature(fn).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


@contextmanager
def patch_ddp_param_and_grad_buffer(buffer_cls):
    """Temporarily patch DDP's imported _ParamAndGradBuffer binding."""
    original_buffer_class = ddp_module._ParamAndGradBuffer
    ddp_module._ParamAndGradBuffer = buffer_cls
    try:
        yield
    finally:
        ddp_module._ParamAndGradBuffer = original_buffer_class


def clone_bucket_group(bucket_group, wrapper_cls):
    """Clone a DDP bucket group into an opt-in subclass while preserving runtime state."""
    if isinstance(bucket_group, wrapper_cls):
        return bucket_group
    wrapped_bucket_group = wrapper_cls.__new__(wrapper_cls)
    wrapped_bucket_group.__dict__ = bucket_group.__dict__.copy()
    return wrapped_bucket_group


def wrap_bucket_groups_with_subclass(
    bucket_groups: List[object],
    wrapper_cls,
    configure_fn: Callable[[object], None],
    param_to_bucket_group: Optional[Dict[torch.nn.Parameter, object]] = None,
) -> List[object]:
    """Replace generic bucket groups with opt-in subclasses and rebuild next links."""
    wrapped_bucket_groups = []
    old_to_new = {}

    for bucket_group in bucket_groups:
        wrapped_bucket_group = clone_bucket_group(bucket_group, wrapper_cls)
        configure_fn(wrapped_bucket_group)
        old_to_new[bucket_group] = wrapped_bucket_group
        wrapped_bucket_groups.append(wrapped_bucket_group)

    for bucket_group, wrapped_bucket_group in old_to_new.items():
        next_bucket_group = getattr(bucket_group, 'next_param_gather_bucket_group', None)
        if next_bucket_group in old_to_new:
            wrapped_bucket_group.next_param_gather_bucket_group = old_to_new[next_bucket_group]

    if param_to_bucket_group is not None:
        for wrapped_bucket_group in wrapped_bucket_groups:
            for bucket in wrapped_bucket_group.buckets:
                for param in bucket.params_list:
                    param_to_bucket_group[param] = wrapped_bucket_group

    return wrapped_bucket_groups


def configure_ordered_bucket_group_scheduler(
    bucket_groups: List[object],
    state_attr: str,
    index_attr: str,
    ready_attr: str,
) -> None:
    """Attach deterministic launch state to bucket groups that must start in rank order."""
    state = {"groups": bucket_groups, "next_index": 0}
    for index, bucket_group in enumerate(bucket_groups):
        setattr(bucket_group, state_attr, state)
        setattr(bucket_group, index_attr, index)
        setattr(bucket_group, ready_attr, False)


def try_start_ordered_bucket_groups(
    bucket_group,
    state_attr: str,
    ready_attr: str,
    start_fn_name: str,
    *args,
    **kwargs,
) -> None:
    """Start ready bucket groups in deterministic order."""
    state = getattr(bucket_group, state_attr, None)
    if state is None:
        getattr(bucket_group, start_fn_name)(*args, **kwargs)
        return

    groups = state["groups"]
    while state["next_index"] < len(groups):
        group = groups[state["next_index"]]
        if not getattr(group, ready_attr, False):
            break
        getattr(group, start_fn_name)(*args, **kwargs)
        state["next_index"] += 1


def reset_ordered_bucket_group_scheduler(bucket_group, state_attr: str, index_attr: str) -> None:
    """Reset ordered scheduler state at the first bucket group."""
    state = getattr(bucket_group, state_attr, None)
    if state is not None and getattr(bucket_group, index_attr, -1) == 0:
        state["next_index"] = 0


def get_global_rank(group, group_rank: int) -> int:
    """Translate a process-group rank to a global rank."""
    if hasattr(dist, "get_global_rank"):
        return dist.get_global_rank(group, group_rank)
    ranks = getattr(group, "ranks", None)
    if ranks is None:
        raise RuntimeError("Cannot map process-group rank to global rank")
    return ranks[group_rank]
