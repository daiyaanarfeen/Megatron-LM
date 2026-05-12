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
import logging
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .. import parallel_state
from ..process_groups_config import ProcessGroupCollection
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


def _default_expert_name_pattern() -> re.Pattern:
    return re.compile(
        r"(?:^|\.)local_experts\.(\d+)(?:\.|$)"
        r"|(?:^|\.)experts\.linear_fc[12]\.(?:weight|bias)(\d+)$"
    )


@dataclass
class NonuniformEPConfig:
    """User-facing opt-in config for nonuniform EP gradient ownership transfer."""

    runtime_config: Optional[dict] = None
    expert_owner: Optional[Dict[int, int]] = None
    expert_name_pattern: Union[str, re.Pattern] = field(
        default_factory=_default_expert_name_pattern
    )
    require_owner_local_expert: bool = True
    grad_transfer_tag_base: int = 711_000
    grad_scatter_tag_base: int = 811_000

    def __post_init__(self):
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
    return parallel_state.create_group(
        ranks,
        timeout=timeout,
        backend=backend,
        pg_options=(
            None if backend == "gloo" else parallel_state.get_nccl_options(desc, nccl_comm_cfgs)
        ),
        group_desc=desc,
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
    for ranks in generator.get_ranks('ep'):
        group = _create_group(ranks, timeout, nccl_comm_cfgs, "ep")
        if rank in ranks:
            _set_parallel_state_attr("_EXPERT_MODEL_PARALLEL_GROUP", group)
            _set_parallel_state_attr("_EXPERT_MODEL_PARALLEL_RANKS", ranks)

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

    min_ep_size = generator.min_k * tp * cp // etp
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
        'edp_group': parallel_state.get_expert_data_parallel_group(),
        'ep_rank': ep_rank,
        'is_edp_eligible': ep_rank < min_ep_size,
        'is_b_leader': ep_rank < min_ep_size,
        'local_expert_indices': expert_placement[ep_rank],
        'expert_placement': expert_placement,
        'expert_gather_map': expert_gather_map,
    }
    set_nonuniform_ep_runtime_config(runtime_config)
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

        for entry in self._nep_entries:
            plan = entry['plan']
            bucket = entry['bucket']
            if entry['is_owner']:
                for _, source_global_rank, recv_buffer in entry['gather_recv_buffers']:
                    works.append(
                        dist.irecv(
                            recv_buffer,
                            src=source_global_rank,
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
                        dst=plan.owner_global_rank,
                        tag=self._grad_transfer_tag(plan),
                    )
                )

        self._nep_gather_handle = _P2PGradTransferHandle(
            works,
            recv_accumulations=recv_accumulations,
            keepalive_buffers=keepalive_buffers,
        )

    def _wait_nep_gather_to_owner(self):
        if self._nep_gather_handle is not None:
            self._nep_gather_handle.wait()
            self._nep_gather_handle = None

    def _start_nep_scatter_from_owner(self):
        works = []
        recv_copies = []
        keepalive_buffers = []

        for entry in self._nep_entries:
            plan = entry['plan']
            bucket = entry['bucket']
            if entry['is_owner']:
                for _, source_global_rank, send_buffer in entry['scatter_send_buffers']:
                    _pack_bucket_slices_into(bucket, plan.bucket_slices, send_buffer)
                    keepalive_buffers.append(send_buffer)
                    works.append(
                        dist.isend(
                            send_buffer,
                            dst=source_global_rank,
                            tag=self._grad_scatter_tag(plan),
                        )
                    )
            else:
                recv_buffer = entry['scatter_recv_buffer']
                works.append(
                    dist.irecv(
                        recv_buffer,
                        src=plan.owner_global_rank,
                        tag=self._grad_scatter_tag(plan),
                    )
                )
                recv_copies.append((bucket, plan.bucket_slices, recv_buffer))

        self._nep_scatter_handle = _P2PGradTransferHandle(
            works,
            recv_copies=recv_copies,
            keepalive_buffers=keepalive_buffers,
        )

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
        self._wait_nep_gather_to_owner()
        return self._start_owner_dp_sync_after_gather(force_all_reduce=force_all_reduce)

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
            self._wait_nep_scatter_from_owner()
            self._copy_back_extra_main_grads()
            return
        result = super().finish_grad_sync(force_all_reduce=force_all_reduce)
        self._start_nep_scatter_from_owner()
        self._wait_nep_scatter_from_owner()
        return result

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
        self._nep_ready = len(self.params) == 0
        self._nep_gather_handle = None
        self._nep_scatter_handle = None
        reset_ordered_bucket_group_scheduler(
            self, '_nep_scheduler_state', '_nep_group_index'
        )


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
