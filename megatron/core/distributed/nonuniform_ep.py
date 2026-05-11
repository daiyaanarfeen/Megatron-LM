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
import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .. import parallel_state
from ..process_groups_config import ProcessGroupCollection
from ..transformer.transformer_config import TransformerConfig
from .distributed_data_parallel import DistributedDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .nonuniform_common import (
    configure_ordered_bucket_group_scheduler,
    filter_kwargs_for_callable,
    get_global_rank,
    get_nonuniform_ep_runtime_config,
    reset_ordered_bucket_group_scheduler,
    try_start_ordered_bucket_groups,
)
from .param_and_grad_buffer import _ParamAndGradBucket, _ParamAndGradBucketGroup


logger = logging.getLogger(__name__)


def _default_expert_name_pattern() -> re.Pattern:
    return re.compile(r"(?:^|\.)local_experts\.(\d+)(?:\.|$)")


@dataclass
class NonuniformEPConfig:
    """User-facing opt-in config for nonuniform EP gradient ownership transfer."""

    runtime_config: Optional[dict] = None
    expert_owner: Optional[Dict[int, int]] = None
    expert_name_pattern: Union[str, re.Pattern] = field(default_factory=_default_expert_name_pattern)
    require_owner_local_expert: bool = True
    grad_transfer_tag_base: int = 711_000
    grad_scatter_tag_base: int = 811_000

    def __post_init__(self):
        if isinstance(self.expert_name_pattern, str):
            self.expert_name_pattern = re.compile(self.expert_name_pattern)

@dataclass
class _ExpertBucketPlan:
    expert_id: int
    owner_ep_rank: int
    owner_global_rank: int
    source_ep_ranks: List[int]
    source_global_ranks: List[int]
    bucket_slices: List[Tuple[int, int]]
    bucket_group_index: int

    @property
    def numel(self) -> int:
        return sum(end - start for start, end in self.bucket_slices)


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
    local_idx = int(match.group(1))
    if local_expert_indices is None:
        return local_idx
    if local_idx >= len(local_expert_indices):
        raise RuntimeError(f"Local expert index {local_idx} is out of range for {name}")
    return int(local_expert_indices[local_idx])


class NonuniformEPParamAndGradBucketGroup(_ParamAndGradBucketGroup):
    """Expert-level bucket group that transfers grads to owner ranks before DP sync."""

    def configure_nonuniform_ep(
        self,
        runtime_config: dict,
        nonuniform_ep_config: NonuniformEPConfig,
        plan: _ExpertBucketPlan,
    ) -> None:
        self._nep_runtime_config = runtime_config
        self._nep_config = nonuniform_ep_config
        self._nep_plan = plan
        self._nep_started = False
        self._nep_ready = False
        self._nep_gather_handle = None
        self._nep_scatter_handle = None
        self._nep_gather_recv_buffers = []
        self._nep_scatter_send_buffers = []
        self._nep_gather_send_buffer = None
        self._nep_scatter_recv_buffer = None

        ep_rank = runtime_config['ep_rank']
        self._nep_is_owner = ep_rank == plan.owner_ep_rank
        if (
            nonuniform_ep_config.require_owner_local_expert
            and self._nep_is_owner
            and plan.expert_id not in runtime_config.get('_local_expert_id_set', set())
        ):
            raise RuntimeError(
                "NEP owner mode requires the owner rank to hold optimizer-visible params "
                f"for expert {plan.expert_id}; owner ep_rank={plan.owner_ep_rank}"
            )
        self._allocate_nep_persistent_grad_buffers()

    def _allocate_nep_persistent_grad_buffers(self):
        """Allocate persistent p2p staging buffers for this expert bucket."""
        plan = self._nep_plan
        bucket = self.buckets[0]
        ep_rank = self._nep_runtime_config['ep_rank']

        if plan.numel == 0:
            return

        if self._nep_is_owner:
            for source_ep_rank, source_global_rank in zip(
                plan.source_ep_ranks, plan.source_global_ranks
            ):
                if source_ep_rank == ep_rank:
                    continue
                self._nep_gather_recv_buffers.append(
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
            # The gather receive buffers are no longer live after gather wait/accumulate.
            # Reuse them as owner-side scatter send buffers to keep leader overhead to one
            # persistent flat grad buffer per non-owner source.
            self._nep_scatter_send_buffers = self._nep_gather_recv_buffers
        else:
            self._nep_gather_send_buffer = torch.empty(
                plan.numel,
                dtype=bucket.grad_data.dtype,
                device=bucket.grad_data.device,
            )
            self._nep_scatter_recv_buffer = torch.empty(
                plan.numel,
                dtype=bucket.grad_data.dtype,
                device=bucket.grad_data.device,
            )

    def _grad_transfer_tag(self) -> int:
        return self._nep_config.grad_transfer_tag_base + self._nep_plan.bucket_group_index

    def _grad_scatter_tag(self) -> int:
        return self._nep_config.grad_scatter_tag_base + self._nep_plan.bucket_group_index

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
        plan = self._nep_plan
        bucket = self.buckets[0]
        works = []
        recv_accumulations = []
        keepalive_buffers = []

        if self._nep_is_owner:
            for _, source_global_rank, recv_buffer in self._nep_gather_recv_buffers:
                works.append(
                    dist.irecv(
                        recv_buffer,
                        src=source_global_rank,
                        tag=self._grad_transfer_tag(),
                    )
                )
                recv_accumulations.append((bucket, plan.bucket_slices, recv_buffer))
        else:
            send_buffer = self._nep_gather_send_buffer
            _pack_bucket_slices_into(bucket, plan.bucket_slices, send_buffer)
            keepalive_buffers.append(send_buffer)
            works.append(
                dist.isend(
                    send_buffer,
                    dst=plan.owner_global_rank,
                    tag=self._grad_transfer_tag(),
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
        plan = self._nep_plan
        bucket = self.buckets[0]
        works = []
        recv_copies = []
        keepalive_buffers = []

        if self._nep_is_owner:
            for _, source_global_rank, send_buffer in self._nep_scatter_send_buffers:
                _pack_bucket_slices_into(bucket, plan.bucket_slices, send_buffer)
                keepalive_buffers.append(send_buffer)
                works.append(
                    dist.isend(
                        send_buffer,
                        dst=source_global_rank,
                        tag=self._grad_scatter_tag(),
                    )
                )
        else:
            recv_buffer = self._nep_scatter_recv_buffer
            works.append(
                dist.irecv(
                    recv_buffer,
                    src=plan.owner_global_rank,
                    tag=self._grad_scatter_tag(),
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
        self._wait_nep_gather_to_owner()
        if not self._nep_is_owner:
            self.grad_reduce_handle = None
            return
        return self._start_owner_dp_sync_after_gather(force_all_reduce=force_all_reduce)

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Finish owner DP sync and scatter synced grads back to source ranks."""
        self.param_gather_dispatched = False
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            if self._nep_is_owner and self.grad_reduce_handle is not None:
                self.grad_reduce_handle.wait()
                self.grad_reduce_handle = None
            self._start_nep_scatter_from_owner()
            self._wait_nep_scatter_from_owner()
            self._copy_back_extra_main_grads()
            return

        if self.is_first_batch:
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
        self._nep_ready = False
        self._nep_gather_handle = None
        self._nep_scatter_handle = None
        reset_ordered_bucket_group_scheduler(
            self, '_nep_scheduler_state', '_nep_group_index'
        )


def _build_expert_bucket_specs(buffers, runtime_config, config, param_to_name):
    local_expert_indices = runtime_config.get('local_expert_indices')
    local_expert_id_set = set(local_expert_indices) if local_expert_indices is not None else set()
    specs = []

    for buffer in buffers:
        expert_to_params = {}
        for param in buffer.param_index_map:
            name = param_to_name.get(param, "")
            expert_id = _local_expert_id_from_name(
                name,
                config.expert_name_pattern,
                local_expert_indices,
            )
            if expert_id is None:
                continue
            local_expert_id_set.add(expert_id)
            expert_to_params.setdefault(expert_id, []).append(param)

        for expert_id, params in expert_to_params.items():
            params = sorted(params, key=lambda param: buffer.param_index_map[param][0])
            starts_ends = [buffer.param_index_map[param][:2] for param in params]
            start = min(start for start, _ in starts_ends)
            end = max(end for _, end in starts_ends)
            total = sum(param_end - param_start for param_start, param_end in starts_ends)
            if total != end - start:
                raise RuntimeError(
                    f"Expert {expert_id} params are not contiguous in the grad buffer"
                )
            specs.append((buffer, expert_id, params, start, end))

    runtime_config['_local_expert_id_set'] = local_expert_id_set
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

    for group_index, (buffer, expert_id, params, start, end) in enumerate(specs):
        owner_ep_rank = _owner_for_expert(
            expert_id,
            runtime_config,
            nonuniform_ep_config.expert_owner,
        )
        source_ep_ranks = _source_ep_ranks_for_expert(expert_id, runtime_config)
        if owner_ep_rank not in source_ep_ranks:
            raise RuntimeError(
                "NEP owner-transfer mode requires the owner rank to physically hold "
                f"expert {expert_id}. owner_ep_rank={owner_ep_rank}, "
                f"source_ep_ranks={source_ep_ranks}. Use an expert placement that "
                "duplicates owner params or choose each expert's physical holder as owner."
            )
        owner_global_rank = get_global_rank(ep_group, owner_ep_rank)
        source_global_ranks = [get_global_rank(ep_group, rank) for rank in source_ep_ranks]
        if edp_group is None and any(rank != owner_ep_rank for rank in source_ep_ranks):
            raise RuntimeError(
                "NEP p2p ownership transfer requires an owner-only expert-data-parallel "
                "group in runtime_config['edp_group'] when any local expert transfers to "
                "a different owner rank."
            )
        plan = _ExpertBucketPlan(
            expert_id=expert_id,
            owner_ep_rank=owner_ep_rank,
            owner_global_rank=owner_global_rank,
            source_ep_ranks=source_ep_ranks,
            source_global_ranks=source_global_ranks,
            bucket_slices=[(0, end - start)],
            bucket_group_index=group_index,
        )

        param_data = buffer.param_data[start:end] if buffer.param_data is not None else None
        bucket = _ParamAndGradBucket(
            params=params,
            param_data=param_data,
            grad_data=buffer.grad_data[start:end],
            offset=start,
            numel_unpadded=end - start,
            gradient_scaling_factor=buffer.gradient_scaling_factor,
            bucket_id=group_index,
            param_index_map=buffer.param_index_map,
            params_with_extra_main_grads=[],
        )
        is_owner_rank = runtime_config['ep_rank'] == owner_ep_rank
        for param in params:
            param.nonuniform_ep_expert_id = expert_id
            param.nonuniform_ep_owner_rank = owner_ep_rank
        collective_group = (
            edp_group if (is_owner_rank and edp_group is not None) else buffer.data_parallel_group
        )
        bucket_group = NonuniformEPParamAndGradBucketGroup(
            [bucket],
            ddp_config,
            collective_group,
            collective_group.size(),
        )
        bucket_group.configure_nonuniform_ep(
            runtime_config,
            nonuniform_ep_config,
            plan,
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
