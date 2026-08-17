# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""
Nonuniform Tensor Parallelism (NTP) - Non-intrusive implementation.

This module provides fault tolerance for tensor-parallel training by allowing
a subset of TP ranks ("spares") to handle failures while "core" ranks continue computation.

All NTP logic is contained in this module as subclasses of core components,
making it non-intrusive to the main codebase.

Usage:
    Instead of using the standard classes, use the NTP variants:
    - NonuniformTPDistributedDataParallel instead of DistributedDataParallel
    - Call initialize_nonuniform_tp_process_groups() after initialize_model_parallel()
"""

import fnmatch
import functools
import logging
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .. import parallel_state
from ..optimizer import param_layout as optimizer_param_layout
from ..process_groups_config import ProcessGroupCollection
from ..transformer.cuda_graphs import is_graph_capturing
from ..transformer.transformer_config import TransformerConfig
from ..utils import PARAM_READY_CALLBACK_ATTR, log_on_each_pipeline_stage
from .distributed_data_parallel import DistributedDataParallel, _BucketParamReadyCallback
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .nonuniform_common import (
    NonuniformTPTopologyRankGenerator,
    all_to_all_with_output_views,
    configure_post_sync_handle_tracker,
    create_nonuniform_process_group,
    initialize_nonuniform_attention_process_groups,
    initialize_nonuniform_expert_gtp_aliases,
    load_nonuniform_nccl_communicator_configs,
    patch_ddp_param_and_grad_buffer,
    record_post_sync_handles,
    set_nonuniform_parallel_state_attr,
    wrap_bucket_groups_with_subclass,
)
from .param_and_grad_buffer import _ParamAndGradBucketGroup, _ParamAndGradBuffer

logger = logging.getLogger(__name__)


@dataclass
class PerBufferParamLayout(optimizer_param_layout.PerBufferParamLayout):
    """Native Megatron parameter layout extended with NTP side-gradient ranges."""

    side_grad_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(
        default_factory=dict
    )


def _ntp_get_non_active_ranks(
    ntp_config: "NonuniformTPConfig", dp_rank: int, cp_rank: int = 0, pp_rank: int = 0
) -> Optional[List[int]]:
    """Return configured inactive local TP ranks, accepting both legacy and tuple keys."""
    if not ntp_config.non_active_ranks_per_dp:
        return None

    rank_key = (dp_rank, cp_rank, pp_rank)
    if rank_key in ntp_config.non_active_ranks_per_dp:
        return ntp_config.non_active_ranks_per_dp[rank_key]
    if dp_rank in ntp_config.non_active_ranks_per_dp:
        return ntp_config.non_active_ranks_per_dp[dp_rank]
    return None


def _ntp_get_current_topology_metadata(
    ntp_config: "NonuniformTPConfig",
) -> Optional[Dict[str, int]]:
    """Return topology coordinates for the current global rank, if topology mode is active."""
    if not ntp_config.topology_rank_metadata:
        return None
    rank = dist.get_rank()
    metadata = ntp_config.topology_rank_metadata.get(rank)
    return dict(metadata) if metadata is not None else None


def _ntp_current_rank_is_reduced_dp(ntp_config: "NonuniformTPConfig") -> bool:
    """Return True if this rank belongs to a DP replica configured with reduced TP."""
    if ntp_config.tp_spares == 0:
        return False

    metadata = _ntp_get_current_topology_metadata(ntp_config)
    if metadata is not None:
        return metadata['active_tp_size'] < ntp_config.tp_base

    dp_rank = parallel_state.get_data_parallel_rank()
    if ntp_config.non_active_ranks_per_dp:
        cp_rank = parallel_state.get_context_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        return _ntp_get_non_active_ranks(ntp_config, dp_rank, cp_rank, pp_rank) is not None
    return dp_rank < ntp_config.num_reduced_tp_dp_ranks


def _ntp_current_rank_should_dp_sync(ntp_config: "NonuniformTPConfig") -> bool:
    """Return True if this rank should participate in data-parallel grad sync."""
    if ntp_config.tp_spares == 0:
        return True

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    reduced_tp_size = ntp_config.tp_base - ntp_config.tp_spares
    metadata = _ntp_get_current_topology_metadata(ntp_config)
    if metadata is not None:
        if metadata['active_tp_size'] < ntp_config.tp_base:
            return True
        return tp_rank < reduced_tp_size

    # Reduced DP replicas only contain active TP ranks after NTP group reconfiguration.
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size != ntp_config.tp_base:
        return True

    # In healthy full-TP replicas, ranks beyond reduced_tp_size are folded into core ranks by
    # NTP resharding and must not wait for a DP peer from the reduced replica.
    return tp_rank < reduced_tp_size


def _ntp_can_query_parallel_state() -> bool:
    """Return True when distributed/model-parallel state is initialized enough for NTP."""
    if not dist.is_available() or not dist.is_initialized():
        return False
    try:
        parallel_state.get_tensor_model_parallel_world_size()
        parallel_state.get_tensor_model_parallel_rank()
        parallel_state.get_data_parallel_rank()
        parallel_state.get_context_parallel_rank()
        parallel_state.get_pipeline_model_parallel_rank()
    except Exception:
        return False
    return True


def _ntp_param_can_reshard(param: torch.nn.Parameter) -> bool:
    """Return True for tensor-parallel params initialized with NTP split metadata."""
    return (
        hasattr(param, 'tensor_model_parallel')
        and param.tensor_model_parallel
        and hasattr(param, 'partition_dim')
        and hasattr(param, 'send_splits')
        and hasattr(param, 'recv_splits')
    )


def _ntp_should_expand_param_grad(
    param: torch.nn.Parameter, ntp_config: "NonuniformTPConfig"
) -> bool:
    """Return True if healthy core rank needs side_grad storage for this TP parameter."""
    if ntp_config.tp_spares == 0 or not _ntp_param_can_reshard(param):
        return False
    if _ntp_current_rank_is_reduced_dp(ntp_config):
        return False
    if parallel_state.get_tensor_model_parallel_world_size() != ntp_config.tp_base:
        return False
    return parallel_state.get_tensor_model_parallel_rank() < (
        ntp_config.tp_base - ntp_config.tp_spares
    )


def _ntp_extra_partition_dim(param: torch.nn.Parameter, ntp_config: "NonuniformTPConfig") -> int:
    """Return side_grad extent along partition_dim for this healthy core rank."""
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    return int(sum(param.recv_splits[tp_rank][-ntp_config.tp_spares :]))


def _ntp_param_numel(param: torch.nn.Parameter, ntp_config: "NonuniformTPConfig") -> int:
    """Return main grad plus any NTP side grad storage needed for this param."""
    numel = param.data.nelement()
    if _ntp_should_expand_param_grad(param, ntp_config):
        side_shape = list(param.data.shape)
        side_shape[param.partition_dim] = _ntp_extra_partition_dim(param, ntp_config)
        numel += torch.Size(side_shape).numel()
    return numel


def _ntp_empty_like_partition(
    param: torch.nn.Parameter, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Create a zero-width tensor matching a TP param's partition dimension."""
    empty_shape = list(param.shape)
    empty_shape[param.partition_dim] = 0
    return torch.empty(empty_shape, device=param.device, dtype=dtype or param.dtype).contiguous()


def _ntp_split_for_all_to_all(tensor: torch.Tensor, splits: List[int], dim: int):
    """Split tensor for all_to_all, preserving zero-sized entries."""
    return [piece.contiguous() for piece in torch.split(tensor, splits, dim=dim)]


def _ntp_split_views_for_all_to_all(tensor: torch.Tensor, splits: List[int], dim: int):
    """Split tensor into output views for all_to_all receive paths."""
    return list(torch.split(tensor, splits, dim=dim))


def _compute_ntp_per_buffer_param_layout(
    params: List[torch.nn.Parameter],
    bucket_size: Optional[int],
    data_parallel_world_size: int,
    ddp_config: DistributedDataParallelConfig,
    ntp_config: "NonuniformTPConfig",
    param_indices: Optional[List[int]] = None,
) -> PerBufferParamLayout:
    """Compute a buffer layout that includes side_grad storage for healthy core ranks."""

    def _does_param_require_new_bucket(param):
        return getattr(param, "shared_embedding", False)

    param_index_map = {}
    side_grad_index_map = {}
    bucket_indices = []
    per_bucket_numel_unpadded = []

    param_start_index = 0
    bucket_start_index = 0
    bucket_params = set()
    bucket_id = 0

    def _finalize_bucket(param_end_index, bucket_start_index, bucket_id):
        per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
        if ddp_config.use_distributed_optimizer:
            bucket_end_index = optimizer_param_layout.pad_bucket_end(
                param_end_index,
                data_parallel_world_size,
                ddp_config.pad_buckets_for_high_nccl_busbw,
            )
        else:
            bucket_end_index = param_end_index
        bucket_indices.append((bucket_start_index, bucket_end_index))
        return bucket_end_index, bucket_id + 1

    for param in params[::-1]:
        if ddp_config.use_distributed_optimizer:
            param_start_index = optimizer_param_layout.pad_param_start(param_start_index)

        if _does_param_require_new_bucket(param) and len(bucket_params) > 0:
            bucket_start_index, bucket_id = _finalize_bucket(
                param_start_index, bucket_start_index, bucket_id
            )
            bucket_params = set()
            param_start_index = bucket_start_index

        main_numel = param.data.nelement()
        param_main_end_index = param_start_index + main_numel
        param_end_index = param_start_index + _ntp_param_numel(param, ntp_config)
        param_index_map[param] = (param_start_index, param_main_end_index, bucket_id)
        if param_end_index > param_main_end_index:
            side_grad_index_map[param] = (param_main_end_index, param_end_index, bucket_id)
        bucket_params.add(param)

        if (
            bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size
        ) or _does_param_require_new_bucket(param):
            bucket_start_index, bucket_id = _finalize_bucket(
                param_end_index, bucket_start_index, bucket_id
            )
            bucket_params = set()
            param_start_index = bucket_start_index
        else:
            param_start_index = param_end_index

    if len(bucket_params) > 0:
        _finalize_bucket(param_end_index, bucket_start_index, bucket_id)

    return PerBufferParamLayout(
        param_index_map=param_index_map,
        side_grad_index_map=side_grad_index_map,
        bucket_indices=bucket_indices,
        per_bucket_numel_unpadded=per_bucket_numel_unpadded,
        param_indices=param_indices if param_indices is not None else [],
    )


# ======================================================================================
# NTP Configuration
# ======================================================================================


@dataclass
class NonuniformTPConfig:
    """Configuration for Nonuniform Tensor Parallelism (NTP).

    NTP provides fault tolerance for tensor-parallel training by designating
    a subset of TP ranks as "spares" that can handle GPU failures.
    """

    tp_base: int = 8
    """Base for tensor parallelism. This is the number of ranks in healthy tensor parallel groups.
       Used for nonuniform tensor parallelism."""

    tp_spares: int = 0
    """Number of spares for nonuniform tensor parallelism.

       When > 0, (tp_base - tp_spares) ranks handle computation and tp_spares ranks
       provide fault tolerance.
    """

    num_reduced_tp_dp_ranks: int = 1
    """Number of DP ranks that use reduced TP (tp_base - tp_spares). The remaining DP ranks use
       full tp_base. Reduced TP ranks are assumed to come first in the global rank ordering."""

    non_active_ranks_per_dp: Optional[Dict[Tuple[int, int, int], List[int]]] = None
    """Mapping of (DP rank, CP rank, PP rank) to list of non-active (spare) local TP rank IDs.
       This allows specifying arbitrary GPU failures across all parallelism dimensions.
       Example: {(0,0,0): [0,3], (0,1,0): [1,2], (1,0,0): [0,3]} means:
         - DP rank 0, CP rank 0, PP rank 0 has local TP ranks 0,3 as spares
         - DP rank 0, CP rank 1, PP rank 0 has local TP ranks 1,2 as spares
         - DP rank 1, CP rank 0, PP rank 0 has local TP ranks 0,3 as spares
       The number of non-active ranks must be consistent across CP replicas within each DP rank.
       If None, defaults to last tp_spares ranks as non-active."""

    tp_domain_sizes: Optional[List[int]] = None
    """Optional topology-aware active TP size per replica.

       When set, NTP process groups are generated directly from contiguous rank
       blocks whose sizes match this list, instead of patching standard Megatron
       groups after init.
       Values must currently be either tp_base or tp_base - tp_spares so the existing
       reduced/full TP resharding semantics remain unchanged.
    """

    topology_rank_metadata: Optional[Dict[int, Dict[str, int]]] = None
    """Runtime mapping of global rank to topology coordinates, filled during init."""

    optimizer_param_group_alignment_group: Optional[dist.ProcessGroup] = field(
        default=None, init=False, repr=False
    )
    """Active-rank group used by current-main optimizer parameter-group alignment."""


def _get_ntp_optimizer_alignment_ranks(
    ntp_config: NonuniformTPConfig, context_parallel_size: int, world_size: int
) -> List[int]:
    """Return ranks that remain active after legacy NTP spare-rank exits."""
    tp_base = ntp_config.tp_base
    dp_replica_size = tp_base * context_parallel_size
    active_global_ranks = []
    for global_rank in range(world_size):
        dp_replica_id = global_rank // dp_replica_size
        if dp_replica_id >= ntp_config.num_reduced_tp_dp_ranks:
            active_global_ranks.append(global_rank)
            continue

        local_rank_in_dp = global_rank % dp_replica_size
        cp_rank = local_rank_in_dp // tp_base if context_parallel_size > 1 else 0
        local_tp_rank = local_rank_in_dp % tp_base
        active_local_ranks = get_active_ranks_for_dp(
            dp_replica_id, tp_base, ntp_config, cp_rank=cp_rank
        )
        if local_tp_rank in active_local_ranks:
            active_global_ranks.append(global_rank)
    return active_global_ranks


def _initialize_ntp_optimizer_alignment_group(
    ntp_config: NonuniformTPConfig, context_parallel_size: int
) -> None:
    """Create one optimizer metadata group that excludes ranks NTP will terminate."""
    world_size = dist.get_world_size()
    active_ranks = _get_ntp_optimizer_alignment_ranks(ntp_config, context_parallel_size, world_size)
    if len(active_ranks) == world_size:
        group = dist.group.WORLD
    else:
        # Every rank must create this group before inactive ranks exit. Inactive ranks receive
        # NON_GROUP_MEMBER and never expose the group through a DDP wrapper.
        group = dist.new_group(ranks=active_ranks)
    ntp_config.optimizer_param_group_alignment_group = (
        group if dist.get_rank() in active_ranks else None
    )


# ======================================================================================
# Utility Functions for NTP Configuration
# ======================================================================================


def get_active_ranks_for_dp(
    dp_rank: int, tp_base: int, ntp_config: NonuniformTPConfig, cp_rank: int = 0, pp_rank: int = 0
) -> List[int]:
    """
    Get list of active (non-spare) local rank IDs for a given DP rank.

    Args:
        dp_rank: Data parallel rank
        tp_base: Base tensor parallel size
        ntp_config: NTP configuration

    Returns:
        List of local rank IDs that are active (not spare)
    """
    non_active = _ntp_get_non_active_ranks(ntp_config, dp_rank, cp_rank, pp_rank)
    if non_active is not None:
        # Use explicitly specified non-active ranks
        non_active_set = set(non_active)
        active_ranks = [i for i in range(tp_base) if i not in non_active_set]
    else:
        # Default: first (tp_base - tp_spares) ranks are active
        red_tp = tp_base - ntp_config.tp_spares
        active_ranks = list(range(red_tp))

    return active_ranks


def _initialize_nonuniform_tp_topology_process_groups(
    ntp_config: NonuniformTPConfig,
    exit_spares: bool = True,
    context_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    create_gloo_process_groups: bool = True,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
) -> bool:
    """Initialize topology-aware NTP process groups from contiguous TP domains."""
    assert dist.is_initialized()
    if parallel_state.get_tensor_model_parallel_group(check_initialized=False) is not None:
        raise RuntimeError(
            "NTP topology process groups must be initialized before standard model-parallel "
            "groups. Use the nonuniform entrypoint topology flag instead of late patching."
        )
    if get_embedding_ranks is None:
        get_embedding_ranks = parallel_state.default_embedding_ranks
    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = parallel_state.default_position_embedding_ranks

    if context_parallel_size is None:
        if parallel_state.get_context_parallel_group(check_initialized=False) is not None:
            context_parallel_size = parallel_state.get_context_parallel_world_size()
        else:
            context_parallel_size = 1

    generator = NonuniformTPTopologyRankGenerator(
        tp=ntp_config.tp_base, cp=context_parallel_size, tp_domain_sizes=ntp_config.tp_domain_sizes
    )
    world_size = dist.get_world_size()
    if generator.world_size != world_size:
        raise RuntimeError(
            f"NTP topology world_size ({generator.world_size}) != distributed world_size "
            f"({world_size}). Expected sum(tp_domain_sizes) * CP ranks."
        )

    reduced_tp_size = ntp_config.tp_base - ntp_config.tp_spares
    allowed_sizes = {ntp_config.tp_base}
    if ntp_config.tp_spares > 0:
        allowed_sizes.add(reduced_tp_size)
    invalid_sizes = [size for size in ntp_config.tp_domain_sizes if size not in allowed_sizes]
    if invalid_sizes:
        raise RuntimeError(
            "NTP topology currently supports only full TP domains and the configured "
            f"reduced TP size ({reduced_tp_size}); got {invalid_sizes}"
        )

    ntp_config.num_reduced_tp_dp_ranks = sum(
        1 for size in ntp_config.tp_domain_sizes if size < ntp_config.tp_base
    )
    ntp_config.topology_rank_metadata = dict(generator.rank_metadata)

    rank = dist.get_rank()
    timeout = timedelta(minutes=distributed_timeout_minutes)
    nccl_comm_cfgs = load_nonuniform_nccl_communicator_configs(nccl_communicator_config_path)
    initialize_nonuniform_attention_process_groups(
        generator=generator,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
        nccl_comm_cfgs=nccl_comm_cfgs,
        create_gloo_process_groups=create_gloo_process_groups,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
    )

    for ranks in [[global_rank] for global_rank in range(world_size)]:
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "ep")
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_EXPERT_MODEL_PARALLEL_GROUP", group)
            set_nonuniform_parallel_state_attr("_EXPERT_MODEL_PARALLEL_RANKS", ranks)

    for ranks in generator.get_ranks('tp'):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "ep_tp")
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_EXPERT_TENSOR_PARALLEL_GROUP", group)

    for ranks in generator.get_ranks('tp'):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "tp_ep_mp")
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP", group)

    for ranks in generator.get_ranks('tp'):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "tp_ep_pp")
        if rank in ranks:
            set_nonuniform_parallel_state_attr(
                "_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP", group
            )

    for ranks in generator.get_ranks('dp'):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "ep_dp")
        group_gloo = (
            create_nonuniform_process_group(
                ranks, timeout, nccl_comm_cfgs, "EXPERT_DATA_PARALLEL_GROUP_GLOO", "gloo"
            )
            if create_gloo_process_groups
            else None
        )
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_EXPERT_DATA_PARALLEL_GROUP", group)
            set_nonuniform_parallel_state_attr("_EXPERT_DATA_PARALLEL_GROUP_GLOO", group_gloo)
            set_nonuniform_parallel_state_attr("_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP", group)
            set_nonuniform_parallel_state_attr(
                "_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO", group_gloo
            )
            set_nonuniform_parallel_state_attr("_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP", None)

    initialize_nonuniform_expert_gtp_aliases()
    set_nonuniform_parallel_state_attr(
        "_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP", dist.group.WORLD
    )
    parallel_state._set_global_memory_buffer()
    rank_metadata = generator.get_rank_metadata(rank)
    if not rank_metadata['is_active']:
        logger.info("[NTP] Rank %s is inactive in topology mode, exiting", rank)
        if exit_spares:
            sys.exit(0)
        return False
    return True


# ======================================================================================
# Process Group Initialization for NTP
# ======================================================================================


def initialize_nonuniform_tp_process_groups(
    ntp_config: NonuniformTPConfig,
    exit_spares: bool = True,
    context_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    create_gloo_process_groups: bool = True,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
) -> bool:
    """
    Reconfigure TP and CP process groups for nonuniform tensor parallelism.

    Call this function after initialize_model_parallel() to enable NTP.
    Non-active (spare) ranks will exit after group creation.

    Args:
        ntp_config: NTP configuration containing tp_base, tp_spares, num_reduced_tp_dp_ranks,
                    and optionally non_active_ranks_per_dp
    """
    if ntp_config.tp_domain_sizes is not None:
        # Topology mode launches only active ranks, so current-main optimizer metadata may use
        # the existing world group directly.
        ntp_config.optimizer_param_group_alignment_group = dist.group.WORLD
        return _initialize_nonuniform_tp_topology_process_groups(
            ntp_config,
            exit_spares=exit_spares,
            context_parallel_size=context_parallel_size,
            nccl_communicator_config_path=nccl_communicator_config_path,
            distributed_timeout_minutes=distributed_timeout_minutes,
            create_gloo_process_groups=create_gloo_process_groups,
            get_embedding_ranks=get_embedding_ranks,
            get_position_embedding_ranks=get_position_embedding_ranks,
        )

    if ntp_config.tp_spares == 0:
        # No nonuniform TP, nothing to reconfigure
        return True

    tp_base = ntp_config.tp_base
    cp_size = parallel_state.get_context_parallel_world_size()
    _initialize_ntp_optimizer_alignment_group(ntp_config, cp_size)
    rank = dist.get_rank()

    # Calculate which DP replicas use reduced TP
    dp_replica_size = tp_base * cp_size
    num_reduced_dp_ranks = ntp_config.num_reduced_tp_dp_ranks

    # Determine if current rank is in a reduced TP DP replica
    dp_replica_id = rank // dp_replica_size
    if dp_replica_id >= num_reduced_dp_ranks:
        # This rank is in a normal TP DP replica, no reconfiguration needed
        logger.info(
            "[NTP] Rank %s is in normal TP DP replica %s, skipping reconfiguration",
            rank,
            dp_replica_id,
        )
        return True

    local_rank_in_dp = rank % dp_replica_size
    cp_rank_in_dp = local_rank_in_dp // tp_base if cp_size > 1 else 0

    # This rank is in a reduced TP DP replica - need to reconfigure
    # Get active ranks for this DP replica (supports non-contiguous)
    active_local_ranks = get_active_ranks_for_dp(
        dp_replica_id, tp_base, ntp_config, cp_rank=cp_rank_in_dp
    )

    logger.info(
        "[NTP] Rank %s in DP replica %s: active_local_ranks=%s",
        rank,
        dp_replica_id,
        active_local_ranks,
    )

    if cp_size > 1:
        # With CP enabled: recreate TP, CP, and TP-CP groups
        dp_replica_start = dp_replica_id * dp_replica_size

        # Create new TP groups (one per CP slice in this DP replica)
        for cp_rank in range(cp_size):
            cp_slice_start = dp_replica_start + cp_rank * tp_base
            tp_group_ranks = [cp_slice_start + local_tp for local_tp in active_local_ranks]
            tp_group = dist.new_group(ranks=tp_group_ranks)

            if rank in tp_group_ranks:
                parallel_state._TENSOR_MODEL_PARALLEL_GROUP = tp_group
                parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
                parallel_state._MODEL_PARALLEL_GROUP = tp_group
                parallel_state._MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
                logger.info("[NTP] Rank %s created TP group: %s", rank, tp_group_ranks)

        # Create new CP groups (one per active TP position)
        for tp_rank_in_slice in active_local_ranks:
            cp_group_ranks = [
                dp_replica_start + tp_rank_in_slice + i * tp_base for i in range(cp_size)
            ]
            cp_group = dist.new_group(ranks=cp_group_ranks)

            if rank in cp_group_ranks:
                parallel_state._CONTEXT_PARALLEL_GROUP = cp_group
                parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS = cp_group_ranks
                logger.info("[NTP] Rank %s created CP group: %s", rank, cp_group_ranks)

        # Update TENSOR_AND_CONTEXT_PARALLEL_GROUP
        tp_rank_in_slice = local_rank_in_dp % tp_base
        if tp_rank_in_slice in active_local_ranks:
            tp_cp_group_ranks = []
            for cp_r in range(cp_size):
                for active_tp in active_local_ranks:
                    tp_cp_group_ranks.append(dp_replica_start + cp_r * tp_base + active_tp)
            tp_cp_group = dist.new_group(ranks=tp_cp_group_ranks)
            parallel_state._TENSOR_AND_CONTEXT_PARALLEL_GROUP = tp_cp_group
            logger.info("[NTP] Rank %s created TP-CP group: %s", rank, tp_cp_group_ranks)
        else:
            # Non-active (spare) rank - exit
            logger.info("[NTP] Rank %s is a spare rank with CP, exiting", rank)
            if exit_spares:
                sys.exit(0)
            return False
    else:
        # No CP: simpler case
        dp_replica_start = dp_replica_id * dp_replica_size
        tp_group_ranks = [dp_replica_start + local_tp for local_tp in active_local_ranks]

        if rank in tp_group_ranks:
            tp_group = dist.new_group(ranks=tp_group_ranks)
            parallel_state._TENSOR_MODEL_PARALLEL_GROUP = tp_group
            parallel_state._MODEL_PARALLEL_GROUP = tp_group
            parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
            parallel_state._MODEL_PARALLEL_GLOBAL_RANKS = tp_group_ranks
            logger.info("[NTP] Rank %s created TP group: %s", rank, tp_group_ranks)
        else:
            # Non-active (spare) rank - exit
            logger.info("[NTP] Rank %s is a spare rank, exiting", rank)
            if exit_spares:
                sys.exit(0)
            return False

    return True


# ======================================================================================
# Parameter Resharding for NTP
# ======================================================================================


def ntp_map(module: torch.nn.Module, ntp_config: NonuniformTPConfig, num_shards: int):
    """
    Initialize TP-sharded params with mapping between healthy and unhealthy TP sizes.

    Only healthy (full TP) ranks need send_splits and recv_splits to know how to reshard
    parameters when synchronizing with unhealthy (reduced TP) ranks.
    Unhealthy ranks synchronize directly without resharding.

    Args:
        module: Module containing parameters to initialize (e.g., self_attention or mlp)
        ntp_config: NTP configuration containing tp_base and tp_spares
        num_shards: Number of shards (e.g., num_attention_heads or ffn_hidden_size)
    """
    if ntp_config.tp_spares == 0:
        # No nonuniform TP, skip initialization
        return

    # Determine which ranks are active (non-spare) for the current DP rank
    rank = dist.get_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    logger.debug(
        f"[NTP] Rank {rank} [DP {dp_rank}, CP {cp_rank}, PP {pp_rank}] "
        f"ntp_map called with module={type(module).__name__}, num_shards={num_shards}"
    )

    # Reduced-TP ranks synchronize directly; full-TP ranks carry the reshard metadata.
    if _ntp_current_rank_is_reduced_dp(ntp_config):
        # This is an unhealthy rank with reduced TP - skip
        logger.debug(
            "[NTP] Rank %s [DP %s, CP %s, PP %s] Unhealthy rank, skipping",
            rank,
            dp_rank,
            cp_rank,
            pp_rank,
        )
        return

    # This is a healthy rank (full TP) - it needs send/recv splits to communicate
    # with unhealthy ranks that have reduced TP
    logger.debug(
        "[NTP] Rank %s [DP %s] Setting up send/recv splits for healthy rank", rank, dp_rank
    )

    for param in module.parameters():
        # Handle both tensor parallel parameters and vocabulary-parallel parameters that only
        # carry partition_dim metadata.
        if (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (
            hasattr(param, 'partition_dim') and not hasattr(param, 'tensor_model_parallel')
        ):
            # For healthy ranks, compute send/recv splits for communication with unhealthy ranks
            # We need to know how to reshard to match the reduced TP size
            reduced_tp_size = ntp_config.tp_base - ntp_config.tp_spares

            shard_ids = torch.arange(num_shards)
            # Partitions for reduced TP (what unhealthy ranks have)
            sync_partitions = list(shard_ids.chunk(reduced_tp_size))

            # Full partitions for healthy ranks (tp_base ranks)
            comp_partitions = sync_partitions + [
                torch.empty(int(len(shard_ids) / ntp_config.tp_base), dtype=torch.int)
                for _ in range(ntp_config.tp_spares)
            ]

            # Build comp_2_sync: for spare positions, which reduced TP ranks do they map to
            comp_2_sync = [[] for _ in range(ntp_config.tp_base)]
            sync_part_idx = 0

            for spare_part_idx in range(reduced_tp_size, ntp_config.tp_base):
                for shard_part_idx in range(len(comp_partitions[spare_part_idx])):
                    # Take the last shard from the current reduced TP rank
                    comp_partitions[spare_part_idx][shard_part_idx] = comp_partitions[
                        sync_part_idx
                    ][-1]
                    comp_partitions[sync_part_idx] = comp_partitions[sync_part_idx][:-1]
                    comp_2_sync[spare_part_idx].append(sync_part_idx)
                    sync_part_idx = (sync_part_idx + 1) % reduced_tp_size

            # Compute param_splits: how many shards each rank sends to each other rank
            param_splits = [
                torch.bincount(torch.tensor(c2s, dtype=torch.int), minlength=ntp_config.tp_base)
                for c2s in comp_2_sync
            ]

            shard_size = int(param.shape[param.partition_dim] * ntp_config.tp_base / len(shard_ids))
            send_splits = [(p_split * shard_size).tolist() for p_split in param_splits]
            recv_splits = [
                [send_splits[send_idx][recv_idx] for send_idx in range(len(send_splits))]
                for recv_idx in range(ntp_config.tp_base)
            ]
            param.send_splits = send_splits
            param.recv_splits = recv_splits
            logger.debug(
                f"[NTP] Rank {rank} [DP {dp_rank}] Set send_splits and recv_splits "
                f"on parameter id={id(param)}, shape={param.shape}"
            )


def ntp_init(layer: torch.nn.Module, ntp_config: NonuniformTPConfig):
    """
    Initialize nonuniform TP mappings for a TransformerLayer.

    This should be called after the layer is created to set up the send_splits
    and recv_splits attributes on tensor-parallel parameters.

    Args:
        layer: TransformerLayer instance
        ntp_config: NTP configuration containing tp_base and tp_spares
    """
    if ntp_config.tp_spares == 0:
        # No nonuniform TP, skip initialization
        return

    # Initialize self-attention parameters
    if hasattr(layer, 'self_attention'):
        ntp_map(layer.self_attention, ntp_config, layer.self_attention.config.num_attention_heads)

    # Initialize MLP parameters
    if hasattr(layer, 'mlp'):
        ntp_map(layer.mlp, ntp_config, layer.mlp.config.ffn_hidden_size)


def _ntp_start_post_sync_grad_reshard(
    params: List[torch.nn.Parameter], ntp_config: NonuniformTPConfig
):
    """Launch async all-to-all that scatters reduced side grads back to extra ranks."""
    if ntp_config.tp_spares == 0 or not _ntp_can_query_parallel_state():
        return []
    if _ntp_current_rank_is_reduced_dp(ntp_config):
        return []
    if parallel_state.get_tensor_model_parallel_world_size() != ntp_config.tp_base:
        return []

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    reduced_tp_size = ntp_config.tp_base - ntp_config.tp_spares
    handles = []

    for param in params:
        if not _ntp_param_can_reshard(param):
            continue

        if tp_rank < reduced_tp_size:
            if not hasattr(param, 'side_grad') or param.side_grad is None:
                raise RuntimeError(
                    "NTP core rank is missing side_grad storage for a tensor-parallel param"
                )
            input_tensors = [
                _ntp_empty_like_partition(param, dtype=param.side_grad.dtype)
                for _ in range(reduced_tp_size)
            ] + _ntp_split_for_all_to_all(
                param.side_grad,
                param.recv_splits[tp_rank][-ntp_config.tp_spares :],
                param.partition_dim,
            )
            output_tensors = [
                _ntp_empty_like_partition(param, dtype=param.side_grad.dtype)
                for _ in range(ntp_config.tp_base)
            ]
        else:
            input_tensors = [
                _ntp_empty_like_partition(param, dtype=param.main_grad.dtype)
                for _ in range(ntp_config.tp_base)
            ]
            output_tensors = _ntp_split_views_for_all_to_all(
                param.main_grad, param.send_splits[tp_rank], param.partition_dim
            )

        handles.append(
            all_to_all_with_output_views(
                output_tensors, input_tensors, group=tp_group, async_op=True
            )
        )

    return handles


# ======================================================================================
# NTP-aware ParamAndGradBuffer
# ======================================================================================


class NonuniformTPParamAndGradBucketGroup(_ParamAndGradBucketGroup):
    """
    NTP-aware version of _ParamAndGradBucketGroup.
    Skips gradient synchronization for spare GPUs.
    """

    def __init__(self, *args, ntp_config: Optional[NonuniformTPConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_nonuniform_tp(ntp_config)

    def configure_nonuniform_tp(self, ntp_config: Optional[NonuniformTPConfig] = None):
        """Attach NTP runtime state to an existing bucket group."""
        self.ntp_config = ntp_config or NonuniformTPConfig()
        self.ntp_post_sync_state = None

    def _wait_ntp_reshard_handles(self):
        """Wait for NTP all-to-all reshard work touching params in this bucket group."""
        for bucket in self.buckets:
            for param in bucket.params:
                handle = getattr(param, 'ntp_reshard_handle', None)
                if handle is not None:
                    handle.wait()
                    param.ntp_reshard_handle = None

    def _start_ntp_post_sync_reshard(self):
        """Start async post-DP-sync gradient reshard for this bucket group."""
        handles = []
        for bucket in self.buckets:
            handles.extend(_ntp_start_post_sync_grad_reshard(bucket.params_list, self.ntp_config))
        return handles

    def _record_ntp_post_sync_handles(self, handles):
        """Track post-sync handles and wait for all of them at the last bucket group."""
        record_post_sync_handles(self, 'ntp_post_sync_state', handles)

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Start DP grad sync after any pending NTP reshard for this bucket is complete."""
        self._wait_ntp_reshard_handles()
        if not _ntp_current_rank_should_dp_sync(self.ntp_config):
            self.grad_reduce_handle = None
            return
        return super().start_grad_sync(force_all_reduce=force_all_reduce)

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Finish DP grad sync and launch async post-sync NTP gradient reshard."""
        self.param_gather_dispatched = False
        if self.ddp_config.overlap_grad_reduce and self.grad_reduce_finished:
            return
        self._wait_ntp_reshard_handles()
        if not _ntp_current_rank_should_dp_sync(self.ntp_config):
            handles = self._start_ntp_post_sync_reshard()
            self._record_ntp_post_sync_handles(handles)
            if self.ddp_config.overlap_grad_reduce:
                self.grad_reduce_finished = True
            return
        result = super().finish_grad_sync(force_all_reduce=force_all_reduce)
        handles = self._start_ntp_post_sync_reshard()
        self._record_ntp_post_sync_handles(handles)
        return result

    def register_grad_ready(
        self, param: torch.nn.Parameter, force_all_reduce: Optional[bool] = False
    ):
        """Skip DP-ready bookkeeping on ranks that are folded into core TP ranks."""
        if not _ntp_current_rank_should_dp_sync(self.ntp_config):
            return
        return super().register_grad_ready(param, force_all_reduce=force_all_reduce)


class NonuniformTPParamAndGradBuffer(_ParamAndGradBuffer):
    """Native parameter buffer with additional NTP side-gradient ranges."""

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params_with_names: List[Tuple[torch.nn.Parameter, str]],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: Optional[int],
        param_to_name: Dict[torch.nn.Parameter, str],
        gradient_scaling_factor: float,
        param_indices: List[int],
        nccl_ub: bool,
        pg_collection: Optional[ProcessGroupCollection] = None,
        param_layout: Optional[optimizer_param_layout.PerBufferParamLayout] = None,
        *,
        ntp_config: Optional[NonuniformTPConfig] = None,
    ) -> None:
        self.ntp_config = ntp_config or NonuniformTPConfig()
        params = [param for param, _ in params_with_names]

        if self.ntp_config.tp_spares > 0:
            if param_layout is not None:
                raise RuntimeError(
                    "Nonuniform TP cannot combine its expanded-gradient layout with a supplied "
                    "parameter layout"
                )
            param_layout = _compute_ntp_per_buffer_param_layout(
                params,
                bucket_size,
                data_parallel_group.size(),
                ddp_config,
                self.ntp_config,
                param_indices,
            )

        self._ntp_side_grad_index_map = (
            param_layout.side_grad_index_map
            if isinstance(param_layout, PerBufferParamLayout)
            else {}
        )
        super().__init__(
            ddp_config,
            param_dtype,
            grad_dtype,
            params_with_names,
            data_parallel_group,
            bucket_size,
            param_to_name,
            gradient_scaling_factor,
            param_indices,
            nccl_ub,
            pg_collection,
            param_layout=param_layout,
        )

        for param in self.params:
            side_range = self._ntp_side_grad_index_map.get(param)
            if side_range is None:
                continue
            side_start, side_end, _ = side_range
            side_shape = list(param.data.shape)
            side_shape[param.partition_dim] = _ntp_extra_partition_dim(param, self.ntp_config)
            assert torch.Size(side_shape).numel() == side_end - side_start
            param.side_grad = self.grad_data[side_start:side_end].view(side_shape)


# ======================================================================================
# NTP-aware DistributedDataParallel
# ======================================================================================


class NonuniformTPDistributedDataParallel(DistributedDataParallel):
    """
    NTP-aware version of DistributedDataParallel.
    Adds gradient synchronization logic for spare GPUs.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        ntp_config: Optional[NonuniformTPConfig] = None,
        full_param_layout: Optional[optimizer_param_layout.FullParamLayout] = None,
    ):
        self.ntp_config = ntp_config or NonuniformTPConfig()
        if self.ntp_config.tp_spares > 0 and (
            getattr(config, "gtp_weight_remat_size", 1) != 1
            or getattr(config, "expert_gtp_weight_remat_size", 1) != 1
        ):
            raise RuntimeError("Nonuniform TP does not support GTP/EGTP rematerialization")
        if self.ntp_config.tp_spares > 0 and ddp_config.use_distributed_optimizer:
            raise RuntimeError("Nonuniform TP does not support the distributed optimizer")
        if self.ntp_config.tp_spares > 0 and ddp_config.reduce_scatter_with_fp32_accumulation:
            raise RuntimeError(
                "Nonuniform TP does not yet support FP32-accumulating reduce-scatter"
            )

        def _call_parent_init():
            super(NonuniformTPDistributedDataParallel, self).__init__(
                config=config,
                ddp_config=ddp_config,
                module=module,
                disable_bucketing=disable_bucketing,
                pg_collection=pg_collection,
                full_param_layout=full_param_layout,
            )

        # Use NTP-aware buffer class
        if self.ntp_config.tp_spares > 0:
            # DDP imports _ParamAndGradBuffer into its module namespace, so patch that binding
            # while the parent constructor allocates buffers.
            buffer_cls = functools.partial(
                NonuniformTPParamAndGradBuffer, ntp_config=self.ntp_config
            )
            with patch_ddp_param_and_grad_buffer(buffer_cls):
                _call_parent_init()
            self._wrap_bucket_groups_for_ntp()
        else:
            _call_parent_init()

        self._optimizer_param_group_alignment_group = (
            self.ntp_config.optimizer_param_group_alignment_group
        )

    def _wrap_bucket_groups_for_ntp(self):
        """Replace DDP bucket groups with NTP-aware groups and rebuild param lookup."""

        def configure(bucket_group):
            bucket_group.configure_nonuniform_tp(self.ntp_config)

        self.param_to_bucket_group = {}
        self.bucket_groups = wrap_bucket_groups_with_subclass(
            self.bucket_groups,
            NonuniformTPParamAndGradBucketGroup,
            configure,
            self.param_to_bucket_group,
        )
        self.expert_parallel_bucket_groups = wrap_bucket_groups_with_subclass(
            self.expert_parallel_bucket_groups,
            NonuniformTPParamAndGradBucketGroup,
            configure,
            self.param_to_bucket_group,
        )

        all_bucket_groups = self.bucket_groups + self.expert_parallel_bucket_groups
        configure_post_sync_handle_tracker(all_bucket_groups, 'ntp_post_sync_state')
        self._rebind_nonuniform_tp_param_ready_callbacks()

    def _rebind_nonuniform_tp_param_ready_callbacks(self) -> None:
        """Point current-main readiness callbacks at cloned NTP bucket groups."""
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            callback = (
                _BucketParamReadyCallback(self, bucket_group)
                if self.ddp_config.overlap_param_gather
                else None
            )
            for bucket in bucket_group.buckets:
                for param in bucket.params_list:
                    if callback is not None:
                        setattr(param, PARAM_READY_CALLBACK_ATTR, callback)
                    elif hasattr(param, PARAM_READY_CALLBACK_ATTR):
                        delattr(param, PARAM_READY_CALLBACK_ATTR)

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """
        Override to add NTP gradient synchronization between spare and core GPUs.
        """

        def ntp_hook(*unused):
            if is_graph_capturing():
                return

            bucket_group = self.param_to_bucket_group.get(param)
            is_last_microbatch = bucket_group is None or bucket_group.is_last_microbatch
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

            # Add NTP-specific logic
            if (
                self.ntp_config.tp_spares > 0
                and _ntp_param_can_reshard(param)
                and is_last_microbatch
                and not _ntp_current_rank_is_reduced_dp(self.ntp_config)
                and parallel_state.get_tensor_model_parallel_world_size() == self.ntp_config.tp_base
            ):
                tp_rank = parallel_state.get_tensor_model_parallel_rank()
                reduced_tp_size = self.ntp_config.tp_base - self.ntp_config.tp_spares

                if tp_rank < reduced_tp_size:
                    # Core GPU: receive grads from spare GPUs
                    input = [
                        _ntp_empty_like_partition(param, dtype=param.side_grad.dtype)
                        for _ in range(parallel_state.get_tensor_model_parallel_world_size())
                    ]
                    # Split side_grad and send to core GPUs
                    output = [
                        _ntp_empty_like_partition(param, dtype=param.side_grad.dtype)
                        for _ in range(reduced_tp_size)
                    ] + _ntp_split_views_for_all_to_all(
                        param.side_grad, param.recv_splits[tp_rank], dim=param.partition_dim
                    )[
                        -self.ntp_config.tp_spares :
                    ]
                else:
                    # Spare GPU: send grads to core GPUs
                    input = _ntp_split_for_all_to_all(
                        param.main_grad, param.send_splits[tp_rank], dim=param.partition_dim
                    )
                    output = [
                        _ntp_empty_like_partition(param, dtype=param.main_grad.dtype)
                        for _ in range(parallel_state.get_tensor_model_parallel_world_size())
                    ]

                try:
                    handle = all_to_all_with_output_views(
                        output,
                        input,
                        group=parallel_state.get_tensor_model_parallel_group(),
                        async_op=True,
                    )
                    param.ntp_reshard_handle = handle
                except Exception as e:
                    logger.error("[NTP] Rank %s all_to_all error: %s", tp_rank, e)
                    input_contiguity = [i.is_contiguous() for i in input]
                    output_contiguity = [o.is_contiguous() for o in output]
                    logger.error(
                        "[NTP] Rank %s input element contiguity: %s", tp_rank, input_contiguity
                    )
                    logger.error(
                        "[NTP] Rank %s output element contiguity: %s", tp_rank, output_contiguity
                    )
                    raise e

            if param in self.param_to_bucket_group and self.ddp_config.overlap_grad_reduce:
                self.param_to_bucket_group[param].register_grad_ready(param, self.force_all_reduce)

        return ntp_hook
