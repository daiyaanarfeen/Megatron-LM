# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Topology, expert-placement, and process-group helpers for opt-in nonuniform EP."""

import math
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence

from .. import parallel_state

_NONUNIFORM_EP_RUNTIME_CONFIG: Optional[dict] = None


def set_nonuniform_ep_runtime_config(runtime_config: Optional[dict]) -> None:
    """Register opt-in NEP runtime metadata for forward token routing."""
    global _NONUNIFORM_EP_RUNTIME_CONFIG
    _NONUNIFORM_EP_RUNTIME_CONFIG = dict(runtime_config) if runtime_config is not None else None


def get_nonuniform_ep_runtime_config() -> Optional[dict]:
    """Return the opt-in NEP runtime metadata registered by the entrypoint."""
    return _NONUNIFORM_EP_RUNTIME_CONFIG


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
    expert_placement: Optional[Sequence[Sequence[int]]], num_experts: Optional[int] = None
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


def build_expert_axis_permutation(
    expert_placement: Optional[Sequence[Sequence[int]]], num_experts: int
) -> Optional[List[int]]:
    """Return logical expert IDs in physical EP-rank/local-slot order."""
    if expert_placement is None:
        return None
    build_expert_to_ep_rank_map(expert_placement, num_experts)
    return [int(expert_id) for expert_ids in expert_placement for expert_id in expert_ids]


def get_nonuniform_ep_expert_axis_permutation(num_experts: int) -> Optional[List[int]]:
    """Return the placement-aware expert-axis permutation, if NEP is active."""
    runtime_config = get_nonuniform_ep_runtime_config()
    if runtime_config is None:
        return None
    return build_expert_axis_permutation(runtime_config.get('expert_placement'), num_experts)


def compute_nonuniform_ep_owner_expert_slots(
    num_experts: int, min_ep_size: int
) -> List[List[Optional[int]]]:
    """Return balanced logical expert IDs in fixed-width owner slots.

    The first ``num_experts % min_ep_size`` owners receive one additional
    logical expert.  Every owner row is padded with ``None`` to
    ``ceil(num_experts / min_ep_size)`` slots so communication layouts can stay
    uniform without creating dummy expert parameters.
    """
    if min_ep_size <= 0:
        raise RuntimeError(f"min_ep_size must be positive; got {min_ep_size}")
    if num_experts < min_ep_size:
        raise RuntimeError(
            "NEP requires at least one logical expert per minimum-EP rank; "
            f"got num_experts={num_experts}, min_ep_size={min_ep_size}"
        )

    base_count, remainder = divmod(num_experts, min_ep_size)
    slots_per_owner = math.ceil(num_experts / min_ep_size)
    owner_slots = []
    next_expert_id = 0
    for owner_ep_rank in range(min_ep_size):
        logical_count = base_count + (1 if owner_ep_rank < remainder else 0)
        slots = list(range(next_expert_id, next_expert_id + logical_count))
        slots.extend([None] * (slots_per_owner - logical_count))
        owner_slots.append(slots)
        next_expert_id += logical_count
    return owner_slots


def compute_nonuniform_ep_dispatch_slots(
    expert_placement: Sequence[Sequence[int]], num_experts: int
) -> List[List[Optional[int]]]:
    """Pad one replica's logical placement to a uniform Flex-dispatch width."""
    if not expert_placement:
        raise RuntimeError("NEP expert placement must contain at least one EP rank")
    slots_per_rank = math.ceil(num_experts / len(expert_placement))
    dispatch_slots = []
    for ep_rank, expert_ids in enumerate(expert_placement):
        if len(expert_ids) > slots_per_rank:
            raise RuntimeError(
                f"EP rank {ep_rank} owns {len(expert_ids)} experts, exceeding the "
                f"virtual dispatch width {slots_per_rank}"
            )
        slots = [int(expert_id) for expert_id in expert_ids]
        slots.extend([None] * (slots_per_rank - len(slots)))
        dispatch_slots.append(slots)
    return dispatch_slots


def compute_nonuniform_ep_expert_placement(
    num_experts: int,
    local_ep_size: int,
    min_ep_size: int,
    preferred_follower_fanout: Optional[int] = None,
) -> List[List[int]]:
    """Compute logical expert placement for a nonuniform EP replica.

    Divisible configurations retain the established striped/round-robin layout.
    For non-divisible configurations, minimum-EP owners receive balanced
    floor/ceiling expert counts and wider replicas offload only the excess owner
    experts to follower ranks.  The model contains only logical experts; uniform
    virtual slots are built separately for fused token dispatch and owner-layout
    communication.
    """
    if local_ep_size < min_ep_size:
        raise RuntimeError(
            f"local_ep_size ({local_ep_size}) must be >= min_ep_size ({min_ep_size})"
        )
    if num_experts < local_ep_size:
        raise RuntimeError(
            "NEP currently requires at least one logical expert per local EP rank; "
            f"got num_experts={num_experts}, local_ep_size={local_ep_size}"
        )
    if preferred_follower_fanout is not None and preferred_follower_fanout <= 0:
        raise RuntimeError("preferred_follower_fanout must be positive")

    owner_slots = compute_nonuniform_ep_owner_expert_slots(num_experts, min_ep_size)
    owner_expert_ids = [
        [expert_id for expert_id in slots if expert_id is not None] for slots in owner_slots
    ]

    if num_experts % local_ep_size != 0 or num_experts % min_ep_size != 0:
        base_count, remainder = divmod(num_experts, local_ep_size)
        target_counts = [
            base_count + (1 if ep_rank < remainder else 0) for ep_rank in range(local_ep_size)
        ]
        placement = [[] for _ in range(local_ep_size)]
        offloaded_experts = []
        for owner_ep_rank, expert_ids in enumerate(owner_expert_ids):
            keep_count = target_counts[owner_ep_rank]
            if keep_count > len(expert_ids):
                raise RuntimeError(
                    f"Owner rank {owner_ep_rank} target count {keep_count} exceeds its "
                    f"logical owner count {len(expert_ids)}"
                )
            placement[owner_ep_rank].extend(expert_ids[:keep_count])
            offloaded_experts.extend(expert_ids[keep_count:])

        next_offloaded = 0
        for follower_ep_rank in range(min_ep_size, local_ep_size):
            follower_count = target_counts[follower_ep_rank]
            placement[follower_ep_rank].extend(
                offloaded_experts[next_offloaded : next_offloaded + follower_count]
            )
            next_offloaded += follower_count
        if next_offloaded != len(offloaded_experts):
            raise RuntimeError(
                "NEP balanced placement did not consume every offloaded expert: "
                f"used {next_offloaded}, available {len(offloaded_experts)}"
            )
    else:
        experts_per_rank = num_experts // local_ep_size
        experts_per_owner = num_experts // min_ep_size
        experts_offloaded = experts_per_owner - experts_per_rank
        num_followers = local_ep_size - min_ep_size

        placement = [[] for _ in range(local_ep_size)]
        for owner_rank in range(min_ep_size):
            start = owner_rank * experts_per_owner
            placement[owner_rank] = list(range(start, start + experts_per_rank))

        striped_assignment = None
        if preferred_follower_fanout is not None and num_followers > 0 and experts_offloaded > 0:
            minimum_fanout = max(
                preferred_follower_fanout, math.ceil(experts_offloaded / experts_per_rank)
            )
            maximum_fanout = min(experts_offloaded, num_followers)
            for fanout in range(min(minimum_fanout, maximum_fanout), maximum_fanout + 1):
                if experts_offloaded % fanout != 0:
                    continue
                if min_ep_size * fanout % num_followers != 0:
                    continue
                expected_follower_degree = min_ep_size * fanout // num_followers
                chunk_size = experts_offloaded // fanout
                for stride in range(1, num_followers + 1):
                    follower_offsets_by_owner = [
                        [(owner_rank * stride + lane) % num_followers for lane in range(fanout)]
                        for owner_rank in range(min_ep_size)
                    ]
                    follower_degrees = [0] * num_followers
                    for follower_offsets in follower_offsets_by_owner:
                        for follower_offset in set(follower_offsets):
                            follower_degrees[follower_offset] += 1
                    if all(
                        len(set(follower_offsets)) == fanout
                        for follower_offsets in follower_offsets_by_owner
                    ) and all(degree == expected_follower_degree for degree in follower_degrees):
                        striped_assignment = (follower_offsets_by_owner, chunk_size)
                        break
                if striped_assignment is not None:
                    break

        if striped_assignment is not None:
            follower_offsets_by_owner, chunk_size = striped_assignment
            for owner_rank, follower_offsets in enumerate(follower_offsets_by_owner):
                offload_start = owner_rank * experts_per_owner + experts_per_rank
                for lane, follower_offset in enumerate(follower_offsets):
                    follower_rank = min_ep_size + follower_offset
                    chunk_start = offload_start + lane * chunk_size
                    placement[follower_rank].extend(range(chunk_start, chunk_start + chunk_size))
        elif num_followers > 0 and experts_offloaded > 0:
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
    build_expert_to_ep_rank_map(placement, num_experts)

    return placement


class NonuniformEPRankGenerator:
    """Generate process groups for opt-in NEP replicas with different EP sizes."""

    _ATTENTION_KEYS = {'tp', 'cp', 'dp', 'dp-cp', 'tp-dp', 'tp-dp-cp', 'tp-cp'}
    _EXPERT_KEYS = {'etp', 'ep', 'etp-ep', 'edp'}

    def __init__(
        self, tp: int, cp: int, num_tp_cp_per_replica: Sequence[int], etp: Optional[int] = None
    ) -> None:
        self.tp = tp
        self.cp = cp
        self.tp_cp = tp * cp
        self.etp = etp if etp is not None else tp
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
            start = self.replica_offsets[replica_index]
            groups.append(list(range(start, start + num_tp_cp * self.tp_cp)))
        return groups

    def _get_etp_ranks(self) -> List[List[int]]:
        groups = []
        for replica_index, num_tp_cp in enumerate(self.num_tp_cp_per_replica):
            start = self.replica_offsets[replica_index]
            ep_size = num_tp_cp * self.tp_cp // self.etp
            for ep_index in range(ep_size):
                groups.append(
                    list(range(start + ep_index * self.etp, start + (ep_index + 1) * self.etp))
                )
        return groups

    def _get_ep_ranks(self) -> List[List[int]]:
        groups = []
        for replica_index, num_tp_cp in enumerate(self.num_tp_cp_per_replica):
            start = self.replica_offsets[replica_index]
            block = list(range(start, start + num_tp_cp * self.tp_cp))
            for etp_position in range(self.etp):
                groups.append(block[etp_position :: self.etp])
        return groups

    def _get_edp_ranks(self) -> List[List[int]]:
        groups = []
        for position in range(self.min_k * self.tp_cp):
            groups.append(
                [
                    self.replica_offsets[replica_index] + position
                    for replica_index in range(self.num_replicas)
                ]
            )
        return groups


def load_nonuniform_nccl_communicator_configs(path: Optional[str]) -> Dict[str, Any]:
    """Load optional NCCL process-group tuning from a YAML file."""
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


def create_nonuniform_process_group(
    ranks: Sequence[int],
    timeout: timedelta,
    nccl_comm_cfgs: Dict[str, Any],
    desc: str,
    backend: Optional[str] = None,
) -> Any:
    """Create a nonuniform process group with Megatron's native NCCL options."""
    pg_options = (
        None if backend == "gloo" else parallel_state.get_nccl_options(desc, nccl_comm_cfgs)
    )
    return parallel_state.create_group(
        ranks, timeout=timeout, backend=backend, pg_options=pg_options, group_desc=desc
    )


def set_nonuniform_parallel_state_attr(name: str, value: Any) -> None:
    """Set process-group state owned by Megatron's parallel-state module."""
    setattr(parallel_state, name, value)


def initialize_nonuniform_attention_process_groups(
    *,
    generator: Any,
    rank: int,
    world_size: int,
    timeout: timedelta,
    nccl_comm_cfgs: Dict[str, Any],
    create_gloo_process_groups: bool,
    get_embedding_ranks: Callable[[List[int]], List[int]],
    get_position_embedding_ranks: Callable[[List[int]], List[int]],
) -> None:
    """Create the shared attention/model process groups for a nonuniform topology."""
    # Nonuniform topologies currently support only the inactive GTP/EGTP case. Main still
    # expects both singleton groups to exist, even when their configured sizes are one.
    for global_rank in range(world_size):
        ranks = [global_rank]
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "gtp_remat")
        if rank == global_rank:
            set_nonuniform_parallel_state_attr("_GTP_WEIGHT_REMAT_GROUP", group)
            set_nonuniform_parallel_state_attr("_GTP_WEIGHT_REMAT_GLOBAL_RANKS", ranks)

    for global_rank in range(world_size):
        ranks = [global_rank]
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "expt_gtp_remat")
        if rank == global_rank:
            set_nonuniform_parallel_state_attr("_EXPERT_GTP_WEIGHT_REMAT_GROUP", group)
            set_nonuniform_parallel_state_attr("_EXPERT_GTP_WEIGHT_REMAT_GLOBAL_RANKS", ranks)

    for ranks in generator.get_ranks("dp-cp"):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "dp_cp")
        group_gloo = (
            create_nonuniform_process_group(
                ranks, timeout, nccl_comm_cfgs, "DATA_PARALLEL_GROUP_WITH_CP_GLOO", "gloo"
            )
            if create_gloo_process_groups
            else None
        )
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_DATA_PARALLEL_GROUP_WITH_CP", group)
            set_nonuniform_parallel_state_attr("_DATA_PARALLEL_GROUP_WITH_CP_GLOO", group_gloo)
            set_nonuniform_parallel_state_attr("_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP", ranks)
            set_nonuniform_parallel_state_attr("_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP", group)
            set_nonuniform_parallel_state_attr(
                "_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO", group_gloo
            )

    for ranks in generator.get_ranks("dp"):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "dp")
        group_gloo = (
            create_nonuniform_process_group(
                ranks, timeout, nccl_comm_cfgs, "DATA_PARALLEL_GROUP_GLOO", "gloo"
            )
            if create_gloo_process_groups
            else None
        )
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_DATA_PARALLEL_GROUP", group)
            set_nonuniform_parallel_state_attr("_DATA_PARALLEL_GROUP_GLOO", group_gloo)
            set_nonuniform_parallel_state_attr("_DATA_PARALLEL_GLOBAL_RANKS", ranks)

    # GTP is inactive, so main's full data-distribution groups are exact aliases of the
    # replicate groups above. Populate the aliases because current getters default to the
    # GTP-inclusive variants.
    set_nonuniform_parallel_state_attr(
        "_DATA_PARALLEL_GROUP_WITH_GTP_REMAT", parallel_state._DATA_PARALLEL_GROUP
    )
    set_nonuniform_parallel_state_attr(
        "_DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT", parallel_state._DATA_PARALLEL_GROUP_WITH_CP
    )
    set_nonuniform_parallel_state_attr(
        "_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT",
        parallel_state._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP,
    )

    for ranks in generator.get_ranks("cp"):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "cp")
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_CONTEXT_PARALLEL_GROUP", group)
            set_nonuniform_parallel_state_attr("_CONTEXT_PARALLEL_GLOBAL_RANKS", ranks)

    for ranks in generator.get_ranks("tp"):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "tp")
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_TENSOR_MODEL_PARALLEL_GROUP", group)
            set_nonuniform_parallel_state_attr("_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS", ranks)

    for ranks in generator.get_ranks("tp"):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "mp")
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_MODEL_PARALLEL_GROUP", group)
            set_nonuniform_parallel_state_attr("_MODEL_PARALLEL_GLOBAL_RANKS", ranks)

    for ranks in [[global_rank] for global_rank in range(world_size)]:
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "pp")
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_PIPELINE_MODEL_PARALLEL_GROUP", group)
            set_nonuniform_parallel_state_attr("_PIPELINE_GLOBAL_RANKS", ranks)

        embedding_ranks = get_embedding_ranks(ranks)
        embedding_group = create_nonuniform_process_group(
            embedding_ranks, timeout, nccl_comm_cfgs, "embd"
        )
        if rank in embedding_ranks:
            set_nonuniform_parallel_state_attr("_EMBEDDING_GROUP", embedding_group)
            set_nonuniform_parallel_state_attr("_EMBEDDING_GLOBAL_RANKS", embedding_ranks)

        position_embedding_ranks = get_position_embedding_ranks(ranks)
        position_embedding_group = create_nonuniform_process_group(
            position_embedding_ranks, timeout, nccl_comm_cfgs, "pos_embd"
        )
        if rank in position_embedding_ranks:
            set_nonuniform_parallel_state_attr(
                "_POSITION_EMBEDDING_GROUP", position_embedding_group
            )
            set_nonuniform_parallel_state_attr(
                "_POSITION_EMBEDDING_GLOBAL_RANKS", position_embedding_ranks
            )

    for ranks in generator.get_ranks("tp-dp-cp"):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "tp_dp_cp")
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP", group)

    for ranks in generator.get_ranks("tp-dp"):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "tp_dp")
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_TENSOR_AND_DATA_PARALLEL_GROUP", group)

    for ranks in generator.get_ranks("tp-cp"):
        group = create_nonuniform_process_group(ranks, timeout, nccl_comm_cfgs, "tp_cp")
        if rank in ranks:
            set_nonuniform_parallel_state_attr("_TENSOR_AND_CONTEXT_PARALLEL_GROUP", group)


def initialize_nonuniform_expert_gtp_aliases() -> None:
    """Populate current-main expert aliases for an inactive EGTP axis."""
    set_nonuniform_parallel_state_attr(
        "_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP_WITH_EGTP",
        parallel_state._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP,
    )
    set_nonuniform_parallel_state_attr(
        "_EXPERT_DATA_PARALLEL_GROUP_WITH_GTP_REMAT", parallel_state._EXPERT_DATA_PARALLEL_GROUP
    )
    set_nonuniform_parallel_state_attr(
        "_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_WITH_GTP_REMAT",
        parallel_state._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP,
    )
