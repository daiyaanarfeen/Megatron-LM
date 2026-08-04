# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.nonuniform_common import (
    build_expert_axis_permutation,
    build_expert_to_ep_rank_map,
    clear_nonuniform_ep_runtime_config,
    compute_nonuniform_ep_expert_placement,
    get_nonuniform_ep_expert_axis_permutation,
    get_nonuniform_ep_expert_to_ep_rank_map,
    get_nonuniform_ep_local_expert_indices,
    set_nonuniform_ep_runtime_config,
)
from megatron.core.distributed.nonuniform_ep import (
    NonuniformEPApproach,
    NonuniformEPDistributedDataParallel,
    NonuniformEPNCCLParamAndGradBucketGroup,
    _build_nep_nccl_scatter_chunk_ranges,
    _configure_nep_edp_ready_gate,
    _ExpertBucketSpec,
    _get_nep_nccl_scatter_chunks,
    _group_expert_bucket_specs_in_backward_order,
    _nep_benchmark_skip_owner_grad_check_enabled,
    _nep_owner_ddp_config,
    _partition_expert_bucket_specs,
    _source_ep_ranks_for_owner,
    _zero_sm_transfer_ranks_by_owner,
)
from megatron.core.transformer.cuda_graphs import _CudagraphReplayNode, _GraphStatus
from megatron.core.transformer.moe.moe_layer import BaseMoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher


class _FakeWork:
    def __init__(self):
        self.block_calls = 0
        self.wait_calls = 0

    def block_current_stream(self):
        self.block_calls += 1

    def wait(self):
        self.wait_calls += 1


def test_nep_scatter_work_defers_stream_dependency_until_train_is_submitted():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    work = _FakeWork()
    state = {"buffer_slot_handles": {}}
    slot_key = ("slot", 0)
    bucket_group._nep_nccl_async_handles = []
    bucket_group._get_nep_nccl_shared_buffer_state = lambda: state

    bucket_group._record_nep_nccl_work(work, slot_key, block_current_stream=False)

    assert work.block_calls == 0
    assert bucket_group._nep_nccl_async_handles == [work]
    assert state["buffer_slot_handles"][slot_key] == [work]

    descriptor = {"kind": "all_to_all", "submitted": True, "work": work}
    bucket_group._order_nep_nccl_owner_all_to_all_scatter_completion(descriptor)

    assert work.block_calls == 1
    assert descriptor["completion_ordered"]


def test_nep_benchmark_skip_owner_grad_check_is_opt_in(monkeypatch):
    monkeypatch.delenv("MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK", raising=False)
    assert not _nep_benchmark_skip_owner_grad_check_enabled()

    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK", "1")
    assert _nep_benchmark_skip_owner_grad_check_enabled()

    config = DistributedDataParallelConfig(
        check_for_nan_in_grad=True, check_for_large_grads=True, num_buckets=16
    )
    config.bucket_size = 94_486_908
    native_config = _nep_owner_ddp_config(config)

    assert native_config is not config
    assert not native_config.check_for_nan_in_grad
    assert not native_config.check_for_large_grads
    assert native_config.num_buckets == 16
    assert native_config.bucket_size == 94_486_908
    assert config.check_for_nan_in_grad
    assert config.check_for_large_grads


class _FakeDenseBucketGroup:
    def __init__(self, numel):
        self.is_first_batch = True
        self.grad_reduce_handle = None
        self.params = [object()]
        self.per_param_grad_ready_counts = {self.params[0]: 1}
        self.golden_per_param_grad_ready_counts = {}
        self.buckets = [type("Bucket", (), {"grad_data": torch.empty(numel)})()]
        self.start_calls = []

    def start_grad_sync(self, force_all_reduce=False):
        self.start_calls.append(force_all_reduce)
        self.grad_reduce_handle = object()


def _make_expert_bucket_spec(layer: int, expert_id: int) -> _ExpertBucketSpec:
    return _ExpertBucketSpec(
        buffer=None,
        source_bucket_index=0,
        expert_id=expert_id,
        params=[],
        start=0,
        end=1,
        slot_key=(f"decoder.layers.{layer}.mlp.experts.local_experts.{{expert}}.weight",),
    )


def test_zero_sm_transfer_groups_share_followers_without_changing_sources():
    source_ranks_by_owner = {
        owner_ep_rank: [owner_ep_rank, 12 + owner_ep_rank % 4] for owner_ep_rank in range(12)
    }

    transfer_ranks_by_owner = _zero_sm_transfer_ranks_by_owner(
        source_ranks_by_owner, min_ep_size=12
    )

    for lane in range(4):
        owner_ep_ranks = [lane, lane + 4, lane + 8]
        for owner_index, owner_ep_rank in enumerate(owner_ep_ranks):
            helper_ep_rank = owner_ep_ranks[(owner_index + 1) % len(owner_ep_ranks)]
            expected_transfer_ranks = [owner_ep_rank, helper_ep_rank, lane + 12]
            assert transfer_ranks_by_owner[owner_ep_rank] == expected_transfer_ranks
            assert source_ranks_by_owner[owner_ep_rank] == [owner_ep_rank, lane + 12]


def test_zero_sm_transfer_groups_balance_single_owner_followers():
    source_ranks_by_owner = {
        owner_ep_rank: [owner_ep_rank, owner_ep_rank + 4] for owner_ep_rank in range(4)
    }

    transfer_ranks_by_owner = _zero_sm_transfer_ranks_by_owner(source_ranks_by_owner, min_ep_size=4)

    assert transfer_ranks_by_owner == {
        owner_ep_rank: [owner_ep_rank, (owner_ep_rank + 1) % 4, owner_ep_rank + 4]
        for owner_ep_rank in range(4)
    }


def test_zero_sm_transfer_group_includes_synthetic_owner():
    source_ranks_by_owner = {0: [2, 3], 1: [4, 5]}

    transfer_ranks_by_owner = _zero_sm_transfer_ranks_by_owner(source_ranks_by_owner, min_ep_size=2)

    assert transfer_ranks_by_owner == {0: [0, 2, 3], 1: [1, 4, 5]}


def test_zero_sm_synthetic_owner_launches_gather_and_scatter():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    bucket_group._nep_runtime_config = {"ep_rank": 0, "zero_sm_reshard": True}
    bucket_group._nep_nccl_group_index = 0
    bucket_group._nep_nccl_owner_source_ranks = lambda owner: [2, 3]
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [0, 2, 3]
    bucket_group._get_nep_nccl_transfer_group_info = lambda owner: (object(), 0, 3, [0, 2, 3])
    bucket_group._pack_nep_nccl_owner_chunk = lambda *args: calls.append("pack_owner")
    bucket_group._start_nep_nccl_owner_native_gather = lambda *args: calls.append("gather")
    bucket_group._start_nep_nccl_owner_native_scatter = lambda *args: calls.append("scatter")

    bucket_group._start_nep_nccl_owner_all_to_all_gather(0, 0, 0, 16, object(), (0,), async_op=True)
    bucket_group._start_nep_nccl_owner_all_to_all_scatter(
        0, 0, 0, 16, object(), (0,), async_op=True
    )

    assert calls == ["pack_owner", "gather", "scatter"]


def test_process_group_gather_selects_separate_owner_group():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []

    class GroupSelected(Exception):
        pass

    bucket_group._nep_runtime_config = {
        "ep_rank": 0,
        "zero_sm_reshard": False,
        "nep_owner_gather_groups": {0: object()},
    }
    bucket_group._nep_nccl_group_index = 0
    bucket_group._nep_nccl_owner_source_ranks = lambda owner: [0, 4]
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [0, 4]
    bucket_group._pack_nep_nccl_owner_chunk = lambda *args: None

    def select_group(owner, group_key):
        calls.append((owner, group_key))
        raise GroupSelected

    bucket_group._get_nep_nccl_transfer_group_info = select_group

    with pytest.raises(GroupSelected):
        bucket_group._start_nep_nccl_owner_all_to_all_gather(
            0, 0, 0, 16, object(), (0,), async_op=True
        )

    assert calls == [(0, "nep_owner_gather_groups")]


def test_zero_sm_transfer_group_finds_helper_outside_min_ep_ranks():
    source_ranks_by_owner = {0: [0, 1], 1: [2, 3]}

    transfer_ranks_by_owner = _zero_sm_transfer_ranks_by_owner(source_ranks_by_owner, min_ep_size=2)

    assert transfer_ranks_by_owner == {0: [0, 2, 1], 1: [1, 2, 3]}


def test_nonuniform_expert_placement_keeps_low_volume_round_robin_layout():
    placement, gather_map = compute_nonuniform_ep_expert_placement(128, 8, 4)

    assert placement[:4] == [
        list(range(0, 16)),
        list(range(32, 48)),
        list(range(64, 80)),
        list(range(96, 112)),
    ]
    assert placement[4] == [16, 20, 24, 28, 48, 52, 56, 60, 80, 84, 88, 92, 112, 116, 120, 124]
    assert [
        _source_ep_ranks_for_owner(placement, owner_rank, 128, 4) for owner_rank in range(4)
    ] == [[0, 4, 5, 6, 7], [1, 4, 5, 6, 7], [2, 4, 5, 6, 7], [3, 4, 5, 6, 7]]
    assert set(gather_map) == set(range(4, 8))
    assert {owner_rank for _, owner_rank, _ in gather_map[4]} == set(range(4))


def test_zero_sm_expert_placement_limits_balanced_follower_fanout():
    placement, gather_map = compute_nonuniform_ep_expert_placement(
        128, 8, 4, preferred_follower_fanout=2
    )

    assert placement[:4] == [
        list(range(0, 16)),
        list(range(32, 48)),
        list(range(64, 80)),
        list(range(96, 112)),
    ]
    assert placement[4:] == [
        list(range(16, 24)) + list(range(120, 128)),
        list(range(24, 32)) + list(range(48, 56)),
        list(range(56, 64)) + list(range(80, 88)),
        list(range(88, 96)) + list(range(112, 120)),
    ]
    assert [
        _source_ep_ranks_for_owner(placement, owner_rank, 128, 4) for owner_rank in range(4)
    ] == [[0, 4, 5], [1, 5, 6], [2, 6, 7], [3, 4, 7]]
    assert all(len(entries) == 16 for entries in placement)
    assert all(
        len({owner_rank for _, owner_rank, _ in gather_map[follower_rank]}) == 2
        for follower_rank in range(4, 8)
    )


def test_process_group_expert_placement_uses_disjoint_single_follower_groups():
    placement, gather_map = compute_nonuniform_ep_expert_placement(
        128, 8, 4, preferred_follower_fanout=1
    )

    assert placement == [
        list(range(0, 16)),
        list(range(32, 48)),
        list(range(64, 80)),
        list(range(96, 112)),
        list(range(16, 32)),
        list(range(48, 64)),
        list(range(80, 96)),
        list(range(112, 128)),
    ]
    assert [
        _source_ep_ranks_for_owner(placement, owner_rank, 128, 4) for owner_rank in range(4)
    ] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert all(
        len({owner_rank for _, owner_rank, _ in gather_map[follower_rank]}) == 1
        for follower_rank in range(4, 8)
    )


def test_ep64_ep48_round_robin_placement_limits_owner_source_fanout():
    placement, _ = compute_nonuniform_ep_expert_placement(192, 64, 48)

    assert placement[:2] == [list(range(0, 3)), list(range(4, 7))]
    assert placement[48] == [3, 67, 131]
    source_groups = [
        _source_ep_ranks_for_owner(placement, owner_rank, 192, 48) for owner_rank in range(48)
    ]
    assert max(map(len, source_groups)) == 2


class TestNonuniformEPTokenRouting:
    def teardown_method(self, _method):
        clear_nonuniform_ep_runtime_config()

    def test_physical_expert_axis_matches_round_robin_placement(self):
        placement, _ = compute_nonuniform_ep_expert_placement(8, 4, 2)

        assert placement == [[0, 1], [4, 5], [2, 6], [3, 7]]
        assert build_expert_axis_permutation(placement, 8) == [0, 1, 4, 5, 2, 6, 3, 7]
        assert build_expert_to_ep_rank_map(placement, 8) == [0, 0, 2, 3, 1, 1, 2, 3]

    def test_physical_expert_axis_reorders_alltoall_destination_chunks(self):
        dispatcher = MoEAlltoAllTokenDispatcher.__new__(MoEAlltoAllTokenDispatcher)
        dispatcher.expert_axis_permutation = torch.tensor([0, 1, 4, 5, 2, 6, 3, 7])
        logical_expert_ids = torch.repeat_interleave(torch.arange(8), torch.arange(1, 9))
        routing_map = logical_expert_ids[:, None] == torch.arange(8)[None, :]
        probs = torch.arange(routing_map.numel(), dtype=torch.float32).view_as(routing_map)

        physical_map, physical_probs = dispatcher._apply_expert_axis_permutation(routing_map, probs)

        assert physical_map.sum(dim=0).tolist() == [1, 2, 5, 6, 3, 7, 4, 8]
        assert physical_map.sum(dim=0).reshape(4, 2).sum(dim=1).tolist() == [3, 11, 10, 12]
        torch.testing.assert_close(physical_probs, probs[:, dispatcher.expert_axis_permutation])

    def test_runtime_config_exposes_local_and_physical_expert_order(self):
        placement = [[0, 1], [4, 5], [2, 6], [3, 7]]
        set_nonuniform_ep_runtime_config(
            {"local_expert_indices": placement[2], "expert_placement": placement}
        )

        assert get_nonuniform_ep_local_expert_indices() == [2, 6]
        assert get_nonuniform_ep_expert_axis_permutation(8) == [0, 1, 4, 5, 2, 6, 3, 7]
        assert get_nonuniform_ep_expert_to_ep_rank_map(8) == [0, 0, 2, 3, 1, 1, 2, 3]

    def test_expert_placement_rejects_duplicate_holders(self):
        with pytest.raises(RuntimeError, match="one physical holder"):
            build_expert_axis_permutation([[0, 1], [1, 2]], 3)

    def test_expert_placement_rejects_missing_experts(self):
        with pytest.raises(RuntimeError, match="must cover every global expert"):
            build_expert_axis_permutation([[0], [2]], 3)


def test_nep_nccl_slot_groups_preserve_backward_buffer_order():
    specs = [
        _make_expert_bucket_spec(layer, expert_id)
        for layer in (45, 44, 10, 9)
        for expert_id in (0, 1)
    ]

    grouped_specs = _group_expert_bucket_specs_in_backward_order(specs)

    assert [int(slot_key[0].split(".")[2]) for slot_key, _ in grouped_specs] == [45, 44, 10, 9]
    assert [[spec.expert_id for spec in group] for _, group in grouped_specs] == [
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
    ]


def test_nep_nccl_partitions_twelve_slots_into_three_backward_ordered_groups():
    specs = [
        _make_expert_bucket_spec(layer, expert_id)
        for layer in range(12, 0, -1)
        for expert_id in (0, 1)
    ]
    grouped_specs = _group_expert_bucket_specs_in_backward_order(specs)

    partitions = _partition_expert_bucket_specs(grouped_specs, 3)

    assert [len(partition) for partition in partitions] == [4, 4, 4]
    assert [
        [int(slot_key[0].split(".")[2]) for slot_key, _ in partition] for partition in partitions
    ] == [[12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]]


def test_nep_nccl_partition_does_not_split_moe_module_slots():
    grouped_specs = [
        ((f"decoder.layers.{layer}.mlp.experts.{slot}",), [])
        for layer in range(4, 0, -1)
        for slot in ("linear_fc2.weight", "linear_fc1.weight")
    ]

    partitions = _partition_expert_bucket_specs(grouped_specs, 3)

    assert [len(partition) for partition in partitions] == [4, 2, 2]
    layer_to_partition = {}
    for partition_index, partition in enumerate(partitions):
        for slot_key, _ in partition:
            layer = int(slot_key[0].split(".")[2])
            previous = layer_to_partition.setdefault(layer, partition_index)
            assert previous == partition_index


def test_nep_nccl_buffer_slot_reuse_does_not_block_host():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    slot_key = (0, 128, None, None)
    works = [_FakeWork(), _FakeWork()]
    bucket_group._nep_nccl_scheduler_state = {
        "gather_buf_cache": {},
        "buffer_slot_handles": {slot_key: works},
    }

    bucket_group._order_nep_nccl_buffer_slot(slot_key)

    assert all(work.block_calls == 1 for work in works)
    assert all(work.wait_calls == 0 for work in works)
    assert slot_key not in bucket_group._nep_nccl_scheduler_state["buffer_slot_handles"]


def test_nep_nccl_uses_parent_ddp_backward_hook_for_dense_params():
    assert "_make_backward_post_hook" not in NonuniformEPDistributedDataParallel.__dict__
    assert "_start_delayed_dense_grad_syncs" not in NonuniformEPDistributedDataParallel.__dict__


def test_nep_bucket_size_uses_full_replica_native_value(monkeypatch):
    config = SimpleNamespace(num_buckets=16, bucket_size=154_354_044)
    bucket_size = SimpleNamespace(item=lambda: 94_486_908)
    calls = []

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: bucket_size)
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, op, group: calls.append((tensor, op, group))
    )
    monkeypatch.setattr(
        "megatron.core.distributed.nonuniform_ep.parallel_state.get_data_parallel_group",
        lambda with_context_parallel: "dp",
    )

    NonuniformEPDistributedDataParallel._synchronize_bucket_size(config)

    assert config.bucket_size == 94_486_908
    assert calls == [(bucket_size, torch.distributed.ReduceOp.MIN, "dp")]


def test_nep_nccl_owner_tasks_use_bounded_distinct_stream_slots(monkeypatch):
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW", "4")
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_nccl_owner_layout = {"min_ep_size": 3, "num_chunks": 2}

    slots = [
        bucket_group._get_nep_nccl_task_buffer_slot(owner_ep_rank, chunk_index)
        for owner_ep_rank in range(3)
        for chunk_index in range(2)
    ]

    assert slots == [0, 1, 2, 3, 0, 1]

    bucket_group._nep_nccl_group_index = 1
    next_group_slots = [
        bucket_group._get_nep_nccl_task_buffer_slot(owner_ep_rank, chunk_index)
        for owner_ep_rank in range(3)
        for chunk_index in range(2)
    ]
    assert next_group_slots == [2, 3, 0, 1, 2, 3]


@pytest.mark.parametrize(
    ("target_chunks", "max_gather_bytes", "expected_chunks", "expected_chunk_numel"),
    ((2, 1 << 30, 2, 40), (4, 1 << 30, 4, 20), (4, 64, 5, 16)),
)
def test_nep_nccl_owner_layout_targets_balanced_chunks_with_byte_cap(
    monkeypatch, target_chunks, max_gather_bytes, expected_chunks, expected_chunk_numel
):
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS", str(target_chunks))
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES", str(max_gather_bytes))
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_nccl_owner_layout = None
    bucket_group._nep_nccl_slot_numel = 40
    bucket_group._nep_runtime_config = {
        "ep_rank": 0,
        "local_ep_size": 8,
        "min_ep_size": 4,
        "expert_placement": None,
    }
    bucket_group.buckets = [SimpleNamespace(grad_data=torch.empty(1, dtype=torch.float32))]

    layout = bucket_group._get_nep_nccl_owner_layout()

    assert layout["owner_numel"] == 80
    assert layout["target_chunks"] == target_chunks
    assert layout["num_chunks"] == expected_chunks
    assert layout["max_chunk_numel"] == expected_chunk_numel


def test_nep_nccl_one_target_has_identical_scheduler_inputs_to_original(monkeypatch):
    def make_group(target_chunks):
        if target_chunks is None:
            monkeypatch.delenv("MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS", raising=False)
        else:
            monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS", str(target_chunks))
        group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
            NonuniformEPNCCLParamAndGradBucketGroup
        )
        group._nep_nccl_owner_layout = None
        group._nep_nccl_slot_numel = 40
        group._nep_nccl_group_index = 0
        group._nep_runtime_config = {
            "ep_rank": 0,
            "local_ep_size": 8,
            "min_ep_size": 4,
            "expert_placement": None,
        }
        group.buckets = [SimpleNamespace(grad_data=torch.empty(1, dtype=torch.float32))]
        group._get_nep_nccl_owner_layout()
        return group

    original = make_group(None)
    one_chunk = make_group(1)

    def scheduler_inputs(group):
        layout = group._get_nep_nccl_owner_layout()
        return [
            (owner, chunk, start, end, group._get_nep_nccl_task_buffer_slot(owner, chunk))
            for owner in range(layout["min_ep_size"])
            for chunk, (start, end) in enumerate(layout["chunk_ranges"])
        ]

    assert original._get_nep_nccl_owner_layout()["chunk_ranges"] == [(0, 80)]
    assert one_chunk._get_nep_nccl_owner_layout()["chunk_ranges"] == [(0, 80)]
    assert scheduler_inputs(one_chunk) == scheduler_inputs(original)


def test_nep_nccl_owner_layout_rejects_nonpositive_target_chunks(monkeypatch):
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS", "0")
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_nccl_owner_layout = None
    bucket_group._nep_nccl_slot_numel = 40
    bucket_group._nep_runtime_config = {
        "ep_rank": 0,
        "local_ep_size": 8,
        "min_ep_size": 4,
        "expert_placement": None,
    }
    bucket_group.buckets = [SimpleNamespace(grad_data=torch.empty(1, dtype=torch.float32))]

    with pytest.raises(RuntimeError, match="TARGET_CHUNKS must be positive"):
        bucket_group._get_nep_nccl_owner_layout()


@pytest.mark.parametrize(
    ("chunk_start", "chunk_end", "remote_segments", "scatter_chunks", "expected"),
    (
        (0, 80, [(40, 80)], 1, [(0, 80)]),
        (0, 80, [(40, 80)], 2, [(0, 60), (60, 80)]),
        (20, 60, [(40, 60)], 4, [(20, 45), (45, 50), (50, 55), (55, 60)]),
        (0, 60, [(10, 20), (40, 50)], 4, [(0, 15), (15, 20), (20, 45), (45, 60)]),
        (0, 10, [(0, 10)], 4, [(0, 3), (3, 5), (5, 8), (8, 10)]),
    ),
)
def test_nep_nccl_scatter_chunk_ranges_balance_remote_payload(
    chunk_start, chunk_end, remote_segments, scatter_chunks, expected
):
    assert (
        _build_nep_nccl_scatter_chunk_ranges(
            chunk_start, chunk_end, remote_segments, scatter_chunks
        )
        == expected
    )


@pytest.mark.parametrize(
    ("scatter_chunks", "expected_ranges"),
    (
        (None, [(0, 16)]),
        ("1", [(0, 16)]),
        ("2", [(0, 12), (12, 16)]),
        ("4", [(0, 10), (10, 12), (12, 14), (14, 16)]),
    ),
)
def test_nep_nccl_scatter_chunks_share_one_ordered_code_path(
    monkeypatch, scatter_chunks, expected_ranges
):
    if scatter_chunks is None:
        monkeypatch.delenv("MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS", raising=False)
    else:
        monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS", scatter_chunks)
    monkeypatch.delenv("MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER", raising=False)

    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    owner_chunk = torch.arange(16)
    context = {
        "owner_ep_rank": 0,
        "chunk_index": 7,
        "chunk_start": 0,
        "chunk_end": 16,
        "chunk": owner_chunk,
        "buffer_slot_key": ("slot",),
        "async_op": True,
    }
    bucket_group._nep_runtime_config = {
        "ep_rank": 0,
        "zero_sm_reshard": False,
        "edp_ready_gate_enabled": False,
    }
    bucket_group.is_first_batch = False
    bucket_group._nep_nccl_scatter_chunk_ranges = (
        lambda owner, start, end, chunks: _build_nep_nccl_scatter_chunk_ranges(
            start, end, [(8, 16)], chunks
        )
    )
    bucket_group._order_nep_nccl_owner_edp_before_scatter = lambda task: calls.append(
        ("order", task)
    )

    def prepare_scatter(
        owner, chunk_index, start, end, chunk, slot_key, async_op, scatter_chunk_index=0
    ):
        descriptor = {
            "scatter_chunk_index": scatter_chunk_index,
            "start": start,
            "end": end,
            "chunk": chunk.tolist(),
        }
        calls.append(
            (
                "prepare",
                owner,
                chunk_index,
                start,
                end,
                chunk.tolist(),
                slot_key,
                async_op,
                scatter_chunk_index,
            )
        )
        return descriptor

    bucket_group._prepare_nep_nccl_owner_all_to_all_scatter = prepare_scatter
    bucket_group._submit_nep_nccl_owner_all_to_all_scatter = lambda descriptor: calls.append(
        ("submit", descriptor["scatter_chunk_index"])
    )
    bucket_group._order_nep_nccl_owner_all_to_all_scatter_completion = (
        lambda descriptor: calls.append(("order_completion", descriptor["scatter_chunk_index"]))
    )
    bucket_group._finish_nep_nccl_owner_all_to_all_scatter = lambda descriptor: calls.append(
        ("copyback", descriptor["scatter_chunk_index"])
    )
    bucket_group._synchronize_first_batch_zero_sm_phase = lambda owner, phase: calls.append(
        ("sync", owner, phase)
    )
    bucket_group._mark_nep_nccl_task_started = lambda owner, chunk_index: calls.append(
        ("mark", owner, chunk_index)
    )

    bucket_group._start_nep_nccl_owner_task_scatter(context)

    assert calls[0] == ("order", context)
    prepare_calls = [call for call in calls if call[0] == "prepare"]
    assert [(call[3], call[4]) for call in prepare_calls] == expected_ranges
    assert [call[5] for call in prepare_calls] == [
        owner_chunk[start:end].tolist() for start, end in expected_ranges
    ]
    assert [call[8] for call in prepare_calls] == list(range(len(expected_ranges)))
    expected_chunk_indices = list(range(len(expected_ranges)))
    expected_phases = ["prepare"] * len(expected_ranges)
    for _ in expected_chunk_indices:
        expected_phases.extend(["submit", "order_completion", "copyback"])
    assert [call[0] for call in calls[1:-2]] == expected_phases
    assert [call[1] for call in calls if call[0] == "submit"] == expected_chunk_indices
    assert [call[1] for call in calls if call[0] == "order_completion"] == expected_chunk_indices
    assert [call[1] for call in calls if call[0] == "copyback"] == expected_chunk_indices
    assert calls[-2:] == [("sync", 0, "scatter"), ("mark", 0, 7)]


def test_nep_nccl_scatter_chunks_rejects_nonpositive_value(monkeypatch):
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS", "0")

    with pytest.raises(RuntimeError, match="SCATTER_CHUNKS must be positive"):
        _get_nep_nccl_scatter_chunks()


def test_nep_nccl_zero_sm_tasks_share_one_ordered_stream(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_runtime_config = {"zero_sm_reshard": True}
    bucket_group.is_first_batch = False
    bucket_group._nep_nccl_scheduler_state = {}
    bucket_group._nep_nccl_streams = {}
    created_streams = []

    def make_stream(device):
        stream = object()
        created_streams.append((device, stream))
        return stream

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)

    first = bucket_group._get_nep_nccl_comm_stream(0)
    second = bucket_group._get_nep_nccl_comm_stream(1)

    assert first is second
    assert created_streams == [(0, first)]
    assert bucket_group._nep_nccl_scheduler_state["comm_streams"] == {"zero_sm": first}


def test_nep_nccl_process_group_dispatch_tasks_share_one_ordered_stream(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_runtime_config = {"zero_sm_reshard": False}
    bucket_group._nep_dispatch_boundary_launch = True
    bucket_group.is_first_batch = False
    bucket_group._nep_nccl_scheduler_state = {}
    bucket_group._nep_nccl_streams = {}
    created_streams = []

    def make_stream(device):
        stream = object()
        created_streams.append((device, stream))
        return stream

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)

    first = bucket_group._get_nep_nccl_comm_stream(0)
    second = bucket_group._get_nep_nccl_comm_stream(1)

    assert first is second
    assert created_streams == [(0, first)]
    assert bucket_group._nep_nccl_scheduler_state["comm_streams"] == {"dispatch": first}


def test_nep_nccl_parallel_gather_window_uses_bounded_dispatch_streams(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_runtime_config = {"zero_sm_reshard": False}
    bucket_group._nep_dispatch_boundary_launch = True
    bucket_group.is_first_batch = False
    bucket_group._nep_nccl_scheduler_state = {}
    bucket_group._nep_nccl_streams = {}

    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_PARALLEL_GATHER_WINDOW", "2")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: object())

    first = bucket_group._get_nep_nccl_comm_stream(0)
    second = bucket_group._get_nep_nccl_comm_stream(1)
    wrapped = bucket_group._get_nep_nccl_comm_stream(2)

    assert first is not second
    assert wrapped is first
    assert bucket_group._nep_nccl_scheduler_state["comm_streams"] == {
        ("dispatch", 0): first,
        ("dispatch", 1): second,
    }


def test_nep_nccl_edp_ready_gate_uses_host_signaled_stream_wait(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    ready_group = object()
    calls = []
    flag = torch.empty(1, dtype=torch.int32)
    signal_stream = SimpleNamespace(cuda_stream=22)
    current_stream = SimpleNamespace(cuda_stream=11)
    device_ready_event = SimpleNamespace(synchronize=lambda: calls.append("device_ready"))
    bucket_group._nep_runtime_config = {
        "edp_ready_gate_enabled": True,
        "edp_group_gloo": ready_group,
    }

    class FakeReadyWork:
        def wait(self):
            calls.append("host_wait")

    class FakeFuture:
        def result(self):
            calls.append("future_result")

    class FakeExecutor:
        def submit(self, function, *args):
            calls.append("submit")
            function(*args)
            return FakeFuture()

    stream_ops = SimpleNamespace(
        wait_value32=lambda stream, address, value: calls.append(
            ("stream_wait", stream, address, value)
        ),
        write_value32=lambda stream, address, value: calls.append(
            ("stream_write", stream, address, value)
        ),
    )
    bucket_group._nep_nccl_scheduler_state = {
        "gather_buf_cache": {},
        "buffer_slot_handles": {},
        "edp_ready_flags": {0: flag},
        "edp_ready_generations": {0: 0},
        "edp_ready_executor": FakeExecutor(),
        "edp_ready_signal_stream": signal_stream,
    }
    bucket_group._nep_edp_ready_futures = []
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda group, async_op: calls.append(("barrier", group, async_op)) or FakeReadyWork(),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(("device", device)))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current_stream)
    monkeypatch.setattr(
        "megatron.core.distributed.nonuniform_ep.get_cuda_stream_memory_ops", lambda: stream_ops
    )

    bucket_group._start_nep_nccl_edp_ready_gate(0, device_ready_event)
    bucket_group._drain_nep_edp_ready_futures()

    assert calls == [
        ("stream_wait", 11, flag.data_ptr(), 1),
        "submit",
        ("device", 7),
        "device_ready",
        ("barrier", ready_group, True),
        "host_wait",
        ("stream_write", 22, flag.data_ptr(), 1),
        "future_result",
    ]
    assert bucket_group._nep_nccl_scheduler_state["edp_ready_generations"] == {0: 1}
    assert bucket_group._nep_edp_ready_futures == []


def test_nep_edp_ready_future_drain_is_bucket_local():
    calls = []

    class FakeFuture:
        def __init__(self, label):
            self.label = label

        def result(self):
            calls.append(self.label)

    first_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    second_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    first_group._nep_edp_ready_futures = [FakeFuture("first")]
    second_group._nep_edp_ready_futures = [FakeFuture("second")]

    first_group._drain_nep_edp_ready_futures()

    assert calls == ["first"]
    assert first_group._nep_edp_ready_futures == []
    assert len(second_group._nep_edp_ready_futures) == 1


def test_nep_scatter_scheduler_ready_gate_skips_unmatched_edp_barrier(monkeypatch):
    edp_group = SimpleNamespace(size=lambda: 2)
    scatter_ready_group = object()
    state = {}
    bucket_group = SimpleNamespace(
        _nep_nccl_scheduler_state=state,
        _nep_runtime_config={
            "edp_ready_gate_enabled": False,
            "is_edp_eligible": True,
            "edp_group_gloo": edp_group,
            "nep_owner_scatter_ready_groups_gloo": {0: scatter_ready_group},
        },
    )
    barriers = []

    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER", "1")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW", "1")
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: torch.empty(1, dtype=torch.int32))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: object())
    monkeypatch.setattr(torch.distributed, "barrier", lambda group: barriers.append(group))
    monkeypatch.setattr(
        "megatron.core.distributed.nonuniform_ep.get_cuda_stream_memory_ops", lambda: object()
    )

    _configure_nep_edp_ready_gate([bucket_group])

    assert barriers == [scatter_ready_group]
    assert set(state["edp_ready_flags"]) == {0}


def test_nep_scatter_descriptor_ready_gate_uses_dedicated_group():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    phase_group = object()
    scatter_launch_group = object()
    scatter_ready_group = object()
    calls = []
    bucket_group._nep_runtime_config = {
        "nep_owner_transfer_groups_gloo": {3: phase_group},
        "nep_owner_scatter_launch_groups_gloo": {3: scatter_launch_group},
        "nep_owner_scatter_ready_groups_gloo": {3: scatter_ready_group},
    }
    bucket_group._prepare_nep_nccl_stream_ready_gate = (
        lambda slot, group, name: calls.append((slot, group, name)) or "gate"
    )

    gate = bucket_group._prepare_nep_nccl_scatter_descriptor_ready_gate(
        {"kind": "all_to_all", "owner_ep_rank": 3, "chunk_index": 2}, 7
    )

    assert gate == "gate"
    assert calls == [(7, scatter_ready_group, "scatter_descriptor_owner_3_chunk_2")]


def test_nep_nccl_owner_task_orders_scatter_and_defers_native_finish():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    fake_edp_group = type("FakeGroup", (), {"rank": lambda self: 0})()
    fake_work = _FakeWork()
    fake_native_group = SimpleNamespace(grad_reduce_handle=None)

    def start_native_ddp():
        calls.append("native_start")
        fake_native_group.grad_reduce_handle = fake_work

    def finish_native_ddp():
        calls.append("native_finish")
        fake_native_group.grad_reduce_handle.wait()
        fake_native_group.grad_reduce_handle = None

    fake_native_group.start_grad_sync = start_native_ddp
    fake_native_group.finish_grad_sync = finish_native_ddp
    bucket_group.buckets = [type("Bucket", (), {"grad_data": torch.empty(8)})()]
    bucket_group.ddp_config = SimpleNamespace(overlap_grad_reduce=True)
    bucket_group.is_first_batch = False
    bucket_group._nep_runtime_config = {
        "ep_rank": 0,
        "edp_group": fake_edp_group,
        "edp_ready_gate_enabled": True,
    }
    bucket_group._nep_nccl_group_index = 0
    bucket_group._nep_nccl_async_tensors = []
    bucket_group._nep_nccl_active_native_edp_states = []
    bucket_group._nep_nccl_scheduler_state = {"gather_buf_cache": {}, "buffer_slot_handles": {}}
    bucket_group._get_nep_nccl_owner_layout = lambda: {"owner_numel": 8}
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: 0
    bucket_group._prep_nep_nccl_owner_entries_for_sync = lambda owner: None
    bucket_group._start_nep_nccl_owner_all_to_all_gather = lambda *args, **kwargs: calls.append(
        "gather"
    )
    bucket_group._start_nep_nccl_edp_ready_gate = lambda slot, event=None: calls.append("ready")
    bucket_group._get_nep_nccl_native_edp_bucket_group = lambda context: fake_native_group
    bucket_group._synchronize_first_batch_zero_sm_phase = lambda owner, phase: None
    bucket_group._prepare_nep_nccl_owner_all_to_all_scatter = lambda *args, **kwargs: calls.append(
        "scatter"
    )
    bucket_group._submit_nep_nccl_owner_all_to_all_scatter = lambda descriptor: None
    bucket_group._order_nep_nccl_owner_all_to_all_scatter_completion = lambda descriptor: None
    bucket_group._finish_nep_nccl_owner_all_to_all_scatter = lambda descriptor: None
    bucket_group._nep_nccl_scatter_chunk_ranges = lambda owner, start, end, chunks: [(start, end)]
    bucket_group._mark_nep_nccl_task_started = lambda owner, chunk: calls.append("mark")

    bucket_group._start_nep_nccl_owner_task(
        owner_ep_rank=0, chunk_index=0, chunk_start=0, chunk_end=8, async_op=True
    )

    assert calls == ["gather", "ready", "native_start", "scatter", "mark"]
    assert fake_work.block_calls == 1
    assert fake_work.wait_calls == 0
    assert fake_native_group.grad_reduce_handle is fake_work

    bucket_group._finish_nep_nccl_native_edp_reductions()

    assert calls[-1] == "native_finish"
    assert fake_work.wait_calls == 1
    assert fake_native_group.grad_reduce_handle is None
    assert bucket_group._nep_nccl_active_native_edp_states == []


def test_nep_nccl_benchmark_skip_scatter_only_copies_owner_grad(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    context = {
        "owner_ep_rank": 0,
        "chunk_index": 2,
        "chunk_start": 8,
        "chunk_end": 16,
        "chunk": object(),
        "buffer_slot_key": object(),
        "async_op": True,
    }
    bucket_group._nep_runtime_config = {"ep_rank": 0}
    bucket_group._order_nep_nccl_owner_edp_before_scatter = lambda value: calls.append(
        ("order", value)
    )
    bucket_group._copy_nep_nccl_owner_chunk_to_local_grads = (
        lambda owner, start, end, chunk: calls.append(("copy", owner, start, end, chunk))
    )
    bucket_group._start_nep_nccl_owner_all_to_all_scatter = lambda *args, **kwargs: calls.append(
        "network_scatter"
    )
    bucket_group._mark_nep_nccl_task_started = lambda owner, chunk: calls.append(
        ("mark", owner, chunk)
    )
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER", "1")

    bucket_group._start_nep_nccl_owner_task_scatter(context)

    assert calls == [("order", context), ("copy", 0, 8, 16, context["chunk"]), ("mark", 0, 2)]

    calls.clear()
    bucket_group._nep_runtime_config = {"ep_rank": 4}
    bucket_group._start_nep_nccl_owner_task_scatter(context)

    assert calls == [("mark", 0, 2)]


def test_nep_nccl_owner_prep_leaves_gradient_scaling_to_native_ddp():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    grad_data = torch.tensor([2.0, 4.0])
    bucket = SimpleNamespace(
        grad_data=grad_data, gradient_scaling_factor=0.25, params_with_extra_main_grads=[]
    )
    bucket_group._nep_nccl_prepped_experts = set()
    bucket_group._nep_nccl_owner_entries = lambda owner: [{"expert_id": 3, "bucket": bucket}]
    bucket_group._foreach_copy_ = lambda destinations, sources: None

    bucket_group._prep_nep_nccl_owner_entries_for_sync(0)

    torch.testing.assert_close(grad_data, torch.tensor([2.0, 4.0]))
    assert bucket_group._nep_nccl_prepped_experts == {(3, 0)}


def test_nep_nccl_owner_edp_batch_coalesces_chunks_per_logical_group():
    first_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    second_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    for index, group in enumerate((first_group, second_group)):
        group._nep_runtime_config = {"ep_rank": 0}
        group._start_nep_nccl_owner_edp_reduce_contexts = (
            lambda contexts, use_device_readiness, index=index: calls.append(
                (index, contexts, use_device_readiness)
            )
        )

    contexts = [
        {"group": first_group, "owner_ep_rank": 0},
        {"group": first_group, "owner_ep_rank": 0},
        {"group": second_group, "owner_ep_rank": 0},
    ]

    first_group._start_nep_nccl_owner_edp_reduce_batch(contexts, use_device_readiness=False)

    assert calls == [(0, contexts[:2], False), (1, contexts[2:], False)]


def test_nep_nccl_grouped_contexts_order_scatter_once_and_finish_at_final_drain():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    fake_work = _FakeWork()
    native_group = SimpleNamespace(grad_reduce_handle=fake_work)

    def finish_native_ddp():
        calls.append("finish")
        native_group.grad_reduce_handle.wait()
        native_group.grad_reduce_handle = None

    native_group.finish_grad_sync = finish_native_ddp
    contexts = [
        {"owner_ep_rank": 0, "native_edp_started": True},
        {"owner_ep_rank": 0, "native_edp_started": True},
    ]
    native_state = {
        "group": native_group,
        "contexts": contexts,
        "started": True,
        "finished": False,
        "scatter_dependency_ordered": False,
    }
    for context in contexts:
        context["native_edp_state"] = native_state
    bucket_group._nep_runtime_config = {"ep_rank": 0}
    bucket_group.ddp_config = SimpleNamespace(overlap_grad_reduce=True)
    bucket_group._nep_nccl_active_native_edp_states = [native_state]

    bucket_group._order_nep_nccl_owner_edp_before_scatter(contexts[0])
    bucket_group._order_nep_nccl_owner_edp_before_scatter(contexts[1])

    assert calls == []
    assert fake_work.block_calls == 1
    assert fake_work.wait_calls == 0
    assert native_group.grad_reduce_handle is fake_work
    assert not native_state["finished"]

    bucket_group._finish_nep_nccl_native_edp_reductions()

    assert calls == ["finish"]
    assert fake_work.wait_calls == 1
    assert native_state["finished"]
    assert not any(context["native_edp_started"] for context in contexts)
    assert bucket_group._nep_nccl_active_native_edp_states == []


def test_nep_nccl_same_communicator_ready_reuses_ordered_token(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    process_group = SimpleNamespace(size=lambda: 2)
    state = {"buffer_slot_handles": {}}
    calls = []
    first_work = object()
    second_work = object()
    works = [first_work, second_work]

    bucket_group._nep_nccl_async_handles = []
    bucket_group._get_nep_nccl_shared_buffer_state = lambda: state
    bucket_group._order_nep_nccl_buffer_slot = lambda key: calls.append(("order", key))
    bucket_group._record_nep_nccl_work = lambda work, key: calls.append(("record", work, key))

    def fake_all_reduce(token, group, async_op):
        calls.append(("all_reduce", token, group, async_op))
        return works.pop(0)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    key = ("edp", 0)
    bucket_group._start_nep_nccl_same_communicator_ready(process_group, key)
    first_token = calls[1][1]
    bucket_group._start_nep_nccl_same_communicator_ready(process_group, key)

    slot_key = ("same_communicator_ready", "edp", 0)
    assert calls == [
        ("order", slot_key),
        ("all_reduce", first_token, process_group, True),
        ("record", first_work, slot_key),
        ("order", slot_key),
        ("all_reduce", first_token, process_group, True),
        ("record", second_work, slot_key),
    ]


def test_nep_nccl_first_batch_zero_sm_finishes_gather_and_edp_before_scatter(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    fake_edp_group = type("FakeGroup", (), {"rank": lambda self: 0})()
    fake_task_group_gloo = object()
    fake_stream = type("FakeStream", (), {"synchronize": lambda self: calls.append("sync")})()
    bucket_group.buckets = [type("Bucket", (), {"grad_data": torch.empty(8)})()]
    bucket_group.ddp_config = SimpleNamespace(overlap_grad_reduce=True)
    bucket_group.is_first_batch = True
    bucket_group._nep_runtime_config = {
        "ep_rank": 0,
        "edp_group": fake_edp_group,
        "dp_cp_group_gloo": fake_task_group_gloo,
        "edp_ready_gate_enabled": True,
        "zero_sm_reshard": True,
    }
    bucket_group._nep_nccl_group_index = 0
    bucket_group._nep_nccl_async_tensors = []
    bucket_group._nep_nccl_scheduler_state = {"gather_buf_cache": {}, "buffer_slot_handles": {}}
    bucket_group._get_nep_nccl_owner_layout = lambda: {}
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: 0
    bucket_group._nep_nccl_owner_source_ranks = lambda owner: [0, 1]
    bucket_group._prep_nep_nccl_owner_entries_for_sync = lambda owner: None
    bucket_group._start_nep_nccl_owner_all_to_all_gather = lambda *args, **kwargs: calls.append(
        "gather"
    )
    bucket_group._start_nep_nccl_edp_ready_gate = lambda slot, event=None: calls.append("ready")
    fake_native_group = SimpleNamespace(
        grad_reduce_handle=None, start_grad_sync=lambda: calls.append("native_start")
    )
    bucket_group._get_nep_nccl_native_edp_bucket_group = lambda context: fake_native_group
    bucket_group._start_nep_nccl_owner_task_scatter = lambda context: calls.append("scatter")

    def fake_barrier(group):
        assert group is fake_task_group_gloo
        calls.append("task_gloo")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: fake_stream)
    monkeypatch.setattr(torch.distributed, "barrier", fake_barrier)

    bucket_group._start_nep_nccl_owner_task(
        owner_ep_rank=0, chunk_index=0, chunk_start=0, chunk_end=8, async_op=False
    )

    assert calls == [
        "gather",
        "sync",
        "sync",
        "task_gloo",
        "native_start",
        "sync",
        "sync",
        "task_gloo",
        "scatter",
    ]


def test_nep_nccl_first_batch_zero_sm_fences_helper_rank(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    bucket_group.is_first_batch = True
    bucket_group._nep_runtime_config = {"ep_rank": 1, "zero_sm_reshard": True}
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [0, 1, 2]
    fake_stream = type("FakeStream", (), {"synchronize": lambda self: calls.append("sync")})()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: fake_stream)

    bucket_group._synchronize_first_batch_zero_sm_phase(0, "gather")
    bucket_group._synchronize_first_batch_zero_sm_phase(0, "scatter")

    assert calls == ["sync", "sync"]


def test_nep_nccl_zero_sm_helper_waits_for_dispatch_boundary():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group.is_first_batch = False
    bucket_group._nep_dispatch_boundary_launch = True
    bucket_group._nep_dispatch_boundary_ready = False
    bucket_group._nep_dispatch_boundary_graph_replay_ready = False
    bucket_group._nep_nccl_owner_entries = lambda owner: []

    assert not bucket_group._nep_nccl_owner_task_ready(0)

    bucket_group._nep_dispatch_boundary_ready = True

    assert bucket_group._nep_nccl_owner_task_ready(0)


def test_nep_graph_replay_boundary_bypasses_host_grad_count_gate():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    param = object()
    bucket_group.is_first_batch = False
    bucket_group._nep_dispatch_boundary_launch = True
    bucket_group._nep_dispatch_boundary_ready = True
    bucket_group._nep_dispatch_boundary_graph_replay_ready = False
    bucket_group.per_param_grad_ready_counts = {}
    bucket_group.golden_per_param_grad_ready_counts = {param: 1}
    bucket_group._nep_nccl_owner_entries = lambda owner: [
        {"bucket": SimpleNamespace(params_list=[param])}
    ]

    assert not bucket_group._nep_nccl_owner_task_ready(0)

    bucket_group._nep_dispatch_boundary_graph_replay_ready = True

    assert bucket_group._nep_nccl_owner_task_ready(0)


def test_nep_bucket_ready_gather_launches_from_accumulate_grad(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    param = object()
    calls = []
    bucket_group.ddp_config = SimpleNamespace(overlap_grad_reduce=True)
    bucket_group.is_last_microbatch = True
    bucket_group.is_first_batch = False
    bucket_group.param_to_bucket = {param: object()}
    bucket_group.params = [param]
    bucket_group.per_param_grad_ready_counts = {}
    bucket_group.golden_per_param_grad_ready_counts = {param: 1}
    bucket_group._nep_dispatch_boundary_launch = True
    bucket_group._nep_dispatch_boundary_ready = False
    bucket_group._nep_dispatch_boundary_groups = (bucket_group,)
    bucket_group._nep_dispatch_boundary_module_label = "bucket.0"
    bucket_group._nep_dispatch_boundary_callback = lambda groups, label: calls.append(
        (groups, label)
    )
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_BUCKET_READY_GATHER", "1")

    bucket_group.register_grad_ready(param)

    assert bucket_group._nep_dispatch_boundary_ready
    assert calls == [((bucket_group,), "bucket.0")]


def test_nep_combined_group_waits_for_all_constituent_modules():
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    group = SimpleNamespace(
        is_first_batch=False,
        is_last_microbatch=True,
        _nep_nccl_group_index=0,
        _nep_dispatch_boundary_ready=False,
        _nep_dispatch_boundary_graph_replay_ready=False,
        _nep_dispatch_boundary_required_modules={"layer.10.mlp", "layer.8.mlp"},
        _nep_dispatch_boundary_ready_modules=set(),
        _nep_dispatch_boundary_inputs_ready=lambda: True,
    )
    ddp._nep_dispatch_waiting_groups = None
    ddp._nep_dispatch_waiting_module_label = None
    ddp._launch_nep_dispatch_boundary_tasks = lambda groups, label: calls.append((groups, label))

    ddp._mark_nep_dispatch_boundary_ready((group,), "layer.10.mlp")

    assert not group._nep_dispatch_boundary_ready
    assert calls == []

    ddp._mark_nep_dispatch_boundary_ready((group,), "layer.8.mlp")

    assert group._nep_dispatch_boundary_ready
    assert calls == [((group,), "layer.8.mlp")]


def test_nep_marks_graph_replay_boundary_before_launch():
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    group = SimpleNamespace(
        is_first_batch=False,
        is_last_microbatch=True,
        _nep_dispatch_boundary_ready=False,
        _nep_dispatch_boundary_graph_replay_ready=False,
    )
    ddp._nep_dispatch_waiting_groups = None
    ddp._nep_dispatch_waiting_module_label = None
    ddp._launch_nep_dispatch_boundary_tasks = lambda groups, module_label: calls.append(
        (groups, module_label)
    )

    ddp._mark_nep_dispatch_boundary_ready((group,), "decoder.layers.1.mlp", graph_replay=True)

    assert group._nep_dispatch_boundary_ready
    assert group._nep_dispatch_boundary_graph_replay_ready
    assert ddp._nep_dispatch_waiting_groups == (group,)
    assert ddp._nep_dispatch_waiting_module_label == "decoder.layers.1.mlp"
    assert calls == [((group,), "decoder.layers.1.mlp")]


def test_nep_defers_graph_replay_boundary_until_next_pre_hook(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    group = SimpleNamespace(
        is_first_batch=False,
        is_last_microbatch=True,
        _nep_dispatch_boundary_ready=False,
        _nep_dispatch_boundary_graph_replay_ready=False,
    )
    ddp._nep_dispatch_waiting_groups = None
    ddp._nep_dispatch_waiting_module_label = None
    ddp._nep_dispatch_pending_completion_event = None
    ddp._nep_dispatch_pending_completion_future = None
    ddp._launch_nep_dispatch_boundary_tasks = (
        lambda groups, module_label: calls.append((groups, module_label)) or True
    )
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH", "1")

    ddp._mark_nep_dispatch_boundary_ready((group,), "decoder.layers.1.mlp", graph_replay=True)

    assert calls == []
    assert ddp._nep_dispatch_waiting_groups == (group,)
    ddp._launch_waiting_nep_dispatch_boundary_tasks()
    assert calls == [((group,), "decoder.layers.1.mlp")]


def test_nep_deferred_host_progress_records_moe_ready_event_without_launch(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    launches = []
    events = []
    compute_stream = object()

    class FakeEvent:
        def __init__(self):
            self.recorded_stream = None
            events.append(self)

        def record(self, stream):
            self.recorded_stream = stream

    group = SimpleNamespace(
        is_first_batch=False,
        is_last_microbatch=True,
        _nep_dispatch_boundary_ready=False,
        _nep_dispatch_boundary_graph_replay_ready=False,
    )
    ddp._nep_dispatch_waiting_groups = None
    ddp._nep_dispatch_waiting_module_label = None
    ddp._nep_dispatch_deferred_compute_ready_event = None
    ddp._launch_nep_dispatch_boundary_tasks = lambda *args: launches.append(args)
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH", "1")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES", "1")
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)

    ddp._mark_nep_dispatch_boundary_ready((group,), "decoder.layers.1.mlp", graph_replay=True)

    assert launches == []
    assert len(events) == 1
    assert events[0].recorded_stream is compute_stream
    assert ddp._nep_dispatch_deferred_compute_ready_event is events[0]


def test_nep_partial_boundary_launches_immediately_after_dispatch():
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    group = SimpleNamespace(
        is_first_batch=False,
        is_last_microbatch=True,
        _nep_dispatch_boundary_ready=False,
        _nep_dispatch_boundary_graph_replay_ready=False,
    )
    ddp._nep_dispatch_waiting_groups = None
    ddp._nep_dispatch_waiting_module_label = None
    ddp._nep_dispatch_pending_completion_event = None

    def launch(groups, module_label):
        calls.append((groups, module_label))
        return True

    ddp._launch_nep_dispatch_boundary_tasks = launch

    ddp._mark_nep_dispatch_boundary_ready((group,), "decoder.layers.1.mlp")
    assert calls == [((group,), "decoder.layers.1.mlp")]


def test_nep_finds_nearest_full_layer_cuda_graph_manager():
    graph_manager = object()
    named_modules = {
        "": SimpleNamespace(),
        "decoder": SimpleNamespace(),
        "decoder.layers.1": SimpleNamespace(cudagraph_manager=graph_manager),
        "decoder.layers.1.mlp": SimpleNamespace(),
    }

    found = NonuniformEPDistributedDataParallel._find_nep_local_cuda_graph_manager(
        "decoder.layers.1.mlp", named_modules
    )

    assert found is graph_manager


def test_nep_does_not_cross_partial_moe_graph_boundary():
    root_graph_manager = object()
    named_modules = {
        "": SimpleNamespace(cudagraph_manager=root_graph_manager),
        "decoder": SimpleNamespace(),
        "decoder.layers.1": SimpleNamespace(use_partial_cudagraphs=True),
        "decoder.layers.1.mlp": SimpleNamespace(),
    }

    found = NonuniformEPDistributedDataParallel._find_nep_local_cuda_graph_manager(
        "decoder.layers.1.mlp", named_modules
    )

    assert found is None


def test_moe_expert_compute_callbacks_launch_after_dispatch():
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    group = object()
    calls = []
    ddp._mark_nep_dispatch_boundary_ready = lambda groups, module_label: calls.append(
        (groups, module_label)
    )

    class FakeMoE:
        register_expert_compute_input_grad_callback = (
            BaseMoELayer.register_expert_compute_input_grad_callback
        )
        register_expert_compute_dgrad_callback = BaseMoELayer.register_expert_compute_dgrad_callback
        register_expert_compute_output_grad_callback = (
            BaseMoELayer.register_expert_compute_output_grad_callback
        )
        _attach_expert_compute_callbacks = staticmethod(
            BaseMoELayer._attach_expert_compute_callbacks
        )
        _attach_expert_compute_input_grad_callbacks = (
            BaseMoELayer._attach_expert_compute_input_grad_callbacks
        )
        _attach_expert_compute_dgrad_callbacks = BaseMoELayer._attach_expert_compute_dgrad_callbacks
        _attach_expert_compute_output_grad_callbacks = (
            BaseMoELayer._attach_expert_compute_output_grad_callbacks
        )

        def __init__(self):
            self._expert_compute_input_grad_callbacks = []
            self._expert_compute_dgrad_callbacks = []
            self._expert_compute_output_grad_callbacks = []

    module = FakeMoE()
    module.register_expert_compute_input_grad_callback(
        lambda: ddp._mark_nep_dispatch_boundary_ready((group,), "decoder.layers.1.mlp")
    )
    module.register_expert_compute_dgrad_callback(lambda: calls.append("wait_before_dispatch"))
    leaf = torch.ones(4, requires_grad=True)
    expert_input = leaf * 2
    dispatch_input = module._attach_expert_compute_input_grad_callbacks(expert_input)
    dispatched_input = dispatch_input * 3
    expert_boundary = module._attach_expert_compute_dgrad_callbacks(dispatched_input)
    expert_output = expert_boundary * 4
    combine_input = module._attach_expert_compute_output_grad_callbacks(expert_output)

    assert calls == []
    (combine_input * 5).sum().backward()
    assert calls == ["wait_before_dispatch", ((group,), "decoder.layers.1.mlp")]


def test_nep_coalesces_shared_cuda_graph_boundary():
    group_1 = SimpleNamespace(_nep_nccl_group_index=7)
    group_2 = SimpleNamespace(_nep_nccl_group_index=3)

    groups, label = NonuniformEPDistributedDataParallel._coalesce_nep_cuda_graph_boundary(
        [("decoder.layers.0.mlp", (group_1,)), ("decoder.layers.1.mlp", (group_2, group_1))]
    )

    assert groups == (group_2, group_1)
    assert label == "cuda_graph[decoder.layers.0.mlp,decoder.layers.1.mlp]"


def test_nep_finds_only_non_moe_cuda_graph_managers():
    class FakeMoE:
        pass

    class FakeModule:
        def __init__(self, graph_manager, children=()):
            self.cudagraph_manager = graph_manager
            self.children = children

        def modules(self):
            return (self, *self.children)

    safe_manager = object()
    mixed_manager = object()
    named_modules = {
        "safe": FakeModule(safe_manager),
        "safe_duplicate": FakeModule(safe_manager),
        "mixed_leaf": FakeModule(mixed_manager),
        "mixed_parent": FakeModule(mixed_manager, (FakeMoE(),)),
    }

    graph_managers = NonuniformEPDistributedDataParallel._find_nep_non_moe_cuda_graph_managers(
        named_modules, FakeMoE
    )

    assert graph_managers == (safe_manager,)


def test_nep_host_progress_registers_full_pipeline_after_non_moe_replay(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []

    class FakeGraphManager:
        def register_backward_replay_hooks(self, pre_hook=None, post_hook=None):
            self.pre_hook = pre_hook
            self.post_hook = post_hook

    graph_manager = FakeGraphManager()
    ddp.expert_parallel_bucket_groups = []
    ddp.module = SimpleNamespace(named_modules=lambda: [("", SimpleNamespace())])
    ddp.param_to_bucket_group = {}
    ddp._find_nep_non_moe_cuda_graph_managers = lambda named_modules, moe_type: (graph_manager,)
    ddp._progress_nep_dispatch_after_graph_launch = lambda: calls.append("launch_full_pipeline")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES", "1")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES", "1")

    ddp._configure_nep_dispatch_boundary_hooks()
    assert graph_manager.pre_hook is None
    graph_manager.post_hook()

    assert calls == ["launch_full_pipeline"]


def test_local_cuda_graph_backward_replay_hooks_wrap_replay():
    calls = []

    class FakeGraph:
        def replay(self):
            calls.append("replay")

    runner = SimpleNamespace(
        bwd_graph=FakeGraph(),
        status=_GraphStatus.BWD_READY,
        static_grad_outputs=(),
        fwd_graph_input_surface=(),
        backward_replay_pre_hooks=[lambda: calls.append("pre")],
        backward_replay_post_hooks=[lambda: calls.append("post")],
        fp8_enabled=False,
        groundtruth_grad_added_to_main_grad={},
        static_grad_inputs=[],
        num_dgrads=0,
    )
    ctx = SimpleNamespace(runner=runner, saved_tensors=())

    result = _CudagraphReplayNode.backward(ctx)

    assert calls == ["pre", "replay", "post"]
    assert runner.status == _GraphStatus.FWD_READY
    assert result == (None, None)


def test_nep_nccl_dispatch_boundary_orders_full_pipeline():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    context = {
        "owner_ep_rank": 0,
        "chunk_index": 0,
        "chunk_start": 0,
        "chunk_end": 8,
        "chunk": object(),
        "buffer_slot": 0,
        "buffer_slot_key": object(),
        "async_op": True,
    }
    bucket_group._nep_runtime_config = {"ep_rank": 0}
    bucket_group._start_nep_nccl_owner_all_to_all_gather = lambda *args, **kwargs: calls.append(
        "gather"
    )
    bucket_group._start_nep_nccl_owner_edp_reduce = lambda task, use_device_readiness: calls.append(
        ("edp", use_device_readiness)
    )
    bucket_group._start_nep_nccl_owner_task_scatter = lambda task: calls.append("scatter")

    bucket_group._start_nep_nccl_dispatch_boundary_task(context)

    assert calls == ["gather", ("edp", True), "scatter"]


def test_process_group_edp_reduce_uses_native_ddp_after_external_host_gate(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    edp_group = SimpleNamespace(rank=lambda: 0)
    chunk = torch.empty(8)
    bucket_group._nep_runtime_config = {
        "ep_rank": 0,
        "edp_group": edp_group,
        "edp_ready_gate_enabled": False,
    }
    bucket_group.ddp_config = SimpleNamespace(overlap_grad_reduce=True)
    bucket_group.is_first_batch = False
    bucket_group._nep_nccl_group_index = 3
    bucket_group._start_nep_nccl_edp_readiness = lambda slot: calls.append("device_gate")
    bucket_group._synchronize_first_batch_zero_sm_phase = lambda owner, phase: None
    fake_native_group = SimpleNamespace(
        grad_reduce_handle=None, start_grad_sync=lambda: calls.append("native_start")
    )
    bucket_group._get_nep_nccl_native_edp_bucket_group = lambda context: fake_native_group
    context = {
        "owner_ep_rank": 0,
        "chunk_index": 2,
        "chunk_start": 0,
        "chunk_end": 8,
        "chunk": chunk,
        "buffer_slot": 1,
        "buffer_slot_key": ("slot",),
        "async_op": True,
    }

    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE", "1")
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("blocking host gate")),
    )

    bucket_group._start_nep_nccl_owner_edp_reduce(context, use_device_readiness=False)

    assert calls == ["native_start"]


def test_process_group_host_gate_defers_edp_and_scatter_after_gather(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    grad_data = torch.empty(8)
    chunk = torch.empty(8)
    pending_tasks = []
    bucket_group.buckets = [SimpleNamespace(grad_data=grad_data)]
    bucket_group._nep_runtime_config = {
        "ep_rank": 1,
        "zero_sm_reshard": False,
        "edp_ready_gate_enabled": False,
    }
    bucket_group._nep_nccl_async_tensors = []
    bucket_group._nep_nccl_scheduler_state = {
        "gather_buf_cache": {
            ("owner_layout_gather", 0, 8, grad_data.dtype, grad_data.device): chunk
        },
        "buffer_slot_handles": {},
        "pending_owner_tasks": pending_tasks,
    }
    bucket_group._get_nep_nccl_owner_layout = lambda: {}
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk_index: 0
    bucket_group._order_nep_nccl_buffer_slot = lambda key: None
    bucket_group._prep_nep_nccl_owner_entries_for_sync = lambda owner: None
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [0, 1]
    bucket_group._start_nep_nccl_owner_all_to_all_gather = lambda *args, **kwargs: calls.append(
        "gather"
    )
    bucket_group._start_nep_nccl_owner_edp_reduce = lambda *args, **kwargs: calls.append("edp")
    bucket_group._start_nep_nccl_owner_task_scatter = lambda task: calls.append("scatter")

    class FakeEvent:
        def record(self, stream):
            calls.append(("record", stream))

    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE", "1")
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "comm_stream")

    bucket_group._start_nep_nccl_owner_task(
        owner_ep_rank=0,
        chunk_index=2,
        chunk_start=0,
        chunk_end=8,
        async_op=True,
        defer_scatter=True,
    )

    assert calls == ["gather", ("record", "comm_stream")]
    assert len(pending_tasks) == 1
    assert pending_tasks[0]["stage"] == "gather"


def test_nep_dispatch_boundary_enqueues_without_launch_barriers(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    completion_stream = object()

    class FakeEvent:
        def record(self, stream):
            assert stream is completion_stream
            calls.append("record_completion")

        def synchronize(self):
            calls.append("wait_completion")

    def make_group(index):
        group = type("Group", (), {})()
        group._nep_nccl_group_index = index
        group._nep_dispatch_boundary_ready = False
        group._nep_dispatch_boundary_launched = False
        group._nep_dispatch_boundary_launching = False
        group._nep_dispatch_boundary_wait_logged = False
        group._nep_nccl_ready = True
        group._nep_dispatch_boundary_inputs_ready = lambda: True
        group._get_nep_nccl_comm_stream = lambda slot: completion_stream
        return group

    groups = (make_group(4), make_group(5))
    groups[0]._try_start_nep_nccl_ready_tasks = lambda **kwargs: calls.append(("launch", kwargs))

    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda group: (_ for _ in ()).throw(AssertionError("unexpected launch barrier")),
    )
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)

    for group in groups:
        group._nep_dispatch_boundary_ready = True
        group._nep_dispatch_boundary_launching = True
    compute_ready_event = object()
    completion_event = ddp._run_nep_dispatch_boundary_tasks(
        groups, "decoder.layers.1.mlp", compute_ready_event
    )

    assert calls == [
        (
            "launch",
            {
                "force_ready": False,
                "async_op_override": True,
                "compute_ready_event": compute_ready_event,
            },
        ),
        "record_completion",
    ]
    assert isinstance(completion_event, FakeEvent)
    assert all(group._nep_dispatch_boundary_ready for group in groups)
    assert all(group._nep_dispatch_boundary_launched for group in groups)


@pytest.mark.parametrize("same_communicator_ready", [False, True])
def test_nep_dispatch_scheduler_launches_process_group_phases(monkeypatch, same_communicator_ready):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    compute_ready_event = object()

    class FakeStream:
        def wait_event(self, event):
            calls.append(("wait_event", event))

        def wait_stream(self, stream):
            calls.append(("wait_stream", stream))

    class FakeStreamContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    nccl_stream = FakeStream()
    source_group = object()
    source_nccl_group = SimpleNamespace(size=lambda: 2)
    edp_owner_group = SimpleNamespace(size=lambda: 2)
    edp_nccl_group = SimpleNamespace(size=lambda: 2)
    events = []

    class FakeEvent:
        def __init__(self):
            events.append(self)

        def record(self, stream):
            calls.append(("record_event", events.index(self), stream))

    tasks = [
        {
            "group": bucket_group,
            "owner_ep_rank": owner,
            "chunk_index": 0,
            "chunk_start": 0,
            "chunk_end": 8,
        }
        for owner in (0, 1)
    ]
    bucket_group._nep_nccl_scheduler_state = {
        "task_sequence": tasks,
        "task_next_index": 0,
        "pending_owner_tasks": [],
    }
    bucket_group._nep_runtime_config = {
        "ep_rank": 0,
        "zero_sm_reshard": False,
        "edp_ready_gate_enabled": True,
        "edp_group": edp_nccl_group,
        "edp_group_gloo": edp_owner_group,
        "nep_owner_transfer_groups": {0: source_nccl_group},
        "nep_owner_transfer_groups_gloo": {0: source_group},
    }
    bucket_group._nep_nccl_owner_task_ready = lambda owner: True
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [owner, owner + 4]
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: owner
    bucket_group._get_nep_nccl_comm_stream = lambda slot: nccl_stream
    bucket_group._prepare_nep_nccl_owner_task_context = (
        lambda owner, chunk, start, end, async_op: calls.append(("prepare", owner))
        or {
            "group": bucket_group,
            "owner_ep_rank": owner,
            "chunk_index": chunk,
            "chunk_start": start,
            "chunk_end": end,
            "chunk": object(),
            "buffer_slot": owner,
            "buffer_slot_key": object(),
        }
    )
    bucket_group._start_nep_nccl_owner_all_to_all_gather = (
        lambda owner, *args, **kwargs: calls.append(("gather", owner))
    )
    bucket_group._start_nep_nccl_owner_edp_reduce_batch = (
        lambda contexts, use_device_readiness: calls.append(
            (
                "edp_batch",
                [context["owner_ep_rank"] for context in contexts],
                use_device_readiness,
                all(context["gather_done_event"] is events[0] for context in contexts),
            )
        )
    )
    bucket_group._start_nep_nccl_scatter_ready_gate = lambda owner, slot, event: calls.append(
        ("scatter_ready", owner, slot, event is events[1])
    )
    bucket_group._start_nep_nccl_same_communicator_ready = lambda group, key: calls.append(
        ("same_communicator_ready", group, key)
    )
    bucket_group._start_nep_nccl_owner_task_scatter = lambda context: calls.append(
        ("scatter", context["owner_ep_rank"])
    )

    monkeypatch.setattr(torch.cuda, "stream", lambda stream: FakeStreamContext())
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        torch.distributed, "barrier", lambda group: calls.append(("owner_barrier", group))
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: (_ for _ in ()).throw(AssertionError("late stream dependency")),
    )

    monkeypatch.setenv(
        "MEGATRON_NONUNIFORM_EP_SAME_COMM_READY", "1" if same_communicator_ready else "0"
    )
    bucket_group._try_start_nep_nccl_ready_tasks(
        async_op_override=True, compute_ready_event=compute_ready_event
    )

    expected_calls = [
        ("wait_event", compute_ready_event),
        ("prepare", 0),
        ("prepare", 1),
        ("gather", 0),
        ("gather", 1),
        ("record_event", 0, nccl_stream),
        ("owner_barrier", source_group),
        ("owner_barrier", edp_owner_group),
        ("edp_batch", [0, 1], True, True),
        ("record_event", 1, nccl_stream),
        ("owner_barrier", source_group),
        ("scatter_ready", 0, 0, True),
        ("scatter", 0),
        ("scatter", 1),
    ]
    if same_communicator_ready:
        expected_calls = [
            ("wait_event", compute_ready_event),
            ("prepare", 0),
            ("prepare", 1),
            ("gather", 0),
            ("gather", 1),
            ("record_event", 0, nccl_stream),
            ("same_communicator_ready", edp_nccl_group, ("edp", 0)),
            ("edp_batch", [0, 1], True, True),
            ("record_event", 1, nccl_stream),
            ("same_communicator_ready", source_nccl_group, ("scatter", 0)),
            ("scatter", 0),
            ("scatter", 1),
        ]
    assert calls == expected_calls
    assert bucket_group._nep_nccl_scheduler_state["task_next_index"] == 2


@pytest.mark.parametrize("device_ordered_edp", [False, True])
def test_nep_split_host_phases_defer_edp_and_scatter(monkeypatch, device_ordered_edp):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    compute_ready_event = object()

    class FakeStream:
        def wait_event(self, event):
            calls.append(("wait_event", event))

    class FakeStreamContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class FakeEvent:
        def record(self, stream):
            calls.append(("record_event", stream))

    class FakeWork:
        def __init__(self, label):
            self.label = label

        def wait(self):
            calls.append(("wait_barrier", self.label))

    nccl_stream = FakeStream()
    gather_group = object()
    scatter_group = object()
    edp_group = SimpleNamespace(size=lambda: 2)
    task = {
        "group": bucket_group,
        "owner_ep_rank": 0,
        "chunk_index": 0,
        "chunk_start": 0,
        "chunk_end": 8,
    }
    bucket_group._nep_nccl_scheduler_state = {
        "task_sequence": [task],
        "task_next_index": 0,
        "pending_owner_tasks": [],
    }
    bucket_group._nep_runtime_config = {
        "ep_rank": 0,
        "zero_sm_reshard": False,
        "edp_ready_gate_enabled": False,
        "edp_group_gloo": edp_group,
        "nep_owner_transfer_groups_gloo": {0: gather_group},
        "nep_owner_scatter_launch_groups_gloo": {0: scatter_group},
    }
    bucket_group._nep_nccl_owner_task_ready = lambda owner: True
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [0, 4]
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: 0
    bucket_group._get_nep_nccl_comm_stream = lambda slot: nccl_stream
    bucket_group._prepare_nep_nccl_owner_task_context = (
        lambda owner, chunk, start, end, async_op: calls.append(("prepare", owner))
        or {
            "group": bucket_group,
            "owner_ep_rank": owner,
            "chunk_index": chunk,
            "chunk_start": start,
            "chunk_end": end,
            "chunk": object(),
            "buffer_slot": 0,
            "buffer_slot_key": object(),
        }
    )
    bucket_group._start_nep_nccl_owner_all_to_all_gather = (
        lambda owner, *args, **kwargs: calls.append(("gather", owner))
    )
    bucket_group._start_nep_nccl_owner_edp_reduce_batch = (
        lambda contexts, use_device_readiness: calls.append(("edp_batch", use_device_readiness))
    )
    bucket_group._start_nep_nccl_owner_task_scatter = lambda context: calls.append(
        ("scatter", context["owner_ep_rank"])
    )

    barrier_counts = {id(gather_group): 0, id(scatter_group): 0, id(edp_group): 0}

    def fake_barrier(group, async_op=False):
        assert async_op
        barrier_counts[id(group)] += 1
        if group is gather_group:
            label = "gather"
        elif group is scatter_group:
            label = "scatter"
        else:
            label = "edp"
        label = f"{label}_{barrier_counts[id(group)]}"
        calls.append(("submit_barrier", label))
        return FakeWork(label)

    monkeypatch.setattr(torch.cuda, "stream", lambda stream: FakeStreamContext())
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.distributed, "barrier", fake_barrier)
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES", "1")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_SAME_COMM_READY", "0")
    monkeypatch.setenv(
        "MEGATRON_NONUNIFORM_EP_DEVICE_ORDERED_EDP", "1" if device_ordered_edp else "0"
    )

    pending = bucket_group._try_start_nep_nccl_ready_tasks(
        async_op_override=True, compute_ready_event=compute_ready_event
    )

    expected_launch_calls = [
        ("wait_event", compute_ready_event),
        ("prepare", 0),
        ("gather", 0),
        ("record_event", nccl_stream),
    ]
    if device_ordered_edp:
        expected_launch_calls.append(("edp_batch", False))
    else:
        expected_launch_calls.append(("submit_barrier", "gather_1"))
    assert calls == expected_launch_calls
    assert len(pending) == 1

    assert not bucket_group._finish_nep_nccl_process_group_dispatch_batches(
        pending, defer_scatter_submission=True
    )

    expected_finish_calls = []
    if not device_ordered_edp:
        expected_finish_calls.extend(
            [
                ("wait_barrier", "gather_1"),
                ("submit_barrier", "edp_1"),
                ("wait_barrier", "edp_1"),
                ("edp_batch", False),
            ]
        )
    expected_finish_calls.extend([("submit_barrier", "scatter_1"), ("wait_barrier", "scatter_1")])
    assert calls[len(expected_launch_calls) :] == expected_finish_calls
    assert pending[0]["phase"] == "scatter_ready"

    assert bucket_group._finish_nep_nccl_process_group_dispatch_batches(pending)

    assert calls[-1:] == [("scatter", 0)]


def test_nep_split_host_phases_skip_scatter_rendezvous(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []

    class FakeStreamContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    context = {"group": bucket_group, "owner_ep_rank": 0}
    pending = [
        {
            "batch_index": 0,
            "contexts": [context],
            "dispatch_stream": object(),
            "local_transfer_contexts": {0: (context, object(), object())},
            "phase": "edp_launched",
        }
    ]
    bucket_group._start_nep_nccl_owner_task_scatter = lambda value: calls.append(
        ("complete_without_scatter", value)
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: FakeStreamContext())
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("scatter barrier")),
    )
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER", "1")

    assert bucket_group._finish_nep_nccl_process_group_dispatch_batches(pending)

    assert calls == [("complete_without_scatter", context)]
    assert pending[0]["phase"] == "finished"


def test_nep_pipelined_host_phases_order_each_context_before_next_gather(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    compute_ready_event = object()

    class FakeStream:
        def wait_event(self, event):
            calls.append(("wait_event", event))

    class FakeStreamContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class FakeEvent:
        next_index = 0

        def __init__(self):
            self.index = self.next_index
            FakeEvent.next_index += 1

        def record(self, stream):
            calls.append(("record_event", self.index, stream))

    class FakeWork:
        def __init__(self, label):
            self.label = label

        def wait(self):
            calls.append(("wait_barrier", self.label))

    nccl_stream = FakeStream()
    gather_group = object()
    scatter_group = object()
    edp_group = SimpleNamespace(size=lambda: 2)
    tasks = [
        {
            "group": bucket_group,
            "owner_ep_rank": 0,
            "chunk_index": chunk_index,
            "chunk_start": chunk_index * 8,
            "chunk_end": (chunk_index + 1) * 8,
        }
        for chunk_index in range(2)
    ]
    bucket_group._nep_nccl_scheduler_state = {
        "task_sequence": tasks,
        "task_next_index": 0,
        "pending_owner_tasks": [],
    }
    bucket_group._nep_runtime_config = {
        "ep_rank": 0,
        "zero_sm_reshard": False,
        "edp_ready_gate_enabled": False,
        "edp_group_gloo": edp_group,
        "nep_owner_transfer_groups_gloo": {0: gather_group},
        "nep_owner_scatter_launch_groups_gloo": {0: scatter_group},
    }
    bucket_group._nep_nccl_owner_task_ready = lambda owner: True
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [0, 4]
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: chunk
    bucket_group._get_nep_nccl_comm_stream = lambda slot: nccl_stream
    bucket_group._prepare_nep_nccl_owner_task_context = (
        lambda owner, chunk, start, end, async_op: calls.append(("prepare", chunk))
        or {
            "group": bucket_group,
            "owner_ep_rank": owner,
            "chunk_index": chunk,
            "chunk_start": start,
            "chunk_end": end,
            "chunk": object(),
            "buffer_slot": chunk,
            "buffer_slot_key": object(),
        }
    )
    bucket_group._start_nep_nccl_owner_all_to_all_gather = (
        lambda owner, chunk, *args, **kwargs: calls.append(("gather", chunk))
    )
    bucket_group._start_nep_nccl_owner_edp_reduce_batch = (
        lambda contexts, use_device_readiness: calls.append(
            ("edp_batch", [context["chunk_index"] for context in contexts], use_device_readiness)
        )
    )
    bucket_group._start_nep_nccl_owner_task_scatter = lambda context: calls.append(
        ("scatter", context["chunk_index"])
    )

    barrier_counts = {id(gather_group): 0, id(scatter_group): 0, id(edp_group): 0}

    def fake_barrier(group, async_op=False):
        assert async_op
        barrier_counts[id(group)] += 1
        if group is gather_group:
            label = "gather"
        elif group is scatter_group:
            label = "scatter"
        else:
            label = "edp"
        label = f"{label}_{barrier_counts[id(group)]}"
        calls.append(("submit_barrier", label))
        return FakeWork(label)

    monkeypatch.setattr(torch.cuda, "stream", lambda stream: FakeStreamContext())
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.distributed, "barrier", fake_barrier)
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES", "1")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES", "1")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_SAME_COMM_READY", "0")

    pending = bucket_group._try_start_nep_nccl_ready_tasks(
        async_op_override=True, compute_ready_event=compute_ready_event
    )

    assert calls == [
        ("wait_event", compute_ready_event),
        ("prepare", 0),
        ("gather", 0),
        ("record_event", 0, nccl_stream),
        ("submit_barrier", "gather_1"),
    ]
    assert len(pending) == 1
    assert len(pending[0]["remaining_task_batches"]) == 1

    assert bucket_group._finish_nep_nccl_process_group_dispatch_batches(pending)

    assert calls[5:] == [
        ("wait_barrier", "gather_1"),
        ("submit_barrier", "edp_1"),
        ("wait_barrier", "edp_1"),
        ("edp_batch", [0], False),
        ("submit_barrier", "scatter_1"),
        ("wait_barrier", "scatter_1"),
        ("scatter", 0),
        ("prepare", 1),
        ("gather", 1),
        ("record_event", 1, nccl_stream),
        ("submit_barrier", "gather_2"),
        ("wait_barrier", "gather_2"),
        ("submit_barrier", "edp_2"),
        ("wait_barrier", "edp_2"),
        ("edp_batch", [1], False),
        ("submit_barrier", "scatter_2"),
        ("wait_barrier", "scatter_2"),
        ("scatter", 1),
    ]
    assert pending[0]["phase"] == "finished"


def test_nep_pipelined_host_phases_group_disjoint_owner_transfers(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    compute_ready_event = object()
    launched_batches = []

    class FakeStream:
        def wait_event(self, event):
            assert event is compute_ready_event

    tasks = [
        {
            "group": bucket_group,
            "owner_ep_rank": owner,
            "chunk_index": 0,
            "chunk_start": 0,
            "chunk_end": 8,
        }
        for owner in range(4)
    ]
    state = {"task_sequence": tasks, "task_next_index": 0}
    transfer_ranks = {0: [0, 6], 1: [1, 7], 2: [2, 6], 3: [3, 7]}
    bucket_group._nep_nccl_owner_task_ready = lambda owner: True
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: transfer_ranks[owner]
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: owner
    bucket_group._get_nep_nccl_comm_stream = lambda slot: FakeStream()

    def start_batch(task_batch, dispatch_stream, batch_index):
        launched_batches.append([task["owner_ep_rank"] for task in task_batch])
        return {"batch_index": batch_index, "phase": "gather_launched"}

    bucket_group._start_nep_nccl_split_host_phase_batch = start_batch
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES", "1")

    pending = bucket_group._start_nep_nccl_process_group_dispatch_batch(
        state,
        force_ready=True,
        async_op=True,
        compute_ready_event=compute_ready_event,
        split_host_phases=True,
    )

    assert launched_batches == [[0, 1]]
    assert [
        [task["owner_ep_rank"] for task in task_batch]
        for _, task_batch in pending[0]["remaining_task_batches"]
    ] == [[2, 3]]


def test_nep_a2a_scatter_scheduler_preserves_ordered_owner_waves(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    compute_ready_event = object()
    launched_batches = []

    class FakeStream:
        def wait_event(self, event):
            assert event is compute_ready_event

    tasks = [
        {
            "group": bucket_group,
            "owner_ep_rank": owner,
            "chunk_index": 0,
            "chunk_start": 0,
            "chunk_end": 8,
        }
        for owner in range(4)
    ]
    state = {"task_sequence": tasks, "task_next_index": 0}
    transfer_ranks = {0: [0, 6], 1: [1, 7], 2: [2, 6], 3: [3, 7]}
    bucket_group._nep_nccl_owner_task_ready = lambda owner: True
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: transfer_ranks[owner]
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: owner
    bucket_group._get_nep_nccl_comm_stream = lambda slot: FakeStream()

    def start_batch(task_batch, dispatch_stream, batch_index):
        launched_batches.append([task["owner_ep_rank"] for task in task_batch])
        return {"batch_index": batch_index, "phase": "gather_launched"}

    bucket_group._start_nep_nccl_split_host_phase_batch = start_batch
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER", "1")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW", "16")

    pending = bucket_group._start_nep_nccl_process_group_dispatch_batch(
        state,
        force_ready=True,
        async_op=True,
        compute_ready_event=compute_ready_event,
        split_host_phases=True,
    )

    assert launched_batches == [[0, 1]]
    assert [
        [task["owner_ep_rank"] for task in task_batch]
        for _, task_batch in pending[0]["remaining_task_batches"]
    ] == [[2, 3]]


def test_nep_parallel_gather_window_submits_overlapping_owner_waves(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    compute_ready_event = object()
    launched_batches = []
    streams = [SimpleNamespace(wait_event=lambda event: None) for _ in range(2)]

    tasks = [
        {
            "group": bucket_group,
            "owner_ep_rank": owner,
            "chunk_index": 0,
            "chunk_start": 0,
            "chunk_end": 8,
        }
        for owner in range(4)
    ]
    state = {"task_sequence": tasks, "task_next_index": 0}
    transfer_ranks = {0: [0, 6], 1: [1, 7], 2: [2, 6], 3: [3, 7]}
    bucket_group._nep_nccl_owner_task_ready = lambda owner: True
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: transfer_ranks[owner]
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: owner
    bucket_group._get_nep_nccl_comm_stream = lambda slot: streams[slot]

    def start_batch(task_batch, dispatch_stream, batch_index):
        launched_batches.append(
            ([task["owner_ep_rank"] for task in task_batch], dispatch_stream, batch_index)
        )
        return {"batch_index": batch_index, "phase": "gather_launched"}

    bucket_group._start_nep_nccl_split_host_phase_batch = start_batch
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER", "1")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW", "16")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_PARALLEL_GATHER_WINDOW", "2")

    pending = bucket_group._start_nep_nccl_process_group_dispatch_batch(
        state,
        force_ready=True,
        async_op=True,
        compute_ready_event=compute_ready_event,
        split_host_phases=True,
    )

    assert launched_batches == [([0, 1], streams[0], 0), ([2, 3], streams[1], 1)]
    assert len(pending) == 2
    assert pending[0]["remaining_task_batches"] == []
    assert pending[0]["remaining_task_batches"] is pending[1]["remaining_task_batches"]


def test_nep_post_graph_phases_device_align_gather_edp_and_scatter(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    events = []

    class FakeStreamContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class FakeEvent:
        def __init__(self):
            self.index = len(events)
            events.append(self)

        def record(self, stream):
            calls.append(("record_event", self.index, stream))

        def synchronize(self):
            calls.append(("sync_event", self.index))

    class FakeWork:
        def __init__(self, label):
            self.label = label

        def wait(self):
            calls.append(("wait_barrier", self.label))

    dispatch_stream = object()
    source_group = object()
    scatter_group = object()
    edp_group = object()
    context = {"group": bucket_group, "owner_ep_rank": 0}
    gather_done_event = FakeEvent()
    pending = [
        {
            "batch_index": 0,
            "contexts": [context],
            "dispatch_stream": dispatch_stream,
            "local_transfer_contexts": {0: (context, source_group, scatter_group)},
            "local_edp_contexts": {0: (context, edp_group)},
            "gather_barrier_works": [(0, FakeWork("gather"))],
            "gather_done_event": gather_done_event,
        }
    ]
    bucket_group._start_nep_nccl_owner_edp_reduce_batch = (
        lambda contexts, use_device_readiness: calls.append(("edp_batch", use_device_readiness))
    )
    bucket_group._start_nep_nccl_owner_task_scatter = lambda task: calls.append(
        ("scatter", task["owner_ep_rank"])
    )

    def fake_barrier(group, async_op=False):
        assert async_op
        label = "edp" if group is edp_group else "scatter"
        if label == "scatter":
            assert group is scatter_group
        calls.append(("submit_barrier", label))
        return FakeWork(label)

    monkeypatch.setattr(torch.cuda, "stream", lambda stream: FakeStreamContext())
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.distributed, "barrier", fake_barrier)

    phases_finished = bucket_group._finish_nep_nccl_process_group_dispatch_batches(
        pending, device_align_phases=True, finish_all_phases=False
    )

    assert calls == [
        ("wait_barrier", "gather"),
        ("sync_event", 0),
        ("submit_barrier", "edp"),
        ("wait_barrier", "edp"),
        ("edp_batch", False),
        ("record_event", 1, dispatch_stream),
    ]
    assert not phases_finished
    assert pending[0]["phase"] == "edp_launched"

    phases_finished = bucket_group._finish_nep_nccl_process_group_dispatch_batches(
        pending, device_align_phases=True, finish_all_phases=False
    )

    assert calls[6:] == [
        ("sync_event", 1),
        ("submit_barrier", "scatter"),
        ("wait_barrier", "scatter"),
        ("scatter", 0),
    ]
    assert phases_finished
    assert pending[0]["phase"] == "finished"


def test_nep_dispatch_boundary_submits_launch_after_recording_compute_ready(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    compute_stream = object()
    completion_future = object()
    events = []

    class FakeReadyEvent:
        def __init__(self):
            events.append(self)

        def record(self, stream):
            calls.append(("record", stream))

    group = type("Group", (), {})()
    group._nep_nccl_group_index = 4
    group._nep_dispatch_boundary_ready = True
    group._nep_dispatch_boundary_launched = False
    group._nep_dispatch_boundary_launching = False
    group._nep_dispatch_boundary_inputs_ready = lambda: True
    ddp._nep_dispatch_pending_completion_event = None
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nonuniform_ep_runtime_config = {"zero_sm_reshard": True}

    def fake_submit(groups, module_label, ready_event, completion_event, device_index):
        calls.append(
            ("submit_launch", groups, module_label, ready_event, completion_event, device_index)
        )
        return completion_future

    ddp._submit_nep_dispatch_launch_and_completion = fake_submit
    monkeypatch.setattr(torch.cuda, "Event", FakeReadyEvent)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)

    assert ddp._launch_nep_dispatch_boundary_tasks((group,), "decoder.layers.1.mlp")

    assert calls[0] == ("record", compute_stream)
    assert calls[1] == ("submit_launch", (group,), "decoder.layers.1.mlp", events[0], events[1], 7)
    assert ddp._nep_dispatch_pending_completion_event is events[1]
    assert ddp._nep_dispatch_pending_completion_future is completion_future


def test_nep_dispatch_boundary_reuses_deferred_compute_ready_event(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    completion_events = []
    deferred_ready_event = object()

    class FakeCompletionEvent:
        def __init__(self):
            completion_events.append(self)

    group = SimpleNamespace(
        _nep_nccl_group_index=4,
        _nep_dispatch_boundary_ready=True,
        _nep_dispatch_boundary_launched=False,
        _nep_dispatch_boundary_launching=False,
        _nep_dispatch_boundary_inputs_ready=lambda: True,
    )
    ddp._nep_dispatch_deferred_compute_ready_event = deferred_ready_event
    ddp._nep_dispatch_pending_completion_event = None
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nep_dispatch_pending_host_phases = None
    ddp._nonuniform_ep_runtime_config = {"zero_sm_reshard": False}
    ddp._run_nep_dispatch_boundary_tasks = lambda groups, label, ready, completion: calls.append(
        (groups, label, ready, completion)
    )
    monkeypatch.setattr(torch.cuda, "Event", FakeCompletionEvent)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)

    assert ddp._launch_nep_dispatch_boundary_tasks((group,), "decoder.layers.1.mlp")

    assert len(completion_events) == 1
    assert calls == [((group,), "decoder.layers.1.mlp", deferred_ready_event, completion_events[0])]
    assert ddp._nep_dispatch_deferred_compute_ready_event is None


def test_process_group_dispatch_boundary_launches_inline_and_stream_orders_completion(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    compute_stream = SimpleNamespace(wait_event=lambda event: calls.append(("wait_event", event)))
    events = []

    class FakeEvent:
        def __init__(self):
            events.append(self)

        def record(self, stream):
            calls.append(("record", stream))

    group = SimpleNamespace(
        _nep_nccl_group_index=4,
        _nep_dispatch_boundary_ready=True,
        _nep_dispatch_boundary_launched=False,
        _nep_dispatch_boundary_launching=False,
        _nep_dispatch_boundary_inputs_ready=lambda: True,
    )
    ddp._nonuniform_ep_runtime_config = {"zero_sm_reshard": False}
    ddp._nep_dispatch_pending_completion_event = None
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nep_dispatch_waiting_groups = (group,)
    ddp._nep_dispatch_waiting_module_label = "decoder.layers.1.mlp"

    def run_inline(groups, module_label, ready_event, completion_event):
        calls.append(("run_inline", groups, module_label, ready_event, completion_event))
        group._nep_dispatch_boundary_launched = True
        return completion_event

    ddp._run_nep_dispatch_boundary_tasks = run_inline
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)

    assert ddp._launch_nep_dispatch_boundary_tasks((group,), "decoder.layers.1.mlp")
    assert calls == [
        ("record", compute_stream),
        ("run_inline", (group,), "decoder.layers.1.mlp", events[0], events[1]),
    ]
    assert ddp._nep_dispatch_pending_completion_future is None

    ddp._wait_for_nep_dispatch_launch()

    assert calls[-1] == ("wait_event", events[1])
    assert ddp._nep_dispatch_pending_completion_event is None
    assert ddp._nep_dispatch_waiting_groups is None


def test_process_group_split_dispatch_records_completion_after_scatter(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    compute_stream = SimpleNamespace(wait_event=lambda event: calls.append(("wait_event", event)))
    dispatch_stream = object()
    events = []

    class FakeEvent:
        def __init__(self):
            self.index = len(events)
            events.append(self)

        def record(self, stream):
            calls.append(("record", self.index, stream))

    pending = [object()]
    group = SimpleNamespace(
        _nep_nccl_group_index=4,
        _nep_nccl_ready=False,
        _nep_dispatch_boundary_ready=True,
        _nep_dispatch_boundary_launched=False,
        _nep_dispatch_boundary_launching=False,
        _nep_dispatch_boundary_inputs_ready=lambda: True,
        _get_nep_nccl_comm_stream=lambda slot: dispatch_stream,
    )

    def launch_gather(**kwargs):
        calls.append(("launch_gather", kwargs))
        return pending

    def finish_host_phases(phases):
        assert phases is pending
        calls.append("finish_edp_scatter")
        group._nep_nccl_ready = True

    group._try_start_nep_nccl_ready_tasks = launch_gather
    group._finish_nep_nccl_process_group_dispatch_batches = finish_host_phases
    ddp._nonuniform_ep_runtime_config = {"zero_sm_reshard": False}
    ddp._nep_dispatch_pending_completion_event = None
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nep_dispatch_pending_host_phases = None
    ddp._nep_dispatch_waiting_groups = (group,)
    ddp._nep_dispatch_waiting_module_label = "decoder.layers.1.mlp"

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)

    assert ddp._launch_nep_dispatch_boundary_tasks((group,), "decoder.layers.1.mlp")
    assert calls == [
        ("record", 0, compute_stream),
        (
            "launch_gather",
            {"force_ready": False, "async_op_override": True, "compute_ready_event": events[0]},
        ),
    ]
    assert ddp._nep_dispatch_pending_host_phases == (group, pending)

    ddp._wait_for_nep_dispatch_launch()

    assert calls[-3:] == [
        "finish_edp_scatter",
        ("record", 1, dispatch_stream),
        ("wait_event", events[1]),
    ]
    assert ddp._nep_dispatch_pending_host_phases is None
    assert ddp._nep_dispatch_pending_completion_event is None
    assert ddp._nep_dispatch_waiting_groups is None


def test_process_group_dispatch_can_defer_model_ep_fences_until_final_drain(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    compute_stream = SimpleNamespace(wait_event=lambda event: calls.append(("wait", event)))
    first_event = object()
    second_event = object()
    first_group = SimpleNamespace(_nep_dispatch_boundary_launched=True)
    second_group = SimpleNamespace(_nep_dispatch_boundary_launched=True)

    ddp._nonuniform_ep_runtime_config = {"zero_sm_reshard": False}
    ddp._nep_dispatch_waiting_groups = (first_group,)
    ddp._nep_dispatch_waiting_module_label = "decoder.layers.3.mlp"
    ddp._nep_dispatch_pending_completion_event = first_event
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nep_dispatch_pending_host_phases = None
    ddp._nep_dispatch_inflight_completion_events = []
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE", "1")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)

    ddp._wait_for_nep_dispatch_launch()

    assert calls == []
    assert ddp._nep_dispatch_inflight_completion_events == [("decoder.layers.3.mlp", first_event)]
    assert ddp._nep_dispatch_pending_completion_event is None
    assert ddp._nep_dispatch_waiting_groups is None

    ddp._nep_dispatch_waiting_groups = (second_group,)
    ddp._nep_dispatch_waiting_module_label = "decoder.layers.1.mlp"
    ddp._nep_dispatch_pending_completion_event = second_event
    ddp._wait_for_nep_dispatch_launch(final=True)

    assert calls == [("wait", first_event), ("wait", second_event)]
    assert ddp._nep_dispatch_inflight_completion_events == []
    assert ddp._nep_dispatch_pending_completion_event is None
    assert ddp._nep_dispatch_waiting_groups is None


def test_model_ep_a2a_burst_hooks_submit_after_current_burst_completion(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    events = []

    class FakeEvent:
        def __init__(self):
            events.append(self)

        def record(self, stream):
            calls.append(("record_a2a_completion", stream))

    compute_stream = object()
    ddp._nep_model_ep_a2a_burst_depth = 0
    ddp._nep_model_ep_a2a_burst_count = 0
    ddp._nep_dispatch_pending_host_phases = object()

    def finish_pending(*, defer_scatter_submission=False, **kwargs):
        calls.append(("finish", defer_scatter_submission, kwargs.get("scatter_after_event")))
        return True

    ddp._finish_pending_nep_dispatch_host_phases = finish_pending
    ddp._wait_for_nep_dispatch_launch = lambda final=False: calls.append(("wait", final))
    ddp._submit_model_ep_aligned_nep_scatter_chunk = (
        lambda after_event=None: calls.append(("submit_aligned", after_event)) or True
    )
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER", "1")
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)

    ddp.model_ep_a2a_burst_begin()

    assert ddp._nep_model_ep_a2a_burst_depth == 1
    assert calls == []

    ddp.model_ep_a2a_burst_end()

    completion_event = events[0]
    assert ddp._nep_model_ep_a2a_burst_depth == 0
    assert ddp._nep_model_ep_a2a_burst_count == 1
    assert calls == [
        ("record_a2a_completion", compute_stream),
        ("finish", False, completion_event),
        ("wait", False),
        ("submit_aligned", completion_event),
    ]


def test_nep_scatter_queue_preserves_owner_batch_boundaries(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    events = []

    class FakeEvent:
        def __init__(self):
            self.index = len(events)
            events.append(self)

        def record(self, stream):
            calls.append(("record", self.index, stream))

    class FakeScatterStream:
        def wait_event(self, event):
            calls.append(("wait", event))

    def prepare(context):
        train = {"context": context}
        calls.append(("prepare", context["owner"]))
        return train

    def mark(train):
        calls.append(("mark", train["context"]["owner"]))

    group = SimpleNamespace(
        _prepare_nep_nccl_owner_task_scatter_train=prepare,
        _mark_nep_nccl_scatter_train_scheduled=mark,
    )
    scatter_stream = FakeScatterStream()
    completion_event = FakeEvent()
    a2a_completion_event = object()
    ddp._nep_scatter_stream = scatter_stream
    ddp._nep_scatter_batches = []
    ddp._nep_scatter_next_batch_ordinal = 0
    ddp._nep_scatter_next_layer_ordinal = 0
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    ddp._queue_nep_scatter_context_batches(
        [[{"group": group, "owner": 0}], [], [{"group": group, "owner": 4}]],
        completion_event,
        "decoder.layers.3.mlp",
        a2a_completion_event,
    )

    assert ("wait", a2a_completion_event) in calls
    assert [batch["module_label"] for batch in ddp._nep_scatter_batches] == [
        "decoder.layers.3.mlp:owner_batch_0",
        "decoder.layers.3.mlp:owner_batch_1",
        "decoder.layers.3.mlp:owner_batch_2",
    ]
    assert [
        [train["context"]["owner"] for train in batch["trains"]]
        for batch in ddp._nep_scatter_batches
    ] == [[0], [], [4]]
    assert ddp._nep_scatter_batches[1]["submission_complete"]
    assert [batch["schedule_ordinal"] for batch in ddp._nep_scatter_batches] == [0, 1, 2]
    assert [batch["layer_ordinal"] for batch in ddp._nep_scatter_batches] == [0, 0, 0]
    assert ddp._nep_scatter_next_layer_ordinal == 1
    assert ddp._nep_scatter_batches[-1]["completion_event"] is completion_event
    assert ("record", 2, scatter_stream) in calls


def test_nep_scatter_progress_queues_one_chunk_per_completed_a2a_burst(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    events = []

    class FakeChunkEvent:
        def __init__(self):
            self.index = len(events)
            self.complete = False
            events.append(self)

        def record(self, stream):
            calls.append(("record_chunk", self.index, stream))

        def query(self):
            calls.append(("query_chunk", self.index, self.complete))
            return self.complete

        def synchronize(self):
            calls.append(("finish_chunk", self.index))
            self.complete = True

    class FakeCompletionEvent:
        def record(self, stream):
            calls.append(("record_batch", stream))

    class FakeScatterStream:
        def wait_event(self, event):
            calls.append(("wait_a2a", event))

    group = SimpleNamespace(_nep_nccl_group_index=3)
    group._submit_nep_nccl_owner_all_to_all_scatter = lambda descriptor: calls.append(
        ("submit", descriptor)
    )
    group._prepare_nep_nccl_scatter_descriptor_ready_gate = (
        lambda descriptor, slot: calls.append(("prepare_gate", descriptor, slot))
        or f"gate-{descriptor}"
    )
    group._release_nep_nccl_scatter_descriptor_ready_gate = lambda gate: calls.append(
        ("release_gate", gate)
    )
    group._order_nep_nccl_owner_all_to_all_scatter_completion = lambda descriptor: calls.append(
        ("order", descriptor)
    )
    group._finish_nep_nccl_owner_all_to_all_scatter = lambda descriptor: calls.append(
        ("copyback", descriptor)
    )
    group._finish_nep_nccl_scatter_train_submission = lambda train: calls.append(
        ("finish_train", train["next_descriptor"])
    )
    first_train = {
        "group": group,
        "context": {"owner_ep_rank": 2, "chunk_index": 5, "buffer_slot": 0},
        "descriptors": ["scatter-0"],
        "next_descriptor": 0,
        "task_marked": True,
    }
    second_train = {
        "group": group,
        "context": {"owner_ep_rank": 3, "chunk_index": 5, "buffer_slot": 0},
        "descriptors": ["scatter-1"],
        "next_descriptor": 0,
        "task_marked": True,
    }
    scatter_stream = FakeScatterStream()
    ddp._nep_model_ep_a2a_burst_depth = 0
    ddp._nep_scatter_batches = [
        {
            "trains": [first_train],
            "next_train": 0,
            "completion_event": FakeCompletionEvent(),
            "module_label": "decoder.layers.3.mlp:owner_batch_0",
            "submission_complete": False,
        },
        {
            "trains": [second_train],
            "next_train": 0,
            "completion_event": FakeCompletionEvent(),
            "module_label": "decoder.layers.3.mlp:owner_batch_1",
            "submission_complete": False,
        },
    ]
    ddp._nep_scatter_inflight_event = None
    ddp._nep_scatter_stream = scatter_stream

    monkeypatch.setattr(torch.cuda, "Event", FakeChunkEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    first_a2a_event = object()
    assert ddp._submit_nep_scatter_chunk(after_event=first_a2a_event)

    assert [call for call in calls if call[0] == "submit"] == [("submit", "scatter-0")]
    assert calls.index(("prepare_gate", "scatter-0", 0)) < calls.index(("submit", "scatter-0"))
    assert calls.index(("submit", "scatter-0")) < calls.index(("release_gate", "gate-scatter-0"))
    assert ("wait_a2a", first_a2a_event) in calls
    assert first_train["next_descriptor"] == 1
    assert len(ddp._nep_scatter_batches) == 2
    assert ddp._nep_scatter_batches[0]["submission_complete"]

    ddp._nep_model_ep_a2a_burst_depth = 1
    assert not ddp._submit_nep_scatter_chunk(after_event=object(), queue_behind_inflight=True)
    ddp._nep_model_ep_a2a_burst_depth = 0
    assert not ddp._submit_nep_scatter_chunk(after_event=object())
    assert [call for call in calls if call[0] == "submit"] == [("submit", "scatter-0")]

    second_a2a_event = object()
    assert ddp._submit_nep_scatter_chunk(after_event=second_a2a_event, queue_behind_inflight=True)

    assert [call for call in calls if call[0] == "submit"] == [
        ("submit", "scatter-0"),
        ("submit", "scatter-1"),
    ]
    assert ("wait_a2a", second_a2a_event) in calls
    assert calls.count(("finish_train", 1)) == 2
    assert len(ddp._nep_scatter_batches) == 1
    assert ddp._nep_scatter_batches[0]["submission_complete"]
    assert ("record_batch", scatter_stream) in calls
    assert ddp._nep_scatter_inflight_event is events[1]
    assert ("query_chunk", 0, False) in calls

    events[1].complete = True
    assert ddp._submit_nep_scatter_chunk()
    assert ddp._nep_scatter_batches == []
    assert ddp._nep_scatter_inflight_event is None


def test_nep_scatter_progress_after_compute_launch_only_retires_work(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    ddp._nep_model_ep_a2a_burst_depth = 0
    ddp._nep_scatter_batches = [object()]
    ddp._nep_dispatch_pending_host_phases = object()
    ddp._finish_pending_nep_dispatch_host_phases = lambda **kwargs: pytest.fail(
        "compute hooks must not launch participant-dependent host phases"
    )
    ddp._wait_for_nep_dispatch_launch = lambda final=False: pytest.fail(
        "compute hooks must not wait for participant-dependent launches"
    )
    ddp._submit_model_ep_aligned_nep_scatter_chunk = lambda **kwargs: pytest.fail(
        "compute hooks must not launch rank-dependent ticket collectives"
    )
    ddp._retire_nep_scatter_chunk = lambda force=False: calls.append(("retire", force)) or True
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER", "1")

    ddp._progress_nep_scatter_after_compute_launch()

    assert calls == [("retire", False)]

    ddp._nep_model_ep_a2a_burst_depth = 1
    ddp._progress_nep_scatter_after_compute_launch()
    assert calls == [("retire", False)]

    ddp._nep_model_ep_a2a_burst_depth = 0
    ddp._nep_scatter_batches = []
    ddp._progress_nep_scatter_after_compute_launch()

    assert calls == [("retire", False)]


def test_nep_scatter_submission_window_spans_owners_but_not_groups_or_layers():
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    group_0 = SimpleNamespace(_nep_nccl_group_index=0)
    group_1 = SimpleNamespace(_nep_nccl_group_index=1)

    def batch(schedule_ordinal, layer_ordinal, group, owner, descriptors, next_descriptor=0):
        return {
            "schedule_ordinal": schedule_ordinal,
            "layer_ordinal": layer_ordinal,
            "next_train": 0,
            "submission_complete": False,
            "trains": [
                {
                    "group": group,
                    "context": {"owner_ep_rank": owner, "chunk_index": 0},
                    "descriptors": descriptors,
                    "next_descriptor": next_descriptor,
                }
            ],
        }

    completed = batch(10, 3, group_1, 7, [object()])
    completed["submission_complete"] = True
    ddp._nep_scatter_batches = [
        completed,
        batch(11, 4, group_0, 0, [object(), object()], next_descriptor=1),
        batch(12, 4, group_0, 1, [object()]),
        batch(13, 4, group_1, 0, [object()]),
        batch(14, 5, group_0, 0, [object()]),
    ]

    assert ddp._peek_nep_scatter_ticket_window() == (
        "ready",
        2,
        (11, 0, 0, 0, 0, 1),
        (12, 0, 0, 1, 0, 0),
    )


def test_nep_scatter_submission_pipelines_matching_model_ep_tickets(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    ep_group_gloo = object()
    calls = []
    first_ticket = (7, 0, 2, 3, 1, 4)
    last_ticket = (9, 0, 2, 5, 1, 4)
    ddp._nonuniform_ep_runtime_config = {"needs_reshard": True, "ep_group_gloo": ep_group_gloo}
    ddp._nep_scatter_alignment_tensor = None
    ddp._nep_scatter_alignment_work = None
    ddp._peek_nep_scatter_ticket_window = lambda: ("ready", 3, first_ticket, last_ticket)
    ddp._submit_nep_scatter_chunk = (
        lambda after_event=None, queue_behind_inflight=False: calls.append(
            ("submit", after_event, queue_behind_inflight)
        )
        or True
    )

    class FakeWork:
        def __init__(self, index):
            self.index = index

        def wait(self):
            calls.append(("wait", self.index))

    def all_reduce(tensor, op, group, async_op):
        index = len([call for call in calls if call[0] == "launch"])
        calls.append(("launch", index, op, group, async_op, tuple(tensor.tolist())))
        return FakeWork(index)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    first_a2a_event = object()
    second_a2a_event = object()
    assert not ddp._submit_model_ep_aligned_nep_scatter_chunk(after_event=first_a2a_event)
    assert [call[0] for call in calls] == ["launch"]

    assert ddp._submit_model_ep_aligned_nep_scatter_chunk(after_event=second_a2a_event) == 3
    assert [call[:2] for call in calls] == [
        ("launch", 0),
        ("wait", 0),
        ("submit", second_a2a_event),
        ("submit", None),
        ("submit", None),
        ("launch", 1),
    ]
    assert all(call[2] for call in calls if call[0] == "submit")
    expected_payload = (
        (2, 3)
        + first_ticket
        + last_ticket
        + (-2, -3)
        + tuple(-value for value in first_ticket)
        + tuple(-value for value in last_ticket)
    )
    assert calls[0] == (
        "launch",
        0,
        torch.distributed.ReduceOp.MIN,
        ep_group_gloo,
        True,
        expected_payload,
    )


def test_nep_scatter_submission_rejects_different_model_ep_tickets(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    ddp._nonuniform_ep_runtime_config = {"needs_reshard": True, "ep_group_gloo": object()}
    ddp._nep_scatter_alignment_tensor = None
    ddp._nep_scatter_alignment_work = None
    ticket = (7, 0, 2, 3, 1, 4)
    ddp._peek_nep_scatter_ticket_window = lambda: ("ready", 1, ticket, ticket)
    ddp._submit_nep_scatter_chunk = lambda **kwargs: pytest.fail(
        "mismatched windows must not submit"
    )

    class FakeWork:
        def __init__(self, tensor):
            self.tensor = tensor

        def wait(self):
            self.tensor[-1] = -5

    def mismatch_ticket(tensor, op, group, async_op):
        assert async_op
        return FakeWork(tensor)

    monkeypatch.setattr(torch.distributed, "all_reduce", mismatch_ticket)

    assert not ddp._submit_model_ep_aligned_nep_scatter_chunk()
    with pytest.raises(RuntimeError, match="different windows"):
        ddp._submit_model_ep_aligned_nep_scatter_chunk()


def test_nep_scatter_submission_skips_empty_window_and_launches_next_agreement(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    ddp._nonuniform_ep_runtime_config = {"needs_reshard": True, "ep_group_gloo": object()}
    ddp._nep_scatter_alignment_tensor = None
    ddp._nep_scatter_alignment_work = None
    ddp._peek_nep_scatter_ticket_window = lambda: ("empty", 0, None, None)
    ddp._submit_nep_scatter_chunk = lambda **kwargs: pytest.fail("empty windows must not submit")

    class FakeWork:
        def __init__(self, index):
            self.index = index

        def wait(self):
            calls.append(("wait", self.index))

    def all_reduce(tensor, op, group, async_op):
        index = len([call for call in calls if call[0] == "launch"])
        calls.append(("launch", index, tuple(tensor.tolist())))
        return FakeWork(index)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    assert not ddp._submit_model_ep_aligned_nep_scatter_chunk()
    assert not ddp._submit_model_ep_aligned_nep_scatter_chunk()
    assert [call[:2] for call in calls] == [("launch", 0), ("wait", 0), ("launch", 1)]
    expected_payload = (0, 0) + (-1,) * 12 + (0, 0) + (1,) * 12
    assert calls[0][2] == expected_payload


def test_nep_scatter_submission_drain_consumes_pending_ticket(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    ticket = (7, 0, 2, 3, 1, 4)
    payload = (
        (2, 1)
        + ticket
        + ticket
        + (-2, -1)
        + tuple(-value for value in ticket)
        + tuple(-value for value in ticket)
    )

    class FakeWork:
        def wait(self):
            calls.append("wait")

    ddp._nonuniform_ep_runtime_config = {"needs_reshard": True}
    ddp._nep_model_ep_a2a_burst_depth = 0
    ddp._nep_scatter_backward_complete = True
    ddp._nep_scatter_alignment_tensor = torch.tensor(payload, dtype=torch.int64)
    ddp._nep_scatter_alignment_work = FakeWork()
    ddp._nep_scatter_batches = []
    ddp._peek_nep_scatter_ticket_window = lambda: ("ready", 1, ticket, ticket)
    ddp._submit_nep_scatter_chunk = (
        lambda after_event=None, force=False, queue_behind_inflight=False: calls.append(
            ("submit", after_event, force, queue_behind_inflight)
        )
        or True
    )
    ddp._retire_nep_scatter_chunk = lambda force=False: calls.append(("retire", force)) or True
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER", "1")

    ddp._drain_nep_scatter_scheduler()

    assert calls == ["wait", ("submit", None, False, True), ("retire", True)]
    assert ddp._nep_scatter_alignment_work is None


def test_nep_scatter_submission_drain_queues_all_chunks_before_one_wait(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    ddp._nonuniform_ep_runtime_config = {"needs_reshard": False}
    ddp._nep_model_ep_a2a_burst_depth = 0
    ddp._nep_scatter_backward_complete = True
    ddp._nep_scatter_batches = [object(), object(), object()]

    def submit(after_event=None, force=False, queue_behind_inflight=False):
        calls.append(("submit", after_event, force, queue_behind_inflight))
        ddp._nep_scatter_batches.pop(0)
        return True

    ddp._submit_nep_scatter_chunk = submit
    ddp._retire_nep_scatter_chunk = lambda force=False: calls.append(("retire", force)) or True
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER", "1")

    ddp._drain_nep_scatter_scheduler()

    assert calls == [
        ("submit", None, False, True),
        ("submit", None, False, True),
        ("submit", None, False, True),
        ("retire", True),
    ]


def test_nep_scatter_chunk_scheduler_marks_ready_before_host_phase_check(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    host_phases = object()
    completion_event = object()
    context = object()
    group = SimpleNamespace(_nep_nccl_group_index=4, _nep_nccl_ready=False)

    def finish_host_phases(phases, **kwargs):
        assert phases is host_phases
        calls.append("finish_host_phases")
        kwargs["scatter_context_batches"].append([context])
        return True

    boundary_group = SimpleNamespace(
        _finish_nep_nccl_process_group_dispatch_batches=finish_host_phases
    )

    def queue_contexts(context_batches, event, module_label, a2a_event):
        calls.append(("queue", context_batches, event, module_label, a2a_event))
        group._nep_nccl_ready = True

    ddp._queue_nep_scatter_context_batches = queue_contexts
    ddp._nep_dispatch_waiting_groups = (group,)
    ddp._nep_dispatch_waiting_module_label = "decoder.layers.1.mlp"
    ddp._nep_dispatch_pending_completion_event = completion_event
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nep_dispatch_pending_host_phases = (boundary_group, host_phases)
    a2a_event = object()
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER", "1")

    assert ddp._finish_pending_nep_dispatch_host_phases(scatter_after_event=a2a_event)

    assert calls == [
        "finish_host_phases",
        ("queue", [[context]], completion_event, "decoder.layers.1.mlp", a2a_event),
    ]
    assert ddp._nep_dispatch_pending_host_phases is None


def test_nep_finish_grad_sync_drains_deferred_dispatch_completions(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    ddp.ddp_config = SimpleNamespace(overlap_grad_reduce=False)
    ddp.expert_parallel_bucket_groups = []
    ddp.bucket_groups = []
    ddp._wait_for_nep_dispatch_launch = lambda final=False: calls.append(("wait", final))
    parent_type = NonuniformEPDistributedDataParallel.__mro__[1]
    monkeypatch.setattr(
        parent_type,
        "finish_grad_sync",
        lambda self, force_all_reduce=False: calls.append(("parent", force_all_reduce)),
    )

    ddp.finish_grad_sync(force_all_reduce=True)

    assert calls == [("wait", True), ("parent", True)]


def test_nep_post_graph_progress_preserves_model_ep_completion_fence(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    dispatch_stream = object()
    host_phases = object()

    class FakeCompletionEvent:
        def record(self, stream):
            calls.append(("record_completion", stream))

    group = SimpleNamespace(_nep_nccl_group_index=4, _nep_nccl_ready=False)

    def finish_pending(
        phases, device_align_phases=False, finish_all_phases=True, defer_scatter_submission=False
    ):
        assert phases is host_phases
        calls.append(("finish_host_phases", device_align_phases, finish_all_phases))
        if len(calls) == 1:
            return False
        group._nep_nccl_ready = True
        return True

    boundary_group = SimpleNamespace(
        _finish_nep_nccl_process_group_dispatch_batches=finish_pending,
        _get_nep_nccl_comm_stream=lambda slot: dispatch_stream,
    )
    completion_event = FakeCompletionEvent()
    ddp._nep_dispatch_waiting_groups = (group,)
    ddp._nep_dispatch_waiting_module_label = "decoder.layers.1.mlp"
    ddp._nep_dispatch_pending_completion_event = completion_event
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nep_dispatch_pending_host_phases = (boundary_group, host_phases)
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES", "1")

    ddp._progress_nep_dispatch_after_graph_launch()

    assert calls == [("finish_host_phases", True, False)]
    assert ddp._nep_dispatch_pending_host_phases == (boundary_group, host_phases)
    assert ddp._nep_dispatch_pending_completion_event is completion_event
    assert ddp._nep_dispatch_waiting_groups == (group,)

    ddp._progress_nep_dispatch_after_graph_launch()

    assert calls[1:] == [
        ("finish_host_phases", True, False),
        ("record_completion", dispatch_stream),
    ]
    assert ddp._nep_dispatch_pending_host_phases is None
    assert ddp._nep_dispatch_pending_completion_event is completion_event
    assert ddp._nep_dispatch_waiting_groups == (group,)


def test_nep_post_graph_host_progress_launches_full_pipeline_before_model_ep_fence(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    dispatch_stream = object()
    host_phases = object()

    class FakeCompletionEvent:
        def record(self, stream):
            calls.append(("record_completion", stream))

    completion_event = FakeCompletionEvent()
    compute_stream = SimpleNamespace(
        wait_event=lambda event: calls.append(("wait_completion", event))
    )
    group = SimpleNamespace(
        _nep_nccl_group_index=4, _nep_nccl_ready=False, _nep_dispatch_boundary_launched=True
    )

    def finish_pending(
        phases, device_align_phases=False, finish_all_phases=True, defer_scatter_submission=False
    ):
        assert phases is host_phases
        calls.append(("finish_host_phases", device_align_phases, finish_all_phases))
        group._nep_nccl_ready = True
        return True

    boundary_group = SimpleNamespace(
        _finish_nep_nccl_process_group_dispatch_batches=finish_pending,
        _get_nep_nccl_comm_stream=lambda slot: dispatch_stream,
    )
    ddp._nonuniform_ep_runtime_config = {"zero_sm_reshard": False}
    ddp._nep_dispatch_waiting_groups = (group,)
    ddp._nep_dispatch_waiting_module_label = "decoder.layers.1.mlp"
    ddp._nep_dispatch_pending_completion_event = completion_event
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nep_dispatch_pending_host_phases = (boundary_group, host_phases)
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES", "1")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)

    ddp._progress_nep_dispatch_after_graph_launch()

    assert calls == [("finish_host_phases", False, True), ("record_completion", dispatch_stream)]
    assert ddp._nep_dispatch_pending_host_phases is None
    assert ddp._nep_dispatch_pending_completion_event is completion_event

    ddp._wait_for_nep_dispatch_launch()

    assert calls[-1] == ("wait_completion", completion_event)
    assert ddp._nep_dispatch_pending_completion_event is None
    assert ddp._nep_dispatch_waiting_groups is None


def test_nep_deferred_post_graph_launches_full_pipeline_in_order(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    ddp._launch_waiting_nep_dispatch_boundary_tasks = lambda: calls.append("launch_gather")
    ddp._finish_pending_nep_dispatch_host_phases = lambda: calls.append("launch_edp_scatter")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH", "1")
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES", "1")

    ddp._progress_nep_dispatch_after_graph_launch()

    assert calls == ["launch_gather", "launch_edp_scatter"]


def test_nep_model_ep_fence_drains_staged_post_graph_phases(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    dispatch_stream = object()
    host_phases = object()

    class FakeCompletionEvent:
        def record(self, stream):
            calls.append(("record_completion", stream))

    completion_event = FakeCompletionEvent()
    compute_stream = SimpleNamespace(
        wait_event=lambda event: calls.append(("wait_completion", event))
    )

    group = SimpleNamespace(
        _nep_nccl_group_index=4, _nep_nccl_ready=False, _nep_dispatch_boundary_launched=True
    )

    def finish_pending(
        phases, device_align_phases=False, finish_all_phases=True, defer_scatter_submission=False
    ):
        assert phases is host_phases
        calls.append(("finish_host_phases", device_align_phases, finish_all_phases))
        if not finish_all_phases:
            return False
        group._nep_nccl_ready = True
        return True

    boundary_group = SimpleNamespace(
        _finish_nep_nccl_process_group_dispatch_batches=finish_pending,
        _get_nep_nccl_comm_stream=lambda slot: dispatch_stream,
    )
    ddp._nonuniform_ep_runtime_config = {"zero_sm_reshard": False}
    ddp._nep_dispatch_waiting_groups = (group,)
    ddp._nep_dispatch_waiting_module_label = "decoder.layers.1.mlp"
    ddp._nep_dispatch_pending_completion_event = completion_event
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nep_dispatch_pending_host_phases = (boundary_group, host_phases)
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES", "1")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: compute_stream)

    ddp._progress_nep_dispatch_after_graph_launch()
    ddp._wait_for_nep_dispatch_launch()

    assert calls == [
        ("finish_host_phases", True, False),
        ("finish_host_phases", True, True),
        ("record_completion", dispatch_stream),
        ("wait_completion", completion_event),
    ]
    assert ddp._nep_dispatch_pending_host_phases is None
    assert ddp._nep_dispatch_pending_completion_event is None
    assert ddp._nep_dispatch_waiting_groups is None


def test_nep_dispatch_worker_launches_before_completion(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    groups = (object(),)
    compute_ready_event = object()
    completion_event = object()
    boundary_group = object()

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(("device", device)))
    ddp._run_nep_dispatch_boundary_tasks = lambda *args: calls.append(("launch", args))
    ddp._complete_nep_dispatch_boundary = lambda event, group: (
        calls.append(("complete", event, group)) or (1.25, 2.5)
    )

    result = ddp._launch_and_complete_nep_dispatch_boundary(
        groups, "decoder.layers.1.mlp", compute_ready_event, completion_event, boundary_group, 7
    )

    assert calls == [
        ("device", 7),
        ("launch", (groups, "decoder.layers.1.mlp", compute_ready_event, completion_event)),
        ("complete", completion_event, boundary_group),
    ]
    assert result == (1.25, 2.5)


def test_nep_dispatch_completion_worker_fences_event_before_barrier(monkeypatch):
    calls = []
    boundary_group = object()

    class FakeCompletionEvent:
        def synchronize(self):
            calls.append("wait_completion")

    def fake_barrier(group):
        assert group is boundary_group
        calls.append("completion_barrier")

    monkeypatch.setattr(torch.distributed, "barrier", fake_barrier)

    completion_wait_ms, completion_barrier_ms = (
        NonuniformEPDistributedDataParallel._complete_nep_dispatch_boundary(
            FakeCompletionEvent(), boundary_group
        )
    )

    assert calls == ["wait_completion", "completion_barrier"]
    assert completion_wait_ms >= 0.0
    assert completion_barrier_ms >= 0.0


def test_nep_dispatch_wait_joins_completion_worker_before_clearing():
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    ddp._nonuniform_ep_runtime_config = {"zero_sm_reshard": True}

    class FakeCompletionEvent:
        def synchronize(self):
            raise AssertionError("autograd thread must not synchronize the completion event")

    class FakeCompletionFuture:
        def result(self):
            calls.append("join_completion_worker")
            return 1.25, 2.5

    group = SimpleNamespace(_nep_dispatch_boundary_launched=True)
    ddp._nep_dispatch_waiting_groups = (group,)
    ddp._nep_dispatch_waiting_module_label = "decoder.layers.1.mlp"
    ddp._nep_dispatch_pending_completion_event = FakeCompletionEvent()
    ddp._nep_dispatch_pending_completion_future = FakeCompletionFuture()

    ddp._wait_for_nep_dispatch_launch()

    assert calls == ["join_completion_worker"]
    assert ddp._nep_dispatch_pending_completion_event is None
    assert ddp._nep_dispatch_pending_completion_future is None
    assert ddp._nep_dispatch_waiting_groups is None
    assert ddp._nep_dispatch_waiting_module_label is None


def test_nep_dispatch_boundary_waits_for_local_inputs(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    group = type("Group", (), {})()
    group._nep_nccl_group_index = 4
    group._nep_dispatch_boundary_ready = True
    group._nep_dispatch_boundary_launched = False
    group._nep_dispatch_boundary_launching = False
    group._nep_dispatch_boundary_wait_logged = False
    group._nep_dispatch_boundary_inputs_ready = lambda: False
    calls = []
    ddp._nonuniform_ep_runtime_config = {"dp_cp_group_gloo": object()}
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: calls.append("barrier"))

    assert not ddp._launch_nep_dispatch_boundary_tasks((group,), "decoder.layers.1.mlp")
    assert calls == []
    assert group._nep_dispatch_boundary_wait_logged


def test_nep_nccl_host_gate_waits_for_gather_and_peer_owner(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    fake_gloo_group = object()

    class FakeEvent:
        ready = False
        recorded_stream = None

        def query(self):
            return self.ready

        def record(self, stream):
            self.recorded_stream = stream

    class FakeReadyWork:
        completed = False

        def is_completed(self):
            return self.completed

    class FakeStreamContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    gather_event = FakeEvent()
    edp_event = FakeEvent()
    ready_work = FakeReadyWork()
    fake_stream = object()
    context = {
        "group": bucket_group,
        "owner_ep_rank": 0,
        "chunk_index": 0,
        "buffer_slot": 0,
        "stage": "gather",
        "gather_done_event": gather_event,
    }
    bucket_group._nep_runtime_config = {
        "ep_rank": 0,
        "edp_group_gloo": fake_gloo_group,
        "nep_owner_transfer_groups_gloo": {},
    }
    bucket_group._nep_nccl_group_index = 0
    bucket_group._nep_nccl_scheduler_state = {"pending_owner_tasks": [context]}
    bucket_group._get_nep_nccl_comm_stream = lambda slot: fake_stream
    bucket_group._start_nep_nccl_owner_edp_reduce = lambda task, use_device_readiness: calls.append(
        ("edp", use_device_readiness)
    )
    bucket_group._start_nep_nccl_owner_task_scatter = lambda task: calls.append("scatter")

    def fake_barrier(group, async_op):
        assert group is fake_gloo_group
        assert async_op
        calls.append("barrier")
        return ready_work

    monkeypatch.setattr(torch.distributed, "barrier", fake_barrier)
    monkeypatch.setattr(torch.cuda, "Event", lambda: edp_event)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: FakeStreamContext())

    assert not bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == []

    gather_event.ready = True
    assert not bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ["barrier"]

    ready_work.completed = True
    assert not bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ["barrier", ("edp", False)]
    assert edp_event.recorded_stream is fake_stream

    edp_event.ready = True
    assert bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ["barrier", ("edp", False), "scatter"]
    assert bucket_group._nep_nccl_scheduler_state["pending_owner_tasks"] == []


def test_nep_nccl_follower_waits_for_owner_before_scatter(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    source_group = object()

    class FakeEvent:
        def query(self):
            return True

    class FakeReadyWork:
        def __init__(self):
            self.completed = False

        def is_completed(self):
            return self.completed

    class FakeStreamContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    gather_work = FakeReadyWork()
    scatter_work = FakeReadyWork()
    works = [gather_work, scatter_work]
    context = {
        "group": bucket_group,
        "owner_ep_rank": 0,
        "chunk_index": 0,
        "buffer_slot": 0,
        "stage": "gather",
        "gather_done_event": FakeEvent(),
    }
    bucket_group._nep_runtime_config = {
        "ep_rank": 1,
        "nep_owner_transfer_groups_gloo": {0: source_group},
    }
    bucket_group._nep_nccl_group_index = 0
    bucket_group._nep_nccl_scheduler_state = {"pending_owner_tasks": [context]}
    bucket_group._get_nep_nccl_comm_stream = lambda slot: object()
    bucket_group._start_nep_nccl_owner_task_scatter = lambda task: calls.append("scatter")

    def fake_barrier(group, async_op):
        assert group is source_group
        assert async_op
        calls.append("barrier")
        return works.pop(0)

    monkeypatch.setattr(torch.distributed, "barrier", fake_barrier)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: FakeStreamContext())

    assert not bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ["barrier"]

    gather_work.completed = True
    assert not bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ["barrier", "barrier"]

    scatter_work.completed = True
    assert bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ["barrier", "barrier", "scatter"]
    assert bucket_group._nep_nccl_scheduler_state["pending_owner_tasks"] == []


def test_nep_nccl_combined_slot_owner_layout_round_trip():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    first_grad = torch.tensor([10.0, 11.0])
    second_grad = torch.tensor([20.0, 21.0, 22.0])
    bucket_group._nep_runtime_config = {"zero_sm_reshard": False}
    bucket_group._nep_nccl_slot_numel = 5
    bucket_group._nep_nccl_slot_numels = (2, 3)
    bucket_group._nep_nccl_slot_offsets = (0, 2)
    bucket_group._nep_nccl_expert_stride = 5
    bucket_group._nep_nccl_experts_per_owner = 2
    bucket_group._nep_nccl_entries = [
        {
            "expert_id": 1,
            "slot_index": 0,
            "slot_offset": 0,
            "entry_key": (1, 0),
            "bucket": SimpleNamespace(grad_data=first_grad),
            "numel": 2,
        },
        {
            "expert_id": 1,
            "slot_index": 1,
            "slot_offset": 2,
            "entry_key": (1, 1),
            "bucket": SimpleNamespace(grad_data=second_grad),
            "numel": 3,
        },
    ]

    owner_chunk = torch.empty(10)
    bucket_group._pack_nep_nccl_owner_chunk(0, 0, 10, owner_chunk)

    assert torch.count_nonzero(owner_chunk[:5]) == 0
    assert torch.equal(owner_chunk[5:], torch.tensor([10.0, 11.0, 20.0, 21.0, 22.0]))

    bucket_group._copy_nep_nccl_owner_chunk_to_local_grads(0, 0, 10, owner_chunk + 100.0)
    assert torch.equal(first_grad, torch.tensor([110.0, 111.0]))
    assert torch.equal(second_grad, torch.tensor([120.0, 121.0, 122.0]))


def test_nep_nccl_dense_source_payload_round_trip():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    source_grad = torch.tensor([20.0, 21.0, 22.0, 23.0])
    source_bucket = type("Bucket", (), {"grad_data": source_grad})()
    bucket_group._nep_runtime_config = {"expert_placement": [[0, 1], [4, 5], [2, 6], [3, 7]]}
    bucket_group._nep_nccl_slot_numel = 4
    bucket_group._nep_nccl_experts_per_owner = 4
    bucket_group._nep_nccl_entries = [
        {"expert_id": 2, "bucket": source_bucket, "numel": source_grad.numel()}
    ]

    payload = torch.empty(4)
    bucket_group._pack_nep_nccl_source_payload(0, 2, 0, 16, payload)
    assert torch.equal(payload, source_grad)

    owner_chunk = torch.zeros(16)
    bucket_group._accumulate_nep_nccl_source_payload(0, 2, 0, 16, payload, owner_chunk)
    assert torch.equal(owner_chunk[8:12], source_grad)
    assert torch.count_nonzero(owner_chunk[:8]) == 0
    assert torch.count_nonzero(owner_chunk[12:]) == 0

    reduced_payload = torch.empty(4)
    bucket_group._pack_nep_nccl_scatter_payload(0, 2, 0, 16, owner_chunk + 100.0, reduced_payload)
    source_grad.zero_()
    bucket_group._copy_nep_nccl_scatter_payload_to_local_grads(0, 2, 0, 16, reduced_payload)
    assert torch.equal(source_grad, torch.tensor([120.0, 121.0, 122.0, 123.0]))
