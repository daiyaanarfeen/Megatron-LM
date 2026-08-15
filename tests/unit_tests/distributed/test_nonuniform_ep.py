# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.nonuniform_common import (
    build_expert_axis_permutation,
    build_expert_to_ep_rank_map,
    compute_nonuniform_ep_dispatch_slots,
    compute_nonuniform_ep_expert_placement,
    compute_nonuniform_ep_owner_expert_slots,
    get_nonuniform_ep_expert_axis_permutation,
    get_nonuniform_ep_local_expert_indices,
    set_nonuniform_ep_runtime_config,
)
from megatron.core.distributed.nonuniform_ep import (
    NonuniformEPDistributedDataParallel,
    NonuniformEPNCCLParamAndGradBucketGroup,
    _build_nep_nccl_scatter_chunk_ranges,
    _compute_nep_distopt_owner_layout,
    _ExpertBucketSpec,
    _get_nep_nccl_scatter_chunks,
    _group_expert_bucket_specs_in_backward_order,
    _nep_distopt_proxy_name,
    _nep_owner_ddp_config,
    _partition_expert_bucket_specs,
    _source_ep_ranks_for_owner,
)
from megatron.core.optimizer import _get_param_groups_and_buffers
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    _pad_nonuniform_flex_dispatch_slots,
)
from tests.unit_tests.test_utilities import Utils


def test_nep_distopt_owner_layout_uses_only_real_params_and_native_padding():
    config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    params = [
        torch.nn.Parameter(torch.empty(3, dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.empty(5, dtype=torch.bfloat16)),
    ]

    layout = _compute_nep_distopt_owner_layout(params, 2, config)

    assert set(layout.param_index_map) == set(params)
    assert layout.param_index_map[params[1]] == (0, 5, 0)
    assert layout.param_index_map[params[0]] == (64, 67, 0)
    assert layout.per_bucket_numel_unpadded == [67]
    assert layout.bucket_indices[0][1] % 2 == 0
    assert layout.bucket_indices[0][1] >= 67


def test_nep_distopt_proxy_name_materializes_logical_expert():
    assert (
        _nep_distopt_proxy_name(
            ("decoder.layers.3.mlp.experts.local_experts.{expert}.linear_fc1.weight",), 11
        )
        == "decoder.layers.3.mlp.experts.local_experts.11.linear_fc1.weight"
    )


def test_nep_distopt_factory_substitutes_only_in_distopt_mode():
    Utils.initialize_distributed()
    physical_param = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    physical_param.allreduce = False
    owner_param = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    owner_param.allreduce = False
    physical_buffer = object()
    owner_buffer = object()

    class _FakeChunk:
        expert_parallel_buffers = [physical_buffer]

        def named_parameters(self):
            return iter((("physical_expert.weight", physical_param),))

        def get_nonuniform_ep_distributed_optimizer_state(self):
            return ((('logical_expert.weight', owner_param),), (owner_buffer,))

    regular_config = OptimizerConfig(optimizer='adam', bf16=True)
    regular_groups, regular_buffers = _get_param_groups_and_buffers(
        [_FakeChunk()],
        model_chunk_offset=0,
        config=regular_config,
        config_overrides={},
        filter_fn=lambda group: group['is_expert_parallel'],
        buffer_name='expert_parallel_buffers',
    )
    assert [param for group in regular_groups for param in group['params']] == [physical_param]
    assert regular_buffers == {0: [physical_buffer]}

    distopt_config = OptimizerConfig(optimizer='adam', bf16=True, use_distributed_optimizer=True)
    distopt_groups, distopt_buffers = _get_param_groups_and_buffers(
        [_FakeChunk()],
        model_chunk_offset=0,
        config=distopt_config,
        config_overrides={},
        filter_fn=lambda group: group['is_expert_parallel'],
        buffer_name='expert_parallel_buffers',
    )
    assert [param for group in distopt_groups for param in group['params']] == [owner_param]
    assert distopt_buffers == {0: [owner_buffer]}


def test_nep_distopt_param_sync_is_idempotent_per_iteration():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group.ddp_config = SimpleNamespace(
        use_distributed_optimizer=True, overlap_param_gather=False
    )
    bucket_group._nep_distopt_param_sync_representative = True
    bucket_group.param_gather_dispatched = False

    class _NativeGroup:
        def __init__(self):
            self.starts = 0
            self.posts = 0

        def start_param_sync(self, force_sync=False):
            assert not force_sync
            self.starts += 1

        def _post_param_sync(self):
            self.posts += 1

    native_group = _NativeGroup()
    bucket_group._nep_distopt_owner_bundle = {"native_group": native_group}
    scatters = []
    bucket_group._scatter_nep_distopt_owner_params_to_physical_holders = lambda: scatters.append(
        True
    )

    bucket_group.start_param_sync()
    bucket_group.start_param_sync()

    assert native_group.starts == 1
    assert native_group.posts == 1
    assert scatters == [True]


def test_nep_distopt_async_param_sync_uses_native_forward_lifecycle():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group.ddp_config = SimpleNamespace(
        use_distributed_optimizer=True, overlap_param_gather=True
    )
    bucket_group._nep_distopt_param_sync_representative = True
    bucket_group.param_gather_dispatched = False

    class _NativeGroup:
        def __init__(self):
            self.starts = 0
            self.finishes = 0

        def start_param_sync(self, force_sync=False):
            assert not force_sync
            self.starts += 1

        def finish_param_sync(self, skip_next_bucket_dispatch=False):
            assert skip_next_bucket_dispatch
            self.finishes += 1

    class _NextGroup:
        def __init__(self):
            self.starts = 0

        def start_param_sync(self):
            self.starts += 1

    native_group = _NativeGroup()
    next_group = _NextGroup()
    bucket_group._nep_distopt_owner_bundle = {
        "native_group": native_group,
        "param_sync_completed": False,
    }
    bucket_group.next_param_gather_bucket_group = next_group
    scatters = []
    bucket_group._scatter_nep_distopt_owner_params_to_physical_holders = lambda: scatters.append(
        True
    )

    bucket_group.start_param_sync()
    assert native_group.starts == 1
    assert native_group.finishes == 0
    assert scatters == []

    bucket_group.finish_param_sync()
    bucket_group.finish_param_sync()
    assert native_group.finishes == 1
    assert scatters == [True]
    assert next_group.starts == 1

    # The next forward calls finish directly. It must launch a fresh native
    # all-gather rather than returning on the previous iteration's completion.
    bucket_group.param_gather_dispatched = False
    bucket_group.finish_param_sync()
    assert native_group.starts == 2
    assert native_group.finishes == 2
    assert scatters == [True, True]
    assert next_group.starts == 2


def test_nep_distopt_scale_gradients_includes_owner_buffers(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    base_scalings = []
    monkeypatch.setattr(
        "megatron.core.distributed.nonuniform_ep.DistributedDataParallel.scale_gradients",
        lambda _self, scaling_factor: base_scalings.append(scaling_factor),
    )

    class _OwnerBuffer:
        def __init__(self):
            self.scalings = []

        def scale_gradients(self, scaling_factor):
            self.scalings.append(scaling_factor)

    owner_buffers = [_OwnerBuffer(), _OwnerBuffer()]
    ddp.nonuniform_ep_distributed_optimizer_buffers = owner_buffers

    ddp.scale_gradients(0.125)

    assert base_scalings == [0.125]
    assert [buffer.scalings for buffer in owner_buffers] == [[0.125], [0.125]]


def test_nep_distopt_owner_task_skips_gradient_scatter():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group.ddp_config = SimpleNamespace(use_distributed_optimizer=True)
    completed = []
    bucket_group._mark_nep_distopt_task_complete = completed.append
    context = {"owner_ep_rank": 0, "chunk_index": 0}

    bucket_group._start_nep_nccl_owner_task_scatter(context)

    assert completed == [context]


def test_nep_distopt_param_transfer_slots_preserve_adjacent_storage():
    physical_params = [torch.nn.Parameter(torch.empty(numel)) for numel in (5, 8, 6)]

    def make_group(index, physical_param):
        group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
            NonuniformEPNCCLParamAndGradBucketGroup
        )
        group._nep_nccl_group_index = index
        group._get_nep_nccl_owner_layout = lambda: {
            "min_ep_size": 1,
            "owner_numel": physical_param.numel(),
        }
        group._get_nep_nccl_transfer_group_info = lambda owner: (None, None, 1, (0,))
        group._nep_nccl_owner_entries = lambda owner: [
            {
                "bucket": SimpleNamespace(params_list=[physical_param]),
                "numel": physical_param.numel(),
            }
        ]
        group._nep_nccl_owner_expert_ids = lambda owner: ()
        group._nep_nccl_entry_owner_start = lambda entry, owner: 0
        group._nep_runtime_config = {"ep_rank": 0, "ep_group": None}
        return group

    groups = tuple(
        make_group(index, physical_param) for index, physical_param in enumerate(physical_params)
    )
    shared_transfer_buffers = {}
    for index, group in enumerate(groups):
        group._nep_distopt_owner_bundle = {
            "groups": (group,),
            "proxy_by_key": {},
            "param_transfer_buffers": shared_transfer_buffers,
            "param_transfer_slot": index % 2,
            "param_transfer_numel_by_slot": {0: 6, 1: 8},
        }

    storage_by_group = []
    for group in groups:
        group._scatter_nep_distopt_owner_params_to_physical_holders()
        storage_by_group.append(
            shared_transfer_buffers[
                (
                    group._nep_distopt_owner_bundle["param_transfer_slot"],
                    physical_params[0].dtype,
                    physical_params[0].device,
                )
            ]
        )

    assert len(shared_transfer_buffers) == 2
    assert storage_by_group[0] is not storage_by_group[1]
    assert storage_by_group[0] is storage_by_group[2]
    assert storage_by_group[0].numel() == 6
    assert storage_by_group[1].numel() == 8


@pytest.mark.parametrize(
    ("use_distributed_optimizer", "ep_rank", "expected_numel"),
    ((True, 0, 8), (True, 1, 0), (False, 1, 8)),
)
def test_nep_distopt_allocates_owner_layout_scratch_only_on_owner(
    use_distributed_optimizer, ep_rank, expected_numel
):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    grad_data = torch.empty(8)
    bucket_group.buckets = [SimpleNamespace(grad_data=grad_data)]
    bucket_group.ddp_config = SimpleNamespace(use_distributed_optimizer=use_distributed_optimizer)
    bucket_group._nep_runtime_config = {"ep_rank": ep_rank}
    bucket_group._nep_nccl_async_tensors = []
    state = {"gather_buf_cache": {}, "buffer_slot_handles": {}, "buffer_slot_events": {}}
    bucket_group._nep_nccl_scheduler_state = state
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: 7
    bucket_group._prep_nep_nccl_owner_entries_for_sync = lambda owner: None

    context = bucket_group._prepare_nep_nccl_owner_task_context(
        owner_ep_rank=0, chunk_index=0, chunk_start=0, chunk_end=8, async_op=True
    )

    assert context["chunk"].numel() == expected_numel
    full_key = ("owner_layout_gather", 7, 8, grad_data.dtype, grad_data.device)
    assert (full_key in state["gather_buf_cache"]) == (expected_numel == 8)


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


def test_nep_owner_ddp_config_disables_redundant_grad_checks():
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


def test_process_group_gather_selects_separate_owner_group():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []

    class GroupSelected(Exception):
        pass

    bucket_group._nep_runtime_config = {"ep_rank": 0, "nep_owner_gather_groups": {0: object()}}
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


def test_nondivisible_ep8_ep6_uses_balanced_logical_experts_and_virtual_slots():
    owner_slots = compute_nonuniform_ep_owner_expert_slots(16, 6)
    assert owner_slots == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12, 13, None],
        [14, 15, None],
    ]

    reduced_placement, reduced_gather_map = compute_nonuniform_ep_expert_placement(16, 6, 6)
    assert reduced_placement == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13], [14, 15]]
    assert reduced_gather_map == {}
    assert compute_nonuniform_ep_dispatch_slots(reduced_placement, 16) == owner_slots

    full_placement, full_gather_map = compute_nonuniform_ep_expert_placement(16, 8, 6)
    assert full_placement == [[0, 1], [3, 4], [6, 7], [9, 10], [12, 13], [14, 15], [2, 5], [8, 11]]
    assert full_gather_map == {6: [(0, 0, 2), (1, 1, 2)], 7: [(0, 2, 2), (1, 3, 2)]}
    assert compute_nonuniform_ep_dispatch_slots(full_placement, 16) == full_placement
    assert sorted(expert_id for experts in full_placement for expert_id in experts) == list(
        range(16)
    )


def test_flex_metadata_maps_logical_experts_to_padded_ep6_slots():
    owner_slots = compute_nonuniform_ep_owner_expert_slots(16, 6)
    dispatch_slots = _pad_nonuniform_flex_dispatch_slots(owner_slots, 6, "hybridep")
    assert dispatch_slots == [
        [0, 1, 2, None],
        [3, 4, 5, None],
        [6, 7, 8, None],
        [9, 10, 11, None],
        [12, 13, None, None],
        [14, 15, None, None],
    ]

    dispatcher = MoEFlexTokenDispatcher.__new__(MoEFlexTokenDispatcher)
    dispatcher.ep_size = 6
    dispatcher.tp_size = 1
    dispatcher.num_local_expert_slots = 4
    physical_slots = [
        0 if expert_id is None else expert_id for slots in dispatch_slots for expert_id in slots
    ]
    active_slots = [expert_id is not None for slots in dispatch_slots for expert_id in slots]
    dispatcher._dispatch_expert_axis = torch.tensor(physical_slots)
    dispatcher._dispatch_expert_slot_mask = torch.tensor(active_slots)

    routing_map = torch.eye(16, dtype=torch.bool)
    probs = torch.eye(16, dtype=torch.float32)
    physical_map, physical_probs = dispatcher._initialize_metadata(routing_map, probs)

    assert physical_map.shape == (16, 6, 4)
    assert physical_probs.shape == (16, 6, 4)
    flat_map = physical_map.reshape(16, 24)
    flat_probs = physical_probs.reshape(16, 24)
    for physical_index, expert_id in enumerate(physical_slots):
        if active_slots[physical_index]:
            assert flat_map[:, physical_index].equal(routing_map[:, expert_id])
            assert flat_probs[:, physical_index].equal(probs[:, expert_id])
        else:
            assert not flat_map[:, physical_index].any()
            assert torch.count_nonzero(flat_probs[:, physical_index]) == 0


class TestNonuniformEPTokenRouting:
    def teardown_method(self, _method):
        set_nonuniform_ep_runtime_config(None)

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


def test_nep_nccl_partition_splits_module_slots_to_reach_larger_target():
    grouped_specs = [
        ((f"decoder.layers.{layer}.mlp.experts.{slot}",), [])
        for layer in range(4, 0, -1)
        for slot in ("linear_fc2.weight", "linear_fc1.weight")
    ]

    partitions = _partition_expert_bucket_specs(grouped_specs, 6)

    assert [len(partition) for partition in partitions] == [2, 2, 1, 1, 1, 1]
    assert [slot_key for partition in partitions for slot_key, _ in partition] == [
        slot_key for slot_key, _ in grouped_specs
    ]


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


def test_nep_end_iteration_scatter_uses_persistent_task_slots():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_nccl_owner_layout = {"min_ep_size": 3, "num_chunks": 2}
    bucket_group._nep_nccl_scheduler_state = {"group_slot_offsets": (0, 6)}
    bucket_group._nep_nccl_group_index = 0

    first_group_slots = [
        bucket_group._get_nep_nccl_task_buffer_slot(owner_ep_rank, chunk_index)
        for owner_ep_rank in range(3)
        for chunk_index in range(2)
    ]
    bucket_group._nep_nccl_group_index = 1
    second_group_slots = [
        bucket_group._get_nep_nccl_task_buffer_slot(owner_ep_rank, chunk_index)
        for owner_ep_rank in range(3)
        for chunk_index in range(2)
    ]

    assert first_group_slots == list(range(6))
    assert second_group_slots == list(range(6, 12))


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
        group._nep_nccl_scheduler_state = {"group_slot_offsets": (0,)}
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


def test_nep_packed_scatter_orders_every_unique_edp_dependency(monkeypatch):
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS", "1")
    calls = []
    first_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    second_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    for index, group in enumerate((first_group, second_group)):
        group._nep_runtime_config = {}
        group._order_nep_nccl_owner_edp_before_scatter = lambda context, index=index: calls.append(
            ("order", index)
        )

    contexts = [
        {"group": first_group, "owner_ep_rank": 0, "chunk_index": 0, "native_edp_state": object()},
        {"group": second_group, "owner_ep_rank": 0, "chunk_index": 1, "native_edp_state": object()},
    ]
    scatter_context = dict(contexts[0])
    scatter_context["scatter_contexts"] = tuple(contexts)
    first_group._prepare_nep_nccl_owner_all_to_all_scatter_batch = (
        lambda scatter_contexts: calls.append(("prepare", scatter_contexts))
        or {"kind": "all_to_all"}
    )

    train = first_group._prepare_nep_nccl_owner_task_scatter_train(scatter_context)

    assert train["descriptors"] == [{"kind": "all_to_all"}]
    assert calls == [("order", 0), ("order", 1), ("prepare", contexts)]


def test_nep_two_level_scatter_batch_combines_and_splits_payloads():
    runtime_config = {"ep_rank": 0}
    cache = {}
    received = []

    def cached_tensor(tensor_cache, key, numel, dtype, device):
        tensor = torch.empty(numel, dtype=dtype, device=device)
        tensor_cache[key] = tensor
        return tensor

    def configure_group(group, group_index, edp_bucket_index, payload_numel, payload_base):
        group._nep_nccl_group_index = group_index
        group._nep_nccl_edp_bucket_index = edp_bucket_index
        group._nep_runtime_config = runtime_config
        group._nep_nccl_owner_source_ranks = lambda owner: [0, 1]
        group._nep_nccl_owner_transfer_ranks = lambda owner: [0, 1]
        group._nep_nccl_owner_source_payload_numel = lambda owner, source, start, end: (
            payload_numel if source == 1 else 0
        )
        group._pack_nep_nccl_scatter_payload = (
            lambda owner, destination, start, end, chunk, output: output.copy_(
                torch.arange(payload_numel, dtype=output.dtype) + payload_base
            )
        )
        group._copy_nep_nccl_scatter_payload_to_local_grads = (
            lambda owner, source, start, end, payload, index=group_index: received.append(
                (index, payload.clone())
            )
        )

    first_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    second_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    configure_group(first_group, 4, 2, 2, 10)
    configure_group(second_group, 5, 3, 3, 20)
    first_group._get_nep_nccl_transfer_group_info = lambda owner: (object(), 0, 2, [0, 1])
    first_group._get_nep_nccl_shared_buffer_state = lambda: {"gather_buf_cache": cache}
    first_group._get_nep_nccl_cached_tensor = cached_tensor
    contexts = [
        {
            "group": first_group,
            "owner_ep_rank": 0,
            "chunk_index": 0,
            "chunk_start": 0,
            "chunk_end": 4,
            "chunk": torch.empty(4),
            "buffer_slot_key": (0,),
            "async_op": True,
        },
        {
            "group": second_group,
            "owner_ep_rank": 0,
            "chunk_index": 0,
            "chunk_start": 0,
            "chunk_end": 6,
            "chunk": torch.empty(6),
            "buffer_slot_key": (1,),
            "async_op": True,
        },
    ]

    owner_descriptor = first_group._prepare_nep_nccl_owner_all_to_all_scatter_batch(contexts)
    assert owner_descriptor["buffer_slot_key"][1] == ("end_iteration", (2, 3))
    assert owner_descriptor["input_split_sizes"] == [0, 5]
    assert torch.equal(owner_descriptor["scatter_input"], torch.tensor([10, 11, 20, 21, 22]))

    runtime_config["ep_rank"] = 1
    follower_descriptor = first_group._prepare_nep_nccl_owner_all_to_all_scatter_batch(contexts)
    assert follower_descriptor["output_split_sizes"] == [5, 0]
    follower_descriptor["scatter_output"].copy_(torch.tensor([1, 2, 3, 4, 5]))
    follower_descriptor["completion_ordered"] = True
    first_group._finish_nep_nccl_owner_all_to_all_scatter(follower_descriptor)

    assert [index for index, _ in received] == [4, 5]
    assert torch.equal(received[0][1], torch.tensor([1, 2]))
    assert torch.equal(received[1][1], torch.tensor([3, 4, 5]))


def test_nep_two_level_gather_staged_phase_advances_queued_batch():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    dispatch_stream = object()
    next_task_batch = [object()]
    calls = []

    def start_next(task_batch, stream):
        calls.append((task_batch, stream))
        return [{"contexts": [], "dispatch_stream": stream, "phase": "gather_staged"}]

    bucket_group._start_nep_nccl_split_host_phase_batch = start_next
    pending = [
        {
            "contexts": [],
            "dispatch_stream": dispatch_stream,
            "phase": "gather_staged",
            "remaining_task_batches": [next_task_batch],
        }
    ]

    assert bucket_group._finish_nep_nccl_process_group_dispatch_batches(pending)
    assert calls == [(next_task_batch, dispatch_stream)]
    assert pending == [
        {
            "contexts": [],
            "dispatch_stream": dispatch_stream,
            "phase": "finished",
            "remaining_task_batches": [],
        }
    ]


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
    bucket_group._nep_runtime_config = {"ep_rank": 0}
    bucket_group.ddp_config = SimpleNamespace(use_distributed_optimizer=False)
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
    assert [call[0] for call in calls[1:-1]] == expected_phases
    assert [call[1] for call in calls if call[0] == "submit"] == expected_chunk_indices
    assert [call[1] for call in calls if call[0] == "order_completion"] == expected_chunk_indices
    assert [call[1] for call in calls if call[0] == "copyback"] == expected_chunk_indices
    assert calls[-1] == ("mark", 0, 7)


def test_nep_nccl_scatter_chunks_rejects_nonpositive_value(monkeypatch):
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS", "0")

    with pytest.raises(RuntimeError, match="SCATTER_CHUNKS must be positive"):
        _get_nep_nccl_scatter_chunks()


def test_nep_nccl_process_group_dispatch_tasks_share_one_ordered_stream(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_runtime_config = {}
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


def test_nep_nccl_edp_uses_distinct_shared_ordered_stream(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_runtime_config = {}
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

    gather_stream = bucket_group._get_nep_nccl_comm_stream(0)
    first_edp_stream = bucket_group._get_nep_nccl_ordered_edp_stream()
    second_edp_stream = bucket_group._get_nep_nccl_ordered_edp_stream()

    assert first_edp_stream is second_edp_stream
    assert first_edp_stream is not gather_stream
    assert created_streams == [(0, gather_stream), (0, first_edp_stream)]
    assert bucket_group._nep_nccl_scheduler_state["comm_streams"] == {
        "dispatch": gather_stream,
        "edp": first_edp_stream,
    }


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
        group._nep_nccl_group_index = index
        group._nep_nccl_edp_bucket_index = index
        group._start_nep_nccl_owner_edp_reduce_contexts = (
            lambda contexts, index=index: calls.append((index, contexts))
        )

    first_group._nep_nccl_scheduler_state = {
        "pending_edp_contexts": {},
        "expected_edp_contexts": {(0, 0): 2, (1, 0): 1},
    }
    contexts = [
        {"group": first_group, "owner_ep_rank": 0, "chunk_index": 0},
        {"group": first_group, "owner_ep_rank": 0, "chunk_index": 1},
        {"group": second_group, "owner_ep_rank": 0, "chunk_index": 0},
    ]

    first_group._start_nep_nccl_owner_edp_reduce_batch(contexts)

    assert calls == [(0, contexts[:2]), (1, contexts[2:])]


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
        "finished": False,
        "scatter_dependency_ordered": False,
    }
    for context in contexts:
        context["native_edp_state"] = native_state
    bucket_group._nep_runtime_config = {"ep_rank": 0}
    bucket_group.ddp_config = SimpleNamespace(
        overlap_grad_reduce=True, use_distributed_optimizer=False
    )
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


def test_nep_bucket_ready_gather_launches_from_accumulate_grad():
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

    bucket_group.register_grad_ready(param)

    assert bucket_group._nep_dispatch_boundary_ready
    assert calls == [((bucket_group,), "bucket.0")]


def test_nep_bucket_ready_gather_coalesces_groups_from_same_module():
    groups = []
    calls = []
    for _ in range(2):
        param = object()
        group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
            NonuniformEPNCCLParamAndGradBucketGroup
        )
        group.ddp_config = SimpleNamespace(overlap_grad_reduce=True)
        group.is_last_microbatch = True
        group.is_first_batch = False
        group.param_to_bucket = {param: object()}
        group.params = [param]
        group.per_param_grad_ready_counts = {}
        group.golden_per_param_grad_ready_counts = {param: 1}
        group._nep_dispatch_boundary_launch = True
        group._nep_dispatch_boundary_ready = False
        group._nep_dispatch_boundary_module_label = "layer.0.mlp"
        groups.append((group, param))

    dispatch_groups = tuple(group for group, _ in groups)
    for group, _ in groups:
        group._nep_dispatch_boundary_groups = dispatch_groups
        group._nep_dispatch_boundary_callback = lambda callback_groups, label: calls.append(
            (callback_groups, label)
        )

    for group, param in groups:
        group.register_grad_ready(param)

    assert calls == [(dispatch_groups, "layer.0.mlp"), (dispatch_groups, "layer.0.mlp")]


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
    state = {"task_next_index": 0, "task_sequence": [{"group": group} for group in groups]}
    for group in groups:
        group._nep_nccl_scheduler_state = state

    def launch(**kwargs):
        calls.append(("launch", kwargs))
        state["task_next_index"] = len(state["task_sequence"])
        return []

    groups[0]._try_start_nep_nccl_ready_tasks = launch

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


def test_nep_end_iteration_scatter_retains_persistent_payload_without_clone(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    ddp.ddp_config = SimpleNamespace(use_distributed_optimizer=False)
    calls = []
    chunk = object()
    context = {
        "group": SimpleNamespace(
            _mark_nep_nccl_task_started=lambda owner, chunk_index: calls.append(
                (owner, chunk_index)
            )
        ),
        "owner_ep_rank": 0,
        "chunk_index": 1,
        "chunk": chunk,
    }
    completion_event = object()
    ddp._nep_end_iteration_scatter_context_batches = []

    ddp._defer_nep_scatter_context_batches_to_iteration_end([[context]], completion_event, "layer")

    saved_batches, saved_completion, label = ddp._nep_end_iteration_scatter_context_batches[0]
    assert saved_batches == [[context]]
    assert saved_batches[0][0]["chunk"] is chunk
    assert saved_completion is completion_event
    assert label == "layer"
    assert calls == [(0, 1)]


def test_nep_end_iteration_scatter_marks_every_two_level_gather_context():
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    ddp.ddp_config = SimpleNamespace(use_distributed_optimizer=False)
    calls = []
    contexts = []
    for group_index in range(2):
        group = SimpleNamespace(
            _mark_nep_nccl_task_started=lambda owner, chunk, index=group_index: calls.append(
                (index, owner, chunk)
            )
        )
        contexts.append(
            {"group": group, "owner_ep_rank": 0, "chunk_index": group_index, "chunk": object()}
        )
    scatter_context = dict(contexts[0])
    scatter_context["scatter_contexts"] = tuple(contexts)
    ddp._nep_end_iteration_scatter_context_batches = []

    ddp._defer_nep_scatter_context_batches_to_iteration_end([[scatter_context]], object(), "layer")

    assert calls == [(0, 0, 0), (1, 0, 1)]


def test_nep_end_iteration_scatter_materializes_canonical_order():
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []

    def make_group(group_index, edp_bucket_index):
        group = SimpleNamespace(
            _nep_nccl_group_index=group_index, _nep_nccl_edp_bucket_index=edp_bucket_index
        )

        def coalesce(contexts):
            context = dict(contexts[0])
            context["scatter_contexts"] = tuple(contexts)
            return context

        group._coalesce_nep_nccl_scatter_contexts = coalesce
        return group

    group_0 = make_group(0, 0)
    group_1 = make_group(1, 1)
    group_2 = make_group(2, 2)
    contexts = [
        {"group": group_1, "owner_ep_rank": 0, "chunk_index": 1},
        {"group": group_0, "owner_ep_rank": 2, "chunk_index": 0},
        {"group": group_2, "owner_ep_rank": 1, "chunk_index": 1},
        {"group": group_0, "owner_ep_rank": 1, "chunk_index": 0},
    ]
    first_completion_event = object()
    final_completion_event = object()
    ddp._nep_end_iteration_scatter_context_batches = [
        ([contexts[:2]], first_completion_event, "decoder.layers.1.mlp"),
        ([contexts[2:]], final_completion_event, "decoder.layers.3.mlp"),
    ]
    ddp._queue_nep_scatter_context_batches = lambda *args, **kwargs: calls.append((args, kwargs))

    assert ddp._materialize_next_nep_end_iteration_scatter_batch()
    assert not ddp._materialize_next_nep_end_iteration_scatter_batch()

    queued = calls[0][0][0][0]
    assert [context["owner_ep_rank"] for context in queued] == [0, 1, 2]
    assert [
        context["group"]._nep_nccl_group_index for context in queued[1]["scatter_contexts"]
    ] == [0, 2]
    assert calls[0][0][1:] == (final_completion_event, "end_iteration_scatter")
    assert calls[0][1] == {}
    assert ddp._nep_end_iteration_scatter_completion_events == [first_completion_event]
    assert not ddp._nep_end_iteration_scatter_context_batches


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
    ddp._nep_dispatch_waiting_groups = None
    ddp._nep_dispatch_waiting_module_label = None
    ddp._nep_dispatch_pending_completion_event = None
    ddp._nep_dispatch_pending_host_phases = None
    ddp._nonuniform_ep_runtime_config = {"dp_cp_group_gloo": object()}
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: calls.append("barrier"))

    assert not ddp._launch_nep_dispatch_boundary_tasks((group,), "decoder.layers.1.mlp")
    assert calls == []
    assert group._nep_dispatch_boundary_wait_logged


def test_nep_nccl_combined_slot_owner_layout_round_trip():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    first_grad = torch.tensor([10.0, 11.0])
    second_grad = torch.tensor([20.0, 21.0, 22.0])
    bucket_group._nep_runtime_config = {}
    bucket_group._nep_nccl_owner_layout = {"owner_expert_slots": [[0, 1], [2, 3]]}
    bucket_group._nep_nccl_slot_numel = 5
    bucket_group._nep_nccl_slot_numels = (2, 3)
    bucket_group._nep_nccl_slot_offsets = (0, 2)
    bucket_group._nep_nccl_expert_stride = 5
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
    bucket_group._nep_nccl_owner_layout = {"owner_expert_slots": [[0, 1, 2, 3], [4, 5, 6, 7]]}
    bucket_group._nep_nccl_slot_numel = 4
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
