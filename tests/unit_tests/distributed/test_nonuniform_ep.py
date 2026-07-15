# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

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
    _ExpertBucketSpec,
    _group_expert_bucket_specs_in_backward_order,
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


class _FakeDenseBucketGroup:
    def __init__(self, numel):
        self.is_first_batch = True
        self.grad_reduce_handle = None
        self.params = [object()]
        self.per_param_grad_ready_counts = {self.params[0]: 1}
        self.golden_per_param_grad_ready_counts = {}
        self.buckets = [type('Bucket', (), {'grad_data': torch.empty(numel)})()]
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
        slot_key=(f'decoder.layers.{layer}.mlp.experts.local_experts.{{expert}}.weight',),
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
    bucket_group._nep_runtime_config = {'ep_rank': 0, 'zero_sm_reshard': True}
    bucket_group._nep_nccl_group_index = 0
    bucket_group._nep_nccl_owner_source_ranks = lambda owner: [2, 3]
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [0, 2, 3]
    bucket_group._get_nep_nccl_transfer_group_info = lambda owner: (object(), 0, 3, [0, 2, 3])
    bucket_group._pack_nep_nccl_owner_chunk = lambda *args: calls.append('pack_owner')
    bucket_group._start_nep_nccl_owner_native_gather = lambda *args: calls.append('gather')
    bucket_group._start_nep_nccl_owner_native_scatter = lambda *args: calls.append('scatter')

    bucket_group._start_nep_nccl_owner_all_to_all_gather(0, 0, 0, 16, object(), (0,), async_op=True)
    bucket_group._start_nep_nccl_owner_all_to_all_scatter(
        0, 0, 0, 16, object(), (0,), async_op=True
    )

    assert calls == ['pack_owner', 'gather', 'scatter']


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
            {'local_expert_indices': placement[2], 'expert_placement': placement}
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

    assert [int(slot_key[0].split('.')[2]) for slot_key, _ in grouped_specs] == [45, 44, 10, 9]
    assert [[spec.expert_id for spec in group] for _, group in grouped_specs] == [
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
    ]


def test_nep_nccl_buffer_slot_reuse_does_not_block_host():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    slot_key = (0, 128, None, None)
    works = [_FakeWork(), _FakeWork()]
    bucket_group._nep_nccl_scheduler_state = {
        'gather_buf_cache': {},
        'buffer_slot_handles': {slot_key: works},
    }

    bucket_group._order_nep_nccl_buffer_slot(slot_key)

    assert all(work.block_calls == 1 for work in works)
    assert all(work.wait_calls == 0 for work in works)
    assert slot_key not in bucket_group._nep_nccl_scheduler_state['buffer_slot_handles']


def test_nep_nccl_starts_first_batch_dense_groups_before_waiting():
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    ddp.ddp_config = type('DDPConfig', (), {'overlap_grad_reduce': True})()
    ddp.nonuniform_ep_config = type(
        'NonuniformEPConfig', (), {'approach': NonuniformEPApproach.NCCL}
    )()
    ddp.bucket_groups = [_FakeDenseBucketGroup(4), _FakeDenseBucketGroup(8)]

    ddp._start_delayed_dense_grad_syncs(force_all_reduce=True)

    assert [group.start_calls for group in ddp.bucket_groups] == [[True], [True]]
    assert all(group.grad_reduce_handle is not None for group in ddp.bucket_groups)


def test_nep_nccl_owner_tasks_use_bounded_distinct_stream_slots(monkeypatch):
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW", "4")
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_nccl_owner_layout = {'min_ep_size': 3, 'num_chunks': 2}

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


def test_nep_nccl_zero_sm_tasks_share_one_ordered_stream(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_runtime_config = {'zero_sm_reshard': True}
    bucket_group.is_first_batch = False
    bucket_group._nep_nccl_scheduler_state = {}
    bucket_group._nep_nccl_streams = {}
    created_streams = []

    def make_stream(device):
        stream = object()
        created_streams.append((device, stream))
        return stream

    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'Stream', make_stream)

    first = bucket_group._get_nep_nccl_comm_stream(0)
    second = bucket_group._get_nep_nccl_comm_stream(1)

    assert first is second
    assert created_streams == [(0, first)]
    assert bucket_group._nep_nccl_scheduler_state['comm_streams'] == {'zero_sm': first}


def test_nep_nccl_process_group_dispatch_tasks_share_one_ordered_stream(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_runtime_config = {'zero_sm_reshard': False}
    bucket_group._nep_dispatch_boundary_launch = True
    bucket_group.is_first_batch = False
    bucket_group._nep_nccl_scheduler_state = {}
    bucket_group._nep_nccl_streams = {}
    created_streams = []

    def make_stream(device):
        stream = object()
        created_streams.append((device, stream))
        return stream

    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'Stream', make_stream)

    first = bucket_group._get_nep_nccl_comm_stream(0)
    second = bucket_group._get_nep_nccl_comm_stream(1)

    assert first is second
    assert created_streams == [(0, first)]
    assert bucket_group._nep_nccl_scheduler_state['comm_streams'] == {'dispatch': first}


def test_nep_nccl_edp_ready_gate_uses_host_signaled_stream_wait(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    ready_group = object()
    calls = []
    flag = torch.empty(1, dtype=torch.int32)
    signal_stream = SimpleNamespace(cuda_stream=22)
    current_stream = SimpleNamespace(cuda_stream=11)
    device_ready_event = SimpleNamespace(synchronize=lambda: calls.append('device_ready'))
    bucket_group._nep_runtime_config = {
        'edp_ready_gate_enabled': True,
        'edp_group_gloo': ready_group,
    }

    class FakeReadyWork:
        def wait(self):
            calls.append('host_wait')

    class FakeFuture:
        def result(self):
            calls.append('future_result')

    class FakeExecutor:
        def submit(self, function, *args):
            calls.append('submit')
            function(*args)
            return FakeFuture()

    stream_ops = SimpleNamespace(
        wait_value32=lambda stream, address, value: calls.append(
            ('stream_wait', stream, address, value)
        ),
        write_value32=lambda stream, address, value: calls.append(
            ('stream_write', stream, address, value)
        ),
    )
    bucket_group._nep_nccl_scheduler_state = {
        'gather_buf_cache': {},
        'buffer_slot_handles': {},
        'edp_ready_flags': {0: flag},
        'edp_ready_generations': {0: 0},
        'edp_ready_executor': FakeExecutor(),
        'edp_ready_signal_stream': signal_stream,
    }
    bucket_group._nep_edp_ready_futures = []
    monkeypatch.setattr(
        torch.distributed,
        'barrier',
        lambda group, async_op: calls.append(('barrier', group, async_op)) or FakeReadyWork(),
    )
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 7)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: calls.append(('device', device)))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: current_stream)
    monkeypatch.setattr(
        'megatron.core.distributed.nonuniform_ep.get_cuda_stream_memory_ops', lambda: stream_ops
    )

    bucket_group._start_nep_nccl_edp_ready_gate(0, device_ready_event)
    bucket_group._drain_nep_edp_ready_futures()

    assert calls == [
        'submit',
        ('device', 7),
        'device_ready',
        ('barrier', ready_group, True),
        'host_wait',
        ('stream_write', 22, flag.data_ptr(), 1),
        ('stream_wait', 11, flag.data_ptr(), 1),
        'future_result',
    ]
    assert bucket_group._nep_nccl_scheduler_state['edp_ready_generations'] == {0: 1}
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
    first_group._nep_edp_ready_futures = [FakeFuture('first')]
    second_group._nep_edp_ready_futures = [FakeFuture('second')]

    first_group._drain_nep_edp_ready_futures()

    assert calls == ['first']
    assert first_group._nep_edp_ready_futures == []
    assert len(second_group._nep_edp_ready_futures) == 1


def test_nep_nccl_owner_task_gates_edp_all_reduce_after_gather(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    fake_work = _FakeWork()
    fake_edp_group = type('FakeGroup', (), {'rank': lambda self: 0})()
    bucket_group.buckets = [type('Bucket', (), {'grad_data': torch.empty(8)})()]
    bucket_group.ddp_config = type('DDPConfig', (), {'average_in_collective': False})()
    bucket_group.is_first_batch = False
    bucket_group._nep_runtime_config = {
        'ep_rank': 0,
        'edp_group': fake_edp_group,
        'edp_ready_gate_enabled': True,
    }
    bucket_group._nep_nccl_group_index = 0
    bucket_group._nep_nccl_async_tensors = []
    bucket_group._nep_nccl_scheduler_state = {'gather_buf_cache': {}, 'buffer_slot_handles': {}}
    bucket_group._get_nep_nccl_owner_layout = lambda: {}
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: 0
    bucket_group._prep_nep_nccl_owner_entries_for_sync = lambda owner: None
    bucket_group._start_nep_nccl_owner_all_to_all_gather = lambda *args, **kwargs: calls.append(
        'gather'
    )
    bucket_group._start_nep_nccl_edp_ready_gate = lambda slot, event=None: calls.append('ready')
    bucket_group._record_nep_nccl_work = lambda work, slot: calls.append('record')
    bucket_group._start_nep_nccl_owner_task_scatter = lambda context: calls.append('scatter')

    def fake_all_reduce(tensor, op, group, async_op):
        calls.append('all_reduce')
        assert group is fake_edp_group
        return fake_work

    monkeypatch.setattr(torch.distributed, 'all_reduce', fake_all_reduce)

    bucket_group._start_nep_nccl_owner_task(
        owner_ep_rank=0, chunk_index=0, chunk_start=0, chunk_end=8, async_op=True
    )

    assert calls == ['gather', 'ready', 'all_reduce', 'record', 'scatter']


def test_nep_nccl_owner_edp_batch_preserves_context_reductions():
    first_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    second_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    for index, group in enumerate((first_group, second_group)):
        group._nep_runtime_config = {'ep_rank': 0}
        group._start_nep_nccl_owner_edp_reduce = (
            lambda context, use_device_readiness, index=index: calls.append(
                (index, context, use_device_readiness)
            )
        )

    contexts = [{'group': group, 'owner_ep_rank': 0} for group in (first_group, second_group)]

    first_group._start_nep_nccl_owner_edp_reduce_batch(contexts, use_device_readiness=False)

    assert calls == [(0, contexts[0], False), (1, contexts[1], False)]


def test_nep_nccl_same_communicator_ready_reuses_ordered_token(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    process_group = SimpleNamespace(size=lambda: 2)
    state = {'buffer_slot_handles': {}}
    calls = []
    first_work = object()
    second_work = object()
    works = [first_work, second_work]

    bucket_group._nep_nccl_async_handles = []
    bucket_group._get_nep_nccl_shared_buffer_state = lambda: state
    bucket_group._order_nep_nccl_buffer_slot = lambda key: calls.append(('order', key))
    bucket_group._record_nep_nccl_work = lambda work, key: calls.append(('record', work, key))

    def fake_all_reduce(token, group, async_op):
        calls.append(('all_reduce', token, group, async_op))
        return works.pop(0)

    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
    monkeypatch.setattr(torch.distributed, 'all_reduce', fake_all_reduce)

    key = ('edp', 0)
    bucket_group._start_nep_nccl_same_communicator_ready(process_group, key)
    first_token = calls[1][1]
    bucket_group._start_nep_nccl_same_communicator_ready(process_group, key)

    slot_key = ('same_communicator_ready', 'edp', 0)
    assert calls == [
        ('order', slot_key),
        ('all_reduce', first_token, process_group, True),
        ('record', first_work, slot_key),
        ('order', slot_key),
        ('all_reduce', first_token, process_group, True),
        ('record', second_work, slot_key),
    ]


def test_nep_nccl_first_batch_zero_sm_finishes_gather_and_edp_before_scatter(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    fake_edp_group = type('FakeGroup', (), {'rank': lambda self: 0})()
    fake_task_group_gloo = object()
    fake_stream = type('FakeStream', (), {'synchronize': lambda self: calls.append('sync')})()
    bucket_group.buckets = [type('Bucket', (), {'grad_data': torch.empty(8)})()]
    bucket_group.ddp_config = type('DDPConfig', (), {'average_in_collective': False})()
    bucket_group.is_first_batch = True
    bucket_group._nep_runtime_config = {
        'ep_rank': 0,
        'edp_group': fake_edp_group,
        'dp_cp_group_gloo': fake_task_group_gloo,
        'edp_ready_gate_enabled': True,
        'zero_sm_reshard': True,
    }
    bucket_group._nep_nccl_group_index = 0
    bucket_group._nep_nccl_async_tensors = []
    bucket_group._nep_nccl_scheduler_state = {'gather_buf_cache': {}, 'buffer_slot_handles': {}}
    bucket_group._get_nep_nccl_owner_layout = lambda: {}
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: 0
    bucket_group._nep_nccl_owner_source_ranks = lambda owner: [0, 1]
    bucket_group._prep_nep_nccl_owner_entries_for_sync = lambda owner: None
    bucket_group._start_nep_nccl_owner_all_to_all_gather = lambda *args, **kwargs: calls.append(
        'gather'
    )
    bucket_group._start_nep_nccl_edp_ready_gate = lambda slot, event=None: calls.append('ready')
    bucket_group._record_nep_nccl_work = lambda work, slot: calls.append('record')
    bucket_group._start_nep_nccl_owner_task_scatter = lambda context: calls.append('scatter')

    def fake_all_reduce(tensor, op, group, async_op):
        calls.append('all_reduce')
        assert group is fake_edp_group
        return _FakeWork()

    def fake_barrier(group):
        assert group is fake_task_group_gloo
        calls.append('task_gloo')

    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: fake_stream)
    monkeypatch.setattr(torch.distributed, 'all_reduce', fake_all_reduce)
    monkeypatch.setattr(torch.distributed, 'barrier', fake_barrier)

    bucket_group._start_nep_nccl_owner_task(
        owner_ep_rank=0, chunk_index=0, chunk_start=0, chunk_end=8, async_op=False
    )

    assert calls == [
        'gather',
        'sync',
        'sync',
        'task_gloo',
        'all_reduce',
        'record',
        'sync',
        'sync',
        'task_gloo',
        'scatter',
    ]


def test_nep_nccl_first_batch_zero_sm_fences_helper_rank(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    bucket_group.is_first_batch = True
    bucket_group._nep_runtime_config = {'ep_rank': 1, 'zero_sm_reshard': True}
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [0, 1, 2]
    fake_stream = type('FakeStream', (), {'synchronize': lambda self: calls.append('sync')})()
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: fake_stream)

    bucket_group._synchronize_first_batch_zero_sm_phase(0, 'gather')
    bucket_group._synchronize_first_batch_zero_sm_phase(0, 'scatter')

    assert calls == ['sync', 'sync']


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
        {'bucket': SimpleNamespace(params_list=[param])}
    ]

    assert not bucket_group._nep_nccl_owner_task_ready(0)

    bucket_group._nep_dispatch_boundary_graph_replay_ready = True

    assert bucket_group._nep_nccl_owner_task_ready(0)


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

    ddp._mark_nep_dispatch_boundary_ready((group,), 'decoder.layers.1.mlp', graph_replay=True)

    assert group._nep_dispatch_boundary_ready
    assert group._nep_dispatch_boundary_graph_replay_ready
    assert ddp._nep_dispatch_waiting_groups == (group,)
    assert ddp._nep_dispatch_waiting_module_label == 'decoder.layers.1.mlp'
    assert calls == [((group,), 'decoder.layers.1.mlp')]


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
    monkeypatch.setenv('MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH', '1')

    ddp._mark_nep_dispatch_boundary_ready((group,), 'decoder.layers.1.mlp', graph_replay=True)

    assert calls == []
    assert ddp._nep_dispatch_waiting_groups == (group,)
    ddp._launch_waiting_nep_dispatch_boundary_tasks()
    assert calls == [((group,), 'decoder.layers.1.mlp')]


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

    ddp._mark_nep_dispatch_boundary_ready((group,), 'decoder.layers.1.mlp')
    assert calls == [((group,), 'decoder.layers.1.mlp')]


def test_nep_finds_nearest_full_layer_cuda_graph_manager():
    graph_manager = object()
    named_modules = {
        '': SimpleNamespace(),
        'decoder': SimpleNamespace(),
        'decoder.layers.1': SimpleNamespace(cudagraph_manager=graph_manager),
        'decoder.layers.1.mlp': SimpleNamespace(),
    }

    found = NonuniformEPDistributedDataParallel._find_nep_local_cuda_graph_manager(
        'decoder.layers.1.mlp', named_modules
    )

    assert found is graph_manager


def test_nep_does_not_cross_partial_moe_graph_boundary():
    root_graph_manager = object()
    named_modules = {
        '': SimpleNamespace(cudagraph_manager=root_graph_manager),
        'decoder': SimpleNamespace(),
        'decoder.layers.1': SimpleNamespace(use_partial_cudagraphs=True),
        'decoder.layers.1.mlp': SimpleNamespace(),
    }

    found = NonuniformEPDistributedDataParallel._find_nep_local_cuda_graph_manager(
        'decoder.layers.1.mlp', named_modules
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
        lambda: ddp._mark_nep_dispatch_boundary_ready((group,), 'decoder.layers.1.mlp')
    )
    module.register_expert_compute_dgrad_callback(lambda: calls.append('wait_before_dispatch'))
    leaf = torch.ones(4, requires_grad=True)
    expert_input = leaf * 2
    dispatch_input = module._attach_expert_compute_input_grad_callbacks(expert_input)
    dispatched_input = dispatch_input * 3
    expert_boundary = module._attach_expert_compute_dgrad_callbacks(dispatched_input)
    expert_output = expert_boundary * 4
    combine_input = module._attach_expert_compute_output_grad_callbacks(expert_output)

    assert calls == []
    (combine_input * 5).sum().backward()
    assert calls == ['wait_before_dispatch', ((group,), 'decoder.layers.1.mlp')]


def test_nep_coalesces_shared_cuda_graph_boundary():
    group_1 = SimpleNamespace(_nep_nccl_group_index=7)
    group_2 = SimpleNamespace(_nep_nccl_group_index=3)

    groups, label = NonuniformEPDistributedDataParallel._coalesce_nep_cuda_graph_boundary(
        [('decoder.layers.0.mlp', (group_1,)), ('decoder.layers.1.mlp', (group_2, group_1))]
    )

    assert groups == (group_2, group_1)
    assert label == 'cuda_graph[decoder.layers.0.mlp,decoder.layers.1.mlp]'


def test_local_cuda_graph_backward_replay_hooks_wrap_replay():
    calls = []

    class FakeGraph:
        def replay(self):
            calls.append('replay')

    runner = SimpleNamespace(
        bwd_graph=FakeGraph(),
        status=_GraphStatus.BWD_READY,
        static_grad_outputs=(),
        fwd_graph_input_surface=(),
        backward_replay_pre_hooks=[lambda: calls.append('pre')],
        backward_replay_post_hooks=[lambda: calls.append('post')],
        fp8_enabled=False,
        groundtruth_grad_added_to_main_grad={},
        static_grad_inputs=[],
        num_dgrads=0,
    )
    ctx = SimpleNamespace(runner=runner, saved_tensors=())

    result = _CudagraphReplayNode.backward(ctx)

    assert calls == ['pre', 'replay', 'post']
    assert runner.status == _GraphStatus.FWD_READY
    assert result == (None, None)


def test_nep_nccl_dispatch_boundary_orders_full_pipeline():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    context = {
        'owner_ep_rank': 0,
        'chunk_index': 0,
        'chunk_start': 0,
        'chunk_end': 8,
        'chunk': object(),
        'buffer_slot': 0,
        'buffer_slot_key': object(),
        'async_op': True,
    }
    bucket_group._nep_runtime_config = {'ep_rank': 0}
    bucket_group._start_nep_nccl_owner_all_to_all_gather = lambda *args, **kwargs: calls.append(
        'gather'
    )
    bucket_group._start_nep_nccl_owner_edp_reduce = lambda task, use_device_readiness: calls.append(
        ('edp', use_device_readiness)
    )
    bucket_group._start_nep_nccl_owner_task_scatter = lambda task: calls.append('scatter')

    bucket_group._start_nep_nccl_dispatch_boundary_task(context)

    assert calls == ['gather', ('edp', True), 'scatter']


def test_process_group_edp_reduce_uses_external_host_phase_gate(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    edp_group = SimpleNamespace(rank=lambda: 0)
    chunk = object()
    work = object()
    bucket_group._nep_runtime_config = {
        'ep_rank': 0,
        'edp_group': edp_group,
        'edp_ready_gate_enabled': False,
    }
    bucket_group.ddp_config = SimpleNamespace(average_in_collective=False)
    bucket_group.is_first_batch = False
    bucket_group._nep_nccl_group_index = 3
    bucket_group._start_nep_nccl_edp_readiness = lambda slot: calls.append('device_gate')
    bucket_group._record_nep_nccl_work = lambda recorded_work, key: calls.append(
        ('record', recorded_work, key)
    )
    bucket_group._synchronize_first_batch_zero_sm_phase = lambda owner, phase: None
    context = {
        'owner_ep_rank': 0,
        'chunk_index': 2,
        'chunk_start': 0,
        'chunk_end': 8,
        'chunk': chunk,
        'buffer_slot': 1,
        'buffer_slot_key': ('slot',),
        'async_op': True,
    }

    monkeypatch.setenv('MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE', '1')
    monkeypatch.setattr(
        torch.distributed,
        'barrier',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('blocking host gate')),
    )
    monkeypatch.setattr(
        torch.distributed,
        'all_reduce',
        lambda tensor, op, group, async_op: calls.append(('all_reduce', tensor, group, async_op))
        or work,
    )

    bucket_group._start_nep_nccl_owner_edp_reduce(context, use_device_readiness=False)

    assert calls == [('all_reduce', chunk, edp_group, True), ('record', work, ('slot',))]


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
        'ep_rank': 1,
        'zero_sm_reshard': False,
        'edp_ready_gate_enabled': False,
    }
    bucket_group._nep_nccl_async_tensors = []
    bucket_group._nep_nccl_scheduler_state = {
        'gather_buf_cache': {
            ('owner_layout_gather', 0, 8, grad_data.dtype, grad_data.device): chunk
        },
        'buffer_slot_handles': {},
        'pending_owner_tasks': pending_tasks,
    }
    bucket_group._get_nep_nccl_owner_layout = lambda: {}
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk_index: 0
    bucket_group._order_nep_nccl_buffer_slot = lambda key: None
    bucket_group._prep_nep_nccl_owner_entries_for_sync = lambda owner: None
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [0, 1]
    bucket_group._start_nep_nccl_owner_all_to_all_gather = lambda *args, **kwargs: calls.append(
        'gather'
    )
    bucket_group._start_nep_nccl_owner_edp_reduce = lambda *args, **kwargs: calls.append('edp')
    bucket_group._start_nep_nccl_owner_task_scatter = lambda task: calls.append('scatter')

    class FakeEvent:
        def record(self, stream):
            calls.append(('record', stream))

    monkeypatch.setenv('MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE', '1')
    monkeypatch.setattr(torch.cuda, 'Event', FakeEvent)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: 'comm_stream')

    bucket_group._start_nep_nccl_owner_task(
        owner_ep_rank=0,
        chunk_index=2,
        chunk_start=0,
        chunk_end=8,
        async_op=True,
        defer_scatter=True,
    )

    assert calls == ['gather', ('record', 'comm_stream')]
    assert len(pending_tasks) == 1
    assert pending_tasks[0]['stage'] == 'gather'


def test_nep_dispatch_boundary_enqueues_without_launch_barriers(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    completion_stream = object()

    class FakeEvent:
        def record(self, stream):
            assert stream is completion_stream
            calls.append('record_completion')

        def synchronize(self):
            calls.append('wait_completion')

    def make_group(index):
        group = type('Group', (), {})()
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
    groups[0]._try_start_nep_nccl_ready_tasks = lambda **kwargs: calls.append(('launch', kwargs))

    monkeypatch.setattr(
        torch.distributed,
        'barrier',
        lambda group: (_ for _ in ()).throw(AssertionError('unexpected launch barrier')),
    )
    monkeypatch.setattr(torch.cuda, 'Event', FakeEvent)

    for group in groups:
        group._nep_dispatch_boundary_ready = True
        group._nep_dispatch_boundary_launching = True
    compute_ready_event = object()
    completion_event = ddp._run_nep_dispatch_boundary_tasks(
        groups, 'decoder.layers.1.mlp', compute_ready_event
    )

    assert calls == [
        (
            'launch',
            {
                'force_ready': False,
                'async_op_override': True,
                'compute_ready_event': compute_ready_event,
            },
        ),
        'record_completion',
    ]
    assert isinstance(completion_event, FakeEvent)
    assert all(group._nep_dispatch_boundary_ready for group in groups)
    assert all(group._nep_dispatch_boundary_launched for group in groups)


@pytest.mark.parametrize('same_communicator_ready', [False, True])
def test_nep_dispatch_scheduler_launches_process_group_phases(monkeypatch, same_communicator_ready):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    compute_ready_event = object()

    class FakeStream:
        def wait_event(self, event):
            calls.append(('wait_event', event))

        def wait_stream(self, stream):
            calls.append(('wait_stream', stream))

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
            calls.append(('record_event', events.index(self), stream))

    tasks = [
        {
            'group': bucket_group,
            'owner_ep_rank': owner,
            'chunk_index': 0,
            'chunk_start': 0,
            'chunk_end': 8,
        }
        for owner in (0, 1)
    ]
    bucket_group._nep_nccl_scheduler_state = {
        'task_sequence': tasks,
        'task_next_index': 0,
        'pending_owner_tasks': [],
    }
    bucket_group._nep_runtime_config = {
        'ep_rank': 0,
        'zero_sm_reshard': False,
        'edp_ready_gate_enabled': True,
        'edp_group': edp_nccl_group,
        'edp_group_gloo': edp_owner_group,
        'nep_owner_transfer_groups': {0: source_nccl_group},
        'nep_owner_transfer_groups_gloo': {0: source_group},
    }
    bucket_group._nep_nccl_owner_task_ready = lambda owner: True
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [owner, owner + 4]
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: owner
    bucket_group._get_nep_nccl_comm_stream = lambda slot: nccl_stream
    bucket_group._prepare_nep_nccl_owner_task_context = (
        lambda owner, chunk, start, end, async_op: calls.append(('prepare', owner))
        or {
            'group': bucket_group,
            'owner_ep_rank': owner,
            'chunk_index': chunk,
            'chunk_start': start,
            'chunk_end': end,
            'chunk': object(),
            'buffer_slot': owner,
            'buffer_slot_key': object(),
        }
    )
    bucket_group._start_nep_nccl_owner_all_to_all_gather = (
        lambda owner, *args, **kwargs: calls.append(('gather', owner))
    )
    bucket_group._start_nep_nccl_owner_edp_reduce_batch = (
        lambda contexts, use_device_readiness: calls.append(
            (
                'edp_batch',
                [context['owner_ep_rank'] for context in contexts],
                use_device_readiness,
                all(context['gather_done_event'] is events[0] for context in contexts),
            )
        )
    )
    bucket_group._start_nep_nccl_scatter_ready_gate = lambda owner, slot, event: calls.append(
        ('scatter_ready', owner, slot, event is events[1])
    )
    bucket_group._start_nep_nccl_same_communicator_ready = lambda group, key: calls.append(
        ('same_communicator_ready', group, key)
    )
    bucket_group._start_nep_nccl_owner_task_scatter = lambda context: calls.append(
        ('scatter', context['owner_ep_rank'])
    )

    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: FakeStreamContext())
    monkeypatch.setattr(torch.cuda, 'Event', FakeEvent)
    monkeypatch.setattr(
        torch.distributed, 'barrier', lambda group: calls.append(('owner_barrier', group))
    )
    monkeypatch.setattr(
        torch.cuda,
        'current_stream',
        lambda: (_ for _ in ()).throw(AssertionError('late stream dependency')),
    )

    monkeypatch.setenv(
        'MEGATRON_NONUNIFORM_EP_SAME_COMM_READY', '1' if same_communicator_ready else '0'
    )
    bucket_group._try_start_nep_nccl_ready_tasks(
        async_op_override=True, compute_ready_event=compute_ready_event
    )

    expected_calls = [
        ('wait_event', compute_ready_event),
        ('prepare', 0),
        ('prepare', 1),
        ('gather', 0),
        ('gather', 1),
        ('record_event', 0, nccl_stream),
        ('owner_barrier', source_group),
        ('owner_barrier', edp_owner_group),
        ('edp_batch', [0, 1], True, True),
        ('record_event', 1, nccl_stream),
        ('owner_barrier', source_group),
        ('scatter_ready', 0, 0, True),
        ('scatter', 0),
        ('scatter', 1),
    ]
    if same_communicator_ready:
        expected_calls = [
            ('wait_event', compute_ready_event),
            ('prepare', 0),
            ('prepare', 1),
            ('gather', 0),
            ('gather', 1),
            ('record_event', 0, nccl_stream),
            ('same_communicator_ready', edp_nccl_group, ('edp', 0)),
            ('edp_batch', [0, 1], True, True),
            ('record_event', 1, nccl_stream),
            ('same_communicator_ready', source_nccl_group, ('scatter', 0)),
            ('scatter', 0),
            ('scatter', 1),
        ]
    assert calls == expected_calls
    assert bucket_group._nep_nccl_scheduler_state['task_next_index'] == 2


def test_nep_split_host_phases_defer_edp_and_scatter(monkeypatch):
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    calls = []
    compute_ready_event = object()

    class FakeStream:
        def wait_event(self, event):
            calls.append(('wait_event', event))

    class FakeStreamContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class FakeEvent:
        def record(self, stream):
            calls.append(('record_event', stream))

    class FakeWork:
        def __init__(self, label):
            self.label = label

        def wait(self):
            calls.append(('wait_barrier', self.label))

    nccl_stream = FakeStream()
    source_group = object()
    edp_group = SimpleNamespace(size=lambda: 2)
    task = {
        'group': bucket_group,
        'owner_ep_rank': 0,
        'chunk_index': 0,
        'chunk_start': 0,
        'chunk_end': 8,
    }
    bucket_group._nep_nccl_scheduler_state = {
        'task_sequence': [task],
        'task_next_index': 0,
        'pending_owner_tasks': [],
    }
    bucket_group._nep_runtime_config = {
        'ep_rank': 0,
        'zero_sm_reshard': False,
        'edp_ready_gate_enabled': False,
        'edp_group_gloo': edp_group,
        'nep_owner_transfer_groups_gloo': {0: source_group},
    }
    bucket_group._nep_nccl_owner_task_ready = lambda owner: True
    bucket_group._nep_nccl_owner_transfer_ranks = lambda owner: [0, 4]
    bucket_group._get_nep_nccl_task_buffer_slot = lambda owner, chunk: 0
    bucket_group._get_nep_nccl_comm_stream = lambda slot: nccl_stream
    bucket_group._prepare_nep_nccl_owner_task_context = (
        lambda owner, chunk, start, end, async_op: calls.append(('prepare', owner))
        or {
            'group': bucket_group,
            'owner_ep_rank': owner,
            'chunk_index': chunk,
            'chunk_start': start,
            'chunk_end': end,
            'chunk': object(),
            'buffer_slot': 0,
            'buffer_slot_key': object(),
        }
    )
    bucket_group._start_nep_nccl_owner_all_to_all_gather = (
        lambda owner, *args, **kwargs: calls.append(('gather', owner))
    )
    bucket_group._start_nep_nccl_owner_edp_reduce_batch = (
        lambda contexts, use_device_readiness: calls.append(
            ('edp_batch', use_device_readiness)
        )
    )
    bucket_group._start_nep_nccl_owner_task_scatter = lambda context: calls.append(
        ('scatter', context['owner_ep_rank'])
    )

    barrier_counts = {id(source_group): 0, id(edp_group): 0}

    def fake_barrier(group, async_op=False):
        assert async_op
        barrier_counts[id(group)] += 1
        label = 'source' if group is source_group else 'edp'
        label = f"{label}_{barrier_counts[id(group)]}"
        calls.append(('submit_barrier', label))
        return FakeWork(label)

    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: FakeStreamContext())
    monkeypatch.setattr(torch.cuda, 'Event', FakeEvent)
    monkeypatch.setattr(torch.distributed, 'barrier', fake_barrier)
    monkeypatch.setenv('MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES', '1')
    monkeypatch.setenv('MEGATRON_NONUNIFORM_EP_SAME_COMM_READY', '0')

    pending = bucket_group._try_start_nep_nccl_ready_tasks(
        async_op_override=True, compute_ready_event=compute_ready_event
    )

    assert calls == [
        ('wait_event', compute_ready_event),
        ('prepare', 0),
        ('gather', 0),
        ('record_event', nccl_stream),
        ('submit_barrier', 'source_1'),
    ]
    assert len(pending) == 1

    bucket_group._finish_nep_nccl_process_group_dispatch_batches(pending)

    assert calls[5:] == [
        ('wait_barrier', 'source_1'),
        ('submit_barrier', 'edp_1'),
        ('wait_barrier', 'edp_1'),
        ('edp_batch', False),
        ('submit_barrier', 'source_2'),
        ('wait_barrier', 'source_2'),
        ('scatter', 0),
    ]


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
            calls.append(('record', stream))

    group = type('Group', (), {})()
    group._nep_nccl_group_index = 4
    group._nep_dispatch_boundary_ready = True
    group._nep_dispatch_boundary_launched = False
    group._nep_dispatch_boundary_launching = False
    group._nep_dispatch_boundary_inputs_ready = lambda: True
    ddp._nep_dispatch_pending_completion_event = None
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nonuniform_ep_runtime_config = {'zero_sm_reshard': True}

    def fake_submit(groups, module_label, ready_event, completion_event, device_index):
        calls.append(
            ('submit_launch', groups, module_label, ready_event, completion_event, device_index)
        )
        return completion_future

    ddp._submit_nep_dispatch_launch_and_completion = fake_submit
    monkeypatch.setattr(torch.cuda, 'Event', FakeReadyEvent)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 7)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: compute_stream)

    assert ddp._launch_nep_dispatch_boundary_tasks((group,), 'decoder.layers.1.mlp')

    assert calls[0] == ('record', compute_stream)
    assert calls[1] == ('submit_launch', (group,), 'decoder.layers.1.mlp', events[0], events[1], 7)
    assert ddp._nep_dispatch_pending_completion_event is events[1]
    assert ddp._nep_dispatch_pending_completion_future is completion_future


def test_process_group_dispatch_boundary_launches_inline_and_stream_orders_completion(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    compute_stream = SimpleNamespace(wait_event=lambda event: calls.append(('wait_event', event)))
    events = []

    class FakeEvent:
        def __init__(self):
            events.append(self)

        def record(self, stream):
            calls.append(('record', stream))

    group = SimpleNamespace(
        _nep_nccl_group_index=4,
        _nep_dispatch_boundary_ready=True,
        _nep_dispatch_boundary_launched=False,
        _nep_dispatch_boundary_launching=False,
        _nep_dispatch_boundary_inputs_ready=lambda: True,
    )
    ddp._nonuniform_ep_runtime_config = {'zero_sm_reshard': False}
    ddp._nep_dispatch_pending_completion_event = None
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nep_dispatch_waiting_groups = (group,)
    ddp._nep_dispatch_waiting_module_label = 'decoder.layers.1.mlp'

    def run_inline(groups, module_label, ready_event, completion_event):
        calls.append(('run_inline', groups, module_label, ready_event, completion_event))
        group._nep_dispatch_boundary_launched = True
        return completion_event

    ddp._run_nep_dispatch_boundary_tasks = run_inline
    monkeypatch.setattr(torch.cuda, 'Event', FakeEvent)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 7)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: compute_stream)

    assert ddp._launch_nep_dispatch_boundary_tasks((group,), 'decoder.layers.1.mlp')
    assert calls == [
        ('record', compute_stream),
        ('run_inline', (group,), 'decoder.layers.1.mlp', events[0], events[1]),
    ]
    assert ddp._nep_dispatch_pending_completion_future is None

    ddp._wait_for_nep_dispatch_launch()

    assert calls[-1] == ('wait_event', events[1])
    assert ddp._nep_dispatch_pending_completion_event is None
    assert ddp._nep_dispatch_waiting_groups is None


def test_process_group_split_dispatch_records_completion_after_scatter(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    compute_stream = SimpleNamespace(wait_event=lambda event: calls.append(('wait_event', event)))
    dispatch_stream = object()
    events = []

    class FakeEvent:
        def __init__(self):
            self.index = len(events)
            events.append(self)

        def record(self, stream):
            calls.append(('record', self.index, stream))

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
        calls.append(('launch_gather', kwargs))
        return pending

    def finish_host_phases(phases):
        assert phases is pending
        calls.append('finish_edp_scatter')
        group._nep_nccl_ready = True

    group._try_start_nep_nccl_ready_tasks = launch_gather
    group._finish_nep_nccl_process_group_dispatch_batches = finish_host_phases
    ddp._nonuniform_ep_runtime_config = {'zero_sm_reshard': False}
    ddp._nep_dispatch_pending_completion_event = None
    ddp._nep_dispatch_pending_completion_future = None
    ddp._nep_dispatch_pending_host_phases = None
    ddp._nep_dispatch_waiting_groups = (group,)
    ddp._nep_dispatch_waiting_module_label = 'decoder.layers.1.mlp'

    monkeypatch.setattr(torch.cuda, 'Event', FakeEvent)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 7)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: compute_stream)

    assert ddp._launch_nep_dispatch_boundary_tasks((group,), 'decoder.layers.1.mlp')
    assert calls == [
        ('record', 0, compute_stream),
        (
            'launch_gather',
            {
                'force_ready': False,
                'async_op_override': True,
                'compute_ready_event': events[0],
            },
        ),
    ]
    assert ddp._nep_dispatch_pending_host_phases == (group, pending)

    ddp._wait_for_nep_dispatch_launch()

    assert calls[-3:] == [
        'finish_edp_scatter',
        ('record', 1, dispatch_stream),
        ('wait_event', events[1]),
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

    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: calls.append(('device', device)))
    ddp._run_nep_dispatch_boundary_tasks = lambda *args: calls.append(('launch', args))
    ddp._complete_nep_dispatch_boundary = lambda event, group: (
        calls.append(('complete', event, group)) or (1.25, 2.5)
    )

    result = ddp._launch_and_complete_nep_dispatch_boundary(
        groups, 'decoder.layers.1.mlp', compute_ready_event, completion_event, boundary_group, 7
    )

    assert calls == [
        ('device', 7),
        ('launch', (groups, 'decoder.layers.1.mlp', compute_ready_event, completion_event)),
        ('complete', completion_event, boundary_group),
    ]
    assert result == (1.25, 2.5)


def test_nep_dispatch_completion_worker_fences_event_before_barrier(monkeypatch):
    calls = []
    boundary_group = object()

    class FakeCompletionEvent:
        def synchronize(self):
            calls.append('wait_completion')

    def fake_barrier(group):
        assert group is boundary_group
        calls.append('completion_barrier')

    monkeypatch.setattr(torch.distributed, 'barrier', fake_barrier)

    completion_wait_ms, completion_barrier_ms = (
        NonuniformEPDistributedDataParallel._complete_nep_dispatch_boundary(
            FakeCompletionEvent(), boundary_group
        )
    )

    assert calls == ['wait_completion', 'completion_barrier']
    assert completion_wait_ms >= 0.0
    assert completion_barrier_ms >= 0.0


def test_nep_dispatch_wait_joins_completion_worker_before_clearing():
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    calls = []
    ddp._nonuniform_ep_runtime_config = {'zero_sm_reshard': True}

    class FakeCompletionEvent:
        def synchronize(self):
            raise AssertionError('autograd thread must not synchronize the completion event')

    class FakeCompletionFuture:
        def result(self):
            calls.append('join_completion_worker')
            return 1.25, 2.5

    group = SimpleNamespace(_nep_dispatch_boundary_launched=True)
    ddp._nep_dispatch_waiting_groups = (group,)
    ddp._nep_dispatch_waiting_module_label = 'decoder.layers.1.mlp'
    ddp._nep_dispatch_pending_completion_event = FakeCompletionEvent()
    ddp._nep_dispatch_pending_completion_future = FakeCompletionFuture()

    ddp._wait_for_nep_dispatch_launch()

    assert calls == ['join_completion_worker']
    assert ddp._nep_dispatch_pending_completion_event is None
    assert ddp._nep_dispatch_pending_completion_future is None
    assert ddp._nep_dispatch_waiting_groups is None
    assert ddp._nep_dispatch_waiting_module_label is None


def test_nep_dispatch_boundary_waits_for_local_inputs(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    group = type('Group', (), {})()
    group._nep_nccl_group_index = 4
    group._nep_dispatch_boundary_ready = True
    group._nep_dispatch_boundary_launched = False
    group._nep_dispatch_boundary_launching = False
    group._nep_dispatch_boundary_wait_logged = False
    group._nep_dispatch_boundary_inputs_ready = lambda: False
    calls = []
    ddp._nonuniform_ep_runtime_config = {'dp_cp_group_gloo': object()}
    monkeypatch.setattr(torch.distributed, 'barrier', lambda **kwargs: calls.append('barrier'))

    assert not ddp._launch_nep_dispatch_boundary_tasks((group,), 'decoder.layers.1.mlp')
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
        'group': bucket_group,
        'owner_ep_rank': 0,
        'chunk_index': 0,
        'buffer_slot': 0,
        'stage': 'gather',
        'gather_done_event': gather_event,
    }
    bucket_group._nep_runtime_config = {
        'ep_rank': 0,
        'edp_group_gloo': fake_gloo_group,
        'nep_owner_transfer_groups_gloo': {},
    }
    bucket_group._nep_nccl_group_index = 0
    bucket_group._nep_nccl_scheduler_state = {'pending_owner_tasks': [context]}
    bucket_group._get_nep_nccl_comm_stream = lambda slot: fake_stream
    bucket_group._start_nep_nccl_owner_edp_reduce = lambda task, use_device_readiness: calls.append(
        ('edp', use_device_readiness)
    )
    bucket_group._start_nep_nccl_owner_task_scatter = lambda task: calls.append('scatter')

    def fake_barrier(group, async_op):
        assert group is fake_gloo_group
        assert async_op
        calls.append('barrier')
        return ready_work

    monkeypatch.setattr(torch.distributed, 'barrier', fake_barrier)
    monkeypatch.setattr(torch.cuda, 'Event', lambda: edp_event)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: FakeStreamContext())

    assert not bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == []

    gather_event.ready = True
    assert not bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ['barrier']

    ready_work.completed = True
    assert not bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ['barrier', ('edp', False)]
    assert edp_event.recorded_stream is fake_stream

    edp_event.ready = True
    assert bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ['barrier', ('edp', False), 'scatter']
    assert bucket_group._nep_nccl_scheduler_state['pending_owner_tasks'] == []


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
        'group': bucket_group,
        'owner_ep_rank': 0,
        'chunk_index': 0,
        'buffer_slot': 0,
        'stage': 'gather',
        'gather_done_event': FakeEvent(),
    }
    bucket_group._nep_runtime_config = {
        'ep_rank': 1,
        'nep_owner_transfer_groups_gloo': {0: source_group},
    }
    bucket_group._nep_nccl_group_index = 0
    bucket_group._nep_nccl_scheduler_state = {'pending_owner_tasks': [context]}
    bucket_group._get_nep_nccl_comm_stream = lambda slot: object()
    bucket_group._start_nep_nccl_owner_task_scatter = lambda task: calls.append('scatter')

    def fake_barrier(group, async_op):
        assert group is source_group
        assert async_op
        calls.append('barrier')
        return works.pop(0)

    monkeypatch.setattr(torch.distributed, 'barrier', fake_barrier)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: FakeStreamContext())

    assert not bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ['barrier']

    gather_work.completed = True
    assert not bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ['barrier', 'barrier']

    scatter_work.completed = True
    assert bucket_group._progress_nep_nccl_pending_owner_tasks()
    assert calls == ['barrier', 'barrier', 'scatter']
    assert bucket_group._nep_nccl_scheduler_state['pending_owner_tasks'] == []


def test_nep_nccl_dense_source_payload_round_trip():
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    source_grad = torch.tensor([20.0, 21.0, 22.0, 23.0])
    source_bucket = type('Bucket', (), {'grad_data': source_grad})()
    bucket_group._nep_runtime_config = {'expert_placement': [[0, 1], [4, 5], [2, 6], [3, 7]]}
    bucket_group._nep_nccl_slot_numel = 4
    bucket_group._nep_nccl_experts_per_owner = 4
    bucket_group._nep_nccl_entries = [
        {'expert_id': 2, 'bucket': source_bucket, 'numel': source_grad.numel()}
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
