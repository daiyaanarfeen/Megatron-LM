# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for opt-in nonuniform expert parallelism helpers."""

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.distributed.nonuniform_common import (
    NonuniformEPRankGenerator,
    build_expert_to_ep_rank_map,
    clear_nonuniform_ep_runtime_config,
    compute_nonuniform_ep_expert_placement,
    get_nonuniform_ep_expert_to_ep_rank_map,
    get_nonuniform_ep_local_expert_indices,
    set_nonuniform_ep_runtime_config,
)
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.distributed.nonuniform_ep import (
    NonuniformEPConfig,
    NonuniformEPParamAndGradBucketGroup,
    _P2PGradTransferHandle,
    _ExpertBucketPlan,
    _ExpertBucketSpec,
    _accumulate_flat_into_bucket,
    _build_expert_bucket_specs,
    _build_synthetic_owner_bucket_specs,
    _copy_flat_into_bucket,
    _owner_for_expert,
    _pack_bucket_slices,
    build_nonuniform_ep_expert_bucket_groups,
)


class TestNonuniformEPTokenRouting:
    def teardown_method(self, method):
        clear_nonuniform_ep_runtime_config()

    def test_expert_to_ep_rank_map_handles_noncontiguous_placement(self):
        placement = [[0, 1], [4, 5], [2, 6], [3, 7]]

        assert build_expert_to_ep_rank_map(placement, num_experts=8) == [
            0,
            0,
            2,
            3,
            1,
            1,
            2,
            3,
        ]

    def test_expert_to_ep_rank_map_rejects_duplicate_expert_holders(self):
        with pytest.raises(RuntimeError, match="one physical holder"):
            build_expert_to_ep_rank_map([[0, 1], [1, 2]], num_experts=3)

    def test_expert_to_ep_rank_map_rejects_unsorted_local_expert_order(self):
        with pytest.raises(RuntimeError, match="ascending global expert order"):
            build_expert_to_ep_rank_map([[1, 0], [2, 3]], num_experts=4)

    def test_registered_runtime_config_drives_token_routing_metadata(self):
        set_nonuniform_ep_runtime_config(
            {
                'local_expert_indices': [2, 6],
                'expert_placement': [[0, 1], [4, 5], [2, 6], [3, 7]],
            }
        )

        assert get_nonuniform_ep_local_expert_indices() == [2, 6]
        assert get_nonuniform_ep_expert_to_ep_rank_map(8) == [0, 0, 2, 3, 1, 1, 2, 3]

    def test_generated_placement_supports_ep32_ep28_shape(self):
        placement, gather_map = compute_nonuniform_ep_expert_placement(
            num_experts=224,
            local_ep_size=32,
            min_ep_size=28,
        )

        assert len(placement) == 32
        assert all(len(experts) == 7 for experts in placement)
        assert placement[0] == list(range(0, 7))
        assert placement[28] == [7, 39, 71, 103, 135, 167, 199]
        assert gather_map[28][0] == (0, 0, 7)

    def test_rank_generator_builds_ep32_ep28_groups_from_tp2_cp2_etp1(self):
        generator = NonuniformEPRankGenerator(
            tp=2,
            cp=2,
            etp=1,
            num_tp_cp_per_replica=[8, 7],
        )

        assert generator.world_size == 60
        assert [len(group) for group in generator.get_ranks('ep')] == [32, 28]
        assert len(generator.get_ranks('edp')) == 28
        assert generator.get_ranks('edp')[0] == [0, 32]


class TestNonuniformEPPlanning:
    def test_owner_for_expert_uses_min_ep_contiguous_owner_ranges(self):
        runtime_config = {
            'min_ep_size': 2,
            'expert_placement': [[0, 1], [4, 5], [2, 3], [6, 7]],
            'ep_rank': 0,
        }

        assert _owner_for_expert(0, runtime_config, None) == 0
        assert _owner_for_expert(3, runtime_config, None) == 0
        assert _owner_for_expert(4, runtime_config, None) == 1
        assert _owner_for_expert(7, runtime_config, None) == 1

    def test_owner_for_expert_prefers_explicit_map(self):
        runtime_config = {
            'min_ep_size': 2,
            'expert_placement': [[0, 1], [2, 3]],
            'ep_rank': 0,
        }

        assert _owner_for_expert(3, runtime_config, {3: 0}) == 0

    def test_build_expert_bucket_specs_maps_local_names_to_global_experts(self):
        p0 = torch.nn.Parameter(torch.randn(2, 2))
        p1 = torch.nn.Parameter(torch.randn(2, 2))
        buffer = Mock()
        buffer.param_index_map = {
            p0: (0, 4, 0),
            p1: (4, 8, 0),
        }
        runtime_config = {'local_expert_indices': [4, 7]}
        param_to_name = {
            p0: "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
            p1: "decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight",
        }

        specs = _build_expert_bucket_specs(
            [buffer],
            runtime_config,
            NonuniformEPConfig(),
            param_to_name,
        )

        assert [(spec.expert_id, spec.start, spec.end) for spec in specs] == [
            (4, 0, 4),
            (7, 4, 8),
        ]
        assert runtime_config['_local_expert_id_set'] == {4, 7}

    def test_build_expert_bucket_specs_maps_grouped_gemm_names(self):
        p0 = torch.nn.Parameter(torch.randn(2, 2))
        p1 = torch.nn.Parameter(torch.randn(2, 2))
        buffer = Mock()
        buffer.param_index_map = {
            p0: (0, 4, 0),
            p1: (4, 8, 0),
        }
        bucket = Mock()
        bucket.params_list = [p0, p1]
        buffer.buckets = [bucket]
        runtime_config = {'local_expert_indices': [8, 9]}
        param_to_name = {
            p0: "decoder.layers.0.mlp.experts.linear_fc1.weight0",
            p1: "decoder.layers.0.mlp.experts.linear_fc1.weight1",
        }

        specs = _build_expert_bucket_specs(
            [buffer],
            runtime_config,
            NonuniformEPConfig(),
            param_to_name,
        )

        assert [(spec.expert_id, spec.start, spec.end) for spec in specs] == [
            (8, 0, 4),
            (9, 4, 8),
        ]
        assert specs[0].slot_key == ("decoder.layers.0.mlp.experts.linear_fc1.weight{expert}",)
        assert runtime_config['_local_expert_id_set'] == {8, 9}

    def test_synthetic_owner_specs_cover_overflow_experts(self):
        buffer = Mock()
        buffer.grad_data = torch.empty(4)
        runtime_config = {
            'ep_rank': 0,
            'min_ep_size': 2,
            'expert_placement': [[0], [2], [1], [3]],
        }
        local_specs = [
            _ExpertBucketSpec(
                buffer=buffer,
                source_bucket_index=0,
                expert_id=0,
                params=[],
                start=0,
                end=4,
                slot_key=("slot",),
            )
        ]
        specs = _build_synthetic_owner_bucket_specs(
            [buffer],
            local_specs,
            runtime_config,
            NonuniformEPConfig(),
        )

        assert len(specs) == 1
        assert specs[0].buffer is buffer
        assert specs[0].expert_id == 1
        assert specs[0].start == 0
        assert specs[0].end == 4
        assert specs[0].synthetic_owner

    def test_bucket_group_builder_groups_multiple_expert_plans_per_source_bucket(self):
        p0 = torch.nn.Parameter(torch.randn(2, 2))
        p1 = torch.nn.Parameter(torch.randn(2, 2))
        buffer = Mock()
        buffer.param_data = None
        buffer.grad_data = torch.zeros(8)
        buffer.gradient_scaling_factor = 1.0
        buffer.data_parallel_group = Mock()
        buffer.data_parallel_group.size.return_value = 1
        buffer.param_index_map = {
            p0: (0, 4, 0),
            p1: (4, 8, 0),
        }
        source_bucket_0 = Mock()
        source_bucket_0.params_list = [p0]
        source_bucket_1 = Mock()
        source_bucket_1.params_list = [p1]
        buffer.buckets = [source_bucket_0, source_bucket_1]
        ep_group = Mock()
        edp_group = Mock()
        edp_group.size.return_value = 2
        runtime_config = {
            'ep_group': ep_group,
            'edp_group': edp_group,
            'ep_rank': 0,
            'min_ep_size': 2,
            'expert_placement': [[0, 1], [2, 3]],
            'local_expert_indices': [0, 1],
        }
        param_to_name = {
            p0: "decoder.layers.0.mlp.experts.linear_fc1.weight0",
            p1: "decoder.layers.0.mlp.experts.linear_fc1.weight1",
        }
        param_to_bucket_group = {p0: object(), p1: object()}

        with patch(
            "megatron.core.distributed.nonuniform_ep.get_global_rank",
            side_effect=lambda group, rank: rank,
        ):
            bucket_groups = build_nonuniform_ep_expert_bucket_groups(
                [buffer],
                DistributedDataParallelConfig(overlap_grad_reduce=True),
                runtime_config,
                NonuniformEPConfig(),
                param_to_bucket_group,
                param_to_name,
            )

        assert len(bucket_groups) == 1
        assert len(bucket_groups[0].buckets) == 2
        assert [plan.expert_id for plan in bucket_groups[0]._nep_plans] == [0, 1]
        assert param_to_bucket_group[p0] is bucket_groups[0]
        assert param_to_bucket_group[p1] is bucket_groups[0]


class TestNonuniformEPTransfers:
    def test_pack_and_accumulate_bucket_slices(self):
        bucket = Mock()
        bucket.grad_data = torch.arange(8, dtype=torch.float32)

        packed = _pack_bucket_slices(bucket, [(1, 3), (5, 8)])
        assert torch.equal(packed, torch.tensor([1.0, 2.0, 5.0, 6.0, 7.0]))

        _accumulate_flat_into_bucket(bucket, [(1, 3), (5, 8)], torch.ones(5))
        assert torch.equal(
            bucket.grad_data,
            torch.tensor([0.0, 2.0, 3.0, 3.0, 4.0, 6.0, 7.0, 8.0]),
        )

    def test_transfer_handle_wait_accumulates_and_copies_recvs(self):
        recv_bucket = Mock()
        recv_bucket.grad_data = torch.zeros(4)
        copy_bucket = Mock()
        copy_bucket.grad_data = torch.ones(4)
        work = Mock()
        keepalive = torch.empty(4)

        handle = _P2PGradTransferHandle(
            works=[work],
            recv_accumulations=[(recv_bucket, [(0, 4)], torch.ones(4))],
            recv_copies=[(copy_bucket, [(1, 3)], torch.tensor([7.0, 8.0]))],
            keepalive_buffers=[keepalive],
        )

        handle.wait()

        work.wait.assert_called_once()
        assert torch.equal(recv_bucket.grad_data, torch.ones(4))
        assert torch.equal(copy_bucket.grad_data, torch.tensor([1.0, 7.0, 8.0, 1.0]))
        assert handle.keepalive_buffers == []

    def test_transfer_handle_completion_is_nonblocking(self):
        pending = Mock()
        pending.is_completed.return_value = False
        complete = Mock()
        complete.is_completed.return_value = True

        handle = _P2PGradTransferHandle([pending, complete])
        assert not handle.is_completed()

        pending.is_completed.return_value = True
        assert handle.is_completed()

    def test_copy_flat_into_bucket_overwrites_synced_grad_slices(self):
        bucket = Mock()
        bucket.grad_data = torch.zeros(6)

        _copy_flat_into_bucket(bucket, [(1, 3), (4, 6)], torch.tensor([2.0, 3.0, 4.0, 5.0]))

        assert torch.equal(bucket.grad_data, torch.tensor([0.0, 2.0, 3.0, 0.0, 4.0, 5.0]))

    def test_extra_main_grads_copy_into_grad_buffer_before_nep_gather(self):
        group = object.__new__(NonuniformEPParamAndGradBucketGroup)
        param = Mock()
        param.main_grad = torch.tensor([3.0, 4.0])
        param.main_grad_copy_in_grad_buffer = torch.zeros(2)
        bucket = Mock()
        bucket.params_with_extra_main_grads = [param]
        group.buckets = [bucket]

        group._copy_extra_main_grads_to_grad_buffer()

        assert torch.equal(param.main_grad_copy_in_grad_buffer, torch.tensor([3.0, 4.0]))

    def test_owner_dp_sync_does_not_recopy_extra_main_grads_after_gather(self):
        group = object.__new__(NonuniformEPParamAndGradBucketGroup)
        param = object()
        bucket = Mock()
        bucket.params_with_extra_main_grads = [param]
        group.buckets = [bucket]

        def fake_start(self, force_all_reduce=False):
            assert bucket.params_with_extra_main_grads == []
            return "started"

        with patch.object(_ParamAndGradBucketGroup, 'start_grad_sync', new=fake_start):
            assert group._start_owner_dp_sync_after_gather() == "started"

        assert bucket.params_with_extra_main_grads == [param]

    def test_owner_dp_sync_waits_for_gather_completion_without_blocking(self):
        group = object.__new__(NonuniformEPParamAndGradBucketGroup)
        gather_handle = Mock()
        gather_handle.is_completed.return_value = False
        group._nep_is_owner = True
        group._nep_started = True
        group._nep_gather_done = False
        group._nep_gather_handle = gather_handle
        group._nep_owner_dp_sync_started = False
        group._nep_owner_dp_sync_scheduler_state = None
        group._start_owner_dp_sync_after_gather = Mock()

        group._try_start_ready_owner_dp_syncs(nonblocking=True)

        group._start_owner_dp_sync_after_gather.assert_not_called()
        gather_handle.wait.assert_not_called()
        assert not group._nep_owner_dp_sync_started

        gather_handle.is_completed.return_value = True
        group._try_start_ready_owner_dp_syncs(nonblocking=True)

        gather_handle.wait.assert_called_once()
        group._start_owner_dp_sync_after_gather.assert_called_once()
        assert group._nep_owner_dp_sync_started

    def test_finish_pre_sync_drains_non_owner_gather(self):
        group = object.__new__(NonuniformEPParamAndGradBucketGroup)
        group.ddp_config = Mock(overlap_grad_reduce=True)
        group.is_first_batch = False
        group.params = [object()]
        group._nep_started = True
        group._nep_is_owner = False
        group._wait_nep_gather_to_owner = Mock()

        group.finish_nep_pre_sync()

        group._wait_nep_gather_to_owner.assert_called_once()

    def test_scatter_wait_is_deferred_until_last_bucket_group(self):
        first = object.__new__(NonuniformEPParamAndGradBucketGroup)
        last = object.__new__(NonuniformEPParamAndGradBucketGroup)
        state = {'entries': [], 'last_bucket_group': last}
        first._nep_post_sync_state = state
        last._nep_post_sync_state = state
        first_handle = Mock()
        last_handle = Mock()
        first._nep_scatter_handle = first_handle
        last._nep_scatter_handle = last_handle
        first._copy_back_extra_main_grads = Mock()
        last._copy_back_extra_main_grads = Mock()

        first._record_nep_scatter_wait(copy_back_after_wait=True)

        first_handle.wait.assert_not_called()
        first._copy_back_extra_main_grads.assert_not_called()

        last._record_nep_scatter_wait(copy_back_after_wait=False)

        first_handle.wait.assert_called_once()
        last_handle.wait.assert_called_once()
        first._copy_back_extra_main_grads.assert_called_once()
        last._copy_back_extra_main_grads.assert_not_called()
        assert state['entries'] == []

    def test_nep_p2p_uses_transfer_group_local_ranks(self):
        group = object.__new__(NonuniformEPParamAndGradBucketGroup)
        ep_group = object()
        transfer_group = object()
        bucket = Mock()
        bucket.grad_data = torch.ones(4)
        plan = _ExpertBucketPlan(
            expert_id=0,
            tag_slot=0,
            owner_ep_rank=2,
            owner_global_rank=42,
            source_ep_ranks=[1],
            source_global_ranks=[41],
            bucket_slices=[(0, 4)],
            bucket_group_index=0,
        )
        group._nep_runtime_config = {
            'ep_group': ep_group,
            'nep_transfer_group': transfer_group,
        }
        group._nep_config = NonuniformEPConfig()
        group._nep_entries = [
            {
                'bucket': bucket,
                'plan': plan,
                'is_owner': False,
                'gather_recv_buffers': [],
                'scatter_send_buffers': [],
                'gather_send_buffer': torch.empty(4),
                'scatter_recv_buffer': torch.empty(4),
            }
        ]

        with patch("megatron.core.distributed.nonuniform_ep.dist.isend") as isend:
            group._start_nep_gather_to_owner()

        isend.assert_called_once()
        assert isend.call_args.kwargs['group'] is transfer_group
        assert isend.call_args.kwargs['group_dst'] == 2
        assert 'dst' not in isend.call_args.kwargs

    def test_owner_uses_persistent_peer_buffers_for_gather_and_scatter(self):
        group = object.__new__(NonuniformEPParamAndGradBucketGroup)
        bucket = Mock()
        bucket.grad_data = torch.zeros(4)
        group.buckets = [bucket]
        group._nep_runtime_config = {'ep_rank': 0}
        group._nep_plan = _ExpertBucketPlan(
            expert_id=0,
            tag_slot=0,
            owner_ep_rank=0,
            owner_global_rank=10,
            source_ep_ranks=[0, 1, 2],
            source_global_ranks=[10, 11, 12],
            bucket_slices=[(0, 4)],
            bucket_group_index=0,
        )
        group._nep_is_owner = True
        group._nep_gather_recv_buffers = []
        group._nep_scatter_send_buffers = []
        group._nep_gather_send_buffer = None
        group._nep_scatter_recv_buffer = None

        group._allocate_nep_persistent_grad_buffers()

        assert len(group._nep_gather_recv_buffers) == 2
        assert group._nep_scatter_send_buffers is group._nep_gather_recv_buffers
        assert all(buffer.numel() == 4 for _, _, buffer in group._nep_gather_recv_buffers)
