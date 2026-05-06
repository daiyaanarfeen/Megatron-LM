# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for opt-in nonuniform expert parallelism helpers."""

from unittest.mock import Mock, patch

import torch

from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
from megatron.core.distributed.nonuniform_ep import (
    NonuniformEPConfig,
    NonuniformEPParamAndGradBucketGroup,
    _P2PGradTransferHandle,
    _ExpertBucketPlan,
    _accumulate_flat_into_bucket,
    _build_expert_bucket_specs,
    _copy_flat_into_bucket,
    _owner_for_expert,
    _pack_bucket_slices,
)


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

        assert [(expert_id, start, end) for _, expert_id, _, start, end in specs] == [
            (4, 0, 4),
            (7, 4, 8),
        ]
        assert runtime_config['_local_expert_id_set'] == {4, 7}


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

    def test_owner_uses_persistent_peer_buffers_for_gather_and_scatter(self):
        group = object.__new__(NonuniformEPParamAndGradBucketGroup)
        bucket = Mock()
        bucket.grad_data = torch.zeros(4)
        group.buckets = [bucket]
        group._nep_runtime_config = {'ep_rank': 0}
        group._nep_plan = _ExpertBucketPlan(
            expert_id=0,
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
