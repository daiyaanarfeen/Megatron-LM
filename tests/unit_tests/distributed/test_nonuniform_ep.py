# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for opt-in nonuniform expert parallelism helpers."""

from unittest.mock import Mock

import torch

from megatron.core.distributed.nonuniform_ep import (
    NonuniformEPConfig,
    _P2PTransferHandle,
    _accumulate_flat_into_bucket,
    _build_expert_bucket_specs,
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

    def test_transfer_handle_wait_accumulates_recvs_and_zeroes_sent_slices(self):
        recv_bucket = Mock()
        recv_bucket.grad_data = torch.zeros(4)
        send_bucket = Mock()
        send_bucket.grad_data = torch.ones(4)
        work = Mock()

        handle = _P2PTransferHandle(
            works=[work],
            recv_accumulations=[(recv_bucket, [(0, 4)], torch.ones(4))],
            zero_slices=[(send_bucket, [(1, 3)])],
        )

        handle.wait()

        work.wait.assert_called_once()
        assert torch.equal(recv_bucket.grad_data, torch.ones(4))
        assert torch.equal(send_bucket.grad_data, torch.tensor([1.0, 0.0, 0.0, 1.0]))
