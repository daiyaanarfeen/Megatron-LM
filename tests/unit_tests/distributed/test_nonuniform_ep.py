# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.distributed.nonuniform_ep import NonuniformEPNCCLParamAndGradBucketGroup


class _FakeWork:
    def __init__(self):
        self.block_calls = 0
        self.wait_calls = 0

    def block_current_stream(self):
        self.block_calls += 1

    def wait(self):
        self.wait_calls += 1


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


def test_nep_nccl_owner_tasks_use_bounded_distinct_stream_slots(monkeypatch):
    monkeypatch.setenv("MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW", "4")
    bucket_group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
        NonuniformEPNCCLParamAndGradBucketGroup
    )
    bucket_group._nep_nccl_owner_layout = {'num_chunks': 2}

    slots = [
        bucket_group._get_nep_nccl_task_buffer_slot(owner_ep_rank, chunk_index)
        for owner_ep_rank in range(3)
        for chunk_index in range(2)
    ]

    assert slots == [0, 1, 2, 3, 0, 1]


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
