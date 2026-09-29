"""Check compact parameter delivery with real NCCL and nondivisible expert layouts."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from cuda_event_capture_adapter import CudaEventCapture

from megatron.core.distributed.nonuniform_common import (
    compute_nonuniform_ep_expert_placement,
    compute_nonuniform_ep_owner_expert_slots,
)
from megatron.core.distributed.nonuniform_ep import NonuniformEPNCCLParamAndGradBucketGroup


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size == 4
    CudaEventCapture.install_stage_labels()
    capture = CudaEventCapture(schedule=lambda step: torch.profiler.ProfilerAction.RECORD)
    capture.start()
    # Divisible, nondivisible, one/two followers, and singleton owner transfers.
    for num_experts, min_ep in ((16, 2), (23, 3), (13, 2), (4, 3)):
        placement = compute_nonuniform_ep_expert_placement(num_experts, size, min_ep)
        owner_slots = compute_nonuniform_ep_owner_expert_slots(num_experts, min_ep)
        transfers = []
        for owner in range(min_ep):
            expert_ids = set(owner_slots[owner]) - {None}
            ranks = tuple(r for r in range(size) if expert_ids.intersection(placement[r]))
            pg = dist.new_group(ranks, group_desc='nep_owner_transfer') if len(ranks) > 1 else None
            transfers.append((pg, None, len(ranks), ranks))
        shared_buffers = {}
        groups = []
        for index in range(3):
            group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
                NonuniformEPNCCLParamAndGradBucketGroup
            )
            group._nep_nccl_group_index = index
            group._nep_nccl_segment_cache = {}
            group._nep_nccl_slot_numels = (1024, 2048)
            group._nep_nccl_slot_offsets = (0, 1024)
            group._nep_nccl_expert_stride = 3072
            group._nep_nccl_owner_layout = dict(
                min_ep_size=min_ep,
                expert_stride=3072,
                owner_numel=len(owner_slots[0]) * 3072,
                owner_expert_slots=owner_slots,
            )
            group._nep_runtime_config = dict(
                ep_rank=rank, ep_group=dist.group.WORLD, expert_placement=placement
            )
            group._get_nep_nccl_transfer_group_info = lambda owner: transfers[owner]
            group._nep_nccl_entries = []
            for expert in placement[rank]:
                for slot, numel in enumerate(group._nep_nccl_slot_numels):
                    param = torch.nn.Parameter(
                        torch.empty(numel, device='cuda', dtype=torch.bfloat16)
                    )
                    group._nep_nccl_entries.append(
                        dict(
                            expert_id=expert,
                            slot_index=slot,
                            slot_offset=group._nep_nccl_slot_offsets[slot],
                            numel=numel,
                            bucket=SimpleNamespace(params_list=[param]),
                        )
                    )
            group._nep_distopt_owner_bundle = dict(
                groups=(group,),
                proxy_by_key={},
                param_transfer_buffers=shared_buffers,
                param_transfer_slot=index % 2,
                param_transfer_numel_by_slot={
                    0: len(owner_slots[0]) * 3072,
                    1: len(owner_slots[0]) * 3072,
                },
            )
            groups.append(group)
        stream = torch.cuda.Stream()
        for iteration in range(3):
            for index, group in enumerate(groups):
                for entry in group._nep_nccl_entries:
                    entry['bucket'].params_list[0].data.fill_(-1)
                # Rebind storage each iteration, as native DistOpt can do.
                proxies = group._nep_distopt_owner_bundle['proxy_by_key']
                if rank < min_ep:
                    for expert in owner_slots[rank]:
                        if expert is not None:
                            for slot, numel in enumerate(group._nep_nccl_slot_numels):
                                proxies[index, expert, slot] = torch.full(
                                    (numel,),
                                    iteration * 60 + index * 15 + expert + slot,
                                    device='cuda',
                                    dtype=torch.bfloat16,
                                )
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for group in groups:
                    group._scatter_nep_distopt_owner_params_to_physical_holders()
            torch.cuda.current_stream().wait_stream(stream)
            for index, group in enumerate(groups):
                for entry in group._nep_nccl_entries:
                    param = entry['bucket'].params_list[0]
                    expected = (
                        iteration * 60 + index * 15 + entry['expert_id'] + entry['slot_index']
                    )
                    assert torch.equal(param, torch.full_like(param, expected)), (
                        rank,
                        num_experts,
                        min_ep,
                        index,
                    )
                for (
                    pg,
                    count,
                    sends,
                    receives,
                    send_numel,
                    recv_numel,
                    pack,
                    unpack,
                ) in group._get_nep_distopt_param_transfer_plan():
                    if count > 1:
                        ranks = dist.get_process_group_ranks(pg)
                        owner = ranks[0]
                        remote = set(owner_slots[owner]) - {None} - set(placement[owner])
                        assert sends == [
                            len(remote.intersection(placement[r])) * 3072 if rank == owner else 0
                            for r in ranks
                        ]
                        assert receives == [
                            len(remote.intersection(placement[rank])) * 3072 if r == owner else 0
                            for r in ranks
                        ]
                        assert send_numel == sum(sends)
                        assert recv_numel == sum(receives)
                        assert not (send_numel and recv_numel)
            dist.barrier()
            capture.step()
        if rank == 0:
            print(f'COMPACT_PARAMS_PASS E={num_experts} EP={size}/{min_ep}', flush=True)
    capture.stop()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
