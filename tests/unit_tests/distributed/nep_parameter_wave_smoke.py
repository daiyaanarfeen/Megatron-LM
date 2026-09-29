"""Exercise cached parameter waves with real NCCL and shared staging storage."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from cuda_event_capture_adapter import CudaEventCapture

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.nonuniform_common import (
    compute_nonuniform_ep_expert_placement,
    compute_nonuniform_ep_owner_expert_slots,
)
from megatron.core.distributed.nonuniform_ep import NonuniformEPNCCLParamAndGradBucketGroup


def exercise(experts, owners):
    rank, ep, n = dist.get_rank(), dist.get_world_size(), 1024
    placement = compute_nonuniform_ep_expert_placement(
        experts, ep, owners, preferred_follower_fanout=1
    )
    slots = compute_nonuniform_ep_owner_expert_slots(experts, owners)
    members = {
        owner: [owner]
        + [
            r
            for r, ids in enumerate(placement)
            if r != owner and set(ids).intersection(slots[owner])
        ]
        for owner in range(owners)
    }
    transfers = {
        owner: dist.new_group(ranks, group_desc='nep_owner_transfer') if len(ranks) > 1 else None
        for owner, ranks in members.items()
    }
    config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_param_gather=True
    )
    buffers, groups = {}, []
    stream = torch.cuda.Stream()
    lane_count = int(os.getenv('NEP_ABLATION_PARAM_LANES', '0'))
    lane_streams = (torch.cuda.Stream(), torch.cuda.Stream()) if lane_count else None
    max_numel = n * max(
        len([e for e in slots[o] if e is not None and e not in placement[o]]) for o in range(owners)
    )
    for index in range(3):
        group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
            NonuniformEPNCCLParamAndGradBucketGroup
        )
        group.ddp_config = config
        group._nep_distopt_param_sync_representative = True
        group.param_gather_dispatched = False
        group.next_param_gather_bucket_group = None
        group._nep_nccl_group_index = index
        group._nep_runtime_config = {
            'ep_rank': rank,
            'ep_group': dist.group.WORLD,
            'expert_placement': placement,
        }
        group._get_nep_nccl_owner_layout = lambda: {
            'min_ep_size': owners,
            'expert_stride': n,
            'owner_expert_slots': slots,
        }
        group._nep_nccl_expert_stride = n
        group._nep_nccl_segment_cache = {}
        group._nep_nccl_slot_offsets = [0]
        group._nep_nccl_slot_numels = [n]
        group._nep_nccl_owner_transfer_ranks = members.__getitem__
        group._get_nep_nccl_transfer_group_info = lambda o: (
            transfers[o],
            None,
            len(members[o]),
            members[o],
        )
        group._nep_nccl_owner_expert_ids = lambda o: [e for e in slots[o] if e is not None]
        physical = {
            e: torch.nn.Parameter(torch.empty(n, device='cuda', dtype=torch.bfloat16))
            for e in placement[rank]
        }
        entries = {
            o: [
                {
                    'bucket': SimpleNamespace(params_list=[physical[e]]),
                    'expert_id': e,
                    'slot_offset': 0,
                    'slot_index': 0,
                    'numel': n,
                }
                for e in placement[rank]
                if e in slots[o]
            ]
            for o in range(owners)
        }
        group._nep_nccl_owner_entries = entries.__getitem__
        proxies = {
            (index, e, 0): torch.nn.Parameter(torch.empty(n, device='cuda', dtype=torch.bfloat16))
            for e in (slots[rank] if rank < owners else [])
            if e is not None
        }
        group._nep_distopt_owner_bundle = dict(
            groups=(group,),
            native_group=None,
            proxy_by_key=proxies,
            param_transfer_buffers=buffers,
            param_transfer_slot=index % 2,
            param_transfer_numel_by_slot={0: max_numel, 1: max_numel},
            param_transfer_stream=stream,
            param_sync_event=torch.cuda.Event(),
            param_sync_completed=False,
        )
        if lane_streams:
            group._nep_distopt_owner_bundle.update(
                param_lane_streams=lane_streams,
                param_lane_count=lane_count,
                param_lane_ready=torch.cuda.Event(),
            )
            for slot in range(2):
                for lane in range(2):
                    key = (slot, torch.bfloat16, torch.device('cuda', rank), lane)
                    if key not in buffers:
                        buffers[key] = torch.empty(max_numel, device='cuda', dtype=torch.bfloat16)
            group._get_nep_distopt_param_transfer_plan()
        groups.append((group, physical, proxies))
    for left, right in zip(groups, groups[1:]):
        left[0].next_param_gather_bucket_group = right[0]
    for iteration in range(9):
        config.overlap_param_gather = iteration != 6
        for index, (group, physical, proxies) in enumerate(groups):
            group.param_gather_dispatched = False
            for param in physical.values():
                param.data.fill_(-1)
            for (_, expert, _), param in proxies.items():
                param.data.fill_(iteration * 20 + index * 2 + expert)
        dist.barrier()
        if rank == 0:
            torch.cuda._sleep(1000000)
        groups[0][0].start_param_sync(force_sync=iteration == 3)
        for index, (group, physical, _) in enumerate(groups):
            if not config.overlap_param_gather:
                group.start_param_sync()
            group.finish_param_sync()
            for expert, param in physical.items():
                assert torch.equal(
                    param, torch.full_like(param, iteration * 20 + index * 2 + expert)
                )
        if lane_streams:
            assert len(buffers) == 4
        else:
            assert not buffers


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    CudaEventCapture.install_stage_labels()
    capture = CudaEventCapture(schedule=lambda step: torch.profiler.ProfilerAction.RECORD)
    capture.start()
    exercise(16, 2)
    capture.step()
    exercise(17, 3)
    capture.stop()
    if dist.get_rank() == 0:
        print(
            'PARAMETER_WAVE_SMOKE_PASS: real plans, divisible/nondivisible, shared followers, fresh values, buffer reuse, sync/async',
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
