#!/usr/bin/env python3
"""EXACT combined pattern: NVSHMEM put+quiet+sync → NCCL barrier+allreduce+barrier, K>1.
Uses exact Approach B groups. Minimal reproducer for multi-chunk hang."""
import os, torch, torch.distributed as dist

rank = int(os.environ.get('RANK', '0'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
ws = int(os.environ.get('WORLD_SIZE', '1'))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', world_size=ws, rank=rank,
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

from megatron.core.resharding.nvshmem_copy_service.compat import ensure_nvshmem_compat, get_cuda_core_device_class
ensure_nvshmem_compat()
import nvshmem.core

uid = nvshmem.core.get_unique_id(empty=True)
if rank == 0: uid = nvshmem.core.get_unique_id()
uids = [uid]; dist.broadcast_object_list(uids, src=0)
Device = get_cuda_core_device_class()
dev = Device(local_rank); dev.set_current()
nvshmem.core.init(device=dev, uid=uids[0], rank=rank, nranks=ws, initializer_method="uid")
stream = dev.create_stream()
_, sp = stream.__cuda_stream__()
nv_torch = torch.cuda.ExternalStream(sp)

# NVSHMEM buffers
SLOT = 1024 * 1024  # 1MB
slot = nvshmem.core.interop.torch.bytetensor((SLOT,), dtype=torch.uint8)
local_slot = nvshmem.core.interop.torch.bytetensor((SLOT,), dtype=torch.uint8)
slot.zero_(); local_slot.zero_()
torch.cuda.synchronize()
nvshmem.core.barrier_all(stream=stream)
nv_torch.synchronize()

# NCCL groups (exact Approach B pattern)
ep_groups_all = [[0,2], [1,3], [4,6,8,10], [5,7,9,11]]
b_edp_groups_all = [[0,4], [1,5]] + [[r] for r in range(2,4)] + [[r] for r in range(6,12)]
my_ep = my_b_edp = None
for ranks in ep_groups_all:
    g = dist.new_group(ranks)
    if rank in ranks: my_ep = g
for ranks in b_edp_groups_all:
    g = dist.new_group(ranks)
    if rank in ranks: my_b_edp = g

ep_rank = my_ep.rank()
dist.barrier()
print(f"Rank {rank}: ep_rank={ep_rank} b_edp_size={my_b_edp.size()}", flush=True)

buf = torch.ones(1024, device='cuda')
K = 5
for i in range(K):
    # Stage 1: NVSHMEM put + quiet + host sync + NCCL barrier
    local_slot[:4].view(torch.float32).fill_(float(rank + i))
    torch.cuda.synchronize()
    nvshmem.core.put(slot[:SLOT], local_slot[:SLOT], 0, stream=stream)
    nvshmem.core.quiet(stream=stream)
    nv_torch.synchronize()
    print(f"Rank {rank}: iter {i} put done", flush=True)
    dist.barrier(group=my_ep)
    print(f"Rank {rank}: iter {i} bar1 done", flush=True)

    # Stage 2: NCCL allreduce (Approach B edp)
    if ep_rank == 0:
        dist.all_reduce(buf, group=my_b_edp)
    else:
        dist.all_reduce(buf, group=my_b_edp)
    torch.cuda.synchronize()
    print(f"Rank {rank}: iter {i} ar done", flush=True)

    # Stage 3: NVSHMEM put + quiet + host sync + NCCL barrier
    if ep_rank == 0:
        nvshmem.core.put(local_slot[:SLOT], slot[:SLOT], rank, stream=stream)
        nvshmem.core.quiet(stream=stream)
        nv_torch.synchronize()
    dist.barrier(group=my_ep)
    print(f"Rank {rank}: iter {i} done", flush=True)

print(f"Rank {rank}: all {K} iterations done", flush=True)
dist.barrier(); dist.destroy_process_group()
