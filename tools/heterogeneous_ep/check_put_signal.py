#!/usr/bin/env python3
"""Minimal test: nvshmem put_signal + signal_wait across 2 PEs."""
import os, torch, torch.distributed as dist

rank = int(os.environ.get('RANK', '0'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '1'))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', world_size=world_size, rank=rank,
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

from megatron.core.resharding.nvshmem_copy_service.compat import ensure_nvshmem_compat, get_cuda_core_device_class
ensure_nvshmem_compat()
import nvshmem.core

uid = nvshmem.core.get_unique_id(empty=True)
if rank == 0:
    uid = nvshmem.core.get_unique_id()
uids = [uid]
dist.broadcast_object_list(uids, src=0)

Device = get_cuda_core_device_class()
device = Device(local_rank)
device.set_current()
nvshmem.core.init(device=device, uid=uids[0], rank=rank, nranks=world_size, initializer_method="uid")
stream = device.create_stream()
print(f"Rank {rank}: NVSHMEM init OK, PE={nvshmem.core.my_pe()}", flush=True)

# Allocate symmetric data + signal buffers (collective)
data_buf = nvshmem.core.interop.torch.bytetensor((1024,), dtype=torch.uint8)
data_view = data_buf[:4].view(torch.float32)
sig_buf = nvshmem.core.buffer(8)  # 1 x uint64 signal
torch.cuda.synchronize()
nvshmem.core.barrier_all(stream=stream)
_, sp = stream.__cuda_stream__()
torch.cuda.ExternalStream(sp).synchronize()
print(f"Rank {rank}: buffers allocated", flush=True)

# Test: rank 1 does put_signal to rank 0, rank 0 does signal_wait
if rank == 1:
    data_view.fill_(42.0)
    print(f"Rank 1: calling put_signal to PE 0", flush=True)
    nvshmem.core.put_signal(
        data_view,   # dest on PE 0
        data_view,   # src (local)
        sig_buf,     # signal on PE 0
        1,           # signal value
        nvshmem.core.SignalOp.SIGNAL_SET,
        0,           # dest PE
        stream=stream,
    )
    nvshmem.core.quiet(stream=stream)
    torch.cuda.ExternalStream(sp).synchronize()
    print(f"Rank 1: put_signal done", flush=True)
elif rank == 0:
    print(f"Rank 0: calling signal_wait", flush=True)
    nvshmem.core.signal_wait(
        sig_buf,
        1,
        nvshmem.core.ComparisonType.CMP_GE,
        stream=stream,
    )
    torch.cuda.ExternalStream(sp).synchronize()
    print(f"Rank 0: signal_wait done, data={data_view.item()}", flush=True)

nvshmem.core.barrier_all(stream=stream)
torch.cuda.ExternalStream(sp).synchronize()
dist.barrier()
dist.destroy_process_group()
print(f"Rank {rank}: done", flush=True)
