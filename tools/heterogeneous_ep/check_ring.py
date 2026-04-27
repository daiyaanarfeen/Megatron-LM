#!/usr/bin/env python3
"""Minimal test: NVSHMEM ring allreduce between 2 PEs.
Each PE starts with a buffer of 1.0, after ring SUM both should have 2.0."""
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
if rank == 0: uid = nvshmem.core.get_unique_id()
uids = [uid]; dist.broadcast_object_list(uids, src=0)
Device = get_cuda_core_device_class()
dev = Device(local_rank); dev.set_current()
nvshmem.core.init(device=dev, uid=uids[0], rank=rank, nranks=world_size, initializer_method="uid")
stream = dev.create_stream()
_, sp = stream.__cuda_stream__()
nv_torch = torch.cuda.ExternalStream(sp)

# Allocate symmetric buffers
SLOT = 1024 * 4  # 1024 float32 elements = 4KB
send_buf = nvshmem.core.interop.torch.bytetensor((SLOT,), dtype=torch.uint8)
# Double-buffered exchange: ping-pong to avoid remote put overwriting
# data before local add_ finishes reading.
exchange_bufs = [
    nvshmem.core.interop.torch.bytetensor((SLOT,), dtype=torch.uint8),
    nvshmem.core.interop.torch.bytetensor((SLOT,), dtype=torch.uint8),
]
exchange_signals = [nvshmem.core.buffer(8), nvshmem.core.buffer(8)]
send_buf.zero_(); exchange_bufs[0].zero_(); exchange_bufs[1].zero_()
# Zero signals via put-to-self.
zero8 = nvshmem.core.interop.torch.bytetensor((8,), dtype=torch.uint8)
zero8.zero_()
for sig in exchange_signals:
    nvshmem.core.put(sig, zero8, rank, stream=stream)
nvshmem.core.quiet(stream=stream)
torch.cuda.synchronize()
nvshmem.core.barrier_all(stream=stream)
nv_torch.synchronize()

# Test data: each rank has [1.0, 1.0, ..., 1.0] (16 elements)
N = world_size  # ring size
n_elems = 16
data = torch.ones(n_elems, device='cuda', dtype=torch.float32)
dtype = data.dtype
elem = data.element_size()

# Ring topology
my_idx = rank
next_pe = (rank + 1) % N
prev_pe = (rank - 1) % N

# Split into N sub-chunks
sub_n = n_elems // N
sub_offsets = [i * sub_n for i in range(N)]
sub_sizes = [sub_n] * N

sig_base = 1000000
step = 0

print(f"Rank {rank}: data before ring = {data.tolist()}", flush=True)

# Phase 1: Reduce-scatter
for s in range(N - 1):
    send_idx = (my_idx - s) % N
    recv_idx = (my_idx - s - 1) % N
    send_n = sub_sizes[send_idx]
    recv_n = sub_sizes[recv_idx]
    send_off = sub_offsets[send_idx]
    recv_off = sub_offsets[recv_idx]
    send_bytes = send_n * elem
    recv_bytes = recv_n * elem
    parity = step % 2
    xbuf = exchange_bufs[parity]
    xsig = exchange_signals[parity]
    sig_val = sig_base; sig_base += 1; step += 1

    with torch.cuda.stream(nv_torch):
        send_buf[:send_bytes].view(dtype)[:send_n].copy_(data[send_off:send_off+send_n])
    nvshmem.core.put(xbuf[:send_bytes], send_buf[:send_bytes], next_pe, stream=stream)
    nvshmem.core.signal_op(xsig, sig_val, nvshmem.core.SignalOp.SIGNAL_SET,
                           next_pe, stream=stream)
    nvshmem.core.quiet(stream=stream)
    nvshmem.core.signal_wait(xsig, sig_val, nvshmem.core.ComparisonType.CMP_GE,
                             stream=stream)
    with torch.cuda.stream(nv_torch):
        data[recv_off:recv_off+recv_n].add_(xbuf[:recv_bytes].view(dtype)[:recv_n])

nv_torch.synchronize()
print(f"Rank {rank}: data after reduce-scatter = {data.tolist()}", flush=True)

# Phase 2: Allgather
for s in range(N - 1):
    send_idx = (my_idx - s + 1) % N
    recv_idx = (my_idx - s) % N
    send_n = sub_sizes[send_idx]
    recv_n = sub_sizes[recv_idx]
    send_off = sub_offsets[send_idx]
    recv_off = sub_offsets[recv_idx]
    send_bytes = send_n * elem
    recv_bytes = recv_n * elem
    parity = step % 2
    xbuf = exchange_bufs[parity]
    xsig = exchange_signals[parity]
    sig_val = sig_base; sig_base += 1; step += 1

    with torch.cuda.stream(nv_torch):
        send_buf[:send_bytes].view(dtype)[:send_n].copy_(data[send_off:send_off+send_n])
    nvshmem.core.put(xbuf[:send_bytes], send_buf[:send_bytes], next_pe, stream=stream)
    nvshmem.core.signal_op(xsig, sig_val, nvshmem.core.SignalOp.SIGNAL_SET,
                           next_pe, stream=stream)
    nvshmem.core.quiet(stream=stream)
    nvshmem.core.signal_wait(xsig, sig_val, nvshmem.core.ComparisonType.CMP_GE,
                             stream=stream)
    with torch.cuda.stream(nv_torch):
        data[recv_off:recv_off+recv_n].copy_(xbuf[:recv_bytes].view(dtype)[:recv_n])

nv_torch.synchronize()
torch.cuda.synchronize()

expected = float(N)
all_correct = all(abs(x - expected) < 1e-5 for x in data.tolist())
status = "PASS" if all_correct else "FAIL"
print(f"Rank {rank}: data after ring = {data.tolist()}, expected all={expected}, {status}", flush=True)

dist.barrier(); dist.destroy_process_group()
