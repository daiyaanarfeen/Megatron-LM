# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Probe native NCCL zero-CTA Gather/Scatter through the public C API."""

import argparse
import ctypes
import gzip
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

_NCCL_DTYPES = {
    torch.int8: 0,
    torch.uint8: 1,
    torch.int32: 2,
    torch.int64: 4,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
}


class _NativeNCCL:
    """Minimal bindings for collectives that ProcessGroupNCCL does not expose natively."""

    def __init__(self) -> None:
        self._library = ctypes.CDLL("libnccl.so.2")
        self._library.ncclGetErrorString.argtypes = [ctypes.c_int]
        self._library.ncclGetErrorString.restype = ctypes.c_char_p
        self._library.ncclGather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._library.ncclGather.restype = ctypes.c_int
        self._library.ncclScatter.argtypes = list(self._library.ncclGather.argtypes)
        self._library.ncclScatter.restype = ctypes.c_int

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        message = self._library.ncclGetErrorString(result).decode("utf-8")
        raise RuntimeError(f"{operation} failed with NCCL error {result}: {message}")

    @staticmethod
    def _arguments(
        send: torch.Tensor, recv: torch.Tensor, count: int, root: int, comm_ptr: int
    ) -> tuple:
        if send.device.type != "cuda" or recv.device.type != "cuda":
            raise RuntimeError("native NCCL buffers must be CUDA tensors")
        if not send.is_contiguous() or not recv.is_contiguous():
            raise RuntimeError("native NCCL buffers must be contiguous")
        if send.dtype != recv.dtype or send.dtype not in _NCCL_DTYPES:
            raise RuntimeError(f"unsupported native NCCL dtype pair: {send.dtype}, {recv.dtype}")
        stream_ptr = torch.cuda.current_stream(send.device).cuda_stream
        return (
            ctypes.c_void_p(send.data_ptr()),
            ctypes.c_void_p(recv.data_ptr()),
            ctypes.c_size_t(count),
            ctypes.c_int(_NCCL_DTYPES[send.dtype]),
            ctypes.c_int(root),
            ctypes.c_void_p(comm_ptr),
            ctypes.c_void_p(stream_ptr),
        )

    def gather(
        self, send: torch.Tensor, recv: torch.Tensor, count: int, root: int, comm_ptr: int
    ) -> None:
        result = self._library.ncclGather(*self._arguments(send, recv, count, root, comm_ptr))
        self._check(result, "ncclGather")

    def scatter(
        self, send: torch.Tensor, recv: torch.Tensor, count: int, root: int, comm_ptr: int
    ) -> None:
        result = self._library.ncclScatter(*self._arguments(send, recv, count, root, comm_ptr))
        self._check(result, "ncclScatter")


def _new_zero_cta_group(ranks):
    options = dist.ProcessGroupNCCL.Options()
    if not hasattr(options.config, "cta_policy"):
        raise RuntimeError("This PyTorch build does not expose NCCL cta_policy")
    options.config.cta_policy = 2
    return dist.new_group(
        ranks=ranks,
        backend="nccl",
        pg_options=options,
        group_desc="probe_native_zero_cta",
    )


def _get_backend_and_comm_ptr(group):
    backend = group._get_backend(torch.device("cuda", torch.cuda.current_device()))
    get_comm_ptr = getattr(backend, "_comm_ptr", None)
    if get_comm_ptr is None:
        raise RuntimeError("This ProcessGroupNCCL build does not expose _comm_ptr()")
    return backend, int(get_comm_ptr())


def _run_profiled(label: str, operation, trace_dir: Path) -> None:
    rank = dist.get_rank()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    ) as profile:
        for _ in range(4):
            with torch.profiler.record_function(label):
                operation()
            profile.step()
    torch.cuda.synchronize()
    if rank == 0:
        trace_dir.mkdir(parents=True, exist_ok=True)
        raw_path = trace_dir / f"{label}.json"
        compressed_path = trace_dir / f"{label}.json.gz"
        profile.export_chrome_trace(str(raw_path))
        with (
            raw_path.open("rt", encoding="utf-8") as source,
            gzip.open(compressed_path, "wt", encoding="utf-8") as target,
        ):
            json.dump(json.load(source), target)
        raw_path.unlink()
        print(f"[native-zero-cta-probe] trace={compressed_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--elements-per-rank", type=int, default=1 << 20)
    parser.add_argument("--group-ranks", type=int, nargs="+")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("native zero-CTA probe requires at least two ranks")

    group_ranks = args.group_ranks or list(range(world_size))
    if len(group_ranks) < 2 or len(set(group_ranks)) != len(group_ranks):
        raise RuntimeError(f"invalid native zero-CTA probe ranks: {group_ranks}")
    if min(group_ranks) < 0 or max(group_ranks) >= world_size:
        raise RuntimeError(
            f"native zero-CTA probe ranks {group_ranks} exceed world size {world_size}"
        )

    control_group = dist.new_group(
        ranks=list(range(world_size)), backend="gloo", group_desc="probe_control"
    )
    group = _new_zero_cta_group(group_ranks)
    if rank not in group_ranks:
        dist.barrier(group=control_group)
        dist.destroy_process_group()
        return

    group_rank = dist.get_rank(group=group)
    group_size = dist.get_world_size(group=group)
    warmup = torch.ones(1, device="cuda")
    dist.all_reduce(warmup, group=group)
    torch.cuda.synchronize()
    backend, comm_ptr = _get_backend_and_comm_ptr(group)
    pool = torch.cuda.MemPool(backend.mem_allocator)
    with torch.cuda.use_mem_pool(pool):
        small = torch.empty(args.elements_per_rank, dtype=torch.bfloat16, device="cuda")
        large = torch.empty(
            group_size * args.elements_per_rank, dtype=torch.bfloat16, device="cuda"
        )
    try:
        backend.register_mem_pool(pool, symm=True)
    except TypeError:
        backend.register_mem_pool(pool)

    native_nccl = _NativeNCCL()
    small.fill_(group_rank)
    large.fill_(-1)
    _run_profiled(
        "native_zero_gather",
        lambda: native_nccl.gather(small, large, small.numel(), 0, comm_ptr),
        args.trace_dir,
    )
    if group_rank == 0:
        expected = torch.cat(
            [torch.full_like(small, source_rank) for source_rank in range(group_size)]
        )
        torch.testing.assert_close(large, expected)

    if group_rank == 0:
        for destination_rank, destination in enumerate(large.chunk(group_size)):
            destination.fill_(100 + destination_rank)
    small.fill_(-1)
    _run_profiled(
        "native_zero_scatter",
        lambda: native_nccl.scatter(large, small, small.numel(), 0, comm_ptr),
        args.trace_dir,
    )
    torch.testing.assert_close(small, torch.full_like(small, 100 + group_rank))

    if group_rank == 0:
        print(
            f"[native-zero-cta-probe] PASS group_ranks={group_ranks} "
            f"nccl={torch.cuda.nccl.version()} comm_ptr={comm_ptr:#x}",
            flush=True,
        )
    dist.barrier(group=control_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
