# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Minimal bindings for NCCL collectives not exposed natively by ProcessGroupNCCL."""

import ctypes
import threading
from typing import Optional

import torch

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
_native_nccl: Optional["NativeNCCL"] = None
_native_nccl_lock = threading.Lock()


class NativeNCCL:
    """Call native NCCL Gather/Scatter on a ProcessGroupNCCL communicator.

    These calls enqueue directly on the current CUDA stream. They are not represented by
    a ProcessGroupNCCL Work object and are therefore not monitored by its watchdog.
    """

    def __init__(self) -> None:
        self._library = ctypes.CDLL("libnccl.so.2")
        self._library.ncclGetErrorString.argtypes = [ctypes.c_int]
        self._library.ncclGetErrorString.restype = ctypes.c_char_p
        collective_argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._library.ncclGather.argtypes = collective_argtypes
        self._library.ncclGather.restype = ctypes.c_int
        self._library.ncclScatter.argtypes = collective_argtypes
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
            raise RuntimeError("Native NCCL buffers must be CUDA tensors")
        if not send.is_contiguous() or not recv.is_contiguous():
            raise RuntimeError("Native NCCL buffers must be contiguous")
        if send.dtype != recv.dtype or send.dtype not in _NCCL_DTYPES:
            raise RuntimeError(f"Unsupported native NCCL dtype pair: {send.dtype}, {recv.dtype}")
        if count < 0 or count > send.numel():
            raise RuntimeError(
                f"Native NCCL count {count} exceeds the send buffer size {send.numel()}"
            )
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


def get_native_nccl() -> NativeNCCL:
    """Return the process-wide lazy native NCCL binding."""
    global _native_nccl
    if _native_nccl is not None:
        return _native_nccl
    with _native_nccl_lock:
        if _native_nccl is None:
            _native_nccl = NativeNCCL()
    return _native_nccl
