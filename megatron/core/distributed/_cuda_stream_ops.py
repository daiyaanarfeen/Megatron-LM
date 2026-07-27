# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Minimal CUDA driver bindings for stream-ordered memory signaling."""

import ctypes
import threading
from typing import Optional


class CUDAStreamMemoryOps:
    """Enqueue 32-bit wait/write operations without consuming GPU SMs."""

    def __init__(self) -> None:
        self._driver = ctypes.CDLL("libcuda.so.1")
        self._driver.cuStreamWaitValue32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint,
        ]
        self._driver.cuStreamWaitValue32.restype = ctypes.c_int
        self._driver.cuStreamWriteValue32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint,
        ]
        self._driver.cuStreamWriteValue32.restype = ctypes.c_int
        self._driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._driver.cuGetErrorString.restype = ctypes.c_int

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        message = ctypes.c_char_p()
        self._driver.cuGetErrorString(result, ctypes.byref(message))
        detail = message.value.decode() if message.value is not None else "unknown error"
        raise RuntimeError(f"{operation} failed with CUDA error {result}: {detail}")

    def wait_value32(self, stream_ptr: int, address: int, value: int) -> None:
        """Wait until the unsigned value at the address is at least value."""
        self._check(
            self._driver.cuStreamWaitValue32(
                ctypes.c_void_p(stream_ptr),
                ctypes.c_uint64(address),
                ctypes.c_uint32(value),
                ctypes.c_uint(0),
            ),
            "cuStreamWaitValue32",
        )

    def write_value32(self, stream_ptr: int, address: int, value: int) -> None:
        """Write value to the address from the given stream."""
        self._check(
            self._driver.cuStreamWriteValue32(
                ctypes.c_void_p(stream_ptr),
                ctypes.c_uint64(address),
                ctypes.c_uint32(value),
                ctypes.c_uint(0),
            ),
            "cuStreamWriteValue32",
        )


_cuda_stream_memory_ops: Optional[CUDAStreamMemoryOps] = None
_cuda_stream_memory_ops_lock = threading.Lock()


def get_cuda_stream_memory_ops() -> CUDAStreamMemoryOps:
    """Return the process-wide lazy CUDA stream-memory binding."""
    global _cuda_stream_memory_ops
    if _cuda_stream_memory_ops is not None:
        return _cuda_stream_memory_ops
    with _cuda_stream_memory_ops_lock:
        if _cuda_stream_memory_ops is None:
            _cuda_stream_memory_ops = CUDAStreamMemoryOps()
    return _cuda_stream_memory_ops
