#!/usr/bin/env python3
"""Verify CUDA stream wait/write value support on the active device."""

import threading

import torch

from megatron.core.distributed._cuda_stream_ops import get_cuda_stream_memory_ops


def main() -> None:
    print("initializing CUDA", flush=True)
    torch.cuda.init()
    stream_ops = get_cuda_stream_memory_ops()

    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    output = torch.zeros(1, dtype=torch.int32, device="cuda")
    wait_stream = torch.cuda.Stream()
    write_stream = torch.cuda.Stream()
    output.add_(0)
    torch.cuda.synchronize()
    print("warmup complete", flush=True)

    print(f"enqueue wait stream={wait_stream.cuda_stream} flag={flag.data_ptr()}", flush=True)
    stream_ops.wait_value32(wait_stream.cuda_stream, flag.data_ptr(), 1)
    print("wait enqueued", flush=True)
    with torch.cuda.stream(wait_stream):
        output.add_(1)
    print("output kernel enqueued", flush=True)

    errors = []

    def signal() -> None:
        try:
            print("signal thread setting device", flush=True)
            torch.cuda.set_device(flag.device)
            print(f"enqueue write stream={write_stream.cuda_stream}", flush=True)
            stream_ops.write_value32(write_stream.cuda_stream, flag.data_ptr(), 1)
            print("write enqueued", flush=True)
        except Exception as exc:  # pragma: no cover - cluster probe diagnostics
            errors.append(exc)

    thread = threading.Thread(target=signal)
    thread.start()
    thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("signal thread did not finish within 10 seconds")
    if errors:
        raise errors[0]

    print("synchronizing device", flush=True)
    torch.cuda.synchronize()
    print("device synchronized", flush=True)
    assert flag.item() == 1
    assert output.item() == 1
    print(
        "cuda stream wait/write value probe: ok "
        f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name()}"
    )


if __name__ == "__main__":
    main()
