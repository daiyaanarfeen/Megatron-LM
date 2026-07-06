# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Probe NCCL CTA policies for NEP-compatible transfer collectives."""

import argparse
import gzip
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import nccl_allocator


def _new_group(cta_policy: int | None, description: str, max_ctas: int | None = None):
    options = dist.ProcessGroupNCCL.Options()
    if cta_policy is not None:
        if not hasattr(options.config, "cta_policy"):
            raise RuntimeError("This PyTorch build does not expose NCCL cta_policy")
        options.config.cta_policy = cta_policy
    if max_ctas is not None:
        options.config.max_ctas = max_ctas
        options.config.min_ctas = max_ctas
    return dist.new_group(
        ranks=list(range(dist.get_world_size())),
        backend="nccl",
        pg_options=options,
        group_desc=description,
    )


def _warm_up(group) -> None:
    value = torch.ones(1, device="cuda")
    dist.all_reduce(value, group=group)
    torch.cuda.synchronize()


def _fixed_all_to_all(group, send: torch.Tensor, recv: torch.Tensor) -> None:
    dist.all_to_all_single(recv, send, group=group)


def _variable_all_to_all(group, send: torch.Tensor, recv: torch.Tensor) -> None:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    peer = (rank + 1) % world_size
    source = (rank - 1) % world_size
    input_splits = [0] * world_size
    output_splits = [0] * world_size
    input_splits[peer] = send.numel()
    output_splits[source] = recv.numel()
    dist.all_to_all_single(
        recv, send, output_split_sizes=output_splits, input_split_sizes=input_splits, group=group
    )


def _gather(group, send: torch.Tensor, recv: torch.Tensor) -> None:
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    gather_list = list(recv.chunk(world_size)) if rank == 0 else None
    dist.gather(send, gather_list=gather_list, dst=0, group=group)


def _all_gather(group, send: torch.Tensor, recv: torch.Tensor) -> None:
    dist.all_gather_into_tensor(recv, send, group=group)


def _scatter(group, send: torch.Tensor, recv: torch.Tensor) -> None:
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    scatter_list = list(send.chunk(world_size)) if rank == 0 else None
    dist.scatter(recv, scatter_list=scatter_list, src=0, group=group)


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
        path = trace_dir / f"{label}.json.gz"
        raw_path = trace_dir / f"{label}.json"
        profile.export_chrome_trace(str(raw_path))
        with (
            raw_path.open("rt", encoding="utf-8") as source,
            gzip.open(path, "wt", encoding="utf-8") as target,
        ):
            json.dump(json.load(source), target)
        raw_path.unlink()
        print(f"[zero-cta-probe] trace={path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--elements-per-peer", type=int, default=1 << 20)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("zero-CTA probe requires at least two ranks")

    default_group = _new_group(None, "probe_default")
    efficiency_group = _new_group(1, "probe_efficiency")
    limited_group = _new_group(None, "probe_one_cta", max_ctas=1)
    zero_group = _new_group(2, "probe_zero_cta")
    _warm_up(default_group)
    _warm_up(efficiency_group)
    _warm_up(limited_group)
    _warm_up(zero_group)

    # Compile the inline allocator extension once, then let the other ranks load it.
    # Concurrent first-time builds target the same build directory and can race.
    if rank == 0:
        nccl_allocator.init()
    dist.barrier()
    if rank != 0:
        nccl_allocator.init()
    pool = nccl_allocator.create_nccl_mem_pool(symmetric=True)
    with nccl_allocator.MultiGroupMemPoolAllocator(
        pool, [default_group, efficiency_group, limited_group, zero_group], symmetric=True
    ):
        fixed_send = torch.empty(
            world_size * args.elements_per_peer, dtype=torch.float32, device="cuda"
        )
        fixed_recv = torch.empty_like(fixed_send)
        variable_send = torch.empty(args.elements_per_peer, dtype=torch.float32, device="cuda")
        variable_recv = torch.empty_like(variable_send)
        gather_send = torch.empty(args.elements_per_peer, dtype=torch.float32, device="cuda")
        gather_recv = torch.empty(
            world_size * args.elements_per_peer, dtype=torch.float32, device="cuda"
        )
        scatter_send = torch.empty_like(gather_recv)
        scatter_recv = torch.empty_like(gather_send)

    for peer in range(world_size):
        fixed_send[peer * args.elements_per_peer : (peer + 1) * args.elements_per_peer].fill_(
            rank * world_size + peer
        )
    variable_send.fill_(rank)
    gather_send.fill_(rank)
    for peer in range(world_size):
        scatter_send[peer * args.elements_per_peer : (peer + 1) * args.elements_per_peer].fill_(
            100 + peer
        )

    cases = {
        "default_fixed": lambda: _fixed_all_to_all(default_group, fixed_send, fixed_recv),
        "efficiency_fixed": lambda: _fixed_all_to_all(efficiency_group, fixed_send, fixed_recv),
        "limited_fixed": lambda: _fixed_all_to_all(limited_group, fixed_send, fixed_recv),
        "default_variable": lambda: _variable_all_to_all(
            default_group, variable_send, variable_recv
        ),
        "efficiency_variable": lambda: _variable_all_to_all(
            efficiency_group, variable_send, variable_recv
        ),
        "limited_variable": lambda: _variable_all_to_all(
            limited_group, variable_send, variable_recv
        ),
        "zero_gather": lambda: _gather(zero_group, gather_send, gather_recv),
        "zero_all_gather": lambda: _all_gather(zero_group, gather_send, gather_recv),
        "zero_scatter": lambda: _scatter(zero_group, scatter_send, scatter_recv),
        "zero_fixed": lambda: _fixed_all_to_all(zero_group, fixed_send, fixed_recv),
        "zero_variable": lambda: _variable_all_to_all(zero_group, variable_send, variable_recv),
    }
    selected_cases = args.cases or [
        "default_fixed",
        "efficiency_fixed",
        "limited_fixed",
        "default_variable",
        "efficiency_variable",
        "limited_variable",
        "zero_all_gather",
    ]
    unknown_cases = set(selected_cases) - cases.keys()
    if unknown_cases:
        raise RuntimeError(f"Unknown probe cases: {sorted(unknown_cases)}")

    for label in selected_cases:
        _run_profiled(label, cases[label], args.trace_dir)
        dist.barrier()

    expected_fixed = torch.cat(
        [
            torch.full(
                (args.elements_per_peer,),
                source * world_size + rank,
                dtype=torch.float32,
                device="cuda",
            )
            for source in range(world_size)
        ]
    )
    expected_variable = torch.full_like(variable_recv, (rank - 1) % world_size)
    if any(label.endswith("fixed") for label in selected_cases):
        torch.testing.assert_close(fixed_recv, expected_fixed)
    if any(label.endswith("variable") for label in selected_cases):
        torch.testing.assert_close(variable_recv, expected_variable)
    if rank == 0 and any(label in {"zero_gather", "zero_all_gather"} for label in selected_cases):
        expected_gather = torch.cat(
            [torch.full_like(gather_send, source) for source in range(world_size)]
        )
        torch.testing.assert_close(gather_recv, expected_gather)
    if "zero_scatter" in selected_cases:
        torch.testing.assert_close(scatter_recv, torch.full_like(scatter_recv, 100 + rank))
    if rank == 0:
        print(
            f"[zero-cta-probe] PASS world_size={world_size} " f"nccl={torch.cuda.nccl.version()}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
