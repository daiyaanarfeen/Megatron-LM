#!/usr/bin/env python3
"""Parse the minimal dlcluster NEP overlap benchmark outputs."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path


ITER_RE = re.compile(
    r"iteration\s+(?P<iter>\d+)/\s*(?P<total>\d+).*?"
    r"elapsed time per iteration \(ms\): (?P<ms>[0-9.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\): (?P<tflops>[0-9.]+).*?"
    r"global batch size:\s+(?P<gbs>\d+)"
)
DEBUG_RE = re.compile(
    r"NEP_OVERLAP_DEBUG rank=(?P<rank>\d+) group=(?P<group>\d+) .*?"
    r"comm_ms=(?P<comm>[0-9.]+) .*?"
    r"ready_since_comm_start_ms=(?P<ready>[0-9.]+) .*?"
    r"finish_wait_ms=(?P<wait>[0-9.]+)"
)


def parse_iters(path: Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = ITER_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "iter": int(match.group("iter")),
                "ms": float(match.group("ms")),
                "tflops": float(match.group("tflops")),
                "gbs": int(match.group("gbs")),
            }
        )
    return rows


def summarize_iters(rows: list[dict[str, float]], active_gpus: int) -> dict[str, float]:
    selected = [row for row in rows if row["iter"] >= 13]
    avg_ms = sum(row["ms"] for row in selected) / len(selected)
    avg_tflops = sum(row["tflops"] for row in selected) / len(selected)
    gbs = selected[-1]["gbs"]
    return {
        "iters": len(selected),
        "avg_ms": avg_ms,
        "avg_tflops": avg_tflops,
        "samples_per_gpu_s": gbs / (avg_ms / 1000.0) / active_gpus,
    }


def summarize_debug(path: Path) -> dict[str, float]:
    comm = []
    ready = []
    wait = []
    for line in path.read_text(errors="replace").splitlines():
        match = DEBUG_RE.search(line)
        if not match:
            continue
        comm.append(float(match.group("comm")))
        ready.append(float(match.group("ready")))
        wait.append(float(match.group("wait")))
    if not comm:
        return {}
    return {
        "count": len(comm),
        "avg_comm_ms": sum(comm) / len(comm),
        "max_comm_ms": max(comm),
        "avg_ready_since_comm_start_ms": sum(ready) / len(ready),
        "avg_finish_wait_ms": sum(wait) / len(wait),
        "max_finish_wait_ms": max(wait),
        "debug_overlap_fraction": 1.0 - (sum(wait) / sum(comm)),
    }


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _intersect_len(
    intervals: list[tuple[float, float]], others: list[tuple[float, float]]
) -> float:
    total = 0.0
    j = 0
    for start, end in intervals:
        while j < len(others) and others[j][1] <= start:
            j += 1
        k = j
        while k < len(others) and others[k][0] < end:
            total += max(0.0, min(end, others[k][1]) - max(start, others[k][0]))
            k += 1
    return total


def summarize_trace(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]

    categories = Counter(event.get("cat", "") for event in events)
    gpu_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("dur")
        and event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]
    nccl_events = [
        event for event in gpu_events if "nccl" in event.get("name", "").lower()
    ]
    compute_events = [
        event
        for event in gpu_events
        if "nccl" not in event.get("name", "").lower()
        and event.get("cat") == "kernel"
    ]
    nccl_intervals = _merge_intervals(
        [(float(event["ts"]), float(event["ts"]) + float(event["dur"])) for event in nccl_events]
    )
    compute_intervals = _merge_intervals(
        [
            (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
            for event in compute_events
        ]
    )
    nccl_total_us = sum(end - start for start, end in nccl_intervals)
    overlap_us = _intersect_len(nccl_intervals, compute_intervals)
    nccl_names = Counter(event.get("name", "") for event in nccl_events)
    compute_names = Counter(event.get("name", "") for event in compute_events)
    cpu_collective_names = Counter(
        event.get("name", "")
        for event in events
        if event.get("ph") == "X"
        and event.get("dur")
        and any(
            token in event.get("name", "").lower()
            for token in ("all_to_all", "alltoall", "all_reduce", "processgroup", "nep")
        )
    )
    cpu_collective_groups = Counter()
    cpu_collective_examples = []
    for event in events:
        name = event.get("name", "")
        if event.get("ph") != "X" or not event.get("dur"):
            continue
        if not any(
            token in name.lower()
            for token in ("all_to_all", "alltoall", "all_reduce", "processgroup", "nep")
        ):
            continue
        args = event.get("args", {})
        group_key = (
            name,
            args.get("Process Group Name"),
            args.get("Process Group Description"),
            args.get("Collective name"),
        )
        cpu_collective_groups[group_key] += 1
        if len(cpu_collective_examples) < 12:
            cpu_collective_examples.append(
                {
                    "name": name,
                    "dur": event.get("dur"),
                    "args": args,
                }
            )
    return {
        "path": str(path),
        "categories": dict(categories.most_common(12)),
        "gpu_kernel_events": len(gpu_events),
        "nccl_events": len(nccl_events),
        "compute_events": len(compute_events),
        "nccl_total_ms": nccl_total_us / 1000.0,
        "nccl_overlap_with_compute_ms": overlap_us / 1000.0,
        "nccl_overlap_fraction": overlap_us / nccl_total_us if nccl_total_us else 0.0,
        "top_nccl_names": nccl_names.most_common(8),
        "top_compute_names": compute_names.most_common(8),
        "top_cpu_collective_names": cpu_collective_names.most_common(20),
        "top_cpu_collective_groups": cpu_collective_groups.most_common(20),
        "cpu_collective_examples": cpu_collective_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthy-log", type=Path, required=True)
    parser.add_argument("--nep-log", type=Path, required=True)
    parser.add_argument("--healthy-active-gpus", type=int, default=8)
    parser.add_argument("--nep-active-gpus", type=int, default=6)
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--brief", action="store_true")
    args = parser.parse_args()

    for label, path, gpus in (
        ("healthy", args.healthy_log, args.healthy_active_gpus),
        ("nep", args.nep_log, args.nep_active_gpus),
    ):
        rows = parse_iters(path)
        print(label, summarize_iters(rows, gpus))
        print(label + "_debug", summarize_debug(path))

    for trace in args.trace:
        summary = summarize_trace(trace)
        if args.brief:
            print(
                "trace_brief",
                {
                    "path": summary["path"],
                    "nccl_events": summary["nccl_events"],
                    "nccl_total_ms": summary["nccl_total_ms"],
                    "nccl_overlap_with_compute_ms": summary[
                        "nccl_overlap_with_compute_ms"
                    ],
                    "nccl_overlap_fraction": summary["nccl_overlap_fraction"],
                    "top_cpu_collective_names": summary["top_cpu_collective_names"][:8],
                    "top_nccl_names": summary["top_nccl_names"][:4],
                },
            )
        else:
            print("trace", summary)


if __name__ == "__main__":
    main()
