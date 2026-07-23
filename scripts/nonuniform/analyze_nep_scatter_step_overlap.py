#!/usr/bin/env python3
"""Measure NEP Scatter overlap over complete PyTorch profiler steps."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    left = merge_intervals(left)
    right = merge_intervals(right)
    left_index = 0
    right_index = 0
    total = 0.0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def parse_trace(path: Path) -> list[dict[str, float | int | str]]:
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]

    records = {
        event.get("args", {}).get("External id"): event.get("args", {})
        for event in events
        if event.get("name") == "record_param_comms"
    }
    kernels: list[tuple[float, float, str]] = []
    annotations: list[tuple[float, str]] = []
    for event in events:
        if event.get("ph") != "X" or not event.get("dur"):
            continue
        start = float(event["ts"])
        end = start + float(event["dur"])
        if event.get("cat") == "kernel":
            name = str(event.get("name", ""))
            if "nccl" in name.lower():
                group = records.get(
                    event.get("args", {}).get("External id"), {}
                ).get("Process Group Description", "unmapped")
                kernels.append((start, end, str(group)))
            else:
                kernels.append((start, end, "compute"))
        elif event.get("cat") == "user_annotation":
            annotations.append((start, str(event.get("name", ""))))

    steps = sorted(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and str(event.get("name", "")).startswith("ProfilerStep#")
        ),
        key=lambda event: float(event["ts"]),
    )
    rows = []
    for step in steps:
        step_start = float(step["ts"])
        step_end = step_start + float(step["dur"])
        by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
        counts: dict[str, int] = defaultdict(int)
        for kernel_start, kernel_end, group in kernels:
            if kernel_start >= step_end or kernel_end <= step_start:
                continue
            by_group[group].append(
                (max(step_start, kernel_start), min(step_end, kernel_end))
            )
            if step_start <= kernel_start < step_end:
                counts[group] += 1

        scatter = by_group["nep_owner_transfer"]
        gather = by_group["nep_owner_gather"]
        model_ep = by_group["ep"]
        compute = by_group["compute"]
        scatter_us = total_duration(scatter)
        scatter_compute_us = intersection_duration(scatter, compute)
        scatter_model_ep_us = intersection_duration(scatter, model_ep)
        rows.append(
            {
                "step": str(step["name"]),
                "step_ms": (step_end - step_start) / 1000.0,
                "scatter_kernels": counts["nep_owner_transfer"],
                "scatter_ms": scatter_us / 1000.0,
                "gather_ms": total_duration(gather) / 1000.0,
                "model_ep_ms": total_duration(model_ep) / 1000.0,
                "scatter_compute_overlap_ms": scatter_compute_us / 1000.0,
                "scatter_compute_overlap_percent": (
                    100.0 * scatter_compute_us / scatter_us if scatter_us else 0.0
                ),
                "scatter_model_ep_overlap_ms": scatter_model_ep_us / 1000.0,
                "scatter_model_ep_overlap_percent": (
                    100.0 * scatter_model_ep_us / scatter_us if scatter_us else 0.0
                ),
                "scatter_exposed_ms": (scatter_us - scatter_compute_us) / 1000.0,
                "scheduled_scatter_annotations": sum(
                    step_start <= start < step_end
                    and name.startswith("nep_scheduled_scatter_chunk_")
                    for start, name in annotations
                ),
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Trace has no profiler steps")
    return {
        key: statistics.mean(float(row[key]) for row in rows)
        for key in rows[0]
        if key != "step"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path, nargs="+")
    parser.add_argument("--rank", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ranks = args.rank or [0]
    results: dict[str, Any] = {}
    for trace_dir in args.trace_dir:
        rank_results = {}
        for rank in ranks:
            trace_path = trace_dir / f"rank-{rank}.json.gz"
            if not trace_path.is_file():
                raise ValueError(f"Missing trace: {trace_path}")
            rows = parse_trace(trace_path)
            rank_results[str(rank)] = {"rows": rows, "mean": summarize(rows)}
        results[str(trace_dir)] = rank_results

    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
