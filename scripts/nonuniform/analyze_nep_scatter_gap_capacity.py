#!/usr/bin/env python3
"""Measure communication gaps available to deferred NEP Scatter collectives."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


RANK_RE = re.compile(r"rank-(?P<rank>\d+)\.json\.gz$")
EP_GROUP = "ep"
TRANSFER_GROUP = "nep_owner_transfer"


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def clip_intervals(
    intervals: list[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    """Clip intervals to one analysis window."""
    return [
        (max(start, interval_start), min(end, interval_end))
        for interval_start, interval_end in intervals
        if interval_end > start and interval_start < end
    ]


def complement_intervals(
    intervals: list[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    """Return gaps in the union of intervals within one window."""
    cursor = start
    gaps: list[tuple[float, float]] = []
    for interval_start, interval_end in merge_intervals(
        clip_intervals(intervals, start, end)
    ):
        if interval_start > cursor:
            gaps.append((cursor, interval_start))
        cursor = max(cursor, interval_end)
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def total_duration(intervals: list[tuple[float, float]]) -> float:
    """Return interval-union duration."""
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    """Return intersection duration between two interval unions."""
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


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float | int]:
    """Return compact summary statistics."""
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def parse_trace(path: Path) -> list[dict[str, Any]]:
    """Extract Scatter-to-next-Gather windows from one PyTorch trace."""
    rank_match = RANK_RE.search(path.name)
    if rank_match is None:
        raise ValueError(f"cannot parse rank from {path}")
    rank = int(rank_match.group("rank"))

    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]

    steps = sorted(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and str(event.get("name", "")).startswith("ProfilerStep#")
        ),
        key=lambda event: float(event["ts"]),
    )
    records = {
        event.get("args", {}).get("External id"): event
        for event in events
        if event.get("name") == "record_param_comms"
        and event.get("args", {}).get("Collective name") != "wait"
    }
    kernels: list[dict[str, Any]] = []
    for event in events:
        if event.get("cat") != "kernel" or "nccl" not in str(
            event.get("name", "")
        ).lower():
            continue
        record = records.get(event.get("args", {}).get("External id"))
        record_args = record.get("args", {}) if record is not None else {}
        kernels.append(
            {
                "start": float(event["ts"]),
                "end": float(event["ts"]) + float(event.get("dur", 0.0)),
                "group": record_args.get("Process Group Description", "unmapped"),
                "collective": record_args.get("Collective name", "unmapped"),
                "in_nelems": int(record_args.get("In msg nelems", 0)),
                "out_nelems": int(record_args.get("Out msg nelems", 0)),
                "group_ranks": tuple(
                    int(item)
                    for item in ast.literal_eval(
                        str(record_args.get("Process Group Ranks", "[]"))
                    )
                ),
            }
        )

    rows: list[dict[str, Any]] = []
    for step in steps:
        step_start = float(step["ts"])
        step_end = step_start + float(step["dur"])
        step_kernels = [
            kernel
            for kernel in kernels
            if kernel["start"] >= step_start and kernel["start"] < step_end
        ]
        all_nccl = [(kernel["start"], kernel["end"]) for kernel in step_kernels]
        ep_nccl = [
            (kernel["start"], kernel["end"])
            for kernel in step_kernels
            if kernel["group"] == EP_GROUP
        ]
        transfers_by_pair: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
        for kernel in step_kernels:
            if kernel["group"] == TRANSFER_GROUP:
                transfers_by_pair[kernel["group_ranks"]].append(kernel)

        for pair, transfers in sorted(transfers_by_pair.items()):
            transfers.sort(key=lambda kernel: kernel["start"])
            if not pair or rank not in pair:
                raise ValueError(
                    f"{path} {step['name']}: invalid transfer ranks {pair} for rank {rank}"
                )
            role = "owner" if rank == pair[0] else "follower"
            gathers: list[dict[str, Any]] = []
            scatters: list[dict[str, Any]] = []
            for transfer in transfers:
                input_numel = transfer["in_nelems"]
                output_numel = transfer["out_nelems"]
                is_gather = (
                    output_numel > input_numel
                    if role == "owner"
                    else input_numel > output_numel
                )
                is_scatter = (
                    input_numel > output_numel
                    if role == "owner"
                    else output_numel > input_numel
                )
                if is_gather:
                    gathers.append(transfer)
                elif is_scatter:
                    scatters.append(transfer)
                else:
                    raise ValueError(
                        f"{path} {step['name']}: cannot classify owner-transfer "
                        f"kernel with in={input_numel}, out={output_numel}, role={role}"
                    )

            for phase, (gather, next_gather) in enumerate(
                zip(gathers[:-1], gathers[1:]), start=1
            ):
                phase_scatters = [
                    scatter
                    for scatter in scatters
                    if scatter["start"] >= gather["end"]
                    and scatter["start"] < next_gather["start"]
                ]
                if not phase_scatters:
                    raise ValueError(
                        f"{path} {step['name']}: no Scatter kernels in phase {phase}"
                    )
                window_start = phase_scatters[0]["start"]
                window_end = next_gather["start"]
                if window_end <= window_start:
                    raise ValueError(
                        f"{path} {step['name']}: invalid phase-{phase} window"
                    )

                scatter_interval = [
                    (scatter["start"], scatter["end"]) for scatter in phase_scatters
                ]
                clipped_ep = clip_intervals(ep_nccl, window_start, window_end)
                clipped_all = clip_intervals(all_nccl, window_start, window_end)
                no_ep_gaps = complement_intervals(ep_nccl, window_start, window_end)
                no_nccl_gaps = complement_intervals(all_nccl, window_start, window_end)
                ep_duration = total_duration(clipped_ep)
                all_nccl_duration = total_duration(clipped_all)
                scatter_duration = total_duration(scatter_interval)
                scatter_ep_overlap = intersection_duration(scatter_interval, clipped_ep)
                following_ep = sorted(
                    interval for interval in clipped_ep if interval[1] > window_start
                )

                rows.append(
                    {
                        "rank": rank,
                        "pair": list(pair),
                        "role": role,
                        "step": str(step["name"]),
                        "phase": phase,
                        "window_ms": (window_end - window_start) / 1000.0,
                        "ep_residency_ms": ep_duration / 1000.0,
                        "no_ep_residency_ms": (window_end - window_start - ep_duration)
                        / 1000.0,
                        "no_ep_residency_percent": 100.0
                        * (window_end - window_start - ep_duration)
                        / (window_end - window_start),
                        "any_nccl_residency_ms": all_nccl_duration / 1000.0,
                        "no_nccl_residency_ms": (
                            window_end - window_start - all_nccl_duration
                        )
                        / 1000.0,
                        "no_nccl_residency_percent": 100.0
                        * (window_end - window_start - all_nccl_duration)
                        / (window_end - window_start),
                        "largest_no_ep_gap_ms": max(
                            (end - start for start, end in no_ep_gaps), default=0.0
                        )
                        / 1000.0,
                        "largest_no_nccl_gap_ms": max(
                            (end - start for start, end in no_nccl_gaps), default=0.0
                        )
                        / 1000.0,
                        "no_ep_gaps_ge_0_25_ms": sum(
                            end - start >= 250.0 for start, end in no_ep_gaps
                        ),
                        "no_ep_gaps_ge_0_50_ms": sum(
                            end - start >= 500.0 for start, end in no_ep_gaps
                        ),
                        "no_ep_gaps_ge_1_00_ms": sum(
                            end - start >= 1000.0 for start, end in no_ep_gaps
                        ),
                        "no_nccl_gaps_ge_0_25_ms": sum(
                            end - start >= 250.0 for start, end in no_nccl_gaps
                        ),
                        "no_nccl_gaps_ge_0_50_ms": sum(
                            end - start >= 500.0 for start, end in no_nccl_gaps
                        ),
                        "no_nccl_gaps_ge_1_00_ms": sum(
                            end - start >= 1000.0 for start, end in no_nccl_gaps
                        ),
                        "scatter_ms": scatter_duration / 1000.0,
                        "scatter_kernel_count": len(phase_scatters),
                        "scatter_ep_overlap_ms": scatter_ep_overlap / 1000.0,
                        "scatter_ep_overlap_percent": 100.0
                        * scatter_ep_overlap
                        / scatter_duration,
                        "first_ep_start_after_scatter_ms": (
                            (following_ep[0][0] - window_start) / 1000.0
                            if following_ep
                            else None
                        ),
                        "_window_start": window_start,
                        "_window_end": window_end,
                        "_ep_intervals": clipped_ep,
                        "_all_nccl_intervals": clipped_all,
                    }
                )
    return rows


def paired_windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure gaps that are simultaneously available on a transfer pair."""
    by_window: dict[tuple[tuple[int, ...], str, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (tuple(row["pair"]), str(row["step"]), int(row["phase"]))
        by_window[key].append(row)

    paired: list[dict[str, Any]] = []
    for (pair, step, phase), pair_rows in sorted(by_window.items()):
        if len(pair_rows) != len(pair):
            raise ValueError(
                f"pair {pair} {step} phase {phase}: expected {len(pair)} traces, "
                f"found {len(pair_rows)}"
            )
        window_start = max(float(row["_window_start"]) for row in pair_rows)
        window_end = min(float(row["_window_end"]) for row in pair_rows)
        if window_end <= window_start:
            raise ValueError(f"pair {pair} {step} phase {phase}: no shared window")
        ep_intervals = [
            interval for row in pair_rows for interval in row["_ep_intervals"]
        ]
        all_nccl_intervals = [
            interval for row in pair_rows for interval in row["_all_nccl_intervals"]
        ]
        clipped_ep = clip_intervals(ep_intervals, window_start, window_end)
        clipped_all = clip_intervals(all_nccl_intervals, window_start, window_end)
        no_ep_gaps = complement_intervals(ep_intervals, window_start, window_end)
        no_nccl_gaps = complement_intervals(
            all_nccl_intervals, window_start, window_end
        )
        window_duration = window_end - window_start
        ep_duration = total_duration(clipped_ep)
        all_nccl_duration = total_duration(clipped_all)
        paired.append(
            {
                "pair": list(pair),
                "step": step,
                "phase": phase,
                "window_ms": window_duration / 1000.0,
                "ep_residency_ms": ep_duration / 1000.0,
                "no_ep_residency_ms": (window_duration - ep_duration) / 1000.0,
                "no_ep_residency_percent": 100.0
                * (window_duration - ep_duration)
                / window_duration,
                "any_nccl_residency_ms": all_nccl_duration / 1000.0,
                "no_nccl_residency_ms": (window_duration - all_nccl_duration)
                / 1000.0,
                "no_nccl_residency_percent": 100.0
                * (window_duration - all_nccl_duration)
                / window_duration,
                "largest_no_ep_gap_ms": max(
                    (end - start for start, end in no_ep_gaps), default=0.0
                )
                / 1000.0,
                "largest_no_nccl_gap_ms": max(
                    (end - start for start, end in no_nccl_gaps), default=0.0
                )
                / 1000.0,
                "no_ep_gaps_ge_0_25_ms": sum(
                    end - start >= 250.0 for start, end in no_ep_gaps
                ),
                "no_ep_gaps_ge_0_50_ms": sum(
                    end - start >= 500.0 for start, end in no_ep_gaps
                ),
                "no_ep_gaps_ge_1_00_ms": sum(
                    end - start >= 1000.0 for start, end in no_ep_gaps
                ),
                "no_nccl_gaps_ge_0_25_ms": sum(
                    end - start >= 250.0 for start, end in no_nccl_gaps
                ),
                "no_nccl_gaps_ge_0_50_ms": sum(
                    end - start >= 500.0 for start, end in no_nccl_gaps
                ),
                "no_nccl_gaps_ge_1_00_ms": sum(
                    end - start >= 1000.0 for start, end in no_nccl_gaps
                ),
                "scatter_ms": max(float(row["scatter_ms"]) for row in pair_rows),
                "scatter_kernel_count": max(
                    int(row["scatter_kernel_count"]) for row in pair_rows
                ),
                "scatter_ep_overlap_ms": max(
                    float(row["scatter_ep_overlap_ms"]) for row in pair_rows
                ),
                "scatter_ep_overlap_percent": max(
                    float(row["scatter_ep_overlap_percent"]) for row in pair_rows
                ),
                "first_ep_start_after_scatter_ms": min(
                    float(row["first_ep_start_after_scatter_ms"])
                    for row in pair_rows
                    if row["first_ep_start_after_scatter_ms"] is not None
                ),
            }
        )
    return paired


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-rank Scatter windows."""
    metrics = [
        "window_ms",
        "ep_residency_ms",
        "no_ep_residency_ms",
        "no_ep_residency_percent",
        "any_nccl_residency_ms",
        "no_nccl_residency_ms",
        "no_nccl_residency_percent",
        "largest_no_ep_gap_ms",
        "largest_no_nccl_gap_ms",
        "scatter_ms",
        "scatter_kernel_count",
        "scatter_ep_overlap_ms",
        "scatter_ep_overlap_percent",
        "first_ep_start_after_scatter_ms",
    ]
    result: dict[str, Any] = {"windows": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        result[metric] = summarize(values)
    for threshold_metric in (
        "no_ep_gaps_ge_0_25_ms",
        "no_ep_gaps_ge_0_50_ms",
        "no_ep_gaps_ge_1_00_ms",
        "no_nccl_gaps_ge_0_25_ms",
        "no_nccl_gaps_ge_0_50_ms",
        "no_nccl_gaps_ge_1_00_ms",
    ):
        result[threshold_metric] = summarize(
            [float(row[threshold_metric]) for row in rows]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(args.trace_dir.glob("rank-*.json.gz"))
    if not paths:
        raise SystemExit(f"no rank traces found in {args.trace_dir}")
    rows = [row for path in paths for row in parse_trace(path)]
    if not rows:
        raise SystemExit("no Scatter-to-next-Gather windows found")

    paired = paired_windows(rows)
    grouped: dict[str, Any] = {
        "all": aggregate(rows),
        "owner": aggregate([row for row in rows if row["role"] == "owner"]),
        "follower": aggregate([row for row in rows if row["role"] == "follower"]),
        "paired": aggregate(paired),
    }
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    result = {
        "trace_dir": str(args.trace_dir),
        "ranks": sorted({int(row["rank"]) for row in rows}),
        "rows": public_rows,
        "paired_rows": paired,
        "summary": grouped,
        "units": "durations are milliseconds; percentages are wall-clock residency fractions",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
