#!/usr/bin/env python3
"""Characterize healthy EP8 iteration variance and its trace-level sources."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


TIMING_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/.*?elapsed time per iteration \(ms\):\s+(?P<elapsed>[0-9.]+)"
)
RANK_RE = re.compile(r"rank-(?P<rank>\d+)\.json\.gz$")
GROUPED_LINEAR_NAME = "_GroupedLinearBackward"
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def lag_one_correlation(values: list[float]) -> float | None:
    """Return lag-one correlation when both slices have nonzero variance."""
    return pearson(values[:-1], values[1:]) if len(values) >= 3 else None


def linear_slope(values: list[float]) -> float:
    """Return least-squares slope per sample index."""
    if len(values) < 2:
        return 0.0
    indices = [float(index) for index in range(len(values))]
    index_mean = statistics.mean(indices)
    value_mean = statistics.mean(values)
    denominator = sum((index - index_mean) ** 2 for index in indices)
    return sum(
        (index - index_mean) * (value - value_mean)
        for index, value in zip(indices, values)
    ) / denominator


def pearson(left: list[float], right: list[float]) -> float | None:
    """Return Pearson correlation, or None for degenerate inputs."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def summarize_values(values: list[float]) -> dict[str, Any]:
    """Return conventional and robust summary statistics."""
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    robust_sigma = 1.4826 * mad
    outlier_indices = [
        index
        for index, value in enumerate(values)
        if robust_sigma and abs(value - median) > 3.5 * robust_sigma
    ]
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "median": median,
        "std": std,
        "cv_percent": 100.0 * std / mean if mean else 0.0,
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mad": mad,
        "robust_sigma": robust_sigma,
        "outlier_indices": outlier_indices,
        "lag_one_correlation": lag_one_correlation(values),
        "slope_per_iteration": linear_slope(values),
    }


def find_single(root: Path, pattern: str) -> Path:
    """Return the only path matching a recursive glob."""
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one match for {pattern!r}, found {len(matches)}: {matches}")
    return matches[0]


def read_timings(path: Path, start_iteration: int) -> tuple[list[int], list[float]]:
    """Read post-warmup iteration times from a Megatron driver log."""
    iterations: list[int] = []
    elapsed_ms: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        match = TIMING_RE.search(line)
        if not match:
            continue
        iteration = int(match.group("iteration"))
        if iteration >= start_iteration:
            iterations.append(iteration)
            elapsed_ms.append(float(match.group("elapsed")))
    if not elapsed_ms:
        raise ValueError(f"no iterations >= {start_iteration} found in {path}")
    return iterations, elapsed_ms


def event_end(event: dict[str, Any]) -> float:
    """Return a trace event end timestamp in microseconds."""
    return float(event["ts"]) + float(event.get("dur", 0.0))


def clipped_interval(event: dict[str, Any], start: float, end: float) -> tuple[float, float] | None:
    """Clip an event interval to a profiler step."""
    clipped_start = max(start, float(event.get("ts", 0.0)))
    clipped_end = min(end, event_end(event))
    return (clipped_start, clipped_end) if clipped_end > clipped_start else None


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping timestamp intervals."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def union_duration(intervals: list[tuple[float, float]]) -> float:
    """Return interval-union duration in microseconds."""
    return sum(end - start for start, end in merge_intervals(intervals))


def overlap_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    """Return overlap between two interval unions in microseconds."""
    left_merged = merge_intervals(left)
    right_merged = merge_intervals(right)
    left_index = 0
    right_index = 0
    overlap = 0.0
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_start, left_end = left_merged[left_index]
        right_start, right_end = right_merged[right_index]
        overlap += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def is_nccl(event: dict[str, Any]) -> bool:
    """Return whether a GPU event belongs to NCCL."""
    return "nccl" in str(event.get("name", "")).lower()


def nccl_kind(event: dict[str, Any]) -> str:
    """Return a stable collective and dtype label for an NCCL kernel."""
    name = str(event.get("name", ""))
    for collective in (
        "AllGather",
        "ReduceScatter",
        "AllReduce",
        "SendRecv",
        "Broadcast",
        "Reduce",
    ):
        if collective in name:
            dtype_match = re.search(rf"{collective}_Sum_([^_(]+)", name)
            dtype = dtype_match.group(1) if dtype_match else ""
            return f"{collective}_{dtype}" if dtype else collective
    return "Other"


def contained_in(event: dict[str, Any], container: dict[str, Any]) -> bool:
    """Return whether an event is fully contained on a container's CPU thread."""
    return (
        event.get("pid") == container.get("pid")
        and event.get("tid") == container.get("tid")
        and float(event.get("ts", 0.0)) >= float(container["ts"])
        and event_end(event) <= event_end(container)
    )


def grouped_launch_gap(
    index: dict[str, Any], scope: dict[str, Any]
) -> dict[str, Any]:
    """Find the largest host gap between expert launches in a GroupedLinear scope."""
    external_id = scope.get("args", {}).get("External id")
    driver_events = index["driver_by_external_id"].get(external_id, [])
    thread = (scope.get("pid"), scope.get("tid"))
    grouped_calls = [
        event
        for event in index["grouped_calls_by_thread"].get(thread, [])
        if contained_in(event, scope)
    ]
    containers = grouped_calls or [scope]
    gaps: list[dict[str, Any]] = []
    for call_index, container in enumerate(sorted(containers, key=lambda event: float(event["ts"]))):
        launches = sorted(
            (event for event in driver_events if contained_in(event, container)),
            key=lambda event: float(event["ts"]),
        )
        for launch_index in range(1, len(launches)):
            launch = launches[launch_index]
            previous = launches[launch_index - 1]
            kernel = index["gpu_by_correlation"].get(
                launch.get("args", {}).get("correlation")
            )
            gaps.append(
                {
                    "duration_us": float(launch["ts"]) - event_end(previous),
                    "call": call_index,
                    "launch": launch_index,
                    "kernel": kernel.get("name") if kernel else None,
                    "grid": kernel.get("args", {}).get("grid") if kernel else None,
                }
            )
    if not gaps:
        return {"duration_us": 0.0, "call": None, "launch": None, "kernel": None, "grid": None}
    return max(gaps, key=lambda gap: float(gap["duration_us"]))


def build_trace_index(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Index event classes used repeatedly by per-step analysis."""
    gpu_events: list[dict[str, Any]] = []
    grouped_scopes: list[dict[str, Any]] = []
    driver_by_external_id: dict[Any, list[dict[str, Any]]] = {}
    grouped_calls_by_thread: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    energy_python_scopes: dict[str, list[dict[str, Any]]] = {
        "lap": [],
        "get_energy": [],
        "nvml": [],
    }
    for event in events:
        category = event.get("cat")
        if category in GPU_CATEGORIES:
            gpu_events.append(event)
        elif category == "cpu_op" and event.get("name") == GROUPED_LINEAR_NAME:
            grouped_scopes.append(event)
        elif category == "cuda_driver" and "cuLaunchKernel" in str(event.get("name", "")):
            external_id = event.get("args", {}).get("External id")
            driver_by_external_id.setdefault(external_id, []).append(event)
        elif category == "python_function" and "te_general_grouped_gemm" in str(
            event.get("name", "")
        ):
            thread = (event.get("pid"), event.get("tid"))
            grouped_calls_by_thread.setdefault(thread, []).append(event)
        elif category == "python_function":
            name = str(event.get("name", ""))
            if "energy_monitor.py(70): lap" in name:
                energy_python_scopes["lap"].append(event)
            elif "energy_monitor.py(59): _get_energy" in name:
                energy_python_scopes["get_energy"].append(event)
            elif "nvmlDeviceGetTotalEnergyConsumption" in name:
                energy_python_scopes["nvml"].append(event)
    return {
        "gpu_events": gpu_events,
        "gpu_by_correlation": {
            event.get("args", {}).get("correlation"): event for event in gpu_events
        },
        "grouped_scopes": grouped_scopes,
        "driver_by_external_id": driver_by_external_id,
        "grouped_calls_by_thread": grouped_calls_by_thread,
        "energy_python_scopes": energy_python_scopes,
    }


def event_timing(event: dict[str, Any]) -> dict[str, float]:
    """Return absolute start/end/duration timestamps for a trace event."""
    return {
        "start_us": float(event["ts"]),
        "end_us": event_end(event),
        "duration_us": float(event.get("dur", 0.0)),
    }


def summarize_step(index: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    """Summarize one profiler step for one rank."""
    start = float(step["ts"])
    end = event_end(step)
    gpu_events = [
        event
        for event in index["gpu_events"]
        if event.get("cat") in GPU_CATEGORIES and clipped_interval(event, start, end)
    ]
    nccl_intervals = [
        interval
        for event in gpu_events
        if is_nccl(event) and (interval := clipped_interval(event, start, end))
    ]
    non_nccl_intervals = [
        interval
        for event in gpu_events
        if not is_nccl(event) and (interval := clipped_interval(event, start, end))
    ]
    all_intervals = nccl_intervals + non_nccl_intervals
    nccl_intervals_by_kind: dict[str, list[tuple[float, float]]] = {}
    nccl_counts_by_kind: Counter[str] = Counter()
    for event in gpu_events:
        if not is_nccl(event):
            continue
        interval = clipped_interval(event, start, end)
        if interval:
            kind = nccl_kind(event)
            nccl_intervals_by_kind.setdefault(kind, []).append(interval)
            nccl_counts_by_kind[kind] += 1
    grouped_scopes = [
        event
        for event in index["grouped_scopes"]
        if start <= float(event.get("ts", 0.0)) < end
    ]
    launch_gaps = [grouped_launch_gap(index, scope) for scope in grouped_scopes]
    largest_gap = max(
        launch_gaps,
        key=lambda gap: float(gap["duration_us"]),
        default={"duration_us": 0.0, "call": None, "launch": None, "kernel": None, "grid": None},
    )
    nccl_us = union_duration(nccl_intervals)
    non_nccl_us = union_duration(non_nccl_intervals)
    active_us = union_duration(all_intervals)
    duration_us = float(step.get("dur", 0.0))
    default_group_events = sorted(
        (
            event
            for event in gpu_events
            if is_nccl(event)
            and event.get("args", {}).get("Process Group Description") == "default_pg"
        ),
        key=lambda event: float(event["ts"]),
    )
    timer_barriers = [
        event for event in default_group_events if event.get("args", {}).get("dtype") == "Byte"
    ]
    energy_reductions = [
        event for event in default_group_events if event.get("args", {}).get("dtype") == "Long"
    ]
    energy_reduction = energy_reductions[-1] if energy_reductions else None
    checkpoint_reductions = [
        event
        for event in default_group_events
        if event.get("args", {}).get("dtype") == "Int"
        and energy_reduction
        and float(event["ts"]) > event_end(energy_reduction)
    ]
    synchronization: dict[str, dict[str, float]] = {}
    if len(timer_barriers) >= 2:
        synchronization["timer_stop"] = event_timing(timer_barriers[0])
        synchronization["timer_start"] = event_timing(timer_barriers[1])
    if energy_reduction:
        synchronization["energy"] = event_timing(energy_reduction)
    if checkpoint_reductions:
        synchronization["checkpoint"] = event_timing(checkpoint_reductions[-1])
    pre_timer_stop: dict[str, Any] = {}
    if timer_barriers:
        pre_stop_end = float(timer_barriers[0]["ts"])
        pre_stop_nccl = [
            interval
            for event in gpu_events
            if is_nccl(event)
            and (interval := clipped_interval(event, start, pre_stop_end))
        ]
        pre_stop_non_nccl = [
            interval
            for event in gpu_events
            if not is_nccl(event)
            and (interval := clipped_interval(event, start, pre_stop_end))
        ]
        pre_stop_by_kind: dict[str, list[tuple[float, float]]] = {}
        for event in gpu_events:
            if not is_nccl(event):
                continue
            interval = clipped_interval(event, start, pre_stop_end)
            if interval:
                pre_stop_by_kind.setdefault(nccl_kind(event), []).append(interval)
        pre_stop_duration = pre_stop_end - start
        pre_stop_active = union_duration(pre_stop_nccl + pre_stop_non_nccl)
        pre_timer_stop = {
            "duration_us": pre_stop_duration,
            "nccl_us": union_duration(pre_stop_nccl),
            "non_nccl_us": union_duration(pre_stop_non_nccl),
            "overlap_us": overlap_duration(pre_stop_nccl, pre_stop_non_nccl),
            "gpu_active_us": pre_stop_active,
            "gpu_idle_us": max(0.0, pre_stop_duration - pre_stop_active),
            "nccl_by_kind_us": {
                kind: union_duration(intervals)
                for kind, intervals in sorted(pre_stop_by_kind.items())
            },
            "sendrecv_events": [
                event_timing(event)
                for event in gpu_events
                if nccl_kind(event) == "SendRecv"
                and clipped_interval(event, start, pre_stop_end)
            ],
        }
    energy_python_us = {
        name: sum(
            float(event.get("dur", 0.0))
            for event in scopes
            if start <= float(event.get("ts", 0.0)) < end
        )
        for name, scopes in index["energy_python_scopes"].items()
    }
    return {
        "duration_us": duration_us,
        "nccl_us": nccl_us,
        "nccl_by_kind_us": {
            kind: union_duration(intervals)
            for kind, intervals in sorted(nccl_intervals_by_kind.items())
        },
        "nccl_count_by_kind": dict(sorted(nccl_counts_by_kind.items())),
        "non_nccl_us": non_nccl_us,
        "nccl_non_nccl_overlap_us": overlap_duration(nccl_intervals, non_nccl_intervals),
        "gpu_active_us": active_us,
        "gpu_idle_us": max(0.0, duration_us - active_us),
        "grouped_scope_count": len(grouped_scopes),
        "grouped_cpu_sum_us": sum(float(scope.get("dur", 0.0)) for scope in grouped_scopes),
        "grouped_cpu_max_us": max(
            (float(scope.get("dur", 0.0)) for scope in grouped_scopes), default=0.0
        ),
        "grouped_internal_launch_gap_max_us": float(largest_gap["duration_us"]),
        "grouped_internal_launch_gap": largest_gap,
        "synchronization": synchronization,
        "pre_timer_stop": pre_timer_stop,
        "energy_python_us": energy_python_us,
    }


def load_profile(trace_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Load all ranks and align metrics by profiler-step name."""
    aligned: dict[str, dict[int, dict[str, Any]]] = {}
    trace_paths = sorted(trace_dir.glob("rank-*.json.gz"))
    if not trace_paths:
        raise ValueError(f"no rank traces found in {trace_dir}")
    for path in trace_paths:
        match = RANK_RE.search(path.name)
        if not match:
            continue
        rank = int(match.group("rank"))
        with gzip.open(path, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        index = build_trace_index(events)
        steps = sorted(
            (
                event
                for event in events
                if event.get("cat") == "user_annotation"
                and str(event.get("name", "")).startswith("ProfilerStep#")
            ),
            key=lambda event: float(event["ts"]),
        )
        for step in steps:
            aligned.setdefault(str(step["name"]), {})[rank] = summarize_step(index, step)
    expected_ranks = set(range(len(trace_paths)))
    for step_name, rows in aligned.items():
        if set(rows) != expected_ranks:
            raise ValueError(
                f"{step_name}: expected ranks {sorted(expected_ranks)}, got {sorted(rows)}"
            )
    return dict(
        sorted(
            aligned.items(),
            key=lambda item: int(item[0].partition("#")[2]),
        )
    )


def model_ep_sendrecv_sources(
    aligned: dict[str, dict[int, dict[str, Any]]], interval_rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Separate EP8 SendRecv matched service from participant-arrival spread."""
    ranks = sorted(next(iter(aligned.values())))
    if ranks != list(range(16)):
        return None
    groups = (tuple(range(0, 8)), tuple(range(8, 16)))
    service_samples: list[float] = []
    arrival_samples: list[float] = []
    interval_metrics: list[dict[str, float]] = []
    operation_samples: dict[tuple[int, int], list[dict[str, float | int]]] = {}
    for row in interval_rows:
        current = aligned[row["current_step"]]
        group_metrics: list[dict[str, float]] = []
        for group_index, group in enumerate(groups):
            events_by_rank = {
                rank: current[rank]["pre_timer_stop"]["sendrecv_events"] for rank in group
            }
            counts = {len(events) for events in events_by_rank.values()}
            if len(counts) != 1:
                raise ValueError(
                    f"{row['current_step']} group {group}: mismatched SendRecv counts {counts}"
                )
            service_sum = 0.0
            arrival_sum = 0.0
            participant_wait_sum = 0.0
            for operation in range(counts.pop()):
                events = [events_by_rank[rank][operation] for rank in group]
                starts = [event["start_us"] for event in events]
                ends = [event["end_us"] for event in events]
                durations = [event["duration_us"] for event in events]
                service = max(ends) - max(starts)
                arrival = max(starts) - min(starts)
                participant_wait = max(0.0, max(durations) - service)
                service_sum += service
                arrival_sum += arrival
                participant_wait_sum += participant_wait
                service_samples.append(service)
                arrival_samples.append(arrival)
                latest_rank = group[starts.index(max(starts))]
                operation_samples.setdefault((group_index, operation), []).append(
                    {
                        "core_us": row["core_to_timer_stop_us"],
                        "service_us": service,
                        "arrival_spread_us": arrival,
                        "latest_rank": latest_rank,
                    }
                )
            group_metrics.append(
                {
                    "service_sum_us": service_sum,
                    "arrival_spread_sum_us": arrival_sum,
                    "participant_wait_sum_us": participant_wait_sum,
                }
            )
        metrics = {
            "max_group_service_sum_us": max(
                group["service_sum_us"] for group in group_metrics
            ),
            "max_group_arrival_spread_sum_us": max(
                group["arrival_spread_sum_us"] for group in group_metrics
            ),
            "max_group_participant_wait_sum_us": max(
                group["participant_wait_sum_us"] for group in group_metrics
            ),
            "late_rank_group_service_sum_us": group_metrics[
                0 if row["timer_stop_late_rank"] < 8 else 1
            ]["service_sum_us"],
            "late_rank_group_arrival_spread_sum_us": group_metrics[
                0 if row["timer_stop_late_rank"] < 8 else 1
            ]["arrival_spread_sum_us"],
        }
        row["model_ep_sendrecv"] = {"groups": group_metrics, "aggregate": metrics}
        interval_metrics.append(metrics)

    if not interval_metrics:
        return None
    core_values = [row["core_to_timer_stop_us"] for row in interval_rows]
    metric_names = tuple(interval_metrics[0])
    operation_sources = []
    for (group_index, operation), samples in sorted(operation_samples.items()):
        operation_sources.append(
            {
                "group": group_index,
                "operation": operation,
                "arrival_correlation_with_core_time": pearson(
                    [float(sample["core_us"]) for sample in samples],
                    [float(sample["arrival_spread_us"]) for sample in samples],
                ),
                "service_correlation_with_core_time": pearson(
                    [float(sample["core_us"]) for sample in samples],
                    [float(sample["service_us"]) for sample in samples],
                ),
                "arrival_spread_ms": summarize_values(
                    [float(sample["arrival_spread_us"]) / 1000.0 for sample in samples]
                ),
                "service_ms": summarize_values(
                    [float(sample["service_us"]) / 1000.0 for sample in samples]
                ),
                "latest_rank_counts": dict(
                    sorted(Counter(int(sample["latest_rank"]) for sample in samples).items())
                ),
            }
        )
    return {
        "correlations_with_core_time": {
            metric: pearson(core_values, [row[metric] for row in interval_metrics])
            for metric in metric_names
        },
        "interval_summaries_ms": {
            metric: summarize_values([row[metric] / 1000.0 for row in interval_metrics])
            for metric in metric_names
        },
        "per_collective_service_ms": summarize_values(
            [value / 1000.0 for value in service_samples]
        ),
        "per_collective_arrival_spread_ms": summarize_values(
            [value / 1000.0 for value in arrival_samples]
        ),
        "operations": operation_sources,
    }


def synchronization_sources(
    aligned: dict[str, dict[int, dict[str, Any]]], logged_ms: dict[int, float]
) -> dict[str, Any]:
    """Reconstruct logged intervals and split logging-tail from core training time."""
    step_items = list(aligned.items())
    interval_rows: list[dict[str, Any]] = []
    timer_stop_late_counts: Counter[int] = Counter()
    energy_late_counts: Counter[int] = Counter()
    required_previous = {"timer_start", "energy", "checkpoint"}
    for (previous_name, previous), (current_name, current) in zip(
        step_items, step_items[1:]
    ):
        if not all(
            required_previous <= set(metrics["synchronization"])
            for metrics in previous.values()
        ) or not all("timer_stop" in metrics["synchronization"] for metrics in current.values()):
            continue
        iteration = int(current_name.partition("#")[2])
        timer_stop_late_rank = max(
            current,
            key=lambda rank: current[rank]["synchronization"]["timer_stop"]["start_us"],
        )
        energy_late_rank = max(
            previous,
            key=lambda rank: previous[rank]["synchronization"]["energy"]["start_us"],
        )
        timer_stop_late_counts[timer_stop_late_rank] += 1
        energy_late_counts[energy_late_rank] += 1

        previous_start_end = max(
            metrics["synchronization"]["timer_start"]["end_us"]
            for metrics in previous.values()
        )
        previous_checkpoint_end = max(
            metrics["synchronization"]["checkpoint"]["end_us"]
            for metrics in previous.values()
        )
        current_stop_end = max(
            metrics["synchronization"]["timer_stop"]["end_us"]
            for metrics in current.values()
        )
        energy_starts = [
            metrics["synchronization"]["energy"]["start_us"]
            for metrics in previous.values()
        ]
        energy_ends = [
            metrics["synchronization"]["energy"]["end_us"]
            for metrics in previous.values()
        ]
        timer_stop_starts = [
            metrics["synchronization"]["timer_stop"]["start_us"]
            for metrics in current.values()
        ]
        timer_stop_ends = [
            metrics["synchronization"]["timer_stop"]["end_us"]
            for metrics in current.values()
        ]
        energy_host_gaps = {
            rank: previous[rank]["synchronization"]["energy"]["start_us"]
            - previous[rank]["synchronization"]["timer_start"]["end_us"]
            for rank in previous
        }
        energy_host_late_rank = max(energy_host_gaps, key=energy_host_gaps.get)
        late_metrics = current[timer_stop_late_rank]
        pre_stop_metric_names = ("nccl_us", "non_nccl_us", "overlap_us", "gpu_idle_us")
        row = {
            "iteration": iteration,
            "previous_step": previous_name,
            "current_step": current_name,
            "logged_ms": logged_ms.get(iteration),
            "trace_interval_us": current_stop_end - previous_start_end,
            "logging_tail_us": previous_checkpoint_end - previous_start_end,
            "core_to_timer_stop_us": current_stop_end - previous_checkpoint_end,
            "timer_stop_arrival_spread_us": max(timer_stop_starts) - min(timer_stop_starts),
            "timer_stop_service_us": max(timer_stop_ends) - max(timer_stop_starts),
            "timer_stop_late_rank": timer_stop_late_rank,
            "energy_arrival_spread_us": max(energy_starts) - min(energy_starts),
            "energy_service_us": max(energy_ends) - max(energy_starts),
            "energy_late_rank": energy_late_rank,
            "energy_host_gap_max_us": energy_host_gaps[energy_host_late_rank],
            "energy_host_gap_rank": energy_host_late_rank,
            "energy_late_rank_python_us": previous[energy_late_rank]["energy_python_us"],
            "late_rank_grouped_cpu_sum_us": late_metrics["grouped_cpu_sum_us"],
            "late_rank_grouped_cpu_max_us": late_metrics["grouped_cpu_max_us"],
            "late_rank_grouped_launch_gap_us": late_metrics[
                "grouped_internal_launch_gap_max_us"
            ],
            "late_rank_pre_stop": {
                metric: late_metrics["pre_timer_stop"][metric]
                for metric in pre_stop_metric_names
            },
            "all_rank_max_pre_stop": {
                metric: max(
                    metrics["pre_timer_stop"][metric] for metrics in current.values()
                )
                for metric in pre_stop_metric_names
            },
            "late_rank_pre_stop_by_kind_us": late_metrics["pre_timer_stop"][
                "nccl_by_kind_us"
            ],
            "all_rank_max_pre_stop_by_kind_us": {
                kind: max(
                    metrics["pre_timer_stop"]["nccl_by_kind_us"].get(kind, 0.0)
                    for metrics in current.values()
                )
                for kind in {
                    collective
                    for metrics in current.values()
                    for collective in metrics["pre_timer_stop"]["nccl_by_kind_us"]
                }
            },
        }
        if row["logged_ms"] is not None:
            row["reconstruction_error_us"] = row["trace_interval_us"] - 1000.0 * row["logged_ms"]
        interval_rows.append(row)

    if not interval_rows:
        return {
            "interval_count": 0,
            "timer_stop_late_rank_counts": {},
            "energy_late_rank_counts": {},
            "correlations_with_logged_time": {},
            "pre_stop_correlations_with_core_time": {},
            "pre_stop_collective_correlations_with_core_time": {},
            "model_ep_sendrecv": None,
            "summaries_ms": {},
            "reconstruction_error_ms": None,
            "intervals": [],
        }

    numeric_metrics = (
        "trace_interval_us",
        "logging_tail_us",
        "core_to_timer_stop_us",
        "timer_stop_arrival_spread_us",
        "timer_stop_service_us",
        "energy_arrival_spread_us",
        "energy_service_us",
        "energy_host_gap_max_us",
        "late_rank_grouped_cpu_sum_us",
        "late_rank_grouped_cpu_max_us",
        "late_rank_grouped_launch_gap_us",
    )
    logged_rows = [row for row in interval_rows if row["logged_ms"] is not None]
    logged_values = [row["logged_ms"] for row in logged_rows]
    correlations = {
        metric: pearson(logged_values, [row[metric] / 1000.0 for row in logged_rows])
        for metric in numeric_metrics
    }
    core_values = [row["core_to_timer_stop_us"] for row in interval_rows]
    pre_stop_correlations = {
        scope: {
            metric: pearson(
                core_values,
                [row[f"{scope}_pre_stop"][metric] for row in interval_rows],
            )
            for metric in ("nccl_us", "non_nccl_us", "overlap_us", "gpu_idle_us")
        }
        for scope in ("late_rank", "all_rank_max")
    }
    pre_stop_kinds = sorted(
        {
            kind
            for row in interval_rows
            for kind in row["all_rank_max_pre_stop_by_kind_us"]
        }
    )
    pre_stop_collective_correlations = {
        kind: {
            "late_rank": pearson(
                core_values,
                [row["late_rank_pre_stop_by_kind_us"].get(kind, 0.0) for row in interval_rows],
            ),
            "all_rank_max": pearson(
                core_values,
                [
                    row["all_rank_max_pre_stop_by_kind_us"].get(kind, 0.0)
                    for row in interval_rows
                ],
            ),
        }
        for kind in pre_stop_kinds
    }
    summaries = {
        metric: summarize_values([row[metric] / 1000.0 for row in interval_rows])
        for metric in numeric_metrics
    }
    reconstruction_errors = [
        row["reconstruction_error_us"] / 1000.0
        for row in logged_rows
        if "reconstruction_error_us" in row
    ]
    model_ep_sendrecv = model_ep_sendrecv_sources(aligned, interval_rows)
    return {
        "interval_count": len(interval_rows),
        "timer_stop_late_rank_counts": dict(sorted(timer_stop_late_counts.items())),
        "energy_late_rank_counts": dict(sorted(energy_late_counts.items())),
        "correlations_with_logged_time": correlations,
        "pre_stop_correlations_with_core_time": pre_stop_correlations,
        "pre_stop_collective_correlations_with_core_time": pre_stop_collective_correlations,
        "model_ep_sendrecv": model_ep_sendrecv,
        "summaries_ms": summaries,
        "reconstruction_error_ms": (
            summarize_values(reconstruction_errors) if reconstruction_errors else None
        ),
        "intervals": interval_rows,
    }


def profile_sources(
    aligned: dict[str, dict[int, dict[str, Any]]], logged_ms: dict[int, float]
) -> dict[str, Any]:
    """Summarize critical ranks and correlate trace metrics with step duration."""
    metric_names = (
        "nccl_us",
        "non_nccl_us",
        "nccl_non_nccl_overlap_us",
        "gpu_idle_us",
        "grouped_cpu_sum_us",
        "grouped_cpu_max_us",
        "grouped_internal_launch_gap_max_us",
    )
    rows: list[dict[str, Any]] = []
    critical_counts: Counter[int] = Counter()
    for step_name, rank_metrics in aligned.items():
        critical_rank = max(rank_metrics, key=lambda rank: rank_metrics[rank]["duration_us"])
        critical_counts[critical_rank] += 1
        durations = [metrics["duration_us"] for metrics in rank_metrics.values()]
        row: dict[str, Any] = {
            "step": step_name,
            "critical_rank": critical_rank,
            "max_duration_us": max(durations),
            "median_duration_us": statistics.median(durations),
            "rank_spread_us": max(durations) - min(durations),
            "critical": rank_metrics[critical_rank],
            "max_by_metric": {
                metric: max(metrics[metric] for metrics in rank_metrics.values())
                for metric in metric_names
            },
        }
        rows.append(row)

    durations = [row["max_duration_us"] for row in rows]
    correlations: dict[str, dict[str, float | None]] = {}
    for metric in metric_names:
        correlations[metric] = {
            "critical_rank_value": pearson(
                durations, [row["critical"][metric] for row in rows]
            ),
            "all_rank_max": pearson(
                durations, [row["max_by_metric"][metric] for row in rows]
            ),
        }
    rank_summaries: dict[int, dict[str, Any]] = {}
    ranks = sorted(next(iter(aligned.values())))
    for rank in ranks:
        rank_summaries[rank] = summarize_values(
            [rank_metrics[rank]["duration_us"] / 1000.0 for rank_metrics in aligned.values()]
        )
    collective_kinds = sorted(
        {
            kind
            for rank_metrics in aligned.values()
            for metrics in rank_metrics.values()
            for kind in metrics["nccl_by_kind_us"]
        }
    )
    collective_sources: dict[str, Any] = {}
    for kind in collective_kinds:
        critical_values: list[float] = []
        all_rank_max_values: list[float] = []
        max_rank_counts: Counter[int] = Counter()
        for row, rank_metrics in zip(rows, aligned.values()):
            critical_values.append(
                rank_metrics[row["critical_rank"]]["nccl_by_kind_us"].get(kind, 0.0)
            )
            max_rank = max(
                rank_metrics,
                key=lambda rank: rank_metrics[rank]["nccl_by_kind_us"].get(kind, 0.0),
            )
            max_rank_counts[max_rank] += 1
            all_rank_max_values.append(
                rank_metrics[max_rank]["nccl_by_kind_us"].get(kind, 0.0)
            )
        collective_sources[kind] = {
            "critical_rank_correlation": pearson(durations, critical_values),
            "all_rank_max_correlation": pearson(durations, all_rank_max_values),
            "all_rank_max_ms": summarize_values(
                [value / 1000.0 for value in all_rank_max_values]
            ),
            "max_rank_counts": dict(sorted(max_rank_counts.items())),
        }
    return {
        "step_count": len(rows),
        "critical_rank_counts": dict(sorted(critical_counts.items())),
        "global_step_ms": summarize_values([duration / 1000.0 for duration in durations]),
        "rank_step_ms": rank_summaries,
        "correlations": correlations,
        "collective_sources": collective_sources,
        "synchronization_sources": synchronization_sources(aligned, logged_ms),
        "steps": rows,
    }


def print_timing(timing: dict[str, Any]) -> None:
    """Print timing summaries in milliseconds."""
    print("\n== Profiler-free healthy EP8/EP8 timing ==")
    for label, run in timing["runs"].items():
        summary = run["summary"]
        print(
            f"{label}: n={summary['n']} mean={summary['mean']:.3f} median={summary['median']:.3f} "
            f"std={summary['std']:.3f} CV={summary['cv_percent']:.2f}% "
            f"p05/p95={summary['p05']:.3f}/{summary['p95']:.3f} "
            f"min/max={summary['min']:.3f}/{summary['max']:.3f} "
            f"lag1={format_optional(summary['lag_one_correlation'])} "
            f"slope={summary['slope_per_iteration']:.4f} ms/iter"
        )
        if run["outliers"]:
            print(f"  robust outliers: {run['outliers']}")
    pooled = timing["pooled"]
    between = timing["between_run_means"]
    print(
        f"pooled: n={pooled['n']} mean={pooled['mean']:.3f} median={pooled['median']:.3f} "
        f"std={pooled['std']:.3f} CV={pooled['cv_percent']:.2f}% "
        f"p05/p95={pooled['p05']:.3f}/{pooled['p95']:.3f}"
    )
    print(
        f"between-run means: mean={between['mean']:.3f} std={between['std']:.3f} "
        f"CV={between['cv_percent']:.2f}% range={between['min']:.3f}-{between['max']:.3f} ms"
    )
    print("fixed warmup-cutoff sensitivity:")
    for cutoff, values in timing["cutoff_sensitivity"].items():
        pooled = values["pooled"]
        between = values["between_run_means"]
        print(
            f"  iteration>={cutoff}: n={pooled['n']} mean={pooled['mean']:.3f} "
            f"std={pooled['std']:.3f} CV={pooled['cv_percent']:.2f}% "
            f"between-run-CV={between['cv_percent']:.2f}%"
        )


def format_optional(value: float | None) -> str:
    """Format an optional scalar."""
    return "n/a" if value is None else f"{value:.3f}"


def print_profile(label: str, sources: dict[str, Any]) -> None:
    """Print profiler source-attribution summary."""
    summary = sources["global_step_ms"]
    print(f"\n== {label} all-rank trace ({sources['step_count']} steps) ==")
    print(
        f"profiled global step: mean={summary['mean']:.3f} ms std={summary['std']:.3f} ms "
        f"CV={summary['cv_percent']:.2f}% p05/p95={summary['p05']:.3f}/{summary['p95']:.3f} ms"
    )
    print(f"critical-rank counts: {sources['critical_rank_counts']}")
    print("correlation with max-rank step duration:")
    for metric, values in sources["correlations"].items():
        print(
            f"  {metric}: critical={format_optional(values['critical_rank_value'])} "
            f"all-rank-max={format_optional(values['all_rank_max'])}"
        )
    print("NCCL collective-type correlation with max-rank step duration:")
    for kind, values in sorted(
        sources["collective_sources"].items(),
        key=lambda item: abs(item[1]["all_rank_max_correlation"] or 0.0),
        reverse=True,
    ):
        summary = values["all_rank_max_ms"]
        print(
            f"  {kind}: critical={format_optional(values['critical_rank_correlation'])} "
            f"all-rank-max={format_optional(values['all_rank_max_correlation'])} "
            f"mean/std={summary['mean']:.3f}/{summary['std']:.3f} ms"
        )
    synchronization = sources["synchronization_sources"]
    if synchronization["interval_count"]:
        summaries = synchronization["summaries_ms"]
        error = synchronization["reconstruction_error_ms"]
        print(
            f"timer reconstruction: n={synchronization['interval_count']} "
            f"trace={summaries['trace_interval_us']['mean']:.3f} ms "
            f"logging-tail={summaries['logging_tail_us']['mean']:.3f} ms "
            f"core-to-stop={summaries['core_to_timer_stop_us']['mean']:.3f} ms"
        )
        if error:
            print(
                f"  trace-minus-driver error: mean={error['mean']:.3f} ms "
                f"std={error['std']:.3f} ms min/max={error['min']:.3f}/{error['max']:.3f} ms"
            )
        print(
            "  late ranks: timer-stop="
            f"{synchronization['timer_stop_late_rank_counts']} "
            f"energy={synchronization['energy_late_rank_counts']}"
        )
        print("  correlation with logged iteration time:")
        for metric, correlation in sorted(
            synchronization["correlations_with_logged_time"].items(),
            key=lambda item: abs(item[1] or 0.0),
            reverse=True,
        ):
            print(f"    {metric}: {format_optional(correlation)}")
        print("  pre-timer-stop correlation with core duration:")
        for scope, correlations in synchronization[
            "pre_stop_correlations_with_core_time"
        ].items():
            formatted = " ".join(
                f"{metric}={format_optional(correlation)}"
                for metric, correlation in correlations.items()
            )
            print(f"    {scope}: {formatted}")
        print("  pre-timer-stop NCCL type correlation with core duration:")
        for kind, correlations in sorted(
            synchronization[
                "pre_stop_collective_correlations_with_core_time"
            ].items(),
            key=lambda item: abs(item[1]["all_rank_max"] or 0.0),
            reverse=True,
        ):
            print(
                f"    {kind}: late={format_optional(correlations['late_rank'])} "
                f"all-rank-max={format_optional(correlations['all_rank_max'])}"
            )
        model_ep = synchronization["model_ep_sendrecv"]
        if model_ep:
            service = model_ep["per_collective_service_ms"]
            arrival = model_ep["per_collective_arrival_spread_ms"]
            print(
                f"  participant-aligned model-EP: service={service['mean']:.3f}+/-"
                f"{service['std']:.3f} ms/collective arrival-spread={arrival['mean']:.3f}+/-"
                f"{arrival['std']:.3f} ms/collective"
            )
            print("  model-EP correlation with core duration:")
            for metric, correlation in sorted(
                model_ep["correlations_with_core_time"].items(),
                key=lambda item: abs(item[1] or 0.0),
                reverse=True,
            ):
                summary = model_ep["interval_summaries_ms"][metric]
                print(
                    f"    {metric}: r={format_optional(correlation)} "
                    f"mean/std={summary['mean']:.3f}/{summary['std']:.3f} ms"
                )
            print("  highest-variance model-EP arrival ordinals:")
            for operation in sorted(
                model_ep["operations"],
                key=lambda item: item["arrival_spread_ms"]["std"],
                reverse=True,
            )[:6]:
                arrival = operation["arrival_spread_ms"]
                print(
                    f"    group={operation['group']} op={operation['operation']} "
                    f"arrival={arrival['mean']:.3f}+/-{arrival['std']:.3f} ms "
                    f"r={format_optional(operation['arrival_correlation_with_core_time'])} "
                    f"latest={operation['latest_rank_counts']}"
                )
    print("slowest profiled steps:")
    for row in sorted(sources["steps"], key=lambda item: item["max_duration_us"], reverse=True)[:5]:
        critical = row["critical"]
        print(
            f"  {row['step']}: {row['max_duration_us'] / 1000.0:.3f} ms rank={row['critical_rank']} "
            f"spread={row['rank_spread_us'] / 1000.0:.3f} ms "
            f"NCCL={critical['nccl_us'] / 1000.0:.3f} non-NCCL={critical['non_nccl_us'] / 1000.0:.3f} "
            f"idle={critical['gpu_idle_us'] / 1000.0:.3f} grouped={critical['grouped_cpu_sum_us'] / 1000.0:.3f} "
            f"max-launch-gap={critical['grouped_internal_launch_gap_max_us'] / 1000.0:.3f} ms"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="baseline-characterization output root")
    parser.add_argument("--job-id", required=True, help="SLURM job ID")
    parser.add_argument("--start-iteration", type=int, default=5)
    parser.add_argument("--timing-labels", nargs="+", default=["timing_1", "timing_2", "timing_3"])
    parser.add_argument("--profile-labels", nargs="+", default=["profile", "profile_detail"])
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run timing and trace analysis."""
    args = parse_args()
    timing_runs: dict[str, Any] = {}
    all_timing_data: dict[str, tuple[list[int], list[float]]] = {}
    pooled_values: list[float] = []
    run_means: list[float] = []
    for label in args.timing_labels:
        log = find_single(args.root / label, f"**/{args.job_id}/driver_{args.job_id}.log")
        all_iterations, all_elapsed_ms = read_timings(log, 1)
        all_timing_data[label] = (all_iterations, all_elapsed_ms)
        selected = [
            (iteration, elapsed)
            for iteration, elapsed in zip(all_iterations, all_elapsed_ms)
            if iteration >= args.start_iteration
        ]
        iterations = [iteration for iteration, _ in selected]
        elapsed_ms = [elapsed for _, elapsed in selected]
        summary = summarize_values(elapsed_ms)
        timing_runs[label] = {
            "log": str(log),
            "iterations": iterations,
            "elapsed_ms": elapsed_ms,
            "summary": summary,
            "outliers": [
                {"iteration": iterations[index], "elapsed_ms": elapsed_ms[index]}
                for index in summary["outlier_indices"]
            ],
        }
        pooled_values.extend(elapsed_ms)
        run_means.append(summary["mean"])
    cutoff_sensitivity: dict[int, Any] = {}
    for cutoff in (5, 10, 15, 20):
        cutoff_runs = {
            label: [
                elapsed
                for iteration, elapsed in zip(iterations, elapsed_values)
                if iteration >= cutoff
            ]
            for label, (iterations, elapsed_values) in all_timing_data.items()
        }
        if not all(cutoff_runs.values()):
            continue
        cutoff_sensitivity[cutoff] = {
            "pooled": summarize_values(
                [elapsed for elapsed_values in cutoff_runs.values() for elapsed in elapsed_values]
            ),
            "between_run_means": summarize_values(
                [statistics.mean(elapsed_values) for elapsed_values in cutoff_runs.values()]
            ),
        }
    timing = {
        "runs": timing_runs,
        "pooled": summarize_values(pooled_values),
        "between_run_means": summarize_values(run_means),
        "cutoff_sensitivity": cutoff_sensitivity,
    }

    profiles: dict[str, Any] = {}
    for label in args.profile_labels:
        trace_dir = find_single(args.root / label, f"**/{args.job_id}/torch_profile")
        log = find_single(args.root / label, f"**/{args.job_id}/driver_{args.job_id}.log")
        profile_iterations, profile_elapsed_ms = read_timings(log, 1)
        profiles[label] = profile_sources(
            load_profile(trace_dir), dict(zip(profile_iterations, profile_elapsed_ms))
        )

    result = {"job_id": args.job_id, "timing": timing, "profiles": profiles}
    print_timing(timing)
    for label, sources in profiles.items():
        print_profile(label, sources)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote machine-readable results to {args.json_output}")


if __name__ == "__main__":
    main()
