#!/usr/bin/env python3
"""Attribute the exact-uniform EP8/EP4 NEP slowdown from all-rank traces."""

from __future__ import annotations

import argparse
import ast
import bisect
import gzip
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
GROUPED_LINEAR_NAME = "_GroupedLinearBackward"
NCCL_GROUPS = {
    "healthy": {
        "model_ep": "EXPERT_MODEL_PARALLEL_GROUP",
        "expert_edp": "EXPERT_DATA_PARALLEL_GROUP",
        "dense_dp": "DATA_PARALLEL_GROUP_WITH_CP",
        "tp": "TENSOR_MODEL_PARALLEL_GROUP",
    },
    "nep": {
        "model_ep": "ep",
        "expert_edp": "ep_dp",
        "dense_dp": "dp_cp",
        "tp": "tp",
        "owner_transfer": "nep_owner_transfer",
        "owner_gather": "nep_owner_gather",
    },
}
POST_TRAIN_TP_NUMEL = 2_104_704
RANK_RE = re.compile(r"rank-(?P<rank>\d+)\.json\.gz$")


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
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "std": std,
        "cv_percent": 100.0 * std / mean if mean else 0.0,
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


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


def union_duration(intervals: list[tuple[float, float]]) -> float:
    """Return interval-union duration."""
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    """Return intersection duration between two interval unions."""
    left_rows = merge_intervals(left)
    right_rows = merge_intervals(right)
    left_index = 0
    right_index = 0
    total = 0.0
    while left_index < len(left_rows) and right_index < len(right_rows):
        left_start, left_end = left_rows[left_index]
        right_start, right_end = right_rows[right_index]
        total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def is_nccl(event: dict[str, Any]) -> bool:
    """Return whether a GPU event is an NCCL kernel."""
    return event.get("cat") == "kernel" and "nccl" in str(event.get("name", "")).lower()


def event_end(event: dict[str, Any]) -> float:
    """Return a trace event end timestamp."""
    return float(event["ts"]) + float(event.get("dur", 0.0))


def contained_in(event: dict[str, Any], container: dict[str, Any]) -> bool:
    """Return whether an event is fully contained in a same-thread CPU scope."""
    return (
        event.get("pid") == container.get("pid")
        and event.get("tid") == container.get("tid")
        and float(event.get("ts", 0.0)) >= float(container["ts"])
        and event_end(event) <= event_end(container)
    )


def grouped_launch_gap(
    scope: dict[str, Any],
    driver_by_external: dict[Any, list[dict[str, Any]]],
    grouped_calls_by_thread: dict[tuple[Any, Any], list[dict[str, Any]]],
) -> float:
    """Return the largest host gap between expert launches in one grouped scope."""
    driver_events = driver_by_external.get(
        scope.get("args", {}).get("External id"), []
    )
    thread = (scope.get("pid"), scope.get("tid"))
    grouped_calls = [
        event
        for event in grouped_calls_by_thread.get(thread, [])
        if contained_in(event, scope)
    ]
    containers = grouped_calls or [scope]
    largest = 0.0
    for container in containers:
        launches = sorted(
            (event for event in driver_events if contained_in(event, container)),
            key=lambda event: float(event["ts"]),
        )
        for previous, current in zip(launches, launches[1:]):
            largest = max(largest, float(current["ts"]) - event_end(previous))
    return largest


def find_step_index(starts: list[float], ends: list[float], timestamp: float) -> int | None:
    """Find the profiler step containing a timestamp."""
    index = bisect.bisect_right(starts, timestamp) - 1
    if index >= 0 and timestamp < ends[index]:
        return index
    return None


def parse_int_list(value: Any) -> tuple[int, ...]:
    """Parse a profiler string/list containing integer ranks or splits."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return tuple(int(item) for item in value)


def parse_trace(
    path: Path, retain_gpu: bool, expected_steps: int = 16
) -> dict[str, Any]:
    """Read one trace and retain compact metrics for the expected number of steps."""
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]

    rank_match = RANK_RE.search(path.name)
    if rank_match is None:
        raise ValueError(f"cannot parse rank from {path}")
    rank = int(rank_match.group("rank"))
    step_events = sorted(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and str(event.get("name", "")).startswith("ProfilerStep#")
        ),
        key=lambda event: float(event["ts"]),
    )
    step_names = [str(event["name"]) for event in step_events]
    step_starts = [float(event["ts"]) for event in step_events]
    step_ends = [float(event["ts"]) + float(event["dur"]) for event in step_events]
    if len(step_names) != expected_steps:
        raise ValueError(
            f"{path}: expected {expected_steps} profiler steps, found {len(step_names)}"
        )

    grouped_scopes_by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_calls_by_thread: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(
        list
    )
    driver_by_external: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        category = event.get("cat")
        if category == "cpu_op" and event.get("name") == GROUPED_LINEAR_NAME:
            step_index = find_step_index(
                step_starts, step_ends, float(event.get("ts", 0.0))
            )
            if step_index is not None:
                grouped_scopes_by_step[step_names[step_index]].append(event)
        elif category == "python_function" and "te_general_grouped_gemm" in str(
            event.get("name", "")
        ):
            grouped_calls_by_thread[(event.get("pid"), event.get("tid"))].append(event)
        elif category == "cuda_driver" and "cuLaunchKernel" in str(event.get("name", "")):
            driver_by_external[event.get("args", {}).get("External id")].append(event)

    record_by_external: dict[Any, dict[str, Any]] = {}
    records_by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("name") != "record_param_comms":
            continue
        args = event.get("args", {})
        if args.get("Collective name") == "wait":
            continue
        step_index = find_step_index(step_starts, step_ends, float(event.get("ts", 0.0)))
        if step_index is None:
            continue
        external_id = args.get("External id")
        record_by_external[external_id] = event
        records_by_step[step_names[step_index]].append(event)

    nccl_by_external: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    compact_gpu_by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nvjet_by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dispatch_counts: dict[Any, Counter[str]] = defaultdict(Counter)
    for event in events:
        if event.get("cat") not in GPU_CATEGORIES:
            continue
        timestamp = float(event.get("ts", 0.0))
        step_index = find_step_index(step_starts, step_ends, timestamp)
        if step_index is None:
            continue
        args = event.get("args", {})
        external_id = args.get("External id")
        if is_nccl(event):
            nccl_by_external[external_id].append(event)
        if event.get("cat") == "kernel" and str(event.get("name", "")).startswith(
            "nvjet"
        ):
            nvjet_by_step[step_names[step_index]].append(event)
        if not retain_gpu:
            continue
        record = record_by_external.get(external_id)
        record_args = record.get("args", {}) if record is not None else {}
        group = args.get("Process Group Description") or record_args.get(
            "Process Group Description"
        )
        name = str(event.get("name", ""))
        stream_id = args.get("stream")
        compact_gpu_by_step[step_names[step_index]].append(
            {
                "start": timestamp,
                "end": timestamp + float(event.get("dur", 0.0)),
                "cat": event.get("cat"),
                "name": name,
                "stream": stream_id,
                "group": group,
                "is_nccl": is_nccl(event),
            }
        )
        if not is_nccl(event) and stream_id is not None:
            if event.get("cat") == "gpu_memcpy" and "DtoD" in name:
                dispatch_counts[stream_id]["copies"] += 1
            elif "CUDAFunctor_add<float>" in name:
                dispatch_counts[stream_id]["adds"] += 1
            elif "FillFunctor<float>" in name:
                dispatch_counts[stream_id]["fills"] += 1

    collectives_by_step: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for step_name, records in records_by_step.items():
        for record in sorted(records, key=lambda event: float(event["ts"])):
            args = record.get("args", {})
            kernels = nccl_by_external.get(args.get("External id"), [])
            if not kernels:
                continue
            input_elements = int(args.get("In msg nelems") or 0)
            output_elements = int(args.get("Out msg nelems") or 0)
            group = str(args.get("Process Group Description"))
            collectives_by_step[step_name][group].append(
                {
                    "collective": str(args.get("Collective name")),
                    "dtype": str(args.get("dtype")),
                    "input_elements": input_elements,
                    "output_elements": output_elements,
                    "payload_elements": max(input_elements, output_elements),
                    "participants": parse_int_list(args.get("Process Group Ranks") or []),
                    "record_start": float(record["ts"]),
                    "kernel_start": min(float(kernel["ts"]) for kernel in kernels),
                    "kernel_end": max(
                        float(kernel["ts"]) + float(kernel.get("dur", 0.0))
                        for kernel in kernels
                    ),
                }
            )

    dispatch_stream = None
    if dispatch_counts:
        candidates = [
            stream_id
            for stream_id, counts in dispatch_counts.items()
            if counts["copies"] and counts["adds"]
        ]
        if candidates:
            dispatch_stream = max(
                candidates,
                key=lambda stream_id: tuple(
                    dispatch_counts[stream_id][key] for key in ("copies", "adds", "fills")
                ),
            )

    rank_local_work = {}
    for step_name in step_names:
        scopes = grouped_scopes_by_step[step_name]
        scope_gaps = [
            grouped_launch_gap(scope, driver_by_external, grouped_calls_by_thread)
            for scope in scopes
        ]
        nvjet_events = nvjet_by_step[step_name]
        stream_counts = Counter(
            event.get("args", {}).get("stream") for event in nvjet_events
        )
        expert_streams = {stream for stream, _ in stream_counts.most_common(4)}
        expert_gemms = [
            event
            for event in nvjet_events
            if event.get("args", {}).get("stream") in expert_streams
        ]
        rank_local_work[step_name] = {
            "grouped_backward_scope_count": len(scopes),
            "grouped_backward_cpu_sum_ms": sum(
                float(scope.get("dur", 0.0)) for scope in scopes
            )
            / 1000.0,
            "grouped_backward_cpu_max_ms": max(
                (float(scope.get("dur", 0.0)) for scope in scopes), default=0.0
            )
            / 1000.0,
            "grouped_internal_launch_gap_sum_ms": sum(scope_gaps) / 1000.0,
            "grouped_internal_launch_gap_max_ms": max(scope_gaps, default=0.0)
            / 1000.0,
            "expert_gemm_count": len(expert_gemms),
            "expert_gemm_sum_ms": sum(
                float(event.get("dur", 0.0)) for event in expert_gemms
            )
            / 1000.0,
            "expert_gemm_mean_us": (
                statistics.mean(
                    float(event.get("dur", 0.0)) for event in expert_gemms
                )
                if expert_gemms
                else 0.0
            ),
        }

    return {
        "rank": rank,
        "step_spans": {
            name: (start, end)
            for name, start, end in zip(step_names, step_starts, step_ends)
        },
        "collectives": {
            step: dict(groups) for step, groups in collectives_by_step.items()
        },
        "gpu": dict(compact_gpu_by_step),
        "dispatch_stream": dispatch_stream,
        "rank_local_work": rank_local_work,
    }


def clipped_intervals(
    events: list[dict[str, Any]], start: float, end: float
) -> list[tuple[float, float]]:
    """Return event intervals clipped to a window."""
    return [
        (max(start, float(event["start"])), min(end, float(event["end"])))
        for event in events
        if float(event["start"]) < end and float(event["end"]) > start
    ]


def gpu_window_metrics(
    events: list[dict[str, Any]], start: float, end: float
) -> dict[str, float]:
    """Return exact GPU interval accounting for a window."""
    selected = [
        event
        for event in events
        if float(event["start"]) < end and float(event["end"]) > start
    ]
    nccl = [event for event in selected if event["is_nccl"]]
    non_nccl = [event for event in selected if not event["is_nccl"]]
    all_intervals = clipped_intervals(selected, start, end)
    nccl_intervals = clipped_intervals(nccl, start, end)
    non_nccl_intervals = clipped_intervals(non_nccl, start, end)
    span = end - start
    return {
        "span_ms": span / 1000.0,
        "gpu_active_ms": union_duration(all_intervals) / 1000.0,
        "gpu_idle_ms": (span - union_duration(all_intervals)) / 1000.0,
        "nccl_union_ms": union_duration(nccl_intervals) / 1000.0,
        "non_nccl_union_ms": union_duration(non_nccl_intervals) / 1000.0,
        "nccl_non_nccl_overlap_ms": intersection_duration(
            nccl_intervals, non_nccl_intervals
        )
        / 1000.0,
    }


def interval_metrics(
    intervals: list[tuple[float, float]], useful: list[tuple[float, float]]
) -> dict[str, float]:
    """Return residency, useful overlap, and exposure for one phase."""
    residency = union_duration(intervals)
    overlap = intersection_duration(intervals, useful)
    return {
        "residency_ms": residency / 1000.0,
        "useful_overlap_ms": overlap / 1000.0,
        "exposed_ms": (residency - overlap) / 1000.0,
        "useful_overlap_percent": 100.0 * overlap / residency if residency else 0.0,
    }


def dispatch_category(event: dict[str, Any]) -> str:
    """Classify NEP staging-stream kernels."""
    name = str(event["name"])
    if event["cat"] == "gpu_memcpy" and "DtoD" in name:
        return "device_copies"
    if "CUDAFunctor_add<float>" in name:
        return "accumulation_adds"
    if "FillFunctor<float>" in name:
        return "buffer_fills"
    if "NormTwoOps" in name:
        return "gradient_norms"
    if event["cat"] == "gpu_memcpy" and "DtoH" in name:
        return "gradient_check_scalar_copies"
    return "other"


def owner_step_metrics(trace: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    """Calculate full-owner step, backward, group, and reshard metrics."""
    groups = NCCL_GROUPS[mode]
    output: dict[str, dict[str, Any]] = {}
    dispatch_stream = trace["dispatch_stream"]
    for step_name, (step_start, step_end) in trace["step_spans"].items():
        gpu_events = trace["gpu"][step_name]
        row: dict[str, Any] = {"full_step": gpu_window_metrics(gpu_events, step_start, step_end)}
        group_union = {}
        group_intervals = {}
        for canonical, group in groups.items():
            intervals = clipped_intervals(
                [event for event in gpu_events if event["is_nccl"] and event["group"] == group],
                step_start,
                step_end,
            )
            group_intervals[canonical] = intervals
            group_union[canonical] = union_duration(intervals) / 1000.0
        row["nccl_group_union_ms"] = group_union

        model_rows = [
            record
            for record in trace["collectives"][step_name][groups["model_ep"]]
            if record["collective"] == "all_to_allv"
        ]
        if len(model_rows) != 36:
            raise ValueError(f"{mode} {step_name}: expected 36 model-EP operations")
        backward_start = float(model_rows[18]["kernel_start"])
        post_train_candidates = [
            record
            for records in trace["collectives"][step_name].values()
            for record in records
            if record["payload_elements"] == POST_TRAIN_TP_NUMEL
            and record["kernel_start"] > backward_start
        ]
        if not post_train_candidates:
            raise ValueError(f"{mode} {step_name}: post-train TP boundary not found")
        backward_end = min(record["kernel_start"] for record in post_train_candidates)
        row["backward"] = gpu_window_metrics(gpu_events, backward_start, backward_end)

        useful_events = [
            event
            for event in gpu_events
            if not event["is_nccl"]
            and (mode == "healthy" or event["stream"] != dispatch_stream)
        ]
        useful_intervals = clipped_intervals(useful_events, backward_start, backward_end)

        def record_intervals(records: list[dict[str, Any]]) -> list[tuple[float, float]]:
            return [
                (
                    max(backward_start, float(record["kernel_start"])),
                    min(backward_end, float(record["kernel_end"])),
                )
                for record in records
                if record["kernel_start"] < backward_end
                and record["kernel_end"] > backward_start
            ]

        expert_edp_intervals = record_intervals(
            trace["collectives"][step_name][groups["expert_edp"]]
        )
        row["expert_edp_overlap"] = interval_metrics(
            expert_edp_intervals, useful_intervals
        )
        backward_group_intervals = {
            canonical: [
                (max(backward_start, start), min(backward_end, end))
                for start, end in intervals
                if start < backward_end and end > backward_start
            ]
            for canonical, intervals in group_intervals.items()
        }
        row["nccl_group_exposure"] = {
            canonical: interval_metrics(intervals, useful_intervals)
            for canonical, intervals in backward_group_intervals.items()
        }
        row["nccl_group_overlap_ms"] = {
            "expert_edp_with_dense_dp": intersection_duration(
                backward_group_intervals["expert_edp"],
                backward_group_intervals["dense_dp"],
            )
            / 1000.0,
            "expert_edp_with_model_ep": intersection_duration(
                backward_group_intervals["expert_edp"],
                backward_group_intervals["model_ep"],
            )
            / 1000.0,
        }

        if mode == "nep":
            transfer_rows = trace["collectives"][step_name][groups["owner_transfer"]]
            legacy_gather_rows = [
                record
                for record in transfer_rows
                if record["input_elements"] == 0 and record["output_elements"] > 0
            ]
            gather_rows = (
                trace["collectives"][step_name].get(groups["owner_gather"], [])
                + legacy_gather_rows
            )
            scatter_rows = [
                record
                for record in transfer_rows
                if record["input_elements"] > 0 and record["output_elements"] == 0
            ]
            gather_intervals = record_intervals(gather_rows)
            edp_intervals = expert_edp_intervals
            scatter_intervals = record_intervals(scatter_rows)
            owner_transfer_intervals = gather_intervals + scatter_intervals
            row["reshard"] = {
                "gather": interval_metrics(gather_intervals, useful_intervals),
                "expert_edp": interval_metrics(edp_intervals, useful_intervals),
                "scatter": interval_metrics(scatter_intervals, useful_intervals),
                "union": interval_metrics(
                    gather_intervals + edp_intervals + scatter_intervals,
                    useful_intervals,
                ),
            }
            row["nccl_group_overlap_ms"].update(
                {
                    "owner_transfer_with_dense_dp": intersection_duration(
                        owner_transfer_intervals,
                        backward_group_intervals["dense_dp"],
                    )
                    / 1000.0,
                    "owner_transfer_with_model_ep": intersection_duration(
                        owner_transfer_intervals,
                        backward_group_intervals["model_ep"],
                    )
                    / 1000.0,
                }
            )
            dispatch_events = [
                event
                for event in gpu_events
                if not event["is_nccl"]
                and event["stream"] == dispatch_stream
                and event["start"] < backward_end
                and event["end"] > backward_start
            ]
            by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for event in dispatch_events:
                by_category[dispatch_category(event)].append(event)
            row["dispatch_stream"] = {
                "stream": dispatch_stream,
                "event_count": len(dispatch_events),
                "sum_ms": sum(
                    min(backward_end, event["end"])
                    - max(backward_start, event["start"])
                    for event in dispatch_events
                )
                / 1000.0,
                "union_ms": union_duration(
                    clipped_intervals(dispatch_events, backward_start, backward_end)
                )
                / 1000.0,
                "categories": {
                    name: {
                        "count": len(events),
                        "sum_ms": sum(
                            float(event["end"]) - float(event["start"])
                            for event in events
                        )
                        / 1000.0,
                    }
                    for name, events in sorted(by_category.items())
                },
            }
        output[step_name] = row
    return output


def summarize_nested_steps(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Recursively summarize numeric leaves over profiler steps."""
    first = next(iter(rows.values()))

    def recurse(values: list[Any]) -> Any:
        if isinstance(values[0], dict):
            return {key: recurse([value[key] for value in values]) for key in values[0]}
        if isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
            return summarize([float(value) for value in values])
        return values[0]

    return recurse([rows[step] for step in sorted(rows)]) if first else {}


def summarize_rank_roles(
    traces: dict[str, dict[int, dict[str, Any]]]
) -> dict[str, Any]:
    """Summarize rank-local expert work by replica role."""
    roles = {
        "healthy_replica_0": ("healthy", range(0, 8)),
        "healthy_replica_1": ("healthy", range(8, 16)),
        "nep_full_ep8": ("nep", range(0, 8)),
        "nep_reduced_ep4": ("nep", range(8, 12)),
    }
    output = {}
    for role, (mode, ranks) in roles.items():
        rows = [
            row
            for rank in ranks
            for row in traces[mode][rank]["rank_local_work"].values()
        ]
        output[role] = {
            key: summarize([float(row[key]) for row in rows]) for key in rows[0]
        }
    output["per_rank"] = {
        mode: {
            str(rank): summarize_nested_steps(trace["rank_local_work"])
            for rank, trace in rank_traces.items()
        }
        for mode, rank_traces in traces.items()
    }
    return output


def aligned_collective_summary(
    traces: dict[int, dict[str, Any]], ranks: list[int], group: str
) -> dict[str, Any]:
    """Align one process group's collectives across participants."""
    step_names = sorted(traces[ranks[0]]["step_spans"])
    per_step = {}
    by_operation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    latest_rank_counts: Counter[int] = Counter()
    owner_wait_by_latest_rank_ms: Counter[int] = Counter()
    for step_name in step_names:
        rank_rows = [traces[rank]["collectives"][step_name][group] for rank in ranks]
        counts = {len(rows) for rows in rank_rows}
        if len(counts) != 1:
            raise ValueError(f"{group} {step_name}: mismatched counts {counts}")
        operations = []
        for index in range(len(rank_rows[0])):
            rows = [rank_row[index] for rank_row in rank_rows]
            signatures = {
                (row["collective"], row["dtype"], row["payload_elements"])
                for row in rows
            }
            if len(signatures) != 1:
                raise ValueError(f"{group} {step_name} operation {index}: {signatures}")
            starts = [float(row["kernel_start"]) for row in rows]
            ends = [float(row["kernel_end"]) for row in rows]
            latest_rank = ranks[
                max(range(len(ranks)), key=lambda rank_index: starts[rank_index])
            ]
            operation = {
                "payload_elements": int(rows[0]["payload_elements"]),
                "owner_residency_ms": (ends[0] - starts[0]) / 1000.0,
                "owner_participant_wait_ms": max(0.0, max(starts) - starts[0])
                / 1000.0,
                "start_spread_ms": (max(starts) - min(starts)) / 1000.0,
                "matched_service_ms": (max(ends) - max(starts)) / 1000.0,
                "latest_rank": latest_rank,
            }
            operations.append(operation)
            by_operation[index].append(operation)
            latest_rank_counts[latest_rank] += 1
            owner_wait_by_latest_rank_ms[latest_rank] += operation[
                "owner_participant_wait_ms"
            ]
        per_step[step_name] = {
            "operation_count": len(operations),
            "payload_elements": sum(row["payload_elements"] for row in operations),
            "owner_residency_ms": sum(row["owner_residency_ms"] for row in operations),
            "owner_participant_wait_ms": sum(
                row["owner_participant_wait_ms"] for row in operations
            ),
            "start_spread_ms": sum(row["start_spread_ms"] for row in operations),
            "matched_service_ms": sum(row["matched_service_ms"] for row in operations),
        }
    per_operation = {}
    for index, operations in sorted(by_operation.items()):
        per_operation[str(index)] = {
            "payload_elements": operations[0]["payload_elements"],
            "owner_residency_ms": summarize(
                [operation["owner_residency_ms"] for operation in operations]
            ),
            "owner_participant_wait_ms": summarize(
                [operation["owner_participant_wait_ms"] for operation in operations]
            ),
            "start_spread_ms": summarize(
                [operation["start_spread_ms"] for operation in operations]
            ),
            "matched_service_ms": summarize(
                [operation["matched_service_ms"] for operation in operations]
            ),
            "latest_rank_counts": dict(
                sorted(
                    Counter(
                        operation["latest_rank"] for operation in operations
                    ).items()
                )
            ),
        }
    return {
        "ranks": ranks,
        "per_step": per_step,
        "summary": summarize_nested_steps(per_step),
        "per_operation": per_operation,
        "latest_rank_counts": dict(sorted(latest_rank_counts.items())),
        "owner_wait_by_latest_rank_ms": dict(
            sorted(owner_wait_by_latest_rank_ms.items())
        ),
    }


def model_ep_milestones(
    traces: dict[int, dict[str, Any]], ranks: list[int], group: str
) -> dict[str, Any]:
    """Return participant-aligned backward model-EP completion milestones."""
    per_ordinal: dict[int, list[float]] = defaultdict(list)
    for step_name in sorted(traces[ranks[0]]["step_spans"]):
        rank_rows = [
            [
                row
                for row in traces[rank]["collectives"][step_name][group]
                if row["collective"] == "all_to_allv"
            ]
            for rank in ranks
        ]
        if {len(rows) for rows in rank_rows} != {36}:
            raise ValueError(f"{group} {step_name}: expected 36 model-EP operations")
        origin = min(rows[18]["kernel_start"] for rows in rank_rows)
        for index in range(18, 36):
            completion = max(rows[index]["kernel_end"] for rows in rank_rows)
            per_ordinal[index - 18].append((completion - origin) / 1000.0)
    return {str(index): summarize(values) for index, values in sorted(per_ordinal.items())}


def mean_delta(
    healthy: dict[str, Any], nep: dict[str, Any], path: tuple[str, ...]
) -> float:
    """Return NEP minus healthy mean at one nested summary path."""
    left: Any = healthy
    right: Any = nep
    for key in path:
        left = left[key]
        right = right[key]
    return float(right["mean"]) - float(left["mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("slurm_runs/lyris_a3b_ep8_exact_uniform_nep_ab"),
    )
    parser.add_argument("--job-id", default="2463417")
    parser.add_argument("--healthy-trace-dir", type=Path)
    parser.add_argument("--nep-trace-dir", type=Path)
    parser.add_argument("--healthy-steps", type=int, default=16)
    parser.add_argument("--nep-steps", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dirs = {
        "healthy": args.root
        / "healthy_profile"
        / "a3b_repeat14_ep8_ep8_mbs1_mb1_healthy"
        / args.job_id
        / "torch_profile",
        "nep": args.root
        / "nep_profile"
        / "a3b_repeat14_ep8_ep4_mbs1_mb1_1_proportional"
        / args.job_id
        / "torch_profile",
    }
    if args.healthy_trace_dir is not None:
        run_dirs["healthy"] = args.healthy_trace_dir
    if args.nep_trace_dir is not None:
        run_dirs["nep"] = args.nep_trace_dir
    expected_steps = {"healthy": args.healthy_steps, "nep": args.nep_steps}
    rank_counts = {"healthy": 16, "nep": 12}
    traces: dict[str, dict[int, dict[str, Any]]] = {"healthy": {}, "nep": {}}
    for mode in ("healthy", "nep"):
        for rank in range(rank_counts[mode]):
            print(f"[{mode}] parsing rank {rank}/{rank_counts[mode] - 1}", flush=True)
            traces[mode][rank] = parse_trace(
                run_dirs[mode] / f"rank-{rank}.json.gz",
                retain_gpu=rank == 0,
                expected_steps=expected_steps[mode],
            )

    owner_steps = {
        mode: owner_step_metrics(traces[mode][0], mode) for mode in ("healthy", "nep")
    }
    owner_summary = {
        mode: summarize_nested_steps(owner_steps[mode]) for mode in ("healthy", "nep")
    }
    collective_groups = {
        "model_ep": {
            "healthy": ([0, 1, 2, 3, 4, 5, 6, 7], NCCL_GROUPS["healthy"]["model_ep"]),
            "nep": ([0, 1, 2, 3, 4, 5, 6, 7], NCCL_GROUPS["nep"]["model_ep"]),
        },
        "expert_edp": {
            "healthy": ([0, 8], NCCL_GROUPS["healthy"]["expert_edp"]),
            "nep": ([0, 8], NCCL_GROUPS["nep"]["expert_edp"]),
        },
        "dense_dp": {
            "healthy": (
                [0, 2, 4, 6, 8, 10, 12, 14],
                NCCL_GROUPS["healthy"]["dense_dp"],
            ),
            "nep": ([0, 2, 4, 6, 8, 10], NCCL_GROUPS["nep"]["dense_dp"]),
        },
        "tp": {
            "healthy": ([0, 1], NCCL_GROUPS["healthy"]["tp"]),
            "nep": ([0, 1], NCCL_GROUPS["nep"]["tp"]),
        },
    }
    aligned = {
        name: {
            mode: aligned_collective_summary(traces[mode], *collective_groups[name][mode])
            for mode in ("healthy", "nep")
        }
        for name in collective_groups
    }
    aligned["owner_transfer"] = {
        "nep": aligned_collective_summary(
            traces["nep"], [0, 4], NCCL_GROUPS["nep"]["owner_transfer"]
        )
    }
    first_nep_step = next(iter(traces["nep"][0]["step_spans"]))
    first_nep_groups = traces["nep"][0]["collectives"][first_nep_step]
    if NCCL_GROUPS["nep"]["owner_gather"] in first_nep_groups:
        aligned["owner_gather"] = {
            "nep": aligned_collective_summary(
                traces["nep"], [0, 4], NCCL_GROUPS["nep"]["owner_gather"]
            )
        }

    milestones = {
        "healthy": model_ep_milestones(
            traces["healthy"], list(range(8)), NCCL_GROUPS["healthy"]["model_ep"]
        ),
        "nep": model_ep_milestones(
            traces["nep"], list(range(8)), NCCL_GROUPS["nep"]["model_ep"]
        ),
    }
    milestone_delta = {
        ordinal: float(milestones["nep"][ordinal]["mean"])
        - float(milestones["healthy"][ordinal]["mean"])
        for ordinal in milestones["healthy"]
    }

    backward_component_deltas = {
        component: mean_delta(owner_summary["healthy"], owner_summary["nep"], ("backward", component))
        for component in (
            "span_ms",
            "non_nccl_union_ms",
            "nccl_union_ms",
            "nccl_non_nccl_overlap_ms",
            "gpu_idle_ms",
        )
    }
    reconstructed = (
        backward_component_deltas["non_nccl_union_ms"]
        + backward_component_deltas["nccl_union_ms"]
        - backward_component_deltas["nccl_non_nccl_overlap_ms"]
        + backward_component_deltas["gpu_idle_ms"]
    )
    backward_component_deltas["reconstructed_span_delta_ms"] = reconstructed
    backward_component_deltas["reconstruction_error_ms"] = (
        reconstructed - backward_component_deltas["span_ms"]
    )

    output = {
        "job_id": args.job_id,
        "run_dirs": {mode: str(path) for mode, path in run_dirs.items()},
        "owner_step_summary": owner_summary,
        "owner_step_rows": owner_steps,
        "backward_component_deltas_ms": backward_component_deltas,
        "participant_aligned_collectives": aligned,
        "model_ep_backward_milestones_ms": milestones,
        "model_ep_backward_milestone_delta_ms": milestone_delta,
        "rank_local_expert_work": summarize_rank_roles(traces),
    }
    rendered = json.dumps(output, indent=2, sort_keys=True)
    output_path = args.output or args.root / f"slowdown_analysis_{args.job_id}.json"
    output_path.write_text(rendered + "\n")
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
