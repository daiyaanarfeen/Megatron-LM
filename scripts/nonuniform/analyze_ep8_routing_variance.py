#!/usr/bin/env python3
"""Compare healthy EP8 timing and model-EP readiness with routing balance on/off."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_ep8_baseline_variance import (
    build_trace_index,
    event_end,
    grouped_launch_gap,
    is_nccl,
    merge_intervals,
    pearson,
    read_timings,
    summarize_values,
    union_duration,
)


HIDDEN_SIZE = 2688
MODEL_EP_DESCRIPTION = "EXPERT_MODEL_PARALLEL_GROUP"
NATIVE_DDP_DESCRIPTIONS = {
    "DATA_PARALLEL_GROUP_WITH_CP",
    "EXPERT_DATA_PARALLEL_GROUP",
}
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
EP_GROUPS = (tuple(range(0, 8)), tuple(range(8, 16)))
PHASE_NAMES = ("dispatch", "metadata", "combine")


def find_run_dir(root: Path, label: str, job_id: str) -> Path:
    """Find the unique per-case run directory."""
    candidates = [
        path
        for path in (root / label).glob(f"**/{job_id}")
        if path.is_dir() and (path / f"driver_{job_id}.log").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(f"{label}: expected one run directory, found {candidates}")
    return candidates[0]


def clipped_intervals(
    events: list[dict[str, Any]], start: float, end: float
) -> list[tuple[float, float]]:
    """Clip trace events to a device-time window."""
    intervals = []
    for event in events:
        event_start = max(start, float(event["ts"]))
        event_end_time = min(end, event_end(event))
        if event_end_time > event_start:
            intervals.append((event_start, event_end_time))
    return intervals


def window_metrics(
    gpu_events: list[dict[str, Any]], start: float, end: float
) -> dict[str, float | int]:
    """Summarize GPU work between two model-EP operations."""
    if end <= start:
        return {
            "span_us": max(0.0, end - start),
            "non_nccl_us": 0.0,
            "other_nccl_us": 0.0,
            "gpu_idle_us": 0.0,
            "expert_gemm_sum_us": 0.0,
            "expert_gemm_count": 0,
        }
    selected = [
        event
        for event in gpu_events
        if float(event["ts"]) < end and event_end(event) > start
    ]
    non_nccl = [event for event in selected if not is_nccl(event)]
    other_nccl = [event for event in selected if is_nccl(event)]
    active_us = union_duration(merge_intervals(clipped_intervals(selected, start, end)))
    expert_gemms = [
        event
        for event in non_nccl
        if str(event.get("name", "")).startswith(("nvjet", "cutlass"))
    ]
    return {
        "span_us": end - start,
        "non_nccl_us": union_duration(
            merge_intervals(clipped_intervals(non_nccl, start, end))
        ),
        "other_nccl_us": union_duration(
            merge_intervals(clipped_intervals(other_nccl, start, end))
        ),
        "gpu_idle_us": max(0.0, end - start - active_us),
        "expert_gemm_sum_us": sum(
            min(end, event_end(event)) - max(start, float(event["ts"]))
            for event in expert_gemms
        ),
        "expert_gemm_count": len(expert_gemms),
    }


def gpu_step_metrics(
    gpu_events: list[dict[str, Any]], start: float, end: float
) -> dict[str, Any]:
    """Summarize GPU residency and NCCL process groups for one rank-step."""
    nccl_events = [event for event in gpu_events if is_nccl(event)]
    non_nccl_events = [event for event in gpu_events if not is_nccl(event)]
    nccl_intervals = merge_intervals(clipped_intervals(nccl_events, start, end))
    non_nccl_intervals = merge_intervals(
        clipped_intervals(non_nccl_events, start, end)
    )
    active_intervals = merge_intervals(
        clipped_intervals(gpu_events, start, end)
    )
    by_group: dict[str, list[dict[str, Any]]] = {}
    by_group_collective: dict[str, list[dict[str, Any]]] = {}
    for event in nccl_events:
        args = event.get("args", {})
        description = str(args.get("Process Group Description") or "unlabeled")
        collective = str(args.get("Collective name") or "unknown")
        dtype = str(args.get("dtype") or "unknown")
        by_group.setdefault(description, []).append(event)
        by_group_collective.setdefault(
            f"{description}/{collective}/{dtype}", []
        ).append(event)
    return {
        "nccl_us": union_duration(nccl_intervals),
        "non_nccl_us": union_duration(non_nccl_intervals),
        "nccl_non_nccl_overlap_us": max(
            0.0,
            union_duration(nccl_intervals)
            + union_duration(non_nccl_intervals)
            - union_duration(active_intervals),
        ),
        "gpu_idle_us": max(0.0, end - start - union_duration(active_intervals)),
        "nccl_by_process_group_us": {
            key: union_duration(
                merge_intervals(clipped_intervals(events, start, end))
            )
            for key, events in sorted(by_group.items())
        },
        "nccl_by_process_group_collective_us": {
            key: union_duration(
                merge_intervals(clipped_intervals(events, start, end))
            )
            for key, events in sorted(by_group_collective.items())
        },
    }


def trace_step_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Extract host submission, device execution, and local-work metrics per step."""
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    index = build_trace_index(events)
    events_by_thread: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for event in events:
        events_by_thread.setdefault((event.get("pid"), event.get("tid")), []).append(
            event
        )
    thread_starts: dict[tuple[Any, Any], list[float]] = {}
    for thread, thread_events in events_by_thread.items():
        thread_events.sort(key=lambda event: float(event.get("ts", 0.0)))
        thread_starts[thread] = [float(event.get("ts", 0.0)) for event in thread_events]
    gpu_events = [
        event for event in events if event.get("cat") in GPU_CATEGORIES
    ]
    gpu_by_external: dict[Any, list[dict[str, Any]]] = {}
    driver_by_correlation: dict[Any, list[dict[str, Any]]] = {}
    for event in gpu_events:
        gpu_by_external.setdefault(event.get("args", {}).get("External id"), []).append(
            event
        )
    for event in events:
        if event.get("cat") == "cuda_driver" and "cuLaunchKernel" in str(
            event.get("name", "")
        ):
            driver_by_correlation.setdefault(
                event.get("args", {}).get("correlation"), []
            ).append(event)

    steps = sorted(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and str(event.get("name", "")).startswith("ProfilerStep#")
        ),
        key=lambda event: float(event["ts"]),
    )
    rows: dict[str, dict[str, Any]] = {}
    for step in steps:
        start = float(step["ts"])
        end = event_end(step)
        records = sorted(
            (
                event
                for event in events
                if event.get("name") == "record_param_comms"
                and start <= float(event.get("ts", 0.0)) < end
                and event.get("args", {}).get("Process Group Description")
                == MODEL_EP_DESCRIPTION
                and event.get("args", {}).get("Collective name") == "all_to_allv"
            ),
            key=lambda event: float(event["ts"]),
        )
        operations = []
        for record in records:
            external_id = record.get("args", {}).get("External id")
            kernels = [
                event
                for event in gpu_by_external.get(external_id, [])
                if is_nccl(event)
                and event.get("args", {}).get("Process Group Description")
                == MODEL_EP_DESCRIPTION
            ]
            if len(kernels) != 1:
                raise ValueError(
                    f"{path.name} {step['name']} external {external_id}: "
                    f"expected one model-EP kernel, found {len(kernels)}"
                )
            kernel = kernels[0]
            launches = driver_by_correlation.get(
                kernel.get("args", {}).get("correlation"), []
            )
            if len(launches) != 1:
                raise ValueError(
                    f"{path.name} {step['name']} external {external_id}: "
                    f"expected one kernel launch, found {len(launches)}"
                )
            launch = launches[0]
            operations.append(
                {
                    "host_pid": record.get("pid"),
                    "host_tid": record.get("tid"),
                    "host_start_us": float(record["ts"]),
                    "host_end_us": event_end(record),
                    "launch_start_us": float(launch["ts"]),
                    "launch_end_us": event_end(launch),
                    "device_start_us": float(kernel["ts"]),
                    "device_end_us": event_end(kernel),
                    "device_duration_us": float(kernel.get("dur", 0.0)),
                    "in_elements": int(record.get("args", {}).get("In msg nelems", 0)),
                    "out_elements": int(record.get("args", {}).get("Out msg nelems", 0)),
                }
            )
        if len(operations) % 3:
            raise ValueError(
                f"{path.name} {step['name']}: expected operation triplets, "
                f"found {len(operations)}"
            )
        for operation_index, operation in enumerate(operations):
            gap_start = (
                float(operations[operation_index - 1]["host_end_us"])
                if operation_index
                else start
            )
            gap_end = float(operation["host_start_us"])
            thread = (operation["host_pid"], operation["host_tid"])
            thread_events = events_by_thread[thread]
            starts = thread_starts[thread]
            gap_events = thread_events[
                bisect_left(starts, gap_start) : bisect_left(starts, gap_end)
            ]
            grouped_forward = [
                event
                for event in gap_events
                if event.get("cat") == "cpu_op"
                and event.get("name") == "_GroupedLinear"
            ]
            grouped_backward = [
                event
                for event in gap_events
                if event.get("cat") == "cpu_op"
                and event.get("name") == "_GroupedLinearBackward"
            ]
            grouped = grouped_forward + grouped_backward
            stream_syncs = [
                event
                for event in gap_events
                if event.get("cat") == "cuda_runtime"
                and event.get("name") == "cudaStreamSynchronize"
            ]
            check_grads = [
                event
                for event in gap_events
                if event.get("cat") == "python_function"
                and "param_and_grad_buffer.py(302): check_grads"
                in str(event.get("name", ""))
            ]
            grouped_gaps = [grouped_launch_gap(index, scope) for scope in grouped]
            grouped_forward_gaps = [
                grouped_launch_gap(index, scope) for scope in grouped_forward
            ]
            grouped_backward_gaps = [
                grouped_launch_gap(index, scope) for scope in grouped_backward
            ]
            operation.update(
                {
                    "pre_host_gap_us": max(0.0, gap_end - gap_start),
                    "pre_grouped_cpu_sum_us": sum(
                        float(event.get("dur", 0.0)) for event in grouped
                    ),
                    "pre_grouped_cpu_max_us": max(
                        (float(event.get("dur", 0.0)) for event in grouped),
                        default=0.0,
                    ),
                    "pre_grouped_launch_gap_us": max(
                        (float(gap["duration_us"]) for gap in grouped_gaps),
                        default=0.0,
                    ),
                    "pre_grouped_forward_cpu_max_us": max(
                        (
                            float(event.get("dur", 0.0))
                            for event in grouped_forward
                        ),
                        default=0.0,
                    ),
                    "pre_grouped_forward_launch_gap_us": max(
                        (
                            float(gap["duration_us"])
                            for gap in grouped_forward_gaps
                        ),
                        default=0.0,
                    ),
                    "pre_grouped_backward_cpu_max_us": max(
                        (
                            float(event.get("dur", 0.0))
                            for event in grouped_backward
                        ),
                        default=0.0,
                    ),
                    "pre_grouped_backward_launch_gap_us": max(
                        (
                            float(gap["duration_us"])
                            for gap in grouped_backward_gaps
                        ),
                        default=0.0,
                    ),
                    "pre_stream_sync_sum_us": sum(
                        float(event.get("dur", 0.0)) for event in stream_syncs
                    ),
                    "pre_check_grads_sum_us": sum(
                        float(event.get("dur", 0.0)) for event in check_grads
                    ),
                }
            )

        native_ddp_records = sorted(
            (
                event
                for event in events
                if event.get("name") == "record_param_comms"
                and start <= float(event.get("ts", 0.0)) < end
                and event.get("args", {}).get("Process Group Description")
                in NATIVE_DDP_DESCRIPTIONS
                and event.get("args", {}).get("Collective name")
                == "allreduce_coalesced"
                and event.get("args", {}).get("dtype") == "Float"
            ),
            key=lambda event: float(event["ts"]),
        )
        native_ddp_operations = []
        native_ddp_ordinals: Counter[tuple[Any, ...]] = Counter()
        for record in native_ddp_records:
            args = record.get("args", {})
            ranks = tuple(int(rank) for rank in json.loads(args["Process Group Ranks"]))
            alignment_base = (
                str(args["Process Group Description"]),
                ranks,
                str(args["Collective name"]),
                str(args["dtype"]),
                int(args.get("In msg nelems", 0)),
                int(args.get("Out msg nelems", 0)),
            )
            ordinal = native_ddp_ordinals[alignment_base]
            native_ddp_ordinals[alignment_base] += 1
            external_id = args.get("External id")
            kernels = [
                event
                for event in gpu_by_external.get(external_id, [])
                if is_nccl(event)
                and event.get("args", {}).get("Process Group Description")
                == args["Process Group Description"]
            ]
            if len(kernels) != 1:
                raise ValueError(
                    f"{path.name} {step['name']} external {external_id}: "
                    f"expected one native DDP kernel, found {len(kernels)}"
                )
            kernel = kernels[0]
            launches = driver_by_correlation.get(
                kernel.get("args", {}).get("correlation"), []
            )
            if len(launches) != 1:
                raise ValueError(
                    f"{path.name} {step['name']} external {external_id}: "
                    f"expected one native DDP kernel launch, found {len(launches)}"
                )
            launch = launches[0]
            native_ddp_operations.append(
                {
                    "description": alignment_base[0],
                    "ranks": ranks,
                    "collective": alignment_base[2],
                    "dtype": alignment_base[3],
                    "in_elements": alignment_base[4],
                    "out_elements": alignment_base[5],
                    "ordinal": ordinal,
                    "host_start_us": float(record["ts"]),
                    "host_end_us": event_end(record),
                    "launch_start_us": float(launch["ts"]),
                    "launch_end_us": event_end(launch),
                    "device_start_us": float(kernel["ts"]),
                    "device_end_us": event_end(kernel),
                    "device_duration_us": float(kernel.get("dur", 0.0)),
                }
            )

        step_gpu_events = [
            event
            for event in gpu_events
            if float(event["ts"]) < end and event_end(event) > start
        ]
        expert_windows = {}
        for base in range(0, len(operations), 3):
            metadata = operations[base + 1]
            combine = operations[base + 2]
            expert_windows[base // 3] = window_metrics(
                step_gpu_events,
                float(metadata["device_end_us"]),
                float(combine["device_start_us"]),
            )
        grouped_scopes = [
            scope
            for scope in index["grouped_scopes"]
            if start <= float(scope.get("ts", 0.0)) < end
        ]
        largest_grouped_gap = max(
            (grouped_launch_gap(index, scope) for scope in grouped_scopes),
            key=lambda gap: float(gap["duration_us"]),
            default={"duration_us": 0.0},
        )
        rows[str(step["name"])] = {
            "step_start_us": start,
            "step_end_us": end,
            "step_duration_us": float(step.get("dur", 0.0)),
            "operations": operations,
            "native_ddp_operations": native_ddp_operations,
            "gpu_metrics": gpu_step_metrics(step_gpu_events, start, end),
            "expert_windows": expert_windows,
            "grouped_launch_gap_us": float(largest_grouped_gap["duration_us"]),
        }
    return rows


def spread(values: list[float]) -> float:
    """Return max minus min."""
    return max(values) - min(values)


def coefficient_of_variation(values: list[float]) -> float:
    """Return population CV, or zero for a zero mean."""
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def align_profile(trace_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Load all 16 ranks and align profile-step rows."""
    aligned: dict[str, dict[int, dict[str, Any]]] = {}
    paths = sorted(trace_dir.glob("rank-*.json.gz"))
    if len(paths) != 16:
        raise ValueError(f"{trace_dir}: expected 16 traces, found {len(paths)}")
    for path in paths:
        rank = int(path.name.removeprefix("rank-").removesuffix(".json.gz"))
        for step_name, row in trace_step_rows(path).items():
            aligned.setdefault(step_name, {})[rank] = row
    for step_name, rows in aligned.items():
        if set(rows) != set(range(16)):
            raise ValueError(f"{step_name}: incomplete ranks {sorted(rows)}")
        counts = {len(row["operations"]) for row in rows.values()}
        if len(counts) != 1:
            raise ValueError(f"{step_name}: mismatched model-EP counts {counts}")
    return dict(
        sorted(aligned.items(), key=lambda item: int(item[0].partition("#")[2]))
    )


def operation_group_metrics(
    rows: dict[int, dict[str, Any]], group_index: int, operation: int
) -> dict[str, Any]:
    """Decompose one aligned model-EP operation's readiness and service."""
    ranks = EP_GROUPS[group_index]
    values = {rank: rows[rank]["operations"][operation] for rank in ranks}
    starts = {rank: float(values[rank]["device_start_us"]) for rank in ranks}
    ends = {rank: float(values[rank]["device_end_us"]) for rank in ranks}
    launch_ends = {rank: float(values[rank]["launch_end_us"]) for rank in ranks}
    queues = {rank: starts[rank] - launch_ends[rank] for rank in ranks}
    latest_rank = max(ranks, key=starts.get)
    earliest_rank = min(ranks, key=starts.get)
    host_delta = launch_ends[latest_rank] - launch_ends[earliest_rank]
    queue_delta = queues[latest_rank] - queues[earliest_rank]
    start_spread = starts[latest_rank] - starts[earliest_rank]
    if abs((host_delta + queue_delta) - start_spread) > 0.01:
        raise ValueError("host/queue decomposition does not reconstruct device skew")

    if operation:
        previous_ends = {
            rank: float(rows[rank]["operations"][operation - 1]["device_end_us"])
            for rank in ranks
        }
    else:
        previous_ends = {
            rank: float(rows[rank]["step_start_us"]) for rank in ranks
        }
    previous_spread = spread(list(previous_ends.values()))
    phase = PHASE_NAMES[operation % 3]
    triplet = operation // 3
    triplet_count = len(rows[ranks[0]]["operations"]) // 3
    direction = "forward" if triplet < triplet_count // 2 else "backward"
    result = {
        "group": group_index,
        "operation": operation,
        "triplet": triplet,
        "direction": direction,
        "phase": phase,
        "host_submit_spread_us": spread(
            [float(values[rank]["host_start_us"]) for rank in ranks]
        ),
        "host_launch_spread_us": spread(list(launch_ends.values())),
        "device_start_spread_us": start_spread,
        "device_completion_spread_us": spread(list(ends.values())),
        "matched_service_us": max(ends.values()) - max(starts.values()),
        "previous_completion_spread_us": previous_spread,
        "skew_injection_us": start_spread - previous_spread,
        "latest_device_rank": latest_rank,
        "earliest_device_rank": earliest_rank,
        "late_vs_early_host_us": host_delta,
        "late_vs_early_queue_us": queue_delta,
        "max_launch_to_device_us": max(queues.values()),
        "launch_to_device_spread_us": spread(list(queues.values())),
    }
    for metric in (
        "pre_host_gap_us",
        "pre_grouped_cpu_sum_us",
        "pre_grouped_cpu_max_us",
        "pre_grouped_launch_gap_us",
        "pre_grouped_forward_cpu_max_us",
        "pre_grouped_forward_launch_gap_us",
        "pre_grouped_backward_cpu_max_us",
        "pre_grouped_backward_launch_gap_us",
        "pre_stream_sync_sum_us",
        "pre_check_grads_sum_us",
    ):
        metric_values = [float(values[rank][metric]) for rank in ranks]
        result[f"{metric}_spread"] = spread(metric_values)
        result[f"late_vs_early_{metric}"] = (
            float(values[latest_rank][metric]) - float(values[earliest_rank][metric])
        )
    if phase == "dispatch":
        loads = [
            float(values[rank]["out_elements"]) / HIDDEN_SIZE for rank in ranks
        ]
        result.update(
            {
                "routing_cv": coefficient_of_variation(loads),
                "routing_max_over_mean": max(loads) / statistics.mean(loads),
                "routing_token_spread": spread(loads),
            }
        )
    if phase == "combine":
        layer = triplet
        windows = {rank: rows[rank]["expert_windows"][layer] for rank in ranks}
        for metric in (
            "span_us",
            "non_nccl_us",
            "other_nccl_us",
            "gpu_idle_us",
            "expert_gemm_sum_us",
            "expert_gemm_count",
        ):
            result_key = metric if metric.startswith("expert_") else f"expert_{metric}"
            result[f"{result_key}_spread"] = spread(
                [float(windows[rank][metric]) for rank in ranks]
            )
        dispatch = {
            rank: rows[rank]["operations"][operation - 2] for rank in ranks
        }
        loads = [
            float(dispatch[rank]["out_elements"]) / HIDDEN_SIZE for rank in ranks
        ]
        expert_spans = [float(windows[rank]["span_us"]) for rank in ranks]
        expert_non_nccl = [float(windows[rank]["non_nccl_us"]) for rank in ranks]
        result["routing_cv"] = coefficient_of_variation(loads)
        result["routing_max_over_mean"] = max(loads) / statistics.mean(loads)
        result["load_vs_expert_span_correlation"] = pearson(loads, expert_spans)
        result["load_vs_expert_non_nccl_correlation"] = pearson(
            loads, expert_non_nccl
        )
    return result


def native_ddp_alignment_key(operation: dict[str, Any]) -> tuple[Any, ...]:
    """Return a cross-rank key for one native Megatron DDP collective."""
    return (
        operation["description"],
        tuple(operation["ranks"]),
        operation["collective"],
        operation["dtype"],
        operation["in_elements"],
        operation["out_elements"],
        operation["ordinal"],
    )


def native_ddp_group_metrics(
    values: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Separate participant readiness from matched native DDP service."""
    first = next(iter(values.values()))
    ranks = tuple(int(rank) for rank in first["ranks"])
    if set(values) != set(ranks):
        raise ValueError(
            f"{first['description']} expected participants {ranks}, got {sorted(values)}"
        )
    starts = {rank: float(values[rank]["device_start_us"]) for rank in ranks}
    ends = {rank: float(values[rank]["device_end_us"]) for rank in ranks}
    latest_start = max(starts.values())
    return {
        "description": first["description"],
        "ranks": ranks,
        "ordinal": first["ordinal"],
        "elements": max(int(first["in_elements"]), int(first["out_elements"])),
        "device_start_spread_us": spread(list(starts.values())),
        "host_start_spread_us": spread(
            [float(values[rank]["host_start_us"]) for rank in ranks]
        ),
        "matched_service_us": max(ends.values()) - latest_start,
        "max_raw_residency_us": max(
            float(values[rank]["device_duration_us"]) for rank in ranks
        ),
        "latest_rank": max(ranks, key=starts.get),
        "earliest_rank": min(ranks, key=starts.get),
        "rank_residency_us": {
            rank: float(values[rank]["device_duration_us"]) for rank in ranks
        },
        "rank_pre_latest_wait_us": {
            rank: max(0.0, latest_start - starts[rank]) for rank in ranks
        },
    }


def summarize_native_ddp(
    operations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarize native DDP readiness, service, and per-step residency."""
    output = {}
    for description in sorted(NATIVE_DDP_DESCRIPTIONS):
        selected = [
            operation
            for operation in operations
            if operation["description"] == description
        ]
        if not selected:
            continue
        step_names = sorted(
            {str(operation["step"]) for operation in selected},
            key=lambda name: int(name.partition("#")[2]),
        )
        raw_residency_by_step = []
        pre_latest_wait_by_step = []
        matched_service_by_step = []
        for step_name in step_names:
            step_operations = [
                operation
                for operation in selected
                if operation["step"] == step_name
            ]
            rank_residency: dict[int, float] = {}
            rank_wait: dict[int, float] = {}
            group_service: dict[tuple[int, ...], float] = {}
            for operation in step_operations:
                for rank, value in operation["rank_residency_us"].items():
                    rank_residency[rank] = rank_residency.get(rank, 0.0) + float(value)
                for rank, value in operation["rank_pre_latest_wait_us"].items():
                    rank_wait[rank] = rank_wait.get(rank, 0.0) + float(value)
                group = tuple(operation["ranks"])
                group_service[group] = group_service.get(group, 0.0) + float(
                    operation["matched_service_us"]
                )
            raw_residency_by_step.append(max(rank_residency.values()))
            pre_latest_wait_by_step.append(max(rank_wait.values()))
            matched_service_by_step.append(max(group_service.values()))
        output[description] = {
            "operation_count": len(selected),
            "device_start_spread_ms": summarize_values(
                [
                    float(operation["device_start_spread_us"]) / 1000.0
                    for operation in selected
                ]
            ),
            "host_start_spread_ms": summarize_values(
                [
                    float(operation["host_start_spread_us"]) / 1000.0
                    for operation in selected
                ]
            ),
            "matched_service_ms": summarize_values(
                [
                    float(operation["matched_service_us"]) / 1000.0
                    for operation in selected
                ]
            ),
            "max_raw_residency_ms": summarize_values(
                [
                    float(operation["max_raw_residency_us"]) / 1000.0
                    for operation in selected
                ]
            ),
            "critical_rank_residency_per_step_ms": summarize_values(
                [value / 1000.0 for value in raw_residency_by_step]
            ),
            "critical_rank_pre_latest_wait_per_step_ms": summarize_values(
                [value / 1000.0 for value in pre_latest_wait_by_step]
            ),
            "max_group_matched_service_per_step_ms": summarize_values(
                [value / 1000.0 for value in matched_service_by_step]
            ),
        }
    return output


def summarize_critical_gpu(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the slowest rank's GPU residency for each profiled step."""
    process_groups = sorted(
        {
            key
            for row in rows
            for key in row["nccl_by_process_group_us"]
        }
    )
    group_collectives = sorted(
        {
            key
            for row in rows
            for key in row["nccl_by_process_group_collective_us"]
        }
    )
    return {
        "critical_rank_counts": dict(
            sorted(Counter(int(row["rank"]) for row in rows).items())
        ),
        "nccl_ms": summarize_values(
            [float(row["nccl_us"]) / 1000.0 for row in rows]
        ),
        "non_nccl_ms": summarize_values(
            [float(row["non_nccl_us"]) / 1000.0 for row in rows]
        ),
        "nccl_non_nccl_overlap_ms": summarize_values(
            [
                float(row["nccl_non_nccl_overlap_us"]) / 1000.0
                for row in rows
            ]
        ),
        "gpu_idle_ms": summarize_values(
            [float(row["gpu_idle_us"]) / 1000.0 for row in rows]
        ),
        "nccl_by_process_group_ms": {
            key: summarize_values(
                [
                    float(row["nccl_by_process_group_us"].get(key, 0.0))
                    / 1000.0
                    for row in rows
                ]
            )
            for key in process_groups
        },
        "nccl_by_process_group_collective_ms": {
            key: summarize_values(
                [
                    float(
                        row["nccl_by_process_group_collective_us"].get(key, 0.0)
                    )
                    / 1000.0
                    for row in rows
                ]
            )
            for key in group_collectives
        },
        "rows": rows,
    }


def summarize_profile(aligned: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Summarize readiness sources over all steps, replicas, and MoE operations."""
    step_rows = []
    operations = []
    native_ddp_operations = []
    critical_gpu_rows = []
    for step_name, rows in aligned.items():
        durations = [float(rows[rank]["step_duration_us"]) for rank in rows]
        operation_count = len(rows[0]["operations"])
        step_operations = [
            operation_group_metrics(rows, group_index, operation)
            for group_index in range(len(EP_GROUPS))
            for operation in range(operation_count)
        ]
        operations.extend({"step": step_name, **row} for row in step_operations)
        aligned_native_ddp: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
        for rank, row in rows.items():
            for operation in row["native_ddp_operations"]:
                aligned_native_ddp.setdefault(
                    native_ddp_alignment_key(operation), {}
                )[rank] = operation
        native_ddp_operations.extend(
            {
                "step": step_name,
                **native_ddp_group_metrics(values),
            }
            for values in aligned_native_ddp.values()
        )
        critical_rank = max(
            rows, key=lambda rank: float(rows[rank]["step_duration_us"])
        )
        critical_gpu_rows.append(
            {
                "step": step_name,
                "rank": critical_rank,
                **rows[critical_rank]["gpu_metrics"],
            }
        )
        dispatches = [
            row
            for row in step_operations
            if row["phase"] == "dispatch" and row["direction"] == "forward"
        ]
        combines = [row for row in step_operations if row["phase"] == "combine"]
        step_rows.append(
            {
                "step": step_name,
                "critical_step_us": max(durations),
                "rank_step_spread_us": spread(durations),
                "routing_cv_mean": statistics.mean(
                    float(row["routing_cv"]) for row in dispatches
                ),
                "routing_max_over_mean_max": max(
                    float(row["routing_max_over_mean"]) for row in dispatches
                ),
                "model_ep_start_spread_mean_us": statistics.mean(
                    float(row["device_start_spread_us"]) for row in step_operations
                ),
                "model_ep_service_sum_max_group_us": max(
                    sum(
                        float(row["matched_service_us"])
                        for row in step_operations
                        if row["group"] == group_index
                    )
                    for group_index in range(len(EP_GROUPS))
                ),
                "positive_skew_injection_sum_us": sum(
                    max(0.0, float(row["skew_injection_us"]))
                    for row in step_operations
                ),
                "combine_positive_skew_injection_sum_us": sum(
                    max(0.0, float(row["skew_injection_us"])) for row in combines
                ),
                "expert_span_spread_mean_us": statistics.mean(
                    float(row["expert_span_us_spread"]) for row in combines
                ),
                "expert_non_nccl_spread_mean_us": statistics.mean(
                    float(row["expert_non_nccl_us_spread"]) for row in combines
                ),
                "expert_gemm_spread_mean_us": statistics.mean(
                    float(row["expert_gemm_sum_us_spread"]) for row in combines
                ),
                "grouped_launch_gap_max_us": max(
                    float(rows[rank]["grouped_launch_gap_us"]) for rank in rows
                ),
            }
        )

    critical = [float(row["critical_step_us"]) for row in step_rows]
    step_metric_names = tuple(
        key for key in step_rows[0] if key not in {"step", "critical_step_us"}
    )
    phase_summaries = {}
    for direction in ("forward", "backward"):
        for phase in PHASE_NAMES:
            key = f"{direction}_{phase}"
            selected = [
                row
                for row in operations
                if row["direction"] == direction and row["phase"] == phase
            ]
            phase_summaries[key] = {
                "operation_count": len(selected),
                "device_start_spread_ms": summarize_values(
                    [float(row["device_start_spread_us"]) / 1000.0 for row in selected]
                ),
                "host_submit_spread_ms": summarize_values(
                    [float(row["host_submit_spread_us"]) / 1000.0 for row in selected]
                ),
                "matched_service_ms": summarize_values(
                    [float(row["matched_service_us"]) / 1000.0 for row in selected]
                ),
                "skew_injection_ms": summarize_values(
                    [float(row["skew_injection_us"]) / 1000.0 for row in selected]
                ),
                "late_vs_early_host_ms": summarize_values(
                    [float(row["late_vs_early_host_us"]) / 1000.0 for row in selected]
                ),
                "late_vs_early_queue_ms": summarize_values(
                    [float(row["late_vs_early_queue_us"]) / 1000.0 for row in selected]
                ),
                "pre_host_gap_spread_ms": summarize_values(
                    [float(row["pre_host_gap_us_spread"]) / 1000.0 for row in selected]
                ),
                "pre_grouped_cpu_spread_ms": summarize_values(
                    [
                        float(row["pre_grouped_cpu_max_us_spread"]) / 1000.0
                        for row in selected
                    ]
                ),
                "pre_grouped_launch_gap_spread_ms": summarize_values(
                    [
                        float(row["pre_grouped_launch_gap_us_spread"]) / 1000.0
                        for row in selected
                    ]
                ),
                "pre_grouped_forward_cpu_spread_ms": summarize_values(
                    [
                        float(row["pre_grouped_forward_cpu_max_us_spread"])
                        / 1000.0
                        for row in selected
                    ]
                ),
                "pre_grouped_forward_launch_gap_spread_ms": summarize_values(
                    [
                        float(
                            row["pre_grouped_forward_launch_gap_us_spread"]
                        )
                        / 1000.0
                        for row in selected
                    ]
                ),
                "pre_grouped_backward_cpu_spread_ms": summarize_values(
                    [
                        float(row["pre_grouped_backward_cpu_max_us_spread"])
                        / 1000.0
                        for row in selected
                    ]
                ),
                "pre_grouped_backward_launch_gap_spread_ms": summarize_values(
                    [
                        float(
                            row["pre_grouped_backward_launch_gap_us_spread"]
                        )
                        / 1000.0
                        for row in selected
                    ]
                ),
                "pre_stream_sync_spread_ms": summarize_values(
                    [
                        float(row["pre_stream_sync_sum_us_spread"]) / 1000.0
                        for row in selected
                    ]
                ),
                "pre_check_grads_spread_ms": summarize_values(
                    [
                        float(row["pre_check_grads_sum_us_spread"]) / 1000.0
                        for row in selected
                    ]
                ),
            }
    dispatches = [
        row
        for row in operations
        if row["phase"] == "dispatch" and row["direction"] == "forward"
    ]
    combines = [row for row in operations if row["phase"] == "combine"]
    load_vs_expert_span_correlations = [
        float(row["load_vs_expert_span_correlation"])
        for row in combines
        if row["load_vs_expert_span_correlation"] is not None
    ]
    return {
        "step_count": len(step_rows),
        "critical_step_ms": summarize_values([value / 1000.0 for value in critical]),
        "step_correlations_with_critical_time": {
            metric: pearson(
                critical, [float(row[metric]) for row in step_rows]
            )
            for metric in step_metric_names
        },
        "routing": {
            "dispatch_count": len(dispatches),
            "rank_load_cv": summarize_values(
                [float(row["routing_cv"]) for row in dispatches]
            ),
            "rank_load_max_over_mean": summarize_values(
                [float(row["routing_max_over_mean"]) for row in dispatches]
            ),
            "load_cv_vs_combine_start_spread": pearson(
                [float(row["routing_cv"]) for row in combines],
                [float(row["device_start_spread_us"]) for row in combines],
            ),
            "load_cv_vs_expert_span_spread": pearson(
                [float(row["routing_cv"]) for row in combines],
                [float(row["expert_span_us_spread"]) for row in combines],
            ),
            "load_vs_expert_span_rank_correlation": (
                summarize_values(load_vs_expert_span_correlations)
                if load_vs_expert_span_correlations
                else None
            ),
        },
        "readiness": {
            "all_device_start_spread_ms": summarize_values(
                [
                    float(row["device_start_spread_us"]) / 1000.0
                    for row in operations
                ]
            ),
            "all_host_submit_spread_ms": summarize_values(
                [
                    float(row["host_submit_spread_us"]) / 1000.0
                    for row in operations
                ]
            ),
            "all_matched_service_ms": summarize_values(
                [float(row["matched_service_us"]) / 1000.0 for row in operations]
            ),
            "device_spread_vs_host_submit_spread": pearson(
                [float(row["device_start_spread_us"]) for row in operations],
                [float(row["host_submit_spread_us"]) for row in operations],
            ),
            "device_spread_vs_launch_dependency_spread": pearson(
                [float(row["device_start_spread_us"]) for row in operations],
                [float(row["launch_to_device_spread_us"]) for row in operations],
            ),
            "device_spread_vs_pre_host_gap_spread": pearson(
                [float(row["device_start_spread_us"]) for row in operations],
                [float(row["pre_host_gap_us_spread"]) for row in operations],
            ),
            "device_spread_vs_pre_grouped_cpu_spread": pearson(
                [float(row["device_start_spread_us"]) for row in operations],
                [float(row["pre_grouped_cpu_max_us_spread"]) for row in operations],
            ),
            "device_spread_vs_pre_grouped_launch_gap_spread": pearson(
                [float(row["device_start_spread_us"]) for row in operations],
                [
                    float(row["pre_grouped_launch_gap_us_spread"])
                    for row in operations
                ],
            ),
            "device_spread_vs_pre_stream_sync_spread": pearson(
                [float(row["device_start_spread_us"]) for row in operations],
                [float(row["pre_stream_sync_sum_us_spread"]) for row in operations],
            ),
            "host_dominated_count": sum(
                abs(float(row["late_vs_early_host_us"]))
                >= abs(float(row["late_vs_early_queue_us"]))
                for row in operations
            ),
            "queue_dominated_count": sum(
                abs(float(row["late_vs_early_queue_us"]))
                > abs(float(row["late_vs_early_host_us"]))
                for row in operations
            ),
            "latest_rank_counts": dict(
                sorted(Counter(int(row["latest_device_rank"]) for row in operations).items())
            ),
            "phase": phase_summaries,
        },
        "expert": {
            "phase_span_spread_ms": summarize_values(
                [float(row["expert_span_us_spread"]) / 1000.0 for row in combines]
            ),
            "non_nccl_spread_ms": summarize_values(
                [
                    float(row["expert_non_nccl_us_spread"]) / 1000.0
                    for row in combines
                ]
            ),
            "gemm_sum_spread_ms": summarize_values(
                [
                    float(row["expert_gemm_sum_us_spread"]) / 1000.0
                    for row in combines
                ]
            ),
        },
        "native_ddp": summarize_native_ddp(native_ddp_operations),
        "critical_gpu": summarize_critical_gpu(critical_gpu_rows),
        "step_rows": step_rows,
        "operations": operations,
    }


def timing_summary(
    root: Path, job_id: str, labels: list[str], start_iteration: int
) -> dict[str, Any]:
    """Summarize profiler-free timing populations."""
    runs = []
    for label in labels:
        run_dir = find_run_dir(root, label, job_id)
        iterations, values = read_timings(
            run_dir / f"driver_{job_id}.log", start_iteration
        )
        runs.append(
            {
                "label": label,
                "iterations": iterations,
                "values_ms": values,
                "summary_ms": summarize_values(values),
            }
        )
    pooled = [value for run in runs for value in run["values_ms"]]
    run_means = [statistics.mean(run["values_ms"]) for run in runs]
    return {
        "runs": runs,
        "pooled_ms": summarize_values(pooled),
        "between_run_means_ms": summarize_values(run_means),
    }


def profile_summary(root: Path, job_id: str, label: str) -> dict[str, Any]:
    """Load and summarize one all-rank profile."""
    run_dir = find_run_dir(root, label, job_id)
    return summarize_profile(align_profile(run_dir / "torch_profile"))


def print_condition(name: str, timing: dict[str, Any], profile: dict[str, Any]) -> None:
    """Print a compact condition summary."""
    pooled = timing["pooled_ms"]
    routing = profile["routing"]
    readiness = profile["readiness"]
    expert = profile["expert"]
    print(
        f"{name}: timing={pooled['mean']:.3f}+/-{pooled['std']:.3f} ms "
        f"CV={pooled['cv_percent']:.2f}% n={pooled['n']}"
    )
    print(
        f"  routing CV={100.0 * routing['rank_load_cv']['mean']:.3f}% "
        f"max/mean={routing['rank_load_max_over_mean']['mean']:.4f}; "
        f"model-EP start spread={readiness['all_device_start_spread_ms']['mean']:.3f} ms; "
        f"service={readiness['all_matched_service_ms']['mean']:.3f} ms"
    )
    print(
        f"  expert span spread={expert['phase_span_spread_ms']['mean']:.3f} ms; "
        f"non-NCCL spread={expert['non_nccl_spread_ms']['mean']:.3f} ms; "
        f"GEMM sum spread={expert['gemm_sum_spread_ms']['mean']:.3f} ms"
    )
    print(
        f"  host/queue dominated operations="
        f"{readiness['host_dominated_count']}/{readiness['queue_dominated_count']}; "
        f"corr(start, host)={readiness['device_spread_vs_host_submit_spread']:.3f}; "
        f"corr(start, dependency)="
        f"{readiness['device_spread_vs_launch_dependency_spread']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--start-iteration", type=int, default=10)
    parser.add_argument(
        "--balanced-timing-labels",
        nargs="+",
        default=["balanced_timing_1", "balanced_timing_2"],
    )
    parser.add_argument(
        "--natural-timing-labels",
        nargs="+",
        default=["natural_timing_1", "natural_timing_2"],
    )
    parser.add_argument("--balanced-profile-label", default="balanced_profile")
    parser.add_argument("--natural-profile-label", default="natural_profile")
    parser.add_argument(
        "--balanced-detail-label", default="balanced_profile_detail"
    )
    parser.add_argument("--natural-detail-label", default="natural_profile_detail")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the timing and causal trace comparison."""
    args = parse_args()
    result = {
        "job_id": args.job_id,
        "balanced": {
            "timing": timing_summary(
                args.root,
                args.job_id,
                args.balanced_timing_labels,
                args.start_iteration,
            ),
            "profile": profile_summary(
                args.root, args.job_id, args.balanced_profile_label
            ),
            "detail": profile_summary(
                args.root, args.job_id, args.balanced_detail_label
            ),
        },
        "natural": {
            "timing": timing_summary(
                args.root,
                args.job_id,
                args.natural_timing_labels,
                args.start_iteration,
            ),
            "profile": profile_summary(
                args.root, args.job_id, args.natural_profile_label
            ),
            "detail": profile_summary(
                args.root, args.job_id, args.natural_detail_label
            ),
        },
    }
    for condition in ("balanced", "natural"):
        print_condition(
            condition, result[condition]["timing"], result[condition]["profile"]
        )
    balanced_mean = result["balanced"]["timing"]["pooled_ms"]["mean"]
    natural_mean = result["natural"]["timing"]["pooled_ms"]["mean"]
    print(
        f"natural/balanced timing={natural_mean / balanced_mean:.4f} "
        f"({100.0 * (natural_mean / balanced_mean - 1.0):+.2f}%)"
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
