#!/usr/bin/env python3
"""Extract EP8/EP4 fragmentation and routing-skew evidence from PyTorch traces."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import Counter
from pathlib import Path


STEP_NAME = "ProfilerStep#3"
HIDDEN_SIZE = 2688
TARGET_DENSE_DP_ELEMENTS = 176_160_768
ITER_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*\d+.*?"
    r"elapsed time per iteration \(ms\): (?P<ms>[0-9.]+)"
)


def load_events(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)["traceEvents"]


def profiler_step(events: list[dict[str, object]]) -> dict[str, object]:
    return next(
        event
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name") == STEP_NAME
    )


def in_window(event: dict[str, object], start: float, end: float) -> bool:
    timestamp = float(event.get("ts", 0.0))
    return start <= timestamp < end


def merge_intervals(
    events: list[dict[str, object]],
    start: float | None = None,
    end: float | None = None,
) -> list[tuple[float, float]]:
    intervals = []
    for event in events:
        event_start = float(event["ts"])
        event_end = event_start + float(event.get("dur", 0.0))
        if start is not None:
            event_start = max(start, event_start)
        if end is not None:
            event_end = min(end, event_end)
        if event_end > event_start:
            intervals.append((event_start, event_end))
    intervals.sort()

    merged: list[tuple[float, float]] = []
    for interval_start, interval_end in intervals:
        if not merged or interval_start > merged[-1][1]:
            merged.append((interval_start, interval_end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval_end))
    return merged


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def intersection_duration(
    left_events: list[dict[str, object]], right_events: list[dict[str, object]]
) -> float:
    left = merge_intervals(left_events)
    right = merge_intervals(right_events)
    left_index = 0
    right_index = 0
    total = 0.0
    while left_index < len(left) and right_index < len(right):
        total += max(
            0.0,
            min(left[left_index][1], right[right_index][1])
            - max(left[left_index][0], right[right_index][0]),
        )
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def process_group_descriptions(healthy: bool) -> tuple[str, str]:
    if healthy:
        return "EXPERT_MODEL_PARALLEL_GROUP", "DATA_PARALLEL_GROUP_WITH_CP"
    return "ep", "dp_cp"


def target_dense_dp_kernel(
    events: list[dict[str, object]], healthy: bool
) -> dict[str, object]:
    step = profiler_step(events)
    start = float(step["ts"])
    end = start + float(step["dur"])
    _, dense_dp_description = process_group_descriptions(healthy)
    collective = next(
        event
        for event in events
        if event.get("name") == "record_param_comms"
        and in_window(event, start, end)
        and event.get("args", {}).get("Process Group Description")
        == dense_dp_description
        and event.get("args", {}).get("In msg nelems")
        == TARGET_DENSE_DP_ELEMENTS
    )
    external_id = collective["args"]["External id"]
    return next(
        event
        for event in events
        if event.get("cat") == "kernel"
        and event.get("args", {}).get("External id") == external_id
    )


def dispatch_loads(events: list[dict[str, object]], healthy: bool) -> list[float]:
    step = profiler_step(events)
    start = float(step["ts"])
    end = start + float(step["dur"])
    model_ep_description, _ = process_group_descriptions(healthy)
    all_to_all = [
        event
        for event in events
        if event.get("name") == "record_param_comms"
        and in_window(event, start, end)
        and event.get("args", {}).get("Process Group Description")
        == model_ep_description
        and event.get("args", {}).get("Collective name") == "all_to_allv"
    ]
    if len(all_to_all) % 3:
        raise ValueError(
            f"Expected dispatch/metadata/combine triplets, got {len(all_to_all)}"
        )
    dispatches = all_to_all[0::3]
    return [
        float(event["args"]["Out msg nelems"]) / HIDDEN_SIZE for event in dispatches
    ]


def routing_stats(
    loads_by_rank: dict[int, list[float]], ranks: range
) -> dict[str, float]:
    event_count = {len(loads_by_rank[rank]) for rank in ranks}
    if len(event_count) != 1:
        raise ValueError(f"Ranks have different dispatch counts: {event_count}")

    per_dispatch_cv = []
    per_dispatch_max_mean = []
    for index in range(event_count.pop()):
        loads = [loads_by_rank[rank][index] for rank in ranks]
        mean = statistics.mean(loads)
        per_dispatch_cv.append(statistics.pstdev(loads) / mean)
        per_dispatch_max_mean.append(max(loads) / mean)

    aggregate_loads = [sum(loads_by_rank[rank]) for rank in ranks]
    aggregate_mean = statistics.mean(aggregate_loads)
    return {
        "mean_tokens_per_dispatch_per_rank": statistics.mean(
            load for rank in ranks for load in loads_by_rank[rank]
        ),
        "mean_per_dispatch_cv": statistics.mean(per_dispatch_cv),
        "mean_per_dispatch_max_over_mean": statistics.mean(per_dispatch_max_mean),
        "aggregate_rank_cv": statistics.pstdev(aggregate_loads) / aggregate_mean,
        "aggregate_rank_max_over_mean": max(aggregate_loads) / aggregate_mean,
    }


def rank_metrics(events: list[dict[str, object]], healthy: bool) -> dict[str, object]:
    step = profiler_step(events)
    start = float(step["ts"])
    end = start + float(step["dur"])
    target = target_dense_dp_kernel(events, healthy)
    target_start = float(target["ts"])

    gpu_events = [
        event
        for event in events
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]
    kernels = [
        event
        for event in gpu_events
        if event.get("cat") == "kernel" and in_window(event, start, end)
    ]
    model_ep_description, _ = process_group_descriptions(healthy)
    model_ep_events = [
        event
        for event in kernels
        if event.get("args", {}).get("Process Group Description")
        == model_ep_description
    ]
    nvjet = [
        event for event in kernels if str(event.get("name", "")).startswith("nvjet")
    ]
    stream_counts = Counter(event["args"]["stream"] for event in nvjet)
    expert_streams = {stream for stream, _ in stream_counts.most_common(4)}
    expert_gemms = [
        event for event in nvjet if event["args"]["stream"] in expert_streams
    ]
    nccl_events = [
        event
        for event in gpu_events
        if event.get("cat") == "kernel"
        and str(event.get("name", "")).startswith("ncclDevKernel")
    ]
    non_nccl_events = [
        event
        for event in gpu_events
        if not (
            event.get("cat") == "kernel"
            and str(event.get("name", "")).startswith("ncclDevKernel")
        )
    ]
    gpu_active_before_target = interval_duration(
        merge_intervals(gpu_events, start, target_start)
    )
    cuda_stream_waits = [
        event
        for event in events
        if event.get("cat") == "cuda_runtime"
        and event.get("name") == "cudaStreamWaitEvent"
        and in_window(event, start, end)
    ]
    graph_launches = [
        event
        for event in events
        if event.get("cat") == "cuda_runtime"
        and event.get("name") == "cudaGraphLaunch"
        and in_window(event, start, end)
    ]
    return {
        "step_start_us": start,
        "step_ms": float(step["dur"]) / 1000.0,
        "target_start_us": target_start,
        "target_duration_ms": float(target["dur"]) / 1000.0,
        "target_end_us": target_start + float(target["dur"]),
        "pre_target_span_ms": (target_start - start) / 1000.0,
        "pre_target_gpu_active_ms": gpu_active_before_target / 1000.0,
        "pre_target_gpu_inactive_ms": (
            target_start - start - gpu_active_before_target
        )
        / 1000.0,
        "pre_target_non_nccl_union_ms": interval_duration(
            merge_intervals(non_nccl_events, start, target_start)
        )
        / 1000.0,
        "pre_target_nccl_union_ms": interval_duration(
            merge_intervals(nccl_events, start, target_start)
        )
        / 1000.0,
        "model_ep_residency_ms": interval_duration(merge_intervals(model_ep_events))
        / 1000.0,
        "expert_gemm_count": len(expert_gemms),
        "expert_gemm_total_ms": sum(float(event["dur"]) for event in expert_gemms)
        / 1000.0,
        "expert_gemm_mean_us": statistics.mean(
            float(event["dur"]) for event in expert_gemms
        ),
        "cuda_stream_wait_count": len(cuda_stream_waits),
        "cuda_stream_wait_api_ms": sum(
            float(event["dur"]) for event in cuda_stream_waits
        )
        / 1000.0,
        "cuda_graph_launch_count": len(graph_launches),
        "cuda_graph_launch_api_ms": sum(
            float(event["dur"]) for event in graph_launches
        )
        / 1000.0,
        "dispatch_loads": dispatch_loads(events, healthy),
        "target_kernel": {
            "name": target["name"],
            "grid": target["args"].get("grid"),
            "block": target["args"].get("block"),
            "blocks_per_sm": target["args"].get("blocks per SM"),
            "stream": target["args"].get("stream"),
            "process_group_ranks": target["args"].get("Process Group Ranks"),
        },
    }


def mean_fields(
    rows: list[dict[str, object]], fields: tuple[str, ...]
) -> dict[str, float]:
    return {
        field: statistics.mean(float(row[field]) for row in rows) for field in fields
    }


def all_rank_summary(
    trace_dir: Path, healthy: bool, rank_count: int
) -> dict[str, object]:
    rows = []
    loads = {}
    for rank in range(rank_count):
        row = rank_metrics(load_events(trace_dir / f"rank-{rank}.json.gz"), healthy)
        rows.append(row)
        loads[rank] = row.pop("dispatch_loads")

    rank_zero_start = float(rows[0]["step_start_us"])
    for row in rows:
        row["target_start_from_rank0_step_ms"] = (
            float(row["target_start_us"]) - rank_zero_start
        ) / 1000.0
        row["target_end_from_rank0_step_ms"] = (
            float(row["target_end_us"]) - rank_zero_start
        ) / 1000.0

    fields = (
        "expert_gemm_count",
        "expert_gemm_total_ms",
        "expert_gemm_mean_us",
        "pre_target_span_ms",
        "pre_target_gpu_active_ms",
        "pre_target_gpu_inactive_ms",
        "pre_target_non_nccl_union_ms",
        "pre_target_nccl_union_ms",
        "model_ep_residency_ms",
        "cuda_stream_wait_count",
        "cuda_stream_wait_api_ms",
        "cuda_graph_launch_count",
        "cuda_graph_launch_api_ms",
        "target_start_from_rank0_step_ms",
        "target_duration_ms",
        "target_end_from_rank0_step_ms",
    )
    if healthy:
        groups = {"replica_0": range(0, 8), "replica_1": range(8, 16)}
    else:
        groups = {"full_ep8": range(0, 8), "reduced_ep4": range(8, 12)}
    return {
        "groups": {
            name: {
                **mean_fields([rows[rank] for rank in ranks], fields),
                "routing": routing_stats(loads, ranks),
            }
            for name, ranks in groups.items()
        },
        "rank0_target_kernel": rows[0]["target_kernel"],
    }


def rank_zero_residency(trace_dir: Path, healthy: bool) -> dict[str, float]:
    events = load_events(trace_dir / "rank-0.json.gz")
    step = profiler_step(events)
    start = float(step["ts"])
    end = start + float(step["dur"])
    model_ep_description, dense_dp_description = process_group_descriptions(healthy)
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel" and in_window(event, start, end)
    ]
    model_ep = [
        event
        for event in kernels
        if event.get("args", {}).get("Process Group Description")
        == model_ep_description
    ]
    dense_dp = [
        event
        for event in kernels
        if event.get("args", {}).get("Process Group Description")
        == dense_dp_description
    ]
    nccl = [
        event
        for event in kernels
        if str(event.get("name", "")).startswith("ncclDevKernel")
    ]
    non_nccl = [
        event
        for event in kernels
        if not str(event.get("name", "")).startswith("ncclDevKernel")
    ]
    target = target_dense_dp_kernel(events, healthy)
    return {
        "step_ms": float(step["dur"]) / 1000.0,
        "model_ep_kernel_count": len(model_ep),
        "model_ep_residency_ms": interval_duration(merge_intervals(model_ep))
        / 1000.0,
        "dense_dp_residency_ms": interval_duration(merge_intervals(dense_dp))
        / 1000.0,
        "nccl_residency_ms": interval_duration(merge_intervals(nccl)) / 1000.0,
        "non_nccl_kernel_residency_ms": interval_duration(merge_intervals(non_nccl))
        / 1000.0,
        "dense_dp_model_ep_overlap_ms": intersection_duration(dense_dp, model_ep)
        / 1000.0,
        "target_dense_dp_model_ep_overlap_ms": intersection_duration(
            [target], model_ep
        )
        / 1000.0,
    }


def model_ep_payload(trace_dir: Path, healthy: bool) -> dict[str, object]:
    model_ep_description, _ = process_group_descriptions(healthy)
    by_dtype: dict[str, list[int]] = {}
    record_count = 0
    for rank in range(8):
        events = load_events(trace_dir / f"rank-{rank}.json.gz")
        step = profiler_step(events)
        start = float(step["ts"])
        end = start + float(step["dur"])
        records = [
            event
            for event in events
            if event.get("name") == "record_param_comms"
            and in_window(event, start, end)
            and event.get("args", {}).get("Process Group Description")
            == model_ep_description
        ]
        record_count += len(records)
        for event in records:
            args = event["args"]
            totals = by_dtype.setdefault(str(args["dtype"]), [0, 0, 0])
            totals[0] += 1
            totals[1] += int(args["In msg nelems"])
            totals[2] += int(args["Out msg nelems"])
    return {
        "record_count": record_count,
        "by_dtype": {
            dtype: {
                "record_count": totals[0],
                "input_elements": totals[1],
                "output_elements": totals[2],
            }
            for dtype, totals in by_dtype.items()
        },
    }


def clean_iteration_mean(log_path: Path, first_iteration: int = 6) -> float:
    samples = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = ITER_RE.search(line)
        if match and int(match.group("iteration")) >= first_iteration:
            samples.append(float(match.group("ms")))
    if not samples:
        raise ValueError(f"No iterations >= {first_iteration} in {log_path}")
    return statistics.mean(samples)


def case_paths(root: Path, job_id: str) -> dict[str, Path]:
    return {
        "balanced_healthy": root
        / "balanced"
        / "a3b_repeat14_ep8_ep8_mbs2_mb7_healthy"
        / job_id,
        "balanced_nep": root
        / "balanced"
        / "a3b_repeat14_ep8_ep4_mbs2_mb7_7_proportional"
        / job_id,
        "biased_healthy": root
        / "biased"
        / "a3b_repeat14_ep8_ep8_mbs2_mb7_healthy"
        / job_id,
        "biased_nep": root
        / "biased"
        / "a3b_repeat14_ep8_ep4_mbs2_mb7_7_proportional"
        / job_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("slurm_runs/lyris_a3b_ep8_ep4_router_balance"),
    )
    parser.add_argument("--job-id", default="2438661")
    args = parser.parse_args()

    paths = case_paths(args.root, args.job_id)
    summaries = {
        name: all_rank_summary(
            path / "torch_profile",
            healthy=name.endswith("healthy"),
            rank_count=16 if name.endswith("healthy") else 12,
        )
        for name, path in paths.items()
        if name in {"balanced_nep", "biased_nep"}
    }
    residency = {
        name: rank_zero_residency(
            path / "torch_profile", healthy=name.endswith("healthy")
        )
        for name, path in paths.items()
    }
    payloads = {
        name: model_ep_payload(path / "torch_profile", healthy=name.endswith("healthy"))
        for name, path in paths.items()
    }
    timings = {
        name: clean_iteration_mean(path / f"driver_{args.job_id}.log")
        for name, path in paths.items()
    }
    timings["balanced_owner_parity_percent"] = (
        100.0 * timings["balanced_healthy"] / timings["balanced_nep"]
    )
    timings["biased_owner_parity_percent"] = (
        100.0 * timings["biased_healthy"] / timings["biased_nep"]
    )

    print(
        json.dumps(
            {
                "job_id": args.job_id,
                "step": STEP_NAME,
                "timings": timings,
                "all_rank": summaries,
                "rank0_residency": residency,
                "model_ep_payload": payloads,
                "model_ep_payloads_identical": len(
                    {
                        json.dumps(payload, sort_keys=True)
                        for payload in payloads.values()
                    }
                )
                == 1,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
