#!/usr/bin/env python3
"""Analyze the clean top-k=6 healthy/NEP proportional trace pair."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from pathlib import Path


STEP_NAME = "ProfilerStep#3"
ITERATION_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*\d+.*?"
    r"elapsed time per iteration \(ms\): (?P<ms>[0-9.]+).*?"
    r"power per GPU \(W/GPU\): (?P<power>[0-9.]+)"
)
POST_TRAIN_TP_NUMEL = 2_104_704


def load_trace(path: Path) -> tuple[list[dict], float, float]:
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    step = next(
        event
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name") == STEP_NAME
    )
    start = float(step["ts"])
    return events, start, start + float(step["dur"])


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    return interval_duration(
        [
            (max(left_start, right_start), min(left_end, right_end))
            for left_start, left_end in merge_intervals(left)
            for right_start, right_end in merge_intervals(right)
            if min(left_end, right_end) > max(left_start, right_start)
        ]
    )


def actual_collectives(
    events: list[dict], step_start: float, step_end: float, group: str
) -> list[tuple[dict, dict]]:
    kernels_by_external_id: dict[int, list[dict]] = {}
    for event in events:
        args = event.get("args", {})
        if event.get("cat") == "kernel" and "External id" in args:
            kernels_by_external_id.setdefault(args["External id"], []).append(event)

    records = sorted(
        (
            event
            for event in events
            if event.get("name") == "record_param_comms"
            and step_start <= float(event.get("ts", 0.0)) < step_end
            and event.get("args", {}).get("Process Group Description") == group
            and event.get("args", {}).get("Collective name") != "wait"
        ),
        key=lambda event: float(event["ts"]),
    )
    output = []
    for record in records:
        kernels = [
            kernel
            for kernel in kernels_by_external_id.get(record["args"]["External id"], [])
            if str(kernel.get("name", "")).startswith("ncclDevKernel")
        ]
        if kernels:
            output.append((record, kernels[0]))
    return output


def trace_set(trace_dir: Path, rank_count: int) -> list[tuple[list[dict], float, float]]:
    return [load_trace(trace_dir / f"rank-{rank}.json.gz") for rank in range(rank_count)]


def nep_dispatch_stream(events: list[dict], start: float, end: float) -> int:
    """Infer the NEP staging stream from its pack/copy-back kernel signature."""
    counts: dict[int, dict[str, int]] = {}
    for event in events:
        if (
            event.get("cat") not in {"kernel", "gpu_memcpy", "gpu_memset"}
            or not start <= float(event.get("ts", 0.0)) < end
        ):
            continue
        stream = event.get("args", {}).get("stream")
        if stream is None:
            continue
        row = counts.setdefault(stream, {"copies": 0, "adds": 0, "fills": 0})
        name = str(event.get("name", ""))
        if event.get("cat") == "gpu_memcpy" and "DtoD" in name:
            row["copies"] += 1
        elif "CUDAFunctor_add<float>" in name:
            row["adds"] += 1
        elif "FillFunctor<float>" in name:
            row["fills"] += 1

    candidates = [
        stream for stream, row in counts.items() if row["copies"] and row["adds"]
    ]
    if not candidates:
        raise ValueError("Could not identify the NEP staging stream")
    return max(
        candidates,
        key=lambda stream: tuple(counts[stream][key] for key in ("copies", "adds", "fills")),
    )


def matched_service(
    traces: list[tuple[list[dict], float, float]], group: str, ranks: list[int]
) -> dict[str, object]:
    rows = [actual_collectives(*traces[rank], group) for rank in ranks]
    counts = {len(row) for row in rows}
    if len(counts) != 1:
        raise ValueError(f"Mismatched {group} collective counts: {sorted(counts)}")

    operations = []
    for index in range(len(rows[0])):
        starts = [float(row[index][1]["ts"]) for row in rows]
        ends = [
            float(row[index][1]["ts"]) + float(row[index][1]["dur"]) for row in rows
        ]
        latest_index = max(range(len(ranks)), key=lambda item: starts[item])
        operations.append(
            {
                "index": index,
                "payload_elements": max(
                    int(rows[0][index][0]["args"]["In msg nelems"]),
                    int(rows[0][index][0]["args"]["Out msg nelems"]),
                ),
                "owner_residency_ms": float(rows[0][index][1]["dur"]) / 1000.0,
                "owner_launch_wait_ms": max(0.0, max(starts) - starts[0]) / 1000.0,
                "latest_rank": ranks[latest_index],
                "service_ms": (max(ends) - max(starts)) / 1000.0,
            }
        )
    return {
        "count": len(operations),
        "payload_elements": sum(row["payload_elements"] for row in operations),
        "owner_residency_ms": sum(row["owner_residency_ms"] for row in operations),
        "service_ms": sum(row["service_ms"] for row in operations),
        "owner_launch_wait_ms": sum(row["owner_launch_wait_ms"] for row in operations),
        "operations": operations,
    }


def model_ep_kernels(
    traces: list[tuple[list[dict], float, float]], group: str
) -> list[list[dict]]:
    output = []
    for events, start, end in traces[:8]:
        rows = actual_collectives(events, start, end, group)
        output.append(
            [
                kernel
                for record, kernel in rows
                if record.get("args", {}).get("Collective name") == "all_to_allv"
            ]
        )
    if {len(row) for row in output} != {36}:
        raise ValueError(f"Expected 36 model-EP collectives, got {[len(row) for row in output]}")
    return output


def post_train_gpu_start(events: list[dict], start: float, end: float) -> float:
    kernels_by_external_id: dict[int, list[dict]] = {}
    for event in events:
        args = event.get("args", {})
        if event.get("cat") == "kernel" and "External id" in args:
            kernels_by_external_id.setdefault(args["External id"], []).append(event)
    record = next(
        event
        for event in events
        if event.get("name") == "record_param_comms"
        and start <= float(event.get("ts", 0.0)) < end
        and event.get("args", {}).get("In msg nelems") == POST_TRAIN_TP_NUMEL
    )
    return min(
        float(kernel["ts"])
        for kernel in kernels_by_external_id[record["args"]["External id"]]
    )


def owner_backward_summary(
    traces: list[tuple[list[dict], float, float]], model_ep_group: str
) -> dict[str, float]:
    events, step_start, step_end = traces[0]
    model_ep = model_ep_kernels(traces, model_ep_group)
    backward_start = float(model_ep[0][18]["ts"])
    boundary = post_train_gpu_start(events, step_start, step_end)
    gpu_events = [
        event
        for event in events
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        and float(event.get("ts", 0.0)) < boundary
        and float(event.get("ts", 0.0)) + float(event.get("dur", 0.0)) > backward_start
    ]
    nccl = [
        event
        for event in gpu_events
        if event.get("cat") == "kernel"
        and str(event.get("name", "")).startswith("ncclDevKernel")
    ]
    non_nccl = [event for event in gpu_events if event not in nccl]

    def intervals(rows: list[dict]) -> list[tuple[float, float]]:
        return [
            (
                max(backward_start, float(event["ts"])),
                min(boundary, float(event["ts"]) + float(event.get("dur", 0.0))),
            )
            for event in rows
        ]

    all_intervals = intervals(gpu_events)
    nccl_intervals = intervals(nccl)
    non_nccl_intervals = intervals(non_nccl)
    span = boundary - backward_start
    active = interval_duration(all_intervals)
    return {
        "span_ms": span / 1000.0,
        "gpu_active_ms": active / 1000.0,
        "gpu_idle_ms": (span - active) / 1000.0,
        "non_nccl_union_ms": interval_duration(non_nccl_intervals) / 1000.0,
        "nccl_union_ms": interval_duration(nccl_intervals) / 1000.0,
        "nccl_non_nccl_overlap_ms": intersection_duration(
            nccl_intervals, non_nccl_intervals
        )
        / 1000.0,
    }


def model_ep_summary(
    traces: list[tuple[list[dict], float, float]], group: str
) -> dict[str, object]:
    rows = model_ep_kernels(traces, group)
    owner_intervals = [
        (float(kernel["ts"]), float(kernel["ts"]) + float(kernel["dur"]))
        for kernel in rows[0]
    ]
    services = []
    owner_waits = []
    for index in range(36):
        starts = [float(row[index]["ts"]) for row in rows]
        ends = [float(row[index]["ts"]) + float(row[index]["dur"]) for row in rows]
        services.append(max(ends) - max(starts))
        owner_waits.append(max(0.0, max(starts) - starts[0]))

    backward_origin = min(float(row[18]["ts"]) for row in rows)
    backward_completions = [
        (max(float(row[index]["ts"]) + float(row[index]["dur"]) for row in rows) - backward_origin)
        / 1000.0
        for index in range(18, 36)
    ]
    return {
        "owner_residency_ms": interval_duration(owner_intervals) / 1000.0,
        "matched_service_ms": sum(services) / 1000.0,
        "owner_participant_wait_ms": sum(owner_waits) / 1000.0,
        "backward_completion_ms": backward_completions,
    }


def nep_dispatch_stream_summary(trace: tuple[list[dict], float, float]) -> dict[str, object]:
    events, start, end = trace
    dispatch_stream = nep_dispatch_stream(events, start, end)
    rows = [
        event
        for event in events
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        and event.get("args", {}).get("stream") == dispatch_stream
        and start <= float(event.get("ts", 0.0)) < end
        and not (
            event.get("cat") == "kernel"
            and str(event.get("name", "")).startswith("ncclDevKernel")
        )
    ]

    def category(event: dict) -> str:
        name = str(event.get("name", ""))
        if event.get("cat") == "gpu_memcpy" and "DtoD" in name:
            return "device_copies"
        if "CUDAFunctor_add<float>" in name:
            return "accumulation_adds"
        if "FillFunctor<float>" in name:
            return "buffer_fills"
        if "NormTwoOps" in name:
            return "gradient_norms"
        if event.get("cat") == "gpu_memcpy" and "DtoH" in name:
            return "gradient_check_scalar_copies"
        return "other"

    by_category: dict[str, list[dict]] = {}
    for event in rows:
        by_category.setdefault(category(event), []).append(event)
    return {
        "stream": dispatch_stream,
        "kernel_count": len(rows),
        "total_ms": sum(float(event.get("dur", 0.0)) for event in rows) / 1000.0,
        "categories": {
            name: {
                "count": len(category_rows),
                "sum_ms": sum(float(event.get("dur", 0.0)) for event in category_rows)
                / 1000.0,
            }
            for name, category_rows in sorted(by_category.items())
        },
    }


def nep_reshard_overlap(trace: tuple[list[dict], float, float]) -> dict[str, float]:
    events, start, end = trace
    dispatch_stream = nep_dispatch_stream(events, start, end)
    gpu = [
        event
        for event in events
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        and start <= float(event.get("ts", 0.0)) < end
    ]
    reshard = [
        event
        for event in gpu
        if event.get("args", {}).get("Process Group Description")
        in {"ep_dp", "nep_owner_transfer"}
    ]
    useful_non_nccl = [
        event
        for event in gpu
        if event.get("args", {}).get("stream") != dispatch_stream
        and not (
            event.get("cat") == "kernel"
            and str(event.get("name", "")).startswith("ncclDevKernel")
        )
    ]

    def intervals(rows: list[dict]) -> list[tuple[float, float]]:
        return [
            (float(event["ts"]), float(event["ts"]) + float(event.get("dur", 0.0)))
            for event in rows
        ]

    reshard_intervals = intervals(reshard)
    useful_intervals = intervals(useful_non_nccl)
    residency = interval_duration(reshard_intervals)
    overlap = intersection_duration(reshard_intervals, useful_intervals)
    return {
        "reshard_residency_ms": residency / 1000.0,
        "useful_non_nccl_overlap_ms": overlap / 1000.0,
        "useful_overlap_percent": 100.0 * overlap / residency,
    }


def iteration_stats(path: Path, first_iteration: int = 6) -> dict[str, object]:
    rows = []
    for match in ITERATION_RE.finditer(path.read_text(errors="replace")):
        if int(match.group("iteration")) >= first_iteration:
            rows.append((float(match.group("ms")), float(match.group("power"))))
    return {
        "count": len(rows),
        "mean_ms": statistics.mean(row[0] for row in rows),
        "median_ms": statistics.median(row[0] for row in rows),
        "stdev_ms": statistics.pstdev(row[0] for row in rows),
        "min_ms": min(row[0] for row in rows),
        "max_ms": max(row[0] for row in rows),
        "mean_power_w": statistics.mean(row[1] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("slurm_runs/lyris_a3b_ep8_ep4_mbs1_topk_clean_sweep/topk6"),
    )
    parser.add_argument("--job-id", default="2441290")
    args = parser.parse_args()

    healthy_dir = (
        args.root / "a3b_repeat14_ep8_ep8_mbs1_mb1_healthy" / args.job_id
    )
    nep_dir = (
        args.root
        / "a3b_repeat14_ep8_ep4_mbs1_mb1_1_proportional"
        / args.job_id
    )
    healthy = trace_set(healthy_dir / "torch_profile", 16)
    nep = trace_set(nep_dir / "torch_profile", 12)

    healthy_iterations = iteration_stats(healthy_dir / f"driver_{args.job_id}.log")
    nep_iterations = iteration_stats(nep_dir / f"driver_{args.job_id}.log")
    healthy_model_ep = model_ep_summary(healthy, "EXPERT_MODEL_PARALLEL_GROUP")
    nep_model_ep = model_ep_summary(nep, "ep")
    output = {
        "job_id": args.job_id,
        "timing": {
            "healthy": healthy_iterations,
            "nep": nep_iterations,
            "owner_parity_percent": 100.0
            * float(healthy_iterations["mean_ms"])
            / float(nep_iterations["mean_ms"]),
            "gap_ms": float(nep_iterations["mean_ms"])
            - float(healthy_iterations["mean_ms"]),
        },
        "owner_backward": {
            "healthy": owner_backward_summary(healthy, "EXPERT_MODEL_PARALLEL_GROUP"),
            "nep": owner_backward_summary(nep, "ep"),
        },
        "model_ep": {"healthy": healthy_model_ep, "nep": nep_model_ep},
        "expert_edp": {
            "healthy": matched_service(healthy, "EXPERT_DATA_PARALLEL_GROUP", [0, 8]),
            "nep": matched_service(nep, "ep_dp", [0, 8]),
        },
        "nep_owner_transfer": matched_service(nep, "nep_owner_transfer", [0, 4]),
        "dense_dp": {
            "healthy": matched_service(
                healthy, "DATA_PARALLEL_GROUP_WITH_CP", [0, 2, 4, 6, 8, 10, 12, 14]
            ),
            "nep": matched_service(nep, "dp_cp", [0, 2, 4, 6, 8, 10]),
        },
        "nep_dispatch_stream": nep_dispatch_stream_summary(nep[0]),
        "nep_reshard_overlap": nep_reshard_overlap(nep[0]),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
