#!/usr/bin/env python3
"""Attribute reduced-replica GroupedLinear backward skew across profiler ranks."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


PROFILE_RANKS = (8, 9, 10, 11)
GROUPED_LINEAR_NAME = "_GroupedLinearBackward"
TE_BACKWARD_FRAGMENT = "transformer_engine/pytorch/module/grouped_linear.py"


def load_events(path: Path) -> list[dict[str, object]]:
    """Load trace events from a compressed PyTorch Chrome trace."""
    with gzip.open(path, "rt") as stream:
        return json.load(stream)["traceEvents"]


def event_end(event: dict[str, object]) -> float:
    """Return a trace event's end timestamp in microseconds."""
    return float(event["ts"]) + float(event.get("dur", 0.0))


def contained(
    event: dict[str, object], start: float, end: float, *, same_pid: object, same_tid: object
) -> bool:
    """Return whether an event is contained in a CPU scope on the same thread."""
    return (
        event.get("pid") == same_pid
        and event.get("tid") == same_tid
        and float(event.get("ts", 0.0)) >= start
        and event_end(event) <= end
    )


def interval_union_duration(intervals: list[tuple[float, float]]) -> float:
    """Return the union duration of timestamp intervals in microseconds."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def python_self_times(
    events: list[dict[str, object]], scope: dict[str, object]
) -> tuple[dict[str, float], dict[str, object] | None]:
    """Aggregate Python/C-call self time beneath TE's GroupedLinear backward."""
    start = float(scope["ts"])
    end = event_end(scope)
    python_events = [
        event
        for event in events
        if event.get("cat") == "python_function"
        and contained(event, start, end, same_pid=scope.get("pid"), same_tid=scope.get("tid"))
    ]
    roots = [
        event
        for event in python_events
        if TE_BACKWARD_FRAGMENT in str(event.get("name", ""))
        and str(event.get("name", "")).endswith(": backward")
    ]
    if not roots:
        return {}, None
    root = max(roots, key=lambda event: float(event.get("dur", 0.0)))

    by_parent: dict[object, list[dict[str, object]]] = defaultdict(list)
    by_id: dict[object, dict[str, object]] = {}
    for event in python_events:
        args = event.get("args", {})
        python_id = args.get("Python id")
        if python_id is not None:
            by_id[python_id] = event
        by_parent[args.get("Python parent id")].append(event)

    root_id = root.get("args", {}).get("Python id")
    descendants: list[dict[str, object]] = []
    pending = [root_id]
    seen: set[object] = set()
    while pending:
        parent_id = pending.pop()
        if parent_id in seen:
            continue
        seen.add(parent_id)
        for child in by_parent.get(parent_id, []):
            descendants.append(child)
            child_id = child.get("args", {}).get("Python id")
            if child_id is not None:
                pending.append(child_id)

    self_times: dict[str, float] = defaultdict(float)
    for event in [root, *descendants]:
        event_start = float(event["ts"])
        event_stop = event_end(event)
        python_id = event.get("args", {}).get("Python id")
        child_intervals = [
            (max(event_start, float(child["ts"])), min(event_stop, event_end(child)))
            for child in by_parent.get(python_id, [])
        ]
        self_time = max(
            0.0,
            float(event.get("dur", 0.0)) - interval_union_duration(child_intervals),
        )
        self_times[str(event.get("name", ""))] += self_time
    return dict(self_times), root


def summarize_scope(
    events: list[dict[str, object]], scope: dict[str, object]
) -> dict[str, object]:
    """Summarize CPU and correlated GPU work for one GroupedLinear scope."""
    start = float(scope["ts"])
    end = event_end(scope)
    external_id = scope.get("args", {}).get("External id")
    runtime_events = [
        event
        for event in events
        if event.get("cat") == "cuda_runtime"
        and contained(event, start, end, same_pid=scope.get("pid"), same_tid=scope.get("tid"))
    ]
    runtime_by_name: dict[str, list[float]] = defaultdict(list)
    for event in runtime_events:
        runtime_by_name[str(event.get("name", ""))].append(float(event.get("dur", 0.0)))

    gpu_events = [
        event
        for event in events
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        and event.get("args", {}).get("External id") == external_id
    ]
    gpu_intervals = [(float(event["ts"]), event_end(event)) for event in gpu_events]
    driver_events = [
        event
        for event in events
        if event.get("cat") == "cuda_driver"
        and event.get("args", {}).get("External id") == external_id
    ]
    grouped_gemm_calls = [
        event
        for event in events
        if event.get("cat") == "python_function"
        and "te_general_grouped_gemm" in str(event.get("name", ""))
        and contained(event, start, end, same_pid=scope.get("pid"), same_tid=scope.get("tid"))
    ]
    grouped_gemm_calls.sort(key=lambda event: float(event["ts"]))
    kernels_by_correlation = {
        event.get("args", {}).get("correlation"): event for event in gpu_events
    }
    launch_gaps: list[dict[str, object]] = []
    for call_index, grouped_gemm in enumerate(grouped_gemm_calls):
        launches = sorted(
            (
                event
                for event in driver_events
                if event.get("name") == "cuLaunchKernelEx"
                and float(grouped_gemm["ts"]) <= float(event["ts"]) < event_end(grouped_gemm)
            ),
            key=lambda event: float(event["ts"]),
        )
        previous_end = float(grouped_gemm["ts"])
        for launch_index, launch in enumerate(launches):
            kernel = kernels_by_correlation.get(launch.get("args", {}).get("correlation"))
            launch_gaps.append(
                {
                    "call": call_index,
                    "launch": launch_index,
                    "duration_us": float(launch["ts"]) - previous_end,
                    "kernel": kernel.get("name") if kernel else None,
                    "grid": kernel.get("args", {}).get("grid") if kernel else None,
                    "kernel_us": float(kernel.get("dur", 0.0)) if kernel else 0.0,
                }
            )
            previous_end = event_end(launch)
        launch_gaps.append(
            {
                "call": call_index,
                "launch": "tail",
                "duration_us": event_end(grouped_gemm) - previous_end,
                "kernel": None,
                "grid": None,
                "kernel_us": 0.0,
            }
        )
    self_times, root = python_self_times(events, scope)
    return {
        "duration_us": float(scope.get("dur", 0.0)),
        "shape": scope.get("args", {}).get("Input Dims"),
        "gpu_count": len(gpu_events),
        "gpu_sum_us": sum(float(event.get("dur", 0.0)) for event in gpu_events),
        "gpu_union_us": interval_union_duration(gpu_intervals),
        "te_root_us": float(root.get("dur", 0.0)) if root else 0.0,
        "python_self_us": self_times,
        "runtime_count": len(runtime_events),
        "runtime_sum_us": sum(float(event.get("dur", 0.0)) for event in runtime_events),
        "runtime_by_name": {
            name: {"count": len(durations), "sum_us": sum(durations)}
            for name, durations in runtime_by_name.items()
        },
        "driver_count": len(driver_events),
        "driver_sum_us": sum(float(event.get("dur", 0.0)) for event in driver_events),
        "grouped_gemm_us": [float(event.get("dur", 0.0)) for event in grouped_gemm_calls],
        "launch_gaps": sorted(
            launch_gaps, key=lambda item: item["duration_us"], reverse=True
        ),
    }


def case_summaries(trace_dir: Path) -> dict[tuple[str, int], dict[int, dict[str, object]]]:
    """Load and align GroupedLinear scopes by profiler step and sequence number."""
    aligned: dict[tuple[str, int], dict[int, dict[str, object]]] = defaultdict(dict)
    for rank in PROFILE_RANKS:
        events = load_events(trace_dir / f"rank-{rank}.json.gz")
        steps = sorted(
            (
                event
                for event in events
                if event.get("cat") == "user_annotation"
                and str(event.get("name", "")).startswith("ProfilerStep#")
            ),
            key=lambda event: float(event["ts"]),
        )
        if len(steps) != 2:
            raise ValueError(f"rank {rank}: expected two profiler steps, got {len(steps)}")
        for step in steps:
            step_start = float(step["ts"])
            step_end = event_end(step)
            scopes = [
                event
                for event in events
                if event.get("cat") == "cpu_op"
                and event.get("name") == GROUPED_LINEAR_NAME
                and step_start <= float(event.get("ts", 0.0)) < step_end
            ]
            if len(scopes) != 12:
                raise ValueError(
                    f"rank {rank} {step['name']}: expected 12 GroupedLinear scopes, got {len(scopes)}"
                )
            for scope in scopes:
                sequence = int(scope.get("args", {})["Sequence number"])
                aligned[(str(step["name"]), sequence)][rank] = summarize_scope(events, scope)
    return aligned


def print_case(case: str, trace_dir: Path, top: int) -> None:
    """Print the largest rank-9 GroupedLinear duration excesses for one case."""
    aligned = case_summaries(trace_dir)
    complete = {
        key: rows for key, rows in aligned.items() if set(rows) == set(PROFILE_RANKS)
    }
    if len(complete) != 24:
        raise ValueError(f"{case}: expected 24 aligned scopes, got {len(complete)}")

    ranked = sorted(
        complete.items(),
        key=lambda item: item[1][9]["duration_us"]
        - statistics.median(item[1][rank]["duration_us"] for rank in (8, 10, 11)),
        reverse=True,
    )
    means = {
        rank: statistics.mean(rows[rank]["duration_us"] for rows in complete.values())
        for rank in PROFILE_RANKS
    }
    print(f"\n== {case} ==")
    print(
        "mean GroupedLinear scope (ms): "
        + " ".join(f"r{rank}={means[rank] / 1000.0:.3f}" for rank in PROFILE_RANKS)
    )

    for (step, sequence), rows in ranked[:top]:
        control_median = statistics.median(rows[rank]["duration_us"] for rank in (8, 10, 11))
        print(
            f"\n{step} seq={sequence} rank9 excess={((rows[9]['duration_us'] - control_median) / 1000.0):+.3f} ms"
        )
        for rank in PROFILE_RANKS:
            row = rows[rank]
            top_self = sorted(
                row["python_self_us"].items(), key=lambda item: item[1], reverse=True
            )[:4]
            self_text = "; ".join(
                f"{name}={duration / 1000.0:.3f}ms" for name, duration in top_self
            )
            runtime = row["runtime_by_name"]
            event_query = runtime.get("cudaEventQuery", {"count": 0, "sum_us": 0.0})
            top_gaps = "; ".join(
                f"call{gap['call']}/launch{gap['launch']}:{gap['duration_us'] / 1000.0:.3f}ms "
                f"grid={gap['grid']} kernel={gap['kernel']}"
                for gap in row["launch_gaps"][:3]
            )
            print(
                f"  r{rank}: scope={row['duration_us'] / 1000.0:.3f}ms "
                f"TE={row['te_root_us'] / 1000.0:.3f}ms shape={row['shape']} "
                f"GPU={row['gpu_union_us'] / 1000.0:.3f}ms/{row['gpu_count']} "
                f"runtime={row['runtime_sum_us'] / 1000.0:.3f}ms/{row['runtime_count']} "
                f"event_query={event_query['sum_us'] / 1000.0:.3f}ms/{event_query['count']} "
                f"driver={row['driver_sum_us'] / 1000.0:.3f}ms/{row['driver_count']} "
                f"grouped_calls={[round(duration / 1000.0, 3) for duration in row['grouped_gemm_us']]}"
            )
            print(f"      largest untraced launch gaps: {top_gaps}")
            print(f"      top Python self: {self_text}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Rank-9 profiler root containing current_a/post_graph_host/current_a_repeat",
    )
    parser.add_argument("--top", type=int, default=4, help="Scopes to print per case")
    parser.add_argument("--job-id", default="2450747", help="Profiler Slurm job ID")
    return parser.parse_args()


def main() -> None:
    """Analyze all three placement-controlled profiler cases."""
    args = parse_args()
    model_case = "a3b_repeat14_ep8_ep4_mbs1_mb1_1_proportional"
    for case in ("current_a", "post_graph_host", "current_a_repeat"):
        trace_dir = args.root / case / model_case / args.job_id / "torch_profile"
        print_case(case, trace_dir, args.top)


if __name__ == "__main__":
    main()
