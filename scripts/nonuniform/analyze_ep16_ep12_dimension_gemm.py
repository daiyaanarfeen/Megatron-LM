#!/usr/bin/env python3
"""Measure routed-expert GEMM efficiency for the EP16/EP12 dimension experiment."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path


CASES = {
    "base_a": (2688, 1856),
    "high": (5376, 3712),
    "base_b": (2688, 1856),
}
ROLE_RANKS = {
    "full_ep16": range(0, 16),
    "reduced_ep12": range(16, 28),
}
EP_DEGREES = {
    "full_ep16": 16,
    "reduced_ep12": 12,
}
REPLICA_TP_CP_LANES = {
    "full_ep16": 8,
    "reduced_ep12": 6,
}


def _load_events(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)["traceEvents"]


def _profile_steps(events: list[dict[str, object]]) -> list[dict[str, object]]:
    steps = [
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("ProfilerStep#")
    ]
    if not steps:
        raise ValueError("Trace has no ProfilerStep annotations")
    return sorted(steps, key=lambda event: float(event["ts"]))


def _in_step(event: dict[str, object], step: dict[str, object]) -> bool:
    timestamp = float(event.get("ts", 0.0))
    start = float(step["ts"])
    return start <= timestamp < start + float(step["dur"])


def _merge_intervals(
    events: list[dict[str, object]],
) -> list[tuple[float, float]]:
    intervals = sorted(
        (
            float(event["ts"]),
            float(event["ts"]) + float(event.get("dur", 0.0)),
        )
        for event in events
    )
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _intersection_duration(
    left_events: list[dict[str, object]],
    right_events: list[dict[str, object]],
) -> float:
    left = _merge_intervals(left_events)
    right = _merge_intervals(right_events)
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


def _trace_rows(
    path: Path,
    rank: int,
    role: str,
    useful_flops: float,
    expected_kernels: int,
) -> list[dict[str, object]]:
    events = _load_events(path)
    rows = []
    for step in _profile_steps(events):
        nvjet = [
            event
            for event in events
            if event.get("cat") == "kernel"
            and str(event.get("name", "")).startswith("nvjet")
            and _in_step(event, step)
        ]
        stream_counts = Counter(
            int(event.get("args", {}).get("stream", -1)) for event in nvjet
        )
        expert_streams = {
            stream for stream, _ in stream_counts.most_common(4)
        }
        expert_gemms = [
            event
            for event in nvjet
            if int(event.get("args", {}).get("stream", -1)) in expert_streams
        ]
        nccl = [
            event
            for event in events
            if event.get("cat") == "kernel"
            and str(event.get("name", "")).startswith("ncclDevKernel")
            and _in_step(event, step)
        ]
        backward_starts = [
            float(event["ts"])
            for event in events
            if event.get("cat") == "cpu_op"
            and str(event.get("name", "")).startswith(
                "autograd::engine::evaluate_function"
            )
            and _in_step(event, step)
        ]
        if not backward_starts:
            raise ValueError(f"{path}: {step['name']} has no autograd boundary")
        first_backward = min(backward_starts)
        forward_gemms = [
            event for event in expert_gemms if float(event["ts"]) < first_backward
        ]
        backward_gemms = [
            event for event in expert_gemms if float(event["ts"]) >= first_backward
        ]
        if len(expert_gemms) != expected_kernels:
            raise ValueError(
                f"{path}: {step['name']} has {len(expert_gemms)} NVJet kernels, "
                f"expected {expected_kernels}"
            )
        if len(forward_gemms) != expected_kernels // 3:
            raise ValueError(
                f"{path}: {step['name']} has {len(forward_gemms)} forward expert "
                f"kernels, expected {expected_kernels // 3}"
            )
        if len(backward_gemms) != 2 * expected_kernels // 3:
            raise ValueError(
                f"{path}: {step['name']} has {len(backward_gemms)} backward expert "
                f"kernels, expected {2 * expected_kernels // 3}"
            )
        total_us = sum(float(event["dur"]) for event in expert_gemms)
        forward_us = sum(float(event["dur"]) for event in forward_gemms)
        backward_us = sum(float(event["dur"]) for event in backward_gemms)
        forward_union_us = _interval_duration(_merge_intervals(forward_gemms))
        backward_union_us = _interval_duration(_merge_intervals(backward_gemms))
        expert_union_us = _interval_duration(_merge_intervals(expert_gemms))
        expert_nccl_overlap_us = _intersection_duration(expert_gemms, nccl)
        rows.append(
            {
                "rank": rank,
                "role": role,
                "step": step["name"],
                "kernel_count": len(expert_gemms),
                "kernel_sum_ms": total_us / 1000.0,
                "kernel_mean_us": total_us / len(expert_gemms),
                "forward_kernel_sum_ms": forward_us / 1000.0,
                "backward_kernel_sum_ms": backward_us / 1000.0,
                "forward_kernel_union_ms": forward_union_us / 1000.0,
                "backward_kernel_union_ms": backward_union_us / 1000.0,
                "forward_kernel_nccl_overlap_ms": (
                    _intersection_duration(forward_gemms, nccl) / 1000.0
                ),
                "backward_kernel_nccl_overlap_ms": (
                    _intersection_duration(backward_gemms, nccl) / 1000.0
                ),
                "kernel_union_ms": expert_union_us / 1000.0,
                "kernel_nccl_overlap_ms": expert_nccl_overlap_us / 1000.0,
                "useful_flops": useful_flops,
                "effective_tflops": useful_flops / (expert_union_us * 1.0e6),
                "streams": sorted(
                    {
                        int(event.get("args", {}).get("stream", -1))
                        for event in expert_gemms
                    }
                ),
                "kernel_names": Counter(str(event["name"]) for event in expert_gemms),
            }
        )
    return rows


def _find_trace_dir(root: Path, label: str, job_id: str) -> Path:
    matches = list((root / label).glob(f"*/{job_id}/torch_profile"))
    if len(matches) != 1:
        raise ValueError(f"Expected one trace directory for {label}/{job_id}, found {matches}")
    return matches[0]


def _role_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    union_durations = [float(row["kernel_union_ms"]) for row in rows]
    summed_durations = [float(row["kernel_sum_ms"]) for row in rows]
    throughputs = [float(row["effective_tflops"]) for row in rows]
    useful_flops = {float(row["useful_flops"]) for row in rows}
    if len(useful_flops) != 1:
        raise ValueError(f"Rows have different useful FLOPs: {useful_flops}")
    mean_union_duration = statistics.mean(union_durations)
    mean_summed_duration = statistics.mean(summed_durations)
    forward_union_mean = statistics.mean(
        float(row["forward_kernel_union_ms"]) for row in rows
    )
    backward_union_mean = statistics.mean(
        float(row["backward_kernel_union_ms"]) for row in rows
    )
    total_useful_flops = next(iter(useful_flops))
    counts = {int(row["kernel_count"]) for row in rows}
    streams = sorted({stream for row in rows for stream in row["streams"]})
    return {
        "rank_step_samples": len(rows),
        "kernel_count_per_rank_step": counts.pop() if len(counts) == 1 else sorted(counts),
        "kernel_sum_ms_mean": mean_summed_duration,
        "kernel_union_ms_mean": mean_union_duration,
        "kernel_union_ms_stdev": statistics.pstdev(union_durations),
        "kernel_union_ms_cv_percent": (
            100.0
            * statistics.pstdev(union_durations)
            / statistics.mean(union_durations)
        ),
        "kernel_sum_to_union_ratio": mean_summed_duration / mean_union_duration,
        "kernel_mean_us": statistics.mean(float(row["kernel_mean_us"]) for row in rows),
        "kernel_nccl_overlap_ms_mean": statistics.mean(
            float(row["kernel_nccl_overlap_ms"]) for row in rows
        ),
        "forward_kernel_sum_ms_mean": statistics.mean(
            float(row["forward_kernel_sum_ms"]) for row in rows
        ),
        "backward_kernel_sum_ms_mean": statistics.mean(
            float(row["backward_kernel_sum_ms"]) for row in rows
        ),
        "forward_kernel_union_ms_mean": forward_union_mean,
        "backward_kernel_union_ms_mean": backward_union_mean,
        "forward_kernel_nccl_overlap_ms_mean": statistics.mean(
            float(row["forward_kernel_nccl_overlap_ms"]) for row in rows
        ),
        "backward_kernel_nccl_overlap_ms_mean": statistics.mean(
            float(row["backward_kernel_nccl_overlap_ms"]) for row in rows
        ),
        "forward_effective_tflops": (total_useful_flops / 3.0)
        / (forward_union_mean * 1.0e9),
        "backward_effective_tflops": (2.0 * total_useful_flops / 3.0)
        / (backward_union_mean * 1.0e9),
        "effective_tflops_mean": total_useful_flops
        / (mean_union_duration * 1.0e9),
        "sample_effective_tflops_mean": statistics.mean(throughputs),
        "effective_tflops_stdev": statistics.pstdev(throughputs),
        "streams": streams,
    }


def _case_summary(
    root: Path,
    job_id: str,
    label: str,
    hidden_size: int,
    ffn_hidden_size: int,
    sequence_length: int,
    topk: int,
    num_experts: int,
    num_moe_layers: int,
) -> dict[str, object]:
    trace_dir = _find_trace_dir(root, label, job_id)
    role_rows: dict[str, list[dict[str, object]]] = {}
    useful_flops_by_role = {}
    for role, ranks in ROLE_RANKS.items():
        assignments = (
            REPLICA_TP_CP_LANES[role] * sequence_length * topk / EP_DEGREES[role]
        )
        if not assignments.is_integer():
            raise ValueError(f"{role} has fractional routed assignments per GPU")
        useful_flops = 12.0 * num_moe_layers * assignments * hidden_size * ffn_hidden_size
        useful_flops_by_role[role] = useful_flops
        expected_kernels = (
            6 * num_moe_layers * (num_experts // EP_DEGREES[role])
        )
        role_rows[role] = [
            row
            for rank in ranks
            for row in _trace_rows(
                trace_dir / f"rank-{rank}.json.gz",
                rank,
                role,
                useful_flops,
                expected_kernels,
            )
        ]

    if len(set(useful_flops_by_role.values())) != 1:
        raise ValueError(f"Per-GPU useful FLOPs differ by role: {useful_flops_by_role}")
    roles = {role: _role_summary(rows) for role, rows in role_rows.items()}
    full_tflops = float(roles["full_ep16"]["effective_tflops_mean"])
    reduced_tflops = float(roles["reduced_ep12"]["effective_tflops_mean"])
    full_forward = float(roles["full_ep16"]["forward_effective_tflops"])
    reduced_forward = float(roles["reduced_ep12"]["forward_effective_tflops"])
    full_backward = float(roles["full_ep16"]["backward_effective_tflops"])
    reduced_backward = float(roles["reduced_ep12"]["backward_effective_tflops"])
    return {
        "label": label,
        "trace_dir": str(trace_dir),
        "hidden_size": hidden_size,
        "ffn_hidden_size": ffn_hidden_size,
        "useful_expert_tflops_per_gpu_step": next(iter(useful_flops_by_role.values()))
        / 1.0e12,
        "roles": roles,
        "reduced_over_full_efficiency_percent": 100.0 * reduced_tflops / full_tflops,
        "forward_reduced_over_full_efficiency_percent": (
            100.0 * reduced_forward / full_forward
        ),
        "backward_reduced_over_full_efficiency_percent": (
            100.0 * reduced_backward / full_backward
        ),
        "_rows": role_rows,
    }


def _combine_cases(
    summaries: dict[str, dict[str, object]], labels: tuple[str, ...], label: str
) -> dict[str, object]:
    roles = {}
    for role in ROLE_RANKS:
        rows = [
            row
            for case_label in labels
            for row in summaries[case_label]["_rows"][role]
        ]
        roles[role] = _role_summary(rows)
    full_tflops = float(roles["full_ep16"]["effective_tflops_mean"])
    reduced_tflops = float(roles["reduced_ep12"]["effective_tflops_mean"])
    full_forward = float(roles["full_ep16"]["forward_effective_tflops"])
    reduced_forward = float(roles["reduced_ep12"]["forward_effective_tflops"])
    full_backward = float(roles["full_ep16"]["backward_effective_tflops"])
    reduced_backward = float(roles["reduced_ep12"]["backward_effective_tflops"])
    return {
        "label": label,
        "roles": roles,
        "reduced_over_full_efficiency_percent": 100.0 * reduced_tflops / full_tflops,
        "forward_reduced_over_full_efficiency_percent": (
            100.0 * reduced_forward / full_forward
        ),
        "backward_reduced_over_full_efficiency_percent": (
            100.0 * reduced_backward / full_backward
        ),
    }


def _print_table(rows: list[dict[str, object]]) -> None:
    print(
        "| Case | Dimensions HxF | Role | Kernels/step | Active GEMM ms | "
        "Effective TFLOP/s | CV |"
    )
    print("|---|---:|---|---:|---:|---:|---:|")
    for case in rows:
        dimensions = (
            f"{case['hidden_size']}x{case['ffn_hidden_size']}"
            if "hidden_size" in case
            else "2688x1856"
        )
        for role in ROLE_RANKS:
            metrics = case["roles"][role]
            print(
                f"| {case['label']} | {dimensions} | {role} | "
                f"{metrics['kernel_count_per_rank_step']} | "
                f"{metrics['kernel_union_ms_mean']:.3f} | "
                f"{metrics['effective_tflops_mean']:.1f} | "
                f"{metrics['kernel_union_ms_cv_percent']:.2f}% |"
            )
        print(
            f"| {case['label']} ratio | {dimensions} | EP12 / EP16 | - | - | "
            f"{case['reduced_over_full_efficiency_percent']:.2f}% | - |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("slurm_runs/lyris_a3b_ep16_ep12_dimension_gemm_aba"),
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--sequence-length", type=int, default=2016)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--num-experts", type=int, default=144)
    parser.add_argument("--num-moe-layers", type=int, default=2)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    summaries = {
        label: _case_summary(
            args.root,
            args.job_id,
            label,
            hidden_size,
            ffn_hidden_size,
            args.sequence_length,
            args.topk,
            args.num_experts,
            args.num_moe_layers,
        )
        for label, (hidden_size, ffn_hidden_size) in CASES.items()
    }
    base = _combine_cases(summaries, ("base_a", "base_b"), "base_bracket")
    high = summaries["high"]
    base_ratio = float(base["reduced_over_full_efficiency_percent"])
    high_ratio = float(high["reduced_over_full_efficiency_percent"])
    result = {
        "job_id": args.job_id,
        "config": {
            "topology": "EP16/EP12",
            "replica_tp_cp_lanes": "8 6",
            "tensor_parallel": 2,
            "sequence_length": args.sequence_length,
            "topk": args.topk,
            "num_experts": args.num_experts,
            "num_moe_layers": args.num_moe_layers,
            "micro_batch_size": 1,
            "num_microbatches": 1,
            "routing": "exact_uniform",
        },
        "cases": {
            label: {key: value for key, value in summary.items() if key != "_rows"}
            for label, summary in summaries.items()
        },
        "base_bracket": base,
        "hypothesis": {
            "base_efficiency_percent": base_ratio,
            "high_efficiency_percent": high_ratio,
            "improvement_percentage_points": high_ratio - base_ratio,
            "supported_directionally": high_ratio > base_ratio,
        },
    }

    _print_table([summaries["base_a"], high, summaries["base_b"], base])
    print()
    print(
        "Dimension hypothesis: "
        f"EP12/EP16 efficiency {base_ratio:.2f}% -> {high_ratio:.2f}% "
        f"({high_ratio - base_ratio:+.2f} percentage points)."
    )
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
