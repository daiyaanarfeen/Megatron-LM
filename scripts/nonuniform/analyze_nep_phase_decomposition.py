#!/usr/bin/env python3
"""Separate causal NEP iteration overhead from collective service time."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from analyze_exact_uniform_nep_slowdown import (
    NCCL_GROUPS,
    aligned_collective_summary,
    clipped_intervals,
    intersection_duration,
    parse_trace,
    summarize,
    union_duration,
)

PHASES = ("none", "gather", "edp", "scatter")
ITERATION_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/.*?elapsed time per iteration \(ms\):\s*"
    r"(?P<elapsed>[0-9.]+)"
)
OWNER_LANES = {
    "gather": [[owner, owner + 4] for owner in range(4)],
    "edp": [[owner, owner + 8] for owner in range(4)],
    "scatter": [[owner, owner + 4] for owner in range(4)],
}
GROUPS = {
    "gather": NCCL_GROUPS["nep"]["owner_gather"],
    "edp": NCCL_GROUPS["nep"]["expert_edp"],
    "scatter": NCCL_GROUPS["nep"]["owner_transfer"],
}


def find_single(path: Path, pattern: str) -> Path:
    """Return the only file under path matching pattern."""
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} under {path}, found {len(matches)}")
    return matches[0]


def parse_timing_log(path: Path, clean_start: int) -> dict[str, Any]:
    """Parse clean profiler-free iteration times from one driver log."""
    values = []
    iterations = []
    for match in ITERATION_RE.finditer(path.read_text(errors="replace")):
        iteration = int(match.group("iteration"))
        if iteration >= clean_start:
            iterations.append(iteration)
            values.append(float(match.group("elapsed")))
    if not values:
        raise ValueError(f"no iterations >= {clean_start} in {path}")
    return {"iterations": iterations, "values_ms": values, "summary": summarize(values)}


def load_timings(root: Path, job_id: str, clean_start: int) -> dict[str, Any]:
    """Load healthy brackets and the four cumulative NEP phases."""
    labels = ["healthy_pre", "healthy_post"]
    labels += [f"nep_{phase}" for phase in PHASES]
    output = {}
    for label in labels:
        run_dir = root / "timing" / label / job_id
        output[label] = parse_timing_log(find_single(run_dir, "driver_*.log"), clean_start)
    return output


def decompose_timing(timings: dict[str, Any]) -> dict[str, Any]:
    """Build an exactly closing decomposition with bracketed linear drift correction."""
    healthy_pre = float(timings["healthy_pre"]["summary"]["mean"])
    healthy_post = float(timings["healthy_post"]["summary"]["mean"])
    healthy_ms = statistics.mean((healthy_pre, healthy_post))
    raw_phase_means = {phase: float(timings[f"nep_{phase}"]["summary"]["mean"]) for phase in PHASES}
    drift_per_launch = (healthy_post - healthy_pre) / 5.0
    phase_positions = {phase: index + 1 for index, phase in enumerate(PHASES)}
    drift_corrections = {
        phase: drift_per_launch * (phase_positions[phase] - 2.5) for phase in PHASES
    }
    phase_means = {phase: raw_phase_means[phase] - drift_corrections[phase] for phase in PHASES}
    components = {
        "non_collective_nep_control_ms": phase_means["none"] - healthy_ms,
        "gather_causal_ms": phase_means["gather"] - phase_means["none"],
        "edp_causal_ms": phase_means["edp"] - phase_means["gather"],
        "scatter_causal_ms": phase_means["scatter"] - phase_means["edp"],
    }
    total_ms = phase_means["scatter"] - healthy_ms
    raw_components = {
        "non_collective_nep_control_ms": raw_phase_means["none"] - healthy_ms,
        "gather_causal_ms": raw_phase_means["gather"] - raw_phase_means["none"],
        "edp_causal_ms": raw_phase_means["edp"] - raw_phase_means["gather"],
        "scatter_causal_ms": raw_phase_means["scatter"] - raw_phase_means["edp"],
    }
    return {
        "healthy_bracket_mean_ms": healthy_ms,
        "linear_drift_corrected_phase_means_ms": phase_means,
        "raw_phase_means_ms": raw_phase_means,
        "components": components,
        "total_nep_overhead_ms": total_ms,
        "iteration_parity_percent": 100.0 * healthy_ms / phase_means["scatter"],
        "closure_error_ms": sum(components.values()) - total_ms,
        "allocation_drift_diagnostics": {
            "healthy_post_minus_pre_ms": healthy_post - healthy_pre,
            "assumed_linear_drift_per_launch_ms": drift_per_launch,
            "phase_corrections_subtracted_ms": drift_corrections,
            "raw_components_without_drift_correction": raw_components,
        },
    }


def load_profile_traces(
    root: Path, job_id: str, expected_steps: int
) -> dict[str, dict[int, dict[str, Any]]]:
    """Load every rank for the healthy and full-NEP profiles."""
    labels = ["healthy", "nep_scatter"]
    traces = {}
    for label in labels:
        rank_count = 16 if label == "healthy" else 12
        trace_dir = root / "profile" / label / job_id / "torch_profile"
        rank_traces = {}
        for rank in range(rank_count):
            path = trace_dir / f"rank-{rank}.json.gz"
            print(f"[{label}] parsing rank {rank}/{rank_count - 1}", flush=True)
            rank_traces[rank] = parse_trace(
                path, retain_gpu=rank == 0, expected_steps=expected_steps
            )
        traces[label] = rank_traces
    return traces


def dtype_nbytes(dtype: str) -> int:
    """Return element width for profiler communication dtype labels."""
    value = dtype.lower()
    if "bfloat16" in value or "float16" in value or "half" in value:
        return 2
    if "float64" in value or "double" in value or "int64" in value:
        return 8
    if "float" in value or "int32" in value:
        return 4
    if "int8" in value or "uint8" in value or "bool" in value:
        return 1
    raise ValueError(f"unknown profiler dtype: {dtype}")


def aligned_phase_summary(traces: dict[int, dict[str, Any]], phase: str) -> dict[str, Any]:
    """Report participant-matched service independently of arrival waiting."""
    group = GROUPS[phase]
    lane_rows = []
    lane_details = {}
    for lane, ranks in enumerate(OWNER_LANES[phase]):
        aligned = aligned_collective_summary(traces, ranks, group)
        lane_details[str(lane)] = aligned
        for step_name, row in aligned["per_step"].items():
            raw_rows = traces[ranks[0]]["collectives"][step_name][group]
            dtypes = {record["dtype"] for record in raw_rows}
            if len(dtypes) != 1:
                raise ValueError(f"{phase} lane {lane} {step_name}: mixed dtypes {dtypes}")
            payload_bytes = int(row["payload_elements"]) * dtype_nbytes(dtypes.pop())
            matched_service_ms = float(row["matched_service_ms"])
            lane_rows.append(
                {
                    "lane": lane,
                    "step": step_name,
                    **row,
                    "payload_bytes": payload_bytes,
                    "effective_payload_gb_per_s": (
                        payload_bytes / (matched_service_ms * 1.0e6) if matched_service_ms else 0.0
                    ),
                }
            )
    critical_service_by_step = {}
    critical_residency_by_step = {}
    for step_name in sorted({str(row["step"]) for row in lane_rows}):
        step_rows = [row for row in lane_rows if row["step"] == step_name]
        critical_service_by_step[step_name] = max(
            float(row["matched_service_ms"]) for row in step_rows
        )
        critical_residency_by_step[step_name] = max(
            float(row["owner_residency_ms"]) for row in step_rows
        )
    return {
        "group": group,
        "lanes": OWNER_LANES[phase],
        "operation_count": sorted({int(row["operation_count"]) for row in lane_rows}),
        "lane_step_summary": {
            key: summarize([float(row[key]) for row in lane_rows])
            for key in (
                "owner_residency_ms",
                "owner_participant_wait_ms",
                "start_spread_ms",
                "matched_service_ms",
                "payload_bytes",
                "effective_payload_gb_per_s",
            )
        },
        "critical_matched_service_ms": summarize(list(critical_service_by_step.values())),
        "critical_owner_residency_ms": summarize(list(critical_residency_by_step.values())),
        "per_lane": lane_details,
    }


def validate_full_nep_profile(traces: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Verify that the full profile contains Gather, EDP, and Scatter."""
    first_step = sorted(traces["nep_scatter"][0]["step_spans"])[0]
    phase_counts = {}
    for phase in PHASES[1:]:
        group = GROUPS[phase]
        counts = [
            len(traces["nep_scatter"][rank]["collectives"][first_step].get(group, []))
            for rank in range(12)
        ]
        if not any(counts):
            raise ValueError(f"full NEP profile is missing {phase} operations")
        phase_counts[phase] = counts
    return phase_counts


def phase_overlap_with_other_gpu_work(trace: dict[str, Any], phase: str) -> dict[str, Any]:
    """Measure target phase residency overlapped with any non-target GPU work."""
    group = GROUPS[phase]
    rows = []
    for step_name, (step_start, step_end) in trace["step_spans"].items():
        events = trace["gpu"][step_name]
        target = [event for event in events if event["is_nccl"] and event["group"] == group]
        other = [event for event in events if not (event["is_nccl"] and event["group"] == group)]
        target_intervals = clipped_intervals(target, step_start, step_end)
        other_intervals = clipped_intervals(other, step_start, step_end)
        residency = union_duration(target_intervals)
        overlap = intersection_duration(target_intervals, other_intervals)
        rows.append(
            {
                "step": step_name,
                "residency_ms": residency / 1000.0,
                "overlap_ms": overlap / 1000.0,
                "exposed_ms": (residency - overlap) / 1000.0,
                "overlap_percent": 100.0 * overlap / residency if residency else 0.0,
            }
        )
    return {
        "summary": {
            key: summarize([float(row[key]) for row in rows])
            for key in ("residency_ms", "overlap_ms", "exposed_ms", "overlap_percent")
        },
        "per_step": rows,
    }


def analyze_profiles(traces: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Build true collective service and observed-overlap reports."""
    validation = validate_full_nep_profile(traces)
    service = {
        "gather_full": aligned_phase_summary(traces["nep_scatter"], "gather"),
        "edp_full": aligned_phase_summary(traces["nep_scatter"], "edp"),
        "scatter_full": aligned_phase_summary(traces["nep_scatter"], "scatter"),
    }
    full_rank_zero = traces["nep_scatter"][0]
    overlap = {
        phase: phase_overlap_with_other_gpu_work(full_rank_zero, phase)
        for phase in ("gather", "edp", "scatter")
    }

    healthy_edp_lanes = [[owner, owner + 8] for owner in range(8)]
    healthy_edp = []
    healthy_group = NCCL_GROUPS["healthy"]["expert_edp"]
    for ranks in healthy_edp_lanes:
        aligned = aligned_collective_summary(traces["healthy"], ranks, healthy_group)
        healthy_edp.extend(aligned["per_step"].values())
    healthy_service = {
        key: summarize([float(row[key]) for row in healthy_edp])
        for key in ("owner_residency_ms", "owner_participant_wait_ms", "matched_service_ms")
    }
    return {
        "full_nep_profile_phase_counts": validation,
        "true_collective_service": service,
        "healthy_edp_lane_step_summary": healthy_service,
        "full_nep_rank0_overlap_with_any_other_gpu_work": overlap,
    }


def format_report(result: dict[str, Any]) -> str:
    """Render a concise Markdown report from the structured result."""
    timing = result["timing_decomposition"]
    lines = [
        "# NEP EP8/EP4 Phase Decomposition",
        "",
        "## Profiler-free iteration timing",
        "",
        "| Case | Mean (ms) | Std (ms) |",
        "|---|---:|---:|",
    ]
    for label, row in result["timings"].items():
        lines.append(f"| {label} | {row['summary']['mean']:.3f} | {row['summary']['std']:.3f} |")
    drift = timing["allocation_drift_diagnostics"]
    lines += [
        "",
        "## Drift-corrected additive decomposition",
        "",
        f"Healthy post-minus-pre drift: **{drift['healthy_post_minus_pre_ms']:.3f} ms**; "
        f"linear correction per intervening launch: "
        f"**{drift['assumed_linear_drift_per_launch_ms']:.3f} ms**.",
        "",
        "| Component | Added iteration time (ms) |",
        "|---|---:|",
    ]
    for component, value in timing["components"].items():
        lines.append(f"| {component} | {value:.3f} |")
    lines += [
        f"| **Total NEP overhead** | **{timing['total_nep_overhead_ms']:.3f}** |",
        "",
        f"Iteration parity: **{timing['iteration_parity_percent']:.3f}%**. "
        f"Closure error: **{timing['closure_error_ms']:.6f} ms**.",
        "",
        "## True collective times from all-rank traces",
        "",
        "Matched service starts when the last participant arrives; participant wait is reported separately.",
        "",
        "| Profile operation | Critical matched service (ms) | Mean lane service (ms) | Mean owner residency (ms) | Mean wait (ms) | Payload/lane (GiB) | Effective payload (GB/s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    services = result["profiles"]["true_collective_service"]
    for label, row in services.items():
        lane = row["lane_step_summary"]
        lines.append(
            f"| {label} | {row['critical_matched_service_ms']['mean']:.3f} | "
            f"{lane['matched_service_ms']['mean']:.3f} | "
            f"{lane['owner_residency_ms']['mean']:.3f} | "
            f"{lane['owner_participant_wait_ms']['mean']:.3f} | "
            f"{lane['payload_bytes']['mean'] / (1024 ** 3):.3f} | "
            f"{lane['effective_payload_gb_per_s']['mean']:.2f} |"
        )
    healthy_edp = result["profiles"]["healthy_edp_lane_step_summary"]
    lines.append(
        f"\nHealthy EDP mean lane service/residency/wait: "
        f"**{healthy_edp['matched_service_ms']['mean']:.3f} / "
        f"{healthy_edp['owner_residency_ms']['mean']:.3f} / "
        f"{healthy_edp['owner_participant_wait_ms']['mean']:.3f} ms**."
    )
    lines += [
        "",
        "## Full-NEP rank-0 overlap",
        "",
        "| Phase | Residency (ms) | Overlap with other GPU work (ms) | Exposed (ms) | Overlap (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for phase, row in result["profiles"]["full_nep_rank0_overlap_with_any_other_gpu_work"].items():
        summary = row["summary"]
        lines.append(
            f"| {phase} | {summary['residency_ms']['mean']:.3f} | "
            f"{summary['overlap_ms']['mean']:.3f} | {summary['exposed_ms']['mean']:.3f} | "
            f"{summary['overlap_percent']['mean']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--clean-start", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    timings = load_timings(args.root, args.job_id, args.clean_start)
    traces = load_profile_traces(args.root, args.job_id, args.profile_steps)
    result = {
        "root": str(args.root),
        "job_id": args.job_id,
        "timings": timings,
        "timing_decomposition": decompose_timing(timings),
        "profiles": analyze_profiles(traces),
    }
    output = args.output or args.root / f"phase_decomposition_{args.job_id}.json"
    report = args.report or args.root / f"phase_decomposition_{args.job_id}.md"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_text = format_report(result)
    report.write_text(report_text)
    print(report_text)
    print(f"wrote {output}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
