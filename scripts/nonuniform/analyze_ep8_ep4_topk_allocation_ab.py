#!/usr/bin/env python3
"""Analyze the EP8/EP4 top-k A/B and allocation controls."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze_ep8_ep4_trace_evidence import (
    STEP_NAME,
    all_rank_summary,
    clean_iteration_mean,
    in_window,
    load_events,
    model_ep_payload,
    profiler_step,
)


HEALTHY_CASE = "a3b_repeat14_ep8_ep8_mbs2_mb7_healthy"
NEP_CASE = "a3b_repeat14_ep8_ep4_mbs2_mb7_7_proportional"


def case_dir(root: Path, job_id: str, healthy: bool) -> Path:
    return root / (HEALTHY_CASE if healthy else NEP_CASE) / job_id


def timing(path: Path, job_id: str) -> float:
    return clean_iteration_mean(path / f"driver_{job_id}.log")


def sendrecv_stats(trace_dir: Path) -> dict[str, float]:
    durations = []
    for rank in range(8, 12):
        events = load_events(trace_dir / f"rank-{rank}.json.gz")
        step = profiler_step(events)
        start = float(step["ts"])
        end = start + float(step["dur"])
        durations.extend(
            float(event["dur"])
            for event in events
            if event.get("cat") == "kernel"
            and str(event.get("name", "")).startswith("ncclDevKernel_SendRecv")
            and event.get("args", {}).get("Process Group Description") == "ep"
            and in_window(event, start, end)
        )
    durations.sort()
    return {
        "count": len(durations),
        "mean_us": statistics.mean(durations),
        "p95_us": durations[int(0.95 * (len(durations) - 1))],
        "max_us": max(durations),
    }


def nep_trace_summary(path: Path) -> dict[str, object]:
    trace_dir = path / "torch_profile"
    summary = all_rank_summary(trace_dir, healthy=False, rank_count=12)
    return {
        "groups": summary["groups"],
        "reduced_model_ep_sendrecv": sendrecv_stats(trace_dir),
        "model_ep_payload": model_ep_payload(trace_dir, healthy=False),
        "trace_count": len(list(trace_dir.glob("rank-*.json.gz"))),
    }


def pair_summary(root: Path, job_id: str) -> dict[str, object]:
    healthy = case_dir(root, job_id, healthy=True)
    nep = case_dir(root, job_id, healthy=False)
    healthy_ms = timing(healthy, job_id)
    nep_ms = timing(nep, job_id)
    return {
        "healthy_ms": healthy_ms,
        "nep_ms": nep_ms,
        "owner_parity_percent": 100.0 * healthy_ms / nep_ms,
        "nep_overhead_percent": 100.0 * (nep_ms / healthy_ms - 1.0),
        "healthy_trace_count": len(
            list((healthy / "torch_profile").glob("rank-*.json.gz"))
        ),
        "nep": nep_trace_summary(nep),
    }


def control_summary(root: Path, job_id: str) -> dict[str, object]:
    path = case_dir(root, job_id, healthy=False)
    return {
        "nep_ms": timing(path, job_id),
        "nep": nep_trace_summary(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topk-root",
        type=Path,
        default=Path("slurm_runs/lyris_a3b_ep8_ep4_topk_ab"),
    )
    parser.add_argument("--topk-job", default="2439279")
    parser.add_argument(
        "--prior-root",
        type=Path,
        default=Path("slurm_runs/lyris_a3b_ep8_ep4_router_balance/balanced"),
    )
    parser.add_argument("--prior-job", default="2438661")
    parser.add_argument("--block13-job")
    parser.add_argument("--block06-job")
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("slurm_runs/lyris_a3b_ep8_ep4_allocation_control"),
    )
    args = parser.parse_args()

    output = {
        "profiler_step": STEP_NAME,
        "topk6": pair_summary(args.topk_root / "topk6", args.topk_job),
        "topk8": pair_summary(args.topk_root / "topk8", args.topk_job),
        "prior_balanced": pair_summary(args.prior_root, args.prior_job),
    }
    if args.block13_job:
        output["block13_control"] = control_summary(
            args.control_root / "block13", args.block13_job
        )
    if args.block06_job:
        output["block06_control"] = control_summary(
            args.control_root / "block06", args.block06_job
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
