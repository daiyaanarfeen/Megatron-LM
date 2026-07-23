#!/usr/bin/env python3
"""Compare exact-uniform healthy EP8/EP8 with proportional NEP EP8/EP4."""

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

TIMING_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/.*?"
    r"elapsed time per iteration \(ms\):\s+(?P<elapsed>[0-9.]+)"
)
RANK_RE = re.compile(r"rank-(?P<rank>\d+)\.json\.gz$")
MODEL_EP_GROUPS = {"EXPERT_MODEL_PARALLEL_GROUP", "ep"}
TIMING_START = 30
TIMING_END = 100


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "std": std,
        "variance": std * std,
        "cv_percent": 100.0 * std / mean if mean else 0.0,
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def find_run_dir(root: Path, label: str, job_id: str) -> Path:
    candidates = [
        path
        for path in (root / label).glob(f"**/{job_id}")
        if path.is_dir() and (path / f"driver_{job_id}.log").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(f"{label}: expected one run directory, found {candidates}")
    return candidates[0]


def read_timings(path: Path) -> dict[int, float]:
    rows: dict[int, float] = {}
    for match in TIMING_RE.finditer(path.read_text(errors="replace")):
        iteration = int(match.group("iteration"))
        if TIMING_START <= iteration <= TIMING_END:
            rows[iteration] = float(match.group("elapsed"))
    expected = set(range(TIMING_START, TIMING_END + 1))
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        raise ValueError(f"{path}: incomplete timing window; missing {missing}")
    return rows


def condition_timings(
    root: Path, labels: list[str], job_id: str
) -> tuple[dict[str, Any], list[dict[int, float]]]:
    launches = []
    launch_rows = []
    for label in labels:
        run_dir = find_run_dir(root, label, job_id)
        rows = read_timings(run_dir / f"driver_{job_id}.log")
        values = list(rows.values())
        launches.append({"label": label, "run_dir": str(run_dir), **summarize(values)})
        launch_rows.append(rows)

    pooled = [value for rows in launch_rows for value in rows.values()]
    pooled_mean = statistics.mean(pooled)
    centered = [
        value - statistics.mean(rows.values()) + pooled_mean
        for rows in launch_rows
        for value in rows.values()
    ]
    iterations = sorted(set.intersection(*(set(rows) for rows in launch_rows)))
    position_means = [
        statistics.mean(rows[iteration] for rows in launch_rows) for iteration in iterations
    ]
    return (
        {
            "launches": launches,
            "pooled": summarize(pooled),
            "launch_centered": summarize(centered),
            "launch_mean_std": statistics.stdev(
                [float(launch["mean"]) for launch in launches]
            ),
            "repeat_position_correlation": pearson(
                list(launch_rows[0].values()), list(launch_rows[1].values())
            ),
            "position_mean": summarize(position_means),
        },
        launch_rows,
    )


def load_trace(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)["traceEvents"]


def profiler_step_stats(path: Path) -> dict[str, float | int]:
    steps = [
        float(event["dur"]) / 1000.0
        for event in load_trace(path)
        if event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("ProfilerStep#")
    ]
    return summarize(steps)


def split_vector_stats(trace_dir: Path) -> dict[str, Any]:
    by_group_size: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "records": 0,
            "vectors": 0,
            "exact_vectors": 0,
            "unique_vectors": set(),
            "vector_sums": set(),
            "max_abs_deviation": 0,
            "ranks": set(),
        }
    )
    traces = sorted(trace_dir.glob("rank-*.json.gz"))
    if not traces:
        raise ValueError(f"no rank traces found in {trace_dir}")
    for path in traces:
        rank_match = RANK_RE.search(path.name)
        if rank_match is None:
            raise ValueError(f"cannot parse rank from {path}")
        rank = int(rank_match.group("rank"))
        for event in load_trace(path):
            args = event.get("args", {})
            if (
                event.get("name") != "record_param_comms"
                or args.get("Collective name") != "all_to_allv"
                or args.get("Process Group Description") not in MODEL_EP_GROUPS
            ):
                continue
            group_size = int(args["Group size"])
            row = by_group_size[group_size]
            row["records"] += 1
            row["ranks"].add(rank)
            for field in ("In split size", "Out split size"):
                vector = tuple(int(value) for value in ast.literal_eval(args[field]))
                if len(vector) != group_size:
                    raise ValueError(f"{path}: {field} has length {len(vector)}")
                row["vectors"] += 1
                row["unique_vectors"].add(vector)
                row["vector_sums"].add(sum(vector))
                target = sum(vector) / len(vector)
                row["max_abs_deviation"] = max(
                    row["max_abs_deviation"],
                    max(abs(value - target) for value in vector),
                )
                if all(value == vector[0] for value in vector):
                    row["exact_vectors"] += 1

    output = {}
    for group_size, row in sorted(by_group_size.items()):
        output[str(group_size)] = {
            "records": row["records"],
            "vectors": row["vectors"],
            "exact_vectors": row["exact_vectors"],
            "all_vectors_exact": row["exact_vectors"] == row["vectors"],
            "unique_vector_count": len(row["unique_vectors"]),
            "unique_vectors": [list(vector) for vector in sorted(row["unique_vectors"])],
            "vector_sums": sorted(row["vector_sums"]),
            "max_abs_deviation": row["max_abs_deviation"],
            "ranks": sorted(row["ranks"]),
        }
    return {"trace_count": len(traces), "by_group_size": output}


def profile_summary(root: Path, label: str, job_id: str, ranks: list[int]) -> dict[str, Any]:
    run_dir = find_run_dir(root, label, job_id)
    trace_dir = run_dir / "torch_profile"
    rank_steps = {
        str(rank): profiler_step_stats(trace_dir / f"rank-{rank}.json.gz") for rank in ranks
    }
    return {
        "run_dir": str(run_dir),
        "rank_profiler_steps_ms": rank_steps,
        "model_ep_splits": split_vector_stats(trace_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("slurm_runs/lyris_a3b_ep8_exact_uniform_nep_ab"),
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    healthy, healthy_rows = condition_timings(
        args.root, ["healthy_timing_1", "healthy_timing_2"], args.job_id
    )
    nep, nep_rows = condition_timings(
        args.root, ["nep_timing_1", "nep_timing_2"], args.job_id
    )
    iterations = list(range(TIMING_START, TIMING_END + 1))
    position_deltas = [
        statistics.mean(rows[iteration] for rows in nep_rows)
        - statistics.mean(rows[iteration] for rows in healthy_rows)
        for iteration in iterations
    ]

    healthy_mean = float(healthy["pooled"]["mean"])
    nep_mean = float(nep["pooled"]["mean"])
    healthy_residual_std = float(healthy["launch_centered"]["std"])
    nep_residual_std = float(nep["launch_centered"]["std"])
    output = {
        "job_id": args.job_id,
        "timing_window": [TIMING_START, TIMING_END],
        "configuration": {
            "healthy": "TP2, EP8/EP8, topology 4 4, MBS1, one microbatch/replica, GBS8",
            "nep": "TP2, EP8/EP4, topology 4 2, MBS1, one microbatch/replica, GBS6",
            "routing": "exact uniform, 128 experts, top-k 6, frozen expert bias",
        },
        "timing": {"healthy": healthy, "nep": nep},
        "comparison": {
            "mean_gap_ms": nep_mean - healthy_mean,
            "mean_change_percent": 100.0 * (nep_mean / healthy_mean - 1.0),
            "owner_parity_percent": 100.0 * healthy_mean / nep_mean,
            "launch_centered_std_gap_ms": nep_residual_std - healthy_residual_std,
            "launch_centered_std_change_percent": 100.0
            * (nep_residual_std / healthy_residual_std - 1.0),
            "launch_centered_variance_change_percent": 100.0
            * (
                float(nep["launch_centered"]["variance"])
                / float(healthy["launch_centered"]["variance"])
                - 1.0
            ),
            "per_iteration_position_delta_ms": summarize(position_deltas),
        },
        "profiles": {
            "healthy": profile_summary(args.root, "healthy_profile", args.job_id, [0, 8]),
            "nep": profile_summary(args.root, "nep_profile", args.job_id, [0, 8]),
        },
    }
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
