#!/usr/bin/env python3
"""Analyze a full-original a3b healthy EP16 versus NEP EP16/EP12 trace pair."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_exact_uniform_nep_slowdown import (
    NCCL_GROUPS,
    intersection_duration,
    parse_trace,
    union_duration,
)

GPU_GROUPS = {
    "healthy": {
        "expert_edp": NCCL_GROUPS["healthy"]["expert_edp"],
        "dense_dp": NCCL_GROUPS["healthy"]["dense_dp"],
        "tp": NCCL_GROUPS["healthy"]["tp"],
        "dp_scalar": "DATA_PARALLEL_GROUP",
        "mp_scalar": "MODEL_PARALLEL_GROUP",
        "tp_cp_scalar": "TENSOR_AND_CONTEXT_PARALLEL_GROUP",
        "tp_dp_cp_scalar": "TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP",
        "default_pg": "default_pg",
    },
    "nep": {
        "expert_edp": NCCL_GROUPS["nep"]["expert_edp"],
        "dense_dp": NCCL_GROUPS["nep"]["dense_dp"],
        "tp": NCCL_GROUPS["nep"]["tp"],
        "gather": NCCL_GROUPS["nep"]["owner_gather"],
        "scatter": NCCL_GROUPS["nep"]["owner_transfer"],
        "dp_scalar": "dp",
        "mp_scalar": "mp",
        "tp_cp_scalar": "tp_cp",
        "tp_dp_cp_scalar": "tp_dp_cp",
        "default_pg": "default_pg",
    },
}
ENVELOPE_KEYS = (
    "span_ms",
    "gpu_active_ms",
    "gpu_idle_ms",
    "nccl_union_ms",
    "non_nccl_union_ms",
    "nccl_non_nccl_overlap_ms",
)
PHASE_KEYS = (
    "residency_ms",
    "overlap_any_ms",
    "overlap_non_nccl_ms",
    "overlap_other_nccl_ms",
    "exposed_ms",
)


def summary(values: list[float]) -> dict[str, float | int]:
    """Return compact statistics."""
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def intervals(events: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Return event intervals."""
    return [(float(event["start"]), float(event["end"])) for event in events]


def phase_metrics(
    target: list[dict[str, Any]], other: list[dict[str, Any]]
) -> dict[str, float]:
    """Measure target residency and overlap with disjoint work categories."""
    target_intervals = intervals(target)
    other_intervals = intervals(other)
    non_nccl_intervals = intervals([event for event in other if not event["is_nccl"]])
    other_nccl_intervals = intervals([event for event in other if event["is_nccl"]])
    residency = union_duration(target_intervals)
    overlap_any = intersection_duration(target_intervals, other_intervals)
    return {
        "residency_ms": residency / 1000.0,
        "overlap_any_ms": overlap_any / 1000.0,
        "overlap_non_nccl_ms": intersection_duration(
            target_intervals, non_nccl_intervals
        )
        / 1000.0,
        "overlap_other_nccl_ms": intersection_duration(
            target_intervals, other_nccl_intervals
        )
        / 1000.0,
        "exposed_ms": (residency - overlap_any) / 1000.0,
    }


def event_group(event: dict[str, Any]) -> str | None:
    """Return the process-group description of an event."""
    value = event.get("group")
    return str(value) if value is not None else None


def analyze_step(
    trace: dict[str, Any], mode: str, step_name: str
) -> dict[str, Any]:
    """Analyze one profiler step within its first-to-last GPU-event envelope."""
    events = trace["gpu"][step_name]
    if not events:
        raise ValueError(f"{mode} rank {trace['rank']} {step_name}: no GPU events")
    start = min(float(event["start"]) for event in events)
    end = max(float(event["end"]) for event in events)
    all_intervals = intervals(events)
    nccl_events = [event for event in events if event["is_nccl"]]
    non_nccl_events = [event for event in events if not event["is_nccl"]]
    nccl_intervals = intervals(nccl_events)
    non_nccl_intervals = intervals(non_nccl_events)
    active = union_duration(all_intervals)
    span = end - start
    envelope = {
        "span_ms": span / 1000.0,
        "gpu_active_ms": active / 1000.0,
        "gpu_idle_ms": (span - active) / 1000.0,
        "nccl_union_ms": union_duration(nccl_intervals) / 1000.0,
        "non_nccl_union_ms": union_duration(non_nccl_intervals) / 1000.0,
        "nccl_non_nccl_overlap_ms": intersection_duration(
            nccl_intervals, non_nccl_intervals
        )
        / 1000.0,
    }

    known_groups = set(GPU_GROUPS[mode].values())
    phases: dict[str, dict[str, float]] = {}
    for canonical, group in GPU_GROUPS[mode].items():
        target = [
            event
            for event in events
            if event["is_nccl"] and event_group(event) == group
        ]
        other = [
            event
            for event in events
            if not (event["is_nccl"] and event_group(event) == group)
        ]
        phases[canonical] = phase_metrics(target, other)

    other_nccl = [
        event
        for event in events
        if event["is_nccl"] and event_group(event) not in known_groups
    ]
    phases["other_nccl"] = phase_metrics(
        other_nccl,
        [
            event
            for event in events
            if not (event["is_nccl"] and event_group(event) not in known_groups)
        ],
    )

    non_nccl_categories = {
        "hybrid_ep_kernels": [
            event
            for event in non_nccl_events
            if "hybrid_ep" in str(event["name"]).lower()
        ],
        "expert_gemm_kernels": [
            event
            for event in non_nccl_events
            if str(event["name"]).startswith("nvjet")
        ],
    }
    for category, target in non_nccl_categories.items():
        target_ids = {id(event) for event in target}
        phases[category] = phase_metrics(
            target, [event for event in events if id(event) not in target_ids]
        )

    dispatch_stream = trace.get("dispatch_stream") if mode == "nep" else None
    staging = [
        event
        for event in non_nccl_events
        if dispatch_stream is not None and event.get("stream") == dispatch_stream
    ]
    phases["owner_layout_staging"] = phase_metrics(
        staging,
        [
            event
            for event in events
            if not (
                not event["is_nccl"]
                and dispatch_stream is not None
                and event.get("stream") == dispatch_stream
            )
        ],
    )

    if mode == "nep":
        reshard_groups = {
            GPU_GROUPS[mode]["gather"],
            GPU_GROUPS[mode]["expert_edp"],
            GPU_GROUPS[mode]["scatter"],
        }
        reshard_nccl = [
            event
            for event in events
            if event["is_nccl"] and event_group(event) in reshard_groups
        ]
        reshard_nccl_ids = {id(event) for event in reshard_nccl}
        phases["reshard_nccl_union"] = phase_metrics(
            reshard_nccl,
            [event for event in events if id(event) not in reshard_nccl_ids],
        )
        reshard_all = reshard_nccl + staging
        reshard_all_ids = {id(event) for event in reshard_all}
        phases["reshard_with_staging_union"] = phase_metrics(
            reshard_all, [event for event in events if id(event) not in reshard_all_ids]
        )

        native = [event for event in events if id(event) not in reshard_all_ids]
        last_native_end = max(
            (float(event["end"]) for event in native), default=start
        )
        tail_events = [event for event in reshard_all if float(event["end"]) > last_native_end]
        tail_start = last_native_end
        tail_end = max((float(event["end"]) for event in tail_events), default=tail_start)
        post_native_tail = {
            "span_ms": (tail_end - tail_start) / 1000.0,
            "active_ms": union_duration(
                [
                    (max(tail_start, float(event["start"])), float(event["end"]))
                    for event in tail_events
                    if float(event["end"]) > tail_start
                ]
            )
            / 1000.0,
            "event_count": len(tail_events),
        }
    else:
        post_native_tail = {"span_ms": 0.0, "active_ms": 0.0, "event_count": 0}

    collectives = trace["collectives"][step_name]
    collective_counts = {
        canonical: len(collectives.get(group, []))
        for canonical, group in GPU_GROUPS[mode].items()
    }
    collective_counts["other_nccl"] = sum(
        len(rows) for group, rows in collectives.items() if group not in known_groups
    )

    return {
        "envelope": envelope,
        "phases": phases,
        "post_native_reshard_tail": post_native_tail,
        "collective_counts": collective_counts,
        "dispatch_stream": dispatch_stream,
        "rank_local_work": trace["rank_local_work"][step_name],
    }


def summarize_steps(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize scalar step metrics."""
    values = list(rows.values())
    phase_names = sorted({name for row in values for name in row["phases"]})
    return {
        "envelope": {
            key: summary([float(row["envelope"][key]) for row in values])
            for key in ENVELOPE_KEYS
        },
        "phases": {
            phase: {
                key: summary(
                    [float(row["phases"].get(phase, {}).get(key, 0.0)) for row in values]
                )
                for key in PHASE_KEYS
            }
            for phase in phase_names
        },
        "post_native_reshard_tail": {
            key: summary(
                [float(row["post_native_reshard_tail"][key]) for row in values]
            )
            for key in ("span_ms", "active_ms", "event_count")
        },
        "rank_local_work": {
            key: summary([float(row["rank_local_work"][key]) for row in values])
            for key in values[0]["rank_local_work"]
        },
    }


def mean_at(summary_row: dict[str, Any], *path: str) -> float:
    """Read a mean from a nested summary."""
    node = summary_row
    for key in path:
        node = node[key]
    return float(node["mean"])


def compare_owner(healthy: dict[str, Any], nep: dict[str, Any]) -> dict[str, Any]:
    """Compare rank-0 means and produce an exact envelope decomposition."""
    envelope_delta = {
        key: mean_at(nep, "envelope", key) - mean_at(healthy, "envelope", key)
        for key in ENVELOPE_KEYS
    }
    envelope_delta["reconstructed_span_delta_ms"] = (
        envelope_delta["non_nccl_union_ms"]
        + envelope_delta["nccl_union_ms"]
        - envelope_delta["nccl_non_nccl_overlap_ms"]
        + envelope_delta["gpu_idle_ms"]
    )
    common_phases = sorted(set(healthy["phases"]) & set(nep["phases"]))
    phase_deltas = {
        phase: {
            key: mean_at(nep, "phases", phase, key)
            - mean_at(healthy, "phases", phase, key)
            for key in PHASE_KEYS
        }
        for phase in common_phases
    }
    incremental_reshard_exposure = mean_at(
        nep, "phases", "reshard_nccl_union", "exposed_ms"
    ) - mean_at(healthy, "phases", "expert_edp", "exposed_ms")
    return {
        "envelope_delta_ms": envelope_delta,
        "common_phase_deltas_ms": phase_deltas,
        "incremental_reshard_exposure_vs_healthy_edp_ms": incremental_reshard_exposure,
    }



def align_collective_participants(
    mode_ranks: dict[str, Any], canonical: str
) -> dict[str, Any]:
    """Align rank-0/rank-16 records and separate arrival wait from service."""
    rows = []
    rank_zero = mode_ranks["0"]["collective_records"][canonical]
    rank_sixteen = mode_ranks["16"]["collective_records"][canonical]
    if set(rank_zero) != set(rank_sixteen):
        raise ValueError(f"rank-0/rank-16 {canonical} profiler steps differ")
    for step_name in sorted(rank_zero):
        zero_rows = rank_zero[step_name]
        sixteen_rows = rank_sixteen[step_name]
        if len(zero_rows) != len(sixteen_rows):
            raise ValueError(
                f"{step_name}: {canonical} operation count differs: "
                f"{len(zero_rows)} versus {len(sixteen_rows)}"
            )
        for ordinal, (zero, sixteen) in enumerate(zip(zero_rows, sixteen_rows)):
            if (
                zero["payload_elements"] != sixteen["payload_elements"]
                or tuple(zero["participants"]) != tuple(sixteen["participants"])
            ):
                raise ValueError(
                    f"{step_name} {canonical} operation {ordinal} does not match"
                )
            starts = [float(zero["kernel_start"]), float(sixteen["kernel_start"])]
            ends = [float(zero["kernel_end"]), float(sixteen["kernel_end"])]
            last_start = max(starts)
            first_end = min(ends)
            rows.append(
                {
                    "step": step_name,
                    "ordinal": ordinal,
                    "payload_elements": int(zero["payload_elements"]),
                    "participants": list(zero["participants"]),
                    "rank0_residency_ms": (ends[0] - starts[0]) / 1000.0,
                    "rank16_residency_ms": (ends[1] - starts[1]) / 1000.0,
                    "start_spread_ms": abs(starts[0] - starts[1]) / 1000.0,
                    "rank0_pre_last_arrival_wait_ms": max(0.0, last_start - starts[0])
                    / 1000.0,
                    "rank16_pre_last_arrival_wait_ms": max(0.0, last_start - starts[1])
                    / 1000.0,
                    "matched_service_ms": max(0.0, first_end - last_start) / 1000.0,
                    "end_spread_ms": abs(ends[0] - ends[1]) / 1000.0,
                }
            )
    numeric = (
        "rank0_residency_ms",
        "rank16_residency_ms",
        "start_spread_ms",
        "rank0_pre_last_arrival_wait_ms",
        "rank16_pre_last_arrival_wait_ms",
        "matched_service_ms",
        "end_spread_ms",
    )
    return {
        "rows": rows,
        "summary": {key: summary([float(row[key]) for row in rows]) for key in numeric},
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healthy-trace-dir", type=Path, required=True)
    parser.add_argument("--nep-trace-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 16])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "trace_dirs": {
            "healthy": str(args.healthy_trace_dir),
            "nep": str(args.nep_trace_dir),
        },
        "ranks": {},
    }
    for mode, trace_dir in (
        ("healthy", args.healthy_trace_dir),
        ("nep", args.nep_trace_dir),
    ):
        result["ranks"][mode] = {}
        for rank in args.ranks:
            path = trace_dir / f"rank-{rank}.json.gz"
            print(f"[{mode}] parsing {path}", flush=True)
            trace = parse_trace(path, retain_gpu=True, expected_steps=args.steps)
            rows = {
                step_name: analyze_step(trace, mode, step_name)
                for step_name in trace["step_spans"]
            }
            aligned_groups = ("expert_edp", "dense_dp", "tp_dp_cp_scalar", "default_pg")
            collective_records = {
                canonical: {
                    step_name: [
                        {
                            key: record[key]
                            for key in (
                                "payload_elements",
                                "participants",
                                "record_start",
                                "kernel_start",
                                "kernel_end",
                            )
                        }
                        for record in trace["collectives"][step_name].get(
                            GPU_GROUPS[mode][canonical], []
                        )
                    ]
                    for step_name in trace["step_spans"]
                }
                for canonical in aligned_groups
            }
            result["ranks"][mode][str(rank)] = {
                "steps": rows,
                "summary": summarize_steps(rows),
                "collective_records": collective_records,
            }
            del trace
            gc.collect()

    healthy_owner = result["ranks"]["healthy"]["0"]["summary"]
    nep_owner = result["ranks"]["nep"]["0"]["summary"]
    result["owner_comparison"] = compare_owner(healthy_owner, nep_owner)
    result["participant_alignment"] = {
        canonical: {
            mode: align_collective_participants(result["ranks"][mode], canonical)
            for mode in ("healthy", "nep")
        }
        for canonical in ("expert_edp", "dense_dp", "tp_dp_cp_scalar", "default_pg")
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    comparison = result["owner_comparison"]
    delta = comparison["envelope_delta_ms"]
    print(f"wrote {args.output}")
    print(
        "owner GPU-envelope delta: "
        f"{delta['span_ms']:.3f} ms = non-NCCL {delta['non_nccl_union_ms']:+.3f} "
        f"+ NCCL {delta['nccl_union_ms']:+.3f} - overlap "
        f"{delta['nccl_non_nccl_overlap_ms']:+.3f} + idle {delta['gpu_idle_ms']:+.3f}"
    )
    print(
        "incremental exposed Gather/EDP/Scatter union versus healthy EDP: "
        f"{comparison['incremental_reshard_exposure_vs_healthy_edp_ms']:.3f} ms"
    )


if __name__ == "__main__":
    main()
