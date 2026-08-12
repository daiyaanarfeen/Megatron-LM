#!/usr/bin/env python3
"""Attribute an EP16 healthy versus EP16/EP12 NEP slowdown from all-rank traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import analyze_exact_uniform_nep_slowdown as exact_uniform
from analyze_exact_uniform_nep_slowdown import (
    NCCL_GROUPS,
    aligned_collective_summary,
    mean_delta,
    model_ep_milestones,
    owner_step_metrics,
    parse_trace,
    summarize,
    summarize_nested_steps,
)


RANK_COUNTS = {"healthy": 32, "nep": 28}
ROLE_RANKS = {
    "healthy_replica_0": ("healthy", range(0, 16)),
    "healthy_replica_1": ("healthy", range(16, 32)),
    "nep_full_ep16": ("nep", range(0, 16)),
    "nep_reduced_ep12": ("nep", range(16, 28)),
}
POST_TRAIN_TP_NUMEL = 2_362_752


def filter_collective_participants(
    traces: dict[int, dict[str, Any]],
    ranks: list[int],
    group: str,
    participants: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    """Return trace views containing only one participant-scoped process group."""
    filtered = {}
    for rank in ranks:
        trace = traces[rank]
        collectives = {}
        for step_name, groups in trace["collectives"].items():
            collectives[step_name] = dict(groups)
            collectives[step_name][group] = [
                row for row in groups.get(group, []) if row["participants"] == participants
            ]
        filtered[rank] = {**trace, "collectives": collectives}
    return filtered


def summarize_rank_roles(traces: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Summarize rank-local grouped-GEMM work by replica role."""
    output = {}
    for role, (mode, ranks) in ROLE_RANKS.items():
        rows = [
            row
            for rank in ranks
            for row in traces[mode][rank]["rank_local_work"].values()
        ]
        output[role] = {
            key: summarize([float(row[key]) for row in rows]) for key in rows[0]
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healthy-trace-dir", type=Path, required=True)
    parser.add_argument("--nep-trace-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace_dirs = {"healthy": args.healthy_trace_dir, "nep": args.nep_trace_dir}
    traces: dict[str, dict[int, dict[str, Any]]] = {"healthy": {}, "nep": {}}
    for mode in ("healthy", "nep"):
        for rank in range(RANK_COUNTS[mode]):
            traces[mode][rank] = parse_trace(
                trace_dirs[mode] / f"rank-{rank}.json.gz",
                retain_gpu=rank == 0,
                expected_steps=args.steps,
            )

    exact_uniform.POST_TRAIN_TP_NUMEL = POST_TRAIN_TP_NUMEL
    owner_steps = {
        mode: owner_step_metrics(traces[mode][0], mode) for mode in ("healthy", "nep")
    }
    owner_summary = {
        mode: summarize_nested_steps(owner_steps[mode]) for mode in ("healthy", "nep")
    }
    backward_components = (
        "span_ms",
        "non_nccl_union_ms",
        "nccl_union_ms",
        "nccl_non_nccl_overlap_ms",
        "gpu_idle_ms",
    )
    backward_deltas = {
        component: mean_delta(
            owner_summary["healthy"], owner_summary["nep"], ("backward", component)
        )
        for component in backward_components
    }
    backward_deltas["reconstructed_span_delta_ms"] = (
        backward_deltas["non_nccl_union_ms"]
        + backward_deltas["nccl_union_ms"]
        - backward_deltas["nccl_non_nccl_overlap_ms"]
        + backward_deltas["gpu_idle_ms"]
    )

    groups = {
        "model_ep": {
            "healthy": (
                list(range(16)),
                NCCL_GROUPS["healthy"]["model_ep"],
            ),
            "nep": (list(range(16)), NCCL_GROUPS["nep"]["model_ep"]),
        },
        "expert_edp": {
            "healthy": ([0, 16], NCCL_GROUPS["healthy"]["expert_edp"]),
            "nep": ([0, 16], NCCL_GROUPS["nep"]["expert_edp"]),
        },
        "dense_dp": {
            "healthy": (list(range(0, 32, 2)), NCCL_GROUPS["healthy"]["dense_dp"]),
            "nep": (list(range(0, 28, 2)), NCCL_GROUPS["nep"]["dense_dp"]),
        },
        "tp": {
            "healthy": ([0, 1], NCCL_GROUPS["healthy"]["tp"]),
            "nep": ([0, 1], NCCL_GROUPS["nep"]["tp"]),
        },
    }
    aligned = {
        name: {
            mode: aligned_collective_summary(traces[mode], *groups[name][mode])
            for mode in ("healthy", "nep")
        }
        for name in groups
    }

    for name, group_key in (
        ("owner_gather", NCCL_GROUPS["nep"]["owner_gather"]),
        ("owner_transfer", NCCL_GROUPS["nep"]["owner_transfer"]),
    ):
        ranks = [0, 12]
        participants = tuple(ranks)
        filtered = filter_collective_participants(
            traces["nep"], ranks, group_key, participants
        )
        aligned[name] = {
            "nep": aligned_collective_summary(filtered, ranks, group_key)
        }

    milestones = {
        "healthy": model_ep_milestones(
            traces["healthy"], list(range(16)), NCCL_GROUPS["healthy"]["model_ep"]
        ),
        "nep": model_ep_milestones(
            traces["nep"], list(range(16)), NCCL_GROUPS["nep"]["model_ep"]
        ),
    }
    milestone_delta = {
        ordinal: float(milestones["nep"][ordinal]["mean"])
        - float(milestones["healthy"][ordinal]["mean"])
        for ordinal in milestones["healthy"]
    }

    output = {
        "trace_dirs": {mode: str(path) for mode, path in trace_dirs.items()},
        "owner_step_summary": owner_summary,
        "owner_step_rows": owner_steps,
        "backward_component_deltas_ms": backward_deltas,
        "participant_aligned_collectives": aligned,
        "model_ep_backward_milestones_ms": milestones,
        "model_ep_backward_milestone_delta_ms": milestone_delta,
        "rank_local_expert_work": summarize_rank_roles(traces),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
