#!/usr/bin/env python3
"""Build phase, overlap, and cross-rank figures for two NEP failure categories."""

from __future__ import annotations

import argparse
import ast
import bisect
import json
import pickle
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from analyze_workload_parity_pair import event_end, iter_trace_events

PHASES = (
    ("forward_inter", "Forward + inter-MB", "#56B4E9"),
    ("backward", "Autograd backward", "#0072B2"),
    ("grad_tail", "Gradient-finalization tail", "#E69F00"),
    ("optimizer", "Optimizer", "#009E73"),
    ("post_optimizer", "Post-optimizer", "#999999"),
)
OVERLAP_PARTS = (
    ("compute", "With useful compute", "#009E73"),
    ("other", "With other GPU work", "#6A3D9A"),
    ("exposed", "Fully exposed", "#D55E00"),
)
OPS = ("gather", "expert_edp", "scatter", "model_ep", "dense_dp")


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    size: int,
    *,
    bold: bool = False,
    anchor: str | None = None,
    fill: str = "#202124",
) -> None:
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def step_index(steps: list[tuple[str, float, float]], timestamp: float) -> int | None:
    starts = [row[1] for row in steps]
    index = bisect.bisect_right(starts, timestamp) - 1
    if index >= 0 and timestamp < steps[index][2]:
        return index
    return None


def cluster_backward(
    rows: list[tuple[float, float]], expected_microbatches: int
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """Split autograd evaluate scopes at the largest inter-microbatch gaps."""
    merged: list[list[float]] = []
    for start, end in sorted(rows):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    if len(merged) < expected_microbatches:
        raise ValueError(
            f"only {len(merged)} autograd intervals for {expected_microbatches} microbatches"
        )
    gaps = [(merged[index][0] - merged[index - 1][1], index) for index in range(1, len(merged))]
    boundaries = {index for _, index in sorted(gaps, reverse=True)[: expected_microbatches - 1]}
    clusters: list[list[float]] = []
    for index, (start, end) in enumerate(merged):
        if not clusters or index in boundaries:
            clusters.append([start, end])
        else:
            clusters[-1][1] = end
    if len(clusters) != expected_microbatches:
        raise ValueError(f"formed {len(clusters)} backward clusters")
    selected = sorted((gap for gap, index in gaps if index in boundaries))
    rejected = [gap for gap, index in gaps if index not in boundaries]
    return (
        [(start, end) for start, end in clusters],
        {
            "minimum_boundary_gap_ms": (min(selected) / 1000.0 if selected else 0.0),
            "maximum_internal_gap_ms": (max(rejected) / 1000.0 if rejected else 0.0),
        },
    )


def parse_ranks(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return tuple(int(rank) for rank in value if rank is not Ellipsis)


def scan_rank(path: Path, expected_microbatches: int) -> dict[str, Any]:
    """Read compact phase, polling, and collective metrics from one trace."""
    steps: list[tuple[str, float, float]] = []
    backward_rows: list[tuple[float, float]] = []
    optimizer_rows: list[tuple[float, float]] = []
    polling_rows: list[tuple[float, float]] = []
    records: dict[Any, dict[str, Any]] = {}
    nccl: dict[Any, list[tuple[float, float]]] = defaultdict(list)
    for event in iter_trace_events(path):
        category = event.get("cat")
        name = str(event.get("name", ""))
        if category == "user_annotation" and name.startswith("ProfilerStep#"):
            steps.append((name, float(event["ts"]), event_end(event)))
        elif category == "cpu_op" and name.startswith(
            "autograd::engine::evaluate_function:"
        ):
            backward_rows.append((float(event["ts"]), event_end(event)))
        elif category == "user_annotation" and name.startswith("Optimizer.step#"):
            optimizer_rows.append((float(event["ts"]), event_end(event)))
        elif category == "kernel" and "hybrid_ep::device_sync_kernel" in name:
            polling_rows.append((float(event["ts"]), event_end(event)))
        elif name == "record_param_comms":
            args = event.get("args", {})
            if args.get("Collective name") != "wait":
                records[args.get("External id")] = {
                    "ts": float(event["ts"]),
                    "description": str(args.get("Process Group Description", "")),
                    "collective": str(args.get("Collective name", "")),
                    "ranks": parse_ranks(args.get("Process Group Ranks", "[]")),
                    "elements": int(args.get("In msg nelems", 0)),
                }
        elif category == "kernel" and "nccl" in name.lower():
            nccl[event.get("args", {}).get("External id")].append(
                (float(event["ts"]), event_end(event))
            )

    steps.sort(key=lambda row: row[1])
    phases = {}
    polling = {}
    clustering = {}
    for step_name, start, end in steps:
        backward = [
            (left, right)
            for left, right in backward_rows
            if left >= start and right <= end
        ]
        clusters, diagnostics = cluster_backward(backward, expected_microbatches)
        optimizer = [
            (left, right)
            for left, right in optimizer_rows
            if left >= start and right <= end
        ]
        if not optimizer:
            raise ValueError(f"{path}: {step_name} has no optimizer scope")
        optimizer_start = min(left for left, _ in optimizer)
        optimizer_end = max(right for _, right in optimizer)
        backward_ms = sum(right - left for left, right in clusters) / 1000.0
        grad_tail_ms = (optimizer_start - clusters[-1][1]) / 1000.0
        optimizer_ms = (optimizer_end - optimizer_start) / 1000.0
        post_ms = (end - optimizer_end) / 1000.0
        forward_ms = (end - start) / 1000.0 - backward_ms - grad_tail_ms - optimizer_ms - post_ms
        values = {
            "forward_inter": forward_ms,
            "backward": backward_ms,
            "grad_tail": grad_tail_ms,
            "optimizer": optimizer_ms,
            "post_optimizer": post_ms,
        }
        if min(values.values()) < -1.0e-6:
            raise ValueError(f"{path}: negative phase in {step_name}: {values}")
        phases[step_name] = values
        clustering[step_name] = diagnostics
        polling[step_name] = sum(
            right - left
            for left, right in polling_rows
            if left >= start and right <= end
        ) / 1000.0

    collectives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for external_id, record in records.items():
        kernels = nccl.get(external_id, [])
        if not kernels:
            continue
        index = step_index(steps, record["ts"])
        if index is None:
            continue
        collectives[steps[index][0]].append(
            {
                **record,
                "duration_ms": sum(right - left for left, right in kernels) / 1000.0,
            }
        )
    for rows in collectives.values():
        rows.sort(key=lambda row: row["ts"])
    return {
        "phases": phases,
        "polling_ms": polling,
        "collectives": collectives,
        "clustering": clustering,
    }


def role_counts(case: dict[str, Any]) -> tuple[int, int]:
    workload = case["workload"]
    factor = int(workload["tensor_parallel"]) * int(workload["context_parallel"])
    topology = case["topology"]["nep"]
    return int(topology[0]) * factor, int(topology[1]) * factor


def trace_path(case: dict[str, Any], mode: str, rank: int) -> Path:
    comparison = Path(case["comparison_artifact"])
    job_id = comparison.stem.removeprefix("comparison_")
    return comparison.parent / mode / job_id / "torch_profile" / f"rank-{rank}.json.gz"


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def analyze_case(
    case: dict[str, Any], include_reduced: bool, cache_dir: Path
) -> dict[str, Any]:
    full_count, reduced_count = role_counts(case)
    output = {"label": case["label"], "modes": {}}
    for mode in ("healthy", "nep"):
        world = full_count * 2 if mode == "healthy" else full_count + reduced_count
        ranks = range(world if include_reduced else full_count)
        full_ga = int(case["batch_configuration"][mode]["replica_num_microbatches"][0])
        reduced_ga = int(case["batch_configuration"][mode]["replica_num_microbatches"][1])
        scans = {}
        for rank in ranks:
            expected = full_ga if rank < full_count else reduced_ga
            case_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", case["label"])
            cache_path = cache_dir / case_slug / case["job_id"] / mode / f"rank-{rank}.pkl"
            if cache_path.is_file():
                print(
                    f"[{case['label']}] {mode} rank {rank}/{world - 1} cached",
                    flush=True,
                )
                with cache_path.open("rb") as stream:
                    scans[rank] = pickle.load(stream)
                continue
            print(f"[{case['label']}] {mode} rank {rank}/{world - 1}", flush=True)
            scans[rank] = scan_rank(trace_path(case, mode, rank), expected)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as stream:
                pickle.dump(scans[rank], stream)
        owner_ranks = range(reduced_count)
        rank_phase_means = {
            rank: {
                phase: mean([row[phase] for row in scans[rank]["phases"].values()])
                for phase, _, _ in PHASES
            }
            for rank in owner_ranks
        }
        rank0 = rank_phase_means[0]
        phase_ranges = {
            phase: summarize([row[phase] for row in rank_phase_means.values()])
            for phase, _, _ in PHASES
        }
        polling_by_rank = {
            rank: mean(list(scans[rank]["polling_ms"].values()))
            for rank in range(full_count)
        }
        output["modes"][mode] = {
            "rank0_phases_ms": rank0,
            "owner_phase_ranges_ms": phase_ranges,
            "polling_by_full_rank_ms": polling_by_rank,
            "scans": scans,
        }
    return output


def operation_bars(case: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for mode in ("healthy", "nep"):
        operations = case["profile"][mode]["operations_mean"]
        output[mode] = {}
        for operation in OPS:
            row = operations[operation]
            compute = float(row["compute_overlap_ms"])
            other = max(0.0, float(row["any_overlap_ms"]) - compute)
            exposed = float(row["exposed_ms"])
            output[mode][operation] = {
                "compute": compute,
                "other": other,
                "exposed": exposed,
                "residency": float(row["residency_ms"]),
            }
    return output


def polling_wait(case_result: dict[str, Any]) -> dict[str, Any]:
    """Compare total per-rank polling residency without subtracting rank clocks."""
    output = {}
    for mode, mode_result in case_result["modes"].items():
        scans = mode_result["scans"]
        ranks = sorted(scans)
        step_names = sorted(scans[0]["polling_ms"])
        per_step = []
        for step_name in step_names:
            # Per-operation polling records are not retained by the light scan. The
            # robust cross-rank quantity available here is total polling residency.
            totals = [scans[rank]["polling_ms"][step_name] for rank in ranks]
            per_step.append(
                {
                    "rank0_residency_ms": totals[0],
                    "minimum_rank_residency_ms": min(totals),
                    "rank0_excess_ms": max(0.0, totals[0] - min(totals)),
                    "rank_range_ms": max(totals) - min(totals),
                }
            )
        output[mode] = {
            key: mean([row[key] for row in per_step]) for key in per_step[0]
        }
    return output


def edp_wait(case_result: dict[str, Any]) -> dict[str, Any]:
    """Match rank-0 EDP calls with their actual participants by group and ordinal."""
    output = {}
    aliases = {"ep_dp", "EXPERT_DATA_PARALLEL_GROUP"}
    for mode, mode_result in case_result["modes"].items():
        scans = mode_result["scans"]
        step_names = sorted(scans[0]["collectives"])
        per_step = []
        for step_name in step_names:
            owner_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in scans[0]["collectives"][step_name]:
                if row["description"] in aliases and 0 in row["ranks"]:
                    key = (row["ranks"], row["description"], row["collective"])
                    owner_groups[key].append(row)
            service = 0.0
            owner_excess = 0.0
            owner_residency = 0.0
            count = 0
            for key, owner_rows in owner_groups.items():
                group_ranks, description, collective = key
                peer_rows = {}
                for rank in group_ranks:
                    rows = [
                        row
                        for row in scans[rank]["collectives"][step_name]
                        if row["ranks"] == group_ranks
                        and row["description"] == description
                        and row["collective"] == collective
                    ]
                    peer_rows[rank] = rows
                counts = {len(rows) for rows in peer_rows.values()} | {len(owner_rows)}
                if len(counts) != 1:
                    raise ValueError(f"{mode} {step_name} EDP count mismatch: {counts}")
                for index, owner_row in enumerate(owner_rows):
                    durations = [peer_rows[rank][index]["duration_ms"] for rank in group_ranks]
                    minimum = min(durations)
                    service += minimum
                    owner_residency += owner_row["duration_ms"]
                    owner_excess += max(0.0, owner_row["duration_ms"] - minimum)
                    count += 1
            per_step.append(
                {
                    "service_proxy_ms": service,
                    "owner_excess_ms": owner_excess,
                    "owner_residency_ms": owner_residency,
                    "operation_count": count,
                }
            )
        output[mode] = {key: mean([row[key] for row in per_step]) for key in per_step[0]}
    return output


def draw_stacked_bars(
    rows: list[dict[str, Any]],
    parts: tuple[tuple[str, str, str], ...],
    output: Path,
    title: str,
    subtitle: str,
    unit: str = "ms",
) -> None:
    width, height = 1800, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    text(draw, (width / 2, 35), title, 31, bold=True, anchor="ma")
    text(draw, (width / 2, 82), subtitle, 18, anchor="ma")
    left, right, top, bottom = 150, 1720, 190, 790
    maximum = max(sum(max(0.0, row["values"].get(key, 0.0)) for key, _, _ in parts) for row in rows)
    maximum *= 1.15
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((left, y, right, y), fill="#DDDDDD")
        text(draw, (left - 12, y), f"{value:.0f}", 15, anchor="rm")
    spacing = (right - left) / len(rows)
    bar_width = min(230, spacing * 0.62)
    for index, row in enumerate(rows):
        center = left + spacing * (index + 0.5)
        value_start = 0.0
        for key, _, color in parts:
            value = max(0.0, row["values"].get(key, 0.0))
            y0 = bottom - (bottom - top) * value_start / maximum
            value_start += value
            y1 = bottom - (bottom - top) * value_start / maximum
            draw.rectangle((center - bar_width / 2, y1, center + bar_width / 2, y0), fill=color)
            if y0 - y1 > 25:
                text(draw, (center, (y0 + y1) / 2), f"{value:.1f}", 14, bold=True, anchor="mm", fill="white")
        text(draw, (center, bottom + 28), row["label"], 17, bold=True, anchor="ma")
        text(draw, (center, y1 - 10), f"{value_start:.1f} {unit}", 15, bold=True, anchor="ms")
    x = 170
    for _, label, color in parts:
        draw.rectangle((x, 875, x + 22, 897), fill=color)
        text(draw, (x + 31, 886), label, 17, anchor="lm")
        x += 31 + int(draw.textlength(label, font=font(17))) + 55
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def render_category(
    name: str,
    cases: list[dict[str, Any]],
    analysis: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    phase_rows = []
    for case, result in zip(cases, analysis):
        for mode in ("healthy", "nep"):
            phase_rows.append(
                {
                    "label": f"{case['label']}\n{mode}",
                    "values": result["modes"][mode]["rank0_phases_ms"],
                }
            )
    draw_stacked_bars(
        phase_rows,
        PHASES,
        output_dir / f"{name}_1_iteration_phases.png",
        f"{name}: exact rank-0 profiler-step phase accounting",
        "CPU step span; every bar closes exactly. Forward/inter-MB includes time between autograd calls.",
    )

    operation_rows = []
    for case in cases:
        bars = operation_bars(case)
        for operation in OPS:
            if bars["healthy"][operation]["residency"] > 0:
                operation_rows.append(
                    {
                        "label": f"{case['label']}\nH {operation}",
                        "values": bars["healthy"][operation],
                    }
                )
            if bars["nep"][operation]["residency"] > 0:
                operation_rows.append(
                    {
                        "label": f"{case['label']}\nN {operation}",
                        "values": bars["nep"][operation],
                    }
                )
    draw_stacked_bars(
        operation_rows,
        OVERLAP_PARTS,
        output_dir / f"{name}_2_operation_overlap.png",
        f"{name}: rank-0 communication residency and overlap",
        "Operation bars are non-additive; same-family intervals are unioned before overlap is measured.",
    )

    if name == "category1_amortization":
        wait_rows = []
        for case, result in zip(cases, analysis):
            for mode, row in polling_wait(result).items():
                wait_rows.append(
                    {
                        "label": f"{case['label']}\n{mode}",
                        "values": {
                            "baseline": row["minimum_rank_residency_ms"],
                            "excess": row["rank0_excess_ms"],
                        },
                    }
                )
        wait_parts = (
            ("baseline", "Minimum full-rank polling residency", "#56B4E9"),
            ("excess", "Rank-0 excess over minimum", "#D55E00"),
        )
        subtitle = "Clock-independent full-replica rank comparison; totals are polling residency, not transfer time."
    else:
        wait_rows = []
        for case, result in zip(cases, analysis):
            for mode, row in edp_wait(result).items():
                wait_rows.append(
                    {
                        "label": f"{case['label']}\n{mode}",
                        "values": {
                            "baseline": row["service_proxy_ms"],
                            "excess": row["owner_excess_ms"],
                        },
                    }
                )
        wait_parts = (
            ("baseline", "Minimum participant residency (service proxy)", "#CC79A7"),
            ("excess", "Rank-0 excess participant residency", "#D55E00"),
        )
        subtitle = "Matched by actual process-group ranks and ordinal; duration minima avoid cross-rank clock subtraction."
    draw_stacked_bars(
        wait_rows,
        wait_parts,
        output_dir / f"{name}_3_cross_rank_wait.png",
        f"{name}: cross-rank source of the dominant wait",
        subtitle,
    )


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    output = {"label": result["label"], "modes": {}}
    for mode, row in result["modes"].items():
        output["modes"][mode] = {
            key: value for key, value in row.items() if key != "scans"
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category1-summary", type=Path, required=True)
    parser.add_argument("--category2-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    categories = {
        "category1_amortization": json.loads(args.category1_summary.read_text())["cases"],
        "category2_readiness": json.loads(args.category2_summary.read_text())["cases"],
    }
    all_results = {}
    for name, cases in categories.items():
        analyses = [
            analyze_case(
                case,
                include_reduced=name == "category2_readiness",
                cache_dir=args.output_dir / "cache",
            )
            for case in cases
        ]
        render_category(name, cases, analyses, args.output_dir)
        all_results[name] = {
            "cases": [compact_result(result) for result in analyses],
            "operation_overlap": [operation_bars(case) for case in cases],
            "cross_rank": [
                polling_wait(result) if name == "category1_amortization" else edp_wait(result)
                for result in analyses
            ],
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "mechanism_details.json"
    data_path.write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {data_path}")


if __name__ == "__main__":
    main()
