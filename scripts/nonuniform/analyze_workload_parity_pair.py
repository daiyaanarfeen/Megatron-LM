#!/usr/bin/env python3
"""Compare owner-rank overhead composition for two healthy/NEP workload pairs."""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import statistics
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from analyze_exact_uniform_nep_slowdown import (
    intersection_duration,
    is_nccl,
    merge_intervals,
    union_duration,
)


GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
STACK_ORDER = (
    "compute_only",
    "compute_comm_overlap",
    "native_comm_only",
    "expert_edp_only",
    "gather_only",
    "scatter_only",
    "nep_staging_only",
    "comm_comm_overlap",
    "gpu_idle",
)
STACK_LABELS = {
    "compute_only": "Compute only",
    "compute_comm_overlap": "Compute + comm",
    "native_comm_only": "Native comm only",
    "expert_edp_only": "Expert EDP only",
    "gather_only": "NEP Gather only",
    "scatter_only": "NEP Scatter only",
    "nep_staging_only": "NEP staging only",
    "comm_comm_overlap": "Comm + comm",
    "gpu_idle": "GPU idle",
}
STACK_COLORS = {
    "compute_only": "#56B4E9",
    "compute_comm_overlap": "#009E73",
    "native_comm_only": "#0072B2",
    "expert_edp_only": "#CC79A7",
    "gather_only": "#E69F00",
    "scatter_only": "#D55E00",
    "nep_staging_only": "#F0E442",
    "comm_comm_overlap": "#6A3D9A",
    "gpu_idle": "#B8B8B8",
}
RESHARD_CATEGORIES = ("gather", "expert_edp", "scatter", "nep_staging")
TRACKED_CATEGORIES = ("model_ep", "dense_dp", "other_comm", *RESHARD_CATEGORIES)
OVERHEAD_ORDER = (
    "incremental_expert_edp",
    "default_and_other_sync",
    "scatter",
    "nep_staging",
    "gather",
    "unattributed_residual",
)
OVERHEAD_LABELS = {
    "incremental_expert_edp": "Incremental exposed expert EDP",
    "default_and_other_sync": "Default/other sync delta",
    "scatter": "Exposed Scatter",
    "nep_staging": "Exposed NEP staging",
    "gather": "Exposed Gather",
    "unattributed_residual": "Unattributed measured residual",
}
OVERHEAD_COLORS = {
    category: STACK_COLORS[stack_category]
    for category, stack_category in zip(
        OVERHEAD_ORDER,
        (
            "expert_edp_only",
            "native_comm_only",
            "scatter_only",
            "nep_staging_only",
            "gather_only",
            "gpu_idle",
        ),
    )
}


def load_trace_events(path: Path) -> list[dict[str, Any]]:
    """Load trace events from one compressed PyTorch profile."""
    with gzip.open(path, "rt") as stream:
        return json.load(stream)["traceEvents"]


def iter_trace_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield trace events without materializing the complete JSON document."""
    marker = '"traceEvents"'
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        buffer = ""
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError(f"{path}: traceEvents array not found")
            buffer += chunk
            marker_index = buffer.find(marker)
            if marker_index < 0:
                buffer = buffer[-len(marker) :]
                continue
            array_index = buffer.find("[", marker_index + len(marker))
            while array_index < 0:
                chunk = stream.read(chunk_size)
                if not chunk:
                    raise ValueError(f"{path}: unterminated traceEvents key")
                buffer += chunk
                array_index = buffer.find("[", marker_index + len(marker))
            buffer = buffer[array_index + 1 :]
            break

        position = 0
        eof = False
        while True:
            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as error:
                if eof:
                    raise ValueError(f"{path}: incomplete trace event") from error
                buffer = buffer[position:]
                position = 0
                chunk = stream.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            if not isinstance(event, dict):
                raise ValueError(f"{path}: non-object trace event")
            yield event
            position = end
            if position >= chunk_size:
                buffer = buffer[position:]
                position = 0


def event_end(event: dict[str, Any]) -> float:
    """Return one trace event's end timestamp."""
    return float(event["ts"]) + float(event.get("dur", 0.0))


def find_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ordered profiler-step annotations."""
    return sorted(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and str(event.get("name", "")).startswith("ProfilerStep#")
        ),
        key=lambda event: float(event["ts"]),
    )


def scoped_external_ids(
    events: list[dict[str, Any]], prefixes: tuple[str, ...]
) -> set[Any]:
    """Return external IDs launched from CPU scopes with the requested prefixes."""
    scope_intervals: dict[tuple[Any, Any], list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event.get("cat") not in {"cpu_op", "user_annotation"}:
            continue
        if not str(event.get("name", "")).startswith(prefixes):
            continue
        scope_intervals[(event.get("pid"), event.get("tid"))].append(
            (float(event["ts"]), event_end(event))
        )
    merged_scopes = {
        thread: merge_intervals(intervals)
        for thread, intervals in scope_intervals.items()
    }
    starts = {
        thread: [start for start, _ in intervals]
        for thread, intervals in merged_scopes.items()
    }

    external_ids: set[Any] = set()
    for event in events:
        if event.get("cat") not in {"cpu_op", "user_annotation"}:
            continue
        external_id = event.get("args", {}).get("External id")
        if external_id is None:
            continue
        thread = (event.get("pid"), event.get("tid"))
        intervals = merged_scopes.get(thread, [])
        index = bisect.bisect_right(starts.get(thread, []), float(event["ts"])) - 1
        if index >= 0 and float(event["ts"]) < intervals[index][1]:
            external_ids.add(external_id)
    return external_ids


def communication_groups(events: list[dict[str, Any]]) -> dict[Any, str]:
    """Map collective kernel external IDs to process-group descriptions."""
    groups = {}
    for event in events:
        if event.get("name") != "record_param_comms":
            continue
        args = event.get("args", {})
        external_id = args.get("External id")
        group = args.get("Process Group Description")
        if external_id is not None and group is not None:
            groups[external_id] = str(group)
    return groups


def classify_process_group(group: str | None) -> str:
    """Map a process-group description to a canonical communication category."""
    normalized = str(group or "").lower()
    if "owner_gather" in normalized:
        return "gather"
    if "owner_transfer" in normalized:
        return "scatter"
    if normalized == "ep_dp" or "expert_data_parallel" in normalized:
        return "expert_edp"
    if normalized == "dp_cp" or "data_parallel_group_with_cp" in normalized:
        return "dense_dp"
    return "other_comm"


def clip_interval(
    event: dict[str, Any], start: float, end: float
) -> tuple[float, float] | None:
    """Clip a GPU event to one profiler step."""
    clipped_start = max(start, float(event["ts"]))
    clipped_end = min(end, event_end(event))
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end


def classify_gpu_events(
    events: list[dict[str, Any]],
    step: dict[str, Any],
    groups: dict[Any, str],
    hybrid_ep_ids: set[Any],
    nep_ids: set[Any],
) -> dict[str, list[tuple[float, float]]]:
    """Classify GPU intervals in one profiler step."""
    start = float(step["ts"])
    end = event_end(step)
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for event in events:
        if event.get("cat") not in GPU_CATEGORIES:
            continue
        interval = clip_interval(event, start, end)
        if interval is None:
            continue
        args = event.get("args", {})
        external_id = args.get("External id")
        if is_nccl(event):
            group = args.get("Process Group Description") or groups.get(external_id)
            category = classify_process_group(str(group) if group is not None else None)
        elif external_id in hybrid_ep_ids:
            category = "model_ep"
        elif external_id in nep_ids:
            category = "nep_staging"
        else:
            category = "compute"
        intervals[category].append(interval)

    return {category: merge_intervals(rows) for category, rows in intervals.items()}


def exclusive_stack(
    intervals: dict[str, list[tuple[float, float]]], start: float, end: float
) -> dict[str, float]:
    """Produce mutually exclusive GPU-state durations that sum to the step span."""
    changes: dict[float, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    changes[start]
    changes[end]
    for category, rows in intervals.items():
        for interval_start, interval_end in rows:
            changes[interval_start][category] += 1
            changes[interval_end][category] -= 1

    ordered = sorted(changes)
    active_counts: dict[str, int] = defaultdict(int)
    output = {category: 0.0 for category in STACK_ORDER}
    for left, right in zip(ordered, ordered[1:]):
        for category, delta in changes[left].items():
            active_counts[category] += delta
        active = {
            category for category, count in active_counts.items() if count > 0
        }
        duration = right - left
        if not active:
            bucket = "gpu_idle"
        elif "compute" in active:
            bucket = "compute_comm_overlap" if active - {"compute"} else "compute_only"
        elif len(active) > 1:
            bucket = "comm_comm_overlap"
        else:
            category = next(iter(active))
            if category in {"model_ep", "dense_dp", "other_comm"}:
                bucket = "native_comm_only"
            elif category == "expert_edp":
                bucket = "expert_edp_only"
            elif category == "gather":
                bucket = "gather_only"
            elif category == "scatter":
                bucket = "scatter_only"
            elif category == "nep_staging":
                bucket = "nep_staging_only"
            else:
                raise ValueError(f"unhandled interval category: {category}")
        output[bucket] += duration / 1000.0
    return output


def operation_overlap(
    intervals: dict[str, list[tuple[float, float]]], category: str
) -> dict[str, float]:
    """Measure one category's residency, overlap, and fully exposed time."""
    selected = intervals.get(category, [])
    residency = union_duration(selected)
    compute_overlap = intersection_duration(selected, intervals.get("compute", []))
    other = [
        interval
        for other_category, rows in intervals.items()
        if other_category != category
        for interval in rows
    ]
    any_overlap = intersection_duration(selected, other)
    return {
        "residency_ms": residency / 1000.0,
        "compute_overlap_ms": compute_overlap / 1000.0,
        "any_overlap_ms": any_overlap / 1000.0,
        "exposed_ms": (residency - any_overlap) / 1000.0,
        "any_overlap_percent": 100.0 * any_overlap / residency if residency else 0.0,
    }


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average matching numeric dictionaries."""
    return {
        key: statistics.mean(row[key] for row in rows)
        for key in rows[0]
    }


def summarize_trace(path: Path) -> dict[str, Any]:
    """Summarize all profiler steps for one owner-rank trace."""
    events = load_trace_events(path)
    steps = find_steps(events)
    if not steps:
        raise ValueError(f"{path}: no profiler steps found")

    groups = communication_groups(events)
    hybrid_ep_ids = scoped_external_ids(events, ("HybridEP",))
    nep_ids = scoped_external_ids(events, ("nep_",))
    per_step = []
    for step in steps:
        start = float(step["ts"])
        end = event_end(step)
        intervals = classify_gpu_events(events, step, groups, hybrid_ep_ids, nep_ids)
        stack = exclusive_stack(intervals, start, end)
        operations = {
            category: operation_overlap(intervals, category)
            for category in TRACKED_CATEGORIES
        }
        per_step.append(
            {
                "name": str(step["name"]),
                "span_ms": (end - start) / 1000.0,
                "stack_ms": stack,
                "operations": operations,
                "category_residency_ms": {
                    category: union_duration(rows) / 1000.0
                    for category, rows in sorted(intervals.items())
                },
            }
        )

    stack_mean = mean_dict([row["stack_ms"] for row in per_step])
    operations_mean = {
        category: mean_dict([row["operations"][category] for row in per_step])
        for category in TRACKED_CATEGORIES
    }
    span_mean = statistics.mean(float(row["span_ms"]) for row in per_step)
    stack_sum = sum(stack_mean.values())
    if abs(stack_sum - span_mean) > 1e-6:
        raise ValueError(
            f"{path}: exclusive stack does not reconstruct span "
            f"({stack_sum:.6f} != {span_mean:.6f})"
        )
    return {
        "trace": str(path),
        "step_count": len(steps),
        "span_mean_ms": span_mean,
        "stack_mean_ms": stack_mean,
        "operations_mean": operations_mean,
        "per_step": per_step,
    }


def summarize_trace_streaming(path: Path) -> dict[str, Any]:
    """Summarize a trace using bounded memory and the standard classifications."""
    steps = []
    groups = {}
    scope_rows = {
        "hybrid_ep": defaultdict(list),
        "nep": defaultdict(list),
    }
    scope_prefixes = {"hybrid_ep": ("HybridEP",), "nep": ("nep_",)}

    for event in iter_trace_events(path):
        category = event.get("cat")
        name = str(event.get("name", ""))
        if category == "user_annotation" and name.startswith("ProfilerStep#"):
            steps.append(event)
        if name == "record_param_comms":
            args = event.get("args", {})
            external_id = args.get("External id")
            group = args.get("Process Group Description")
            if external_id is not None and group is not None:
                groups[external_id] = str(group)
        if category not in {"cpu_op", "user_annotation"}:
            continue
        for scope_name, prefixes in scope_prefixes.items():
            if name.startswith(prefixes):
                scope_rows[scope_name][(event.get("pid"), event.get("tid"))].append(
                    (float(event["ts"]), event_end(event))
                )

    steps.sort(key=lambda event: float(event["ts"]))
    if not steps:
        raise ValueError(f"{path}: no profiler steps found")

    merged_scopes = {
        scope_name: {
            thread: merge_intervals(intervals)
            for thread, intervals in rows.items()
        }
        for scope_name, rows in scope_rows.items()
    }
    scope_starts = {
        scope_name: {
            thread: [start for start, _ in intervals]
            for thread, intervals in rows.items()
        }
        for scope_name, rows in merged_scopes.items()
    }
    scoped_ids = {"hybrid_ep": set(), "nep": set()}
    for event in iter_trace_events(path):
        if event.get("cat") not in {"cpu_op", "user_annotation"}:
            continue
        external_id = event.get("args", {}).get("External id")
        if external_id is None:
            continue
        thread = (event.get("pid"), event.get("tid"))
        timestamp = float(event["ts"])
        for scope_name, rows in merged_scopes.items():
            intervals = rows.get(thread, [])
            index = bisect.bisect_right(
                scope_starts[scope_name].get(thread, []), timestamp
            ) - 1
            if index >= 0 and timestamp < intervals[index][1]:
                scoped_ids[scope_name].add(external_id)

    raw_intervals = [defaultdict(list) for _ in steps]
    for event in iter_trace_events(path):
        if event.get("cat") not in GPU_CATEGORIES:
            continue
        args = event.get("args", {})
        external_id = args.get("External id")
        if is_nccl(event):
            group = args.get("Process Group Description") or groups.get(external_id)
            category = classify_process_group(str(group) if group is not None else None)
        elif external_id in scoped_ids["hybrid_ep"]:
            category = "model_ep"
        elif external_id in scoped_ids["nep"]:
            category = "nep_staging"
        else:
            category = "compute"
        for index, step in enumerate(steps):
            interval = clip_interval(event, float(step["ts"]), event_end(step))
            if interval is not None:
                raw_intervals[index][category].append(interval)

    per_step = []
    for step, rows in zip(steps, raw_intervals):
        start = float(step["ts"])
        end = event_end(step)
        intervals = {
            category: merge_intervals(category_rows)
            for category, category_rows in rows.items()
        }
        stack = exclusive_stack(intervals, start, end)
        operations = {
            category: operation_overlap(intervals, category)
            for category in TRACKED_CATEGORIES
        }
        per_step.append(
            {
                "name": str(step["name"]),
                "span_ms": (end - start) / 1000.0,
                "stack_ms": stack,
                "operations": operations,
                "category_residency_ms": {
                    category: union_duration(category_rows) / 1000.0
                    for category, category_rows in sorted(intervals.items())
                },
            }
        )

    stack_mean = mean_dict([row["stack_ms"] for row in per_step])
    operations_mean = {
        category: mean_dict([row["operations"][category] for row in per_step])
        for category in TRACKED_CATEGORIES
    }
    span_mean = statistics.mean(float(row["span_ms"]) for row in per_step)
    stack_sum = sum(stack_mean.values())
    if abs(stack_sum - span_mean) > 1e-6:
        raise ValueError(
            f"{path}: exclusive stack does not reconstruct span "
            f"({stack_sum:.6f} != {span_mean:.6f})"
        )
    return {
        "trace": str(path),
        "step_count": len(steps),
        "span_mean_ms": span_mean,
        "stack_mean_ms": stack_mean,
        "operations_mean": operations_mean,
        "per_step": per_step,
    }


def trace_path(comparison_path: Path, mode: str, rank: int) -> Path:
    """Resolve a rank trace adjacent to a comparison artifact."""
    job_id = comparison_path.stem.removeprefix("comparison_")
    path = comparison_path.parent / mode / job_id / "torch_profile" / f"rank-{rank}.json.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def case_summary(
    comparison_path: Path, label: str, rank: int, streaming_traces: bool = False
) -> dict[str, Any]:
    """Combine timing metadata and owner-rank profile metrics for one A/B run."""
    comparison = json.loads(comparison_path.read_text())
    timing = {
        "healthy_mean_ms": float(comparison["healthy"]["mean_elapsed_ms"]),
        "nep_mean_ms": float(comparison["nep"]["mean_elapsed_ms"]),
        "gap_ms": float(comparison["comparison"]["latency_gap_ms"]),
        "parity_percent": float(
            comparison["comparison"]["owner_iteration_time_parity_percent"]
        ),
    }
    summarize = summarize_trace_streaming if streaming_traces else summarize_trace
    profile = {
        mode: summarize(trace_path(comparison_path, mode, rank))
        for mode in ("healthy", "nep")
    }
    profile["delta_span_ms"] = (
        profile["nep"]["span_mean_ms"] - profile["healthy"]["span_mean_ms"]
    )
    profile["trace_parity_percent"] = (
        100.0 * profile["healthy"]["span_mean_ms"] / profile["nep"]["span_mean_ms"]
    )
    healthy_operations = profile["healthy"]["operations_mean"]
    nep_operations = profile["nep"]["operations_mean"]
    measured_attribution = {
        "incremental_expert_edp": max(
            0.0,
            nep_operations["expert_edp"]["exposed_ms"]
            - healthy_operations["expert_edp"]["exposed_ms"],
        ),
        "default_and_other_sync": max(
            0.0,
            nep_operations["other_comm"]["exposed_ms"]
            - healthy_operations["other_comm"]["exposed_ms"],
        ),
        "scatter": nep_operations["scatter"]["exposed_ms"],
        "nep_staging": nep_operations["nep_staging"]["exposed_ms"],
        "gather": nep_operations["gather"]["exposed_ms"],
    }
    attributed_ms = sum(measured_attribution.values())
    measured_attribution["unattributed_residual"] = timing["gap_ms"] - attributed_ms
    profile["measured_gap_attribution_ms"] = measured_attribution
    profile["measured_slowdown_attribution_percent"] = {
        category: 100.0 * value / timing["healthy_mean_ms"]
        for category, value in measured_attribution.items()
    }

    batch = comparison.get("batch_configuration")
    if batch is None:
        workload = comparison["workload"]
        micro_batch_size = int(workload["micro_batch_size"])
        grad_accumulation = int(workload["grad_accumulation_steps"])
        batch = {
            mode: {
                "global_batch_size": int(comparison[mode]["global_batch_size"]),
                "replica_micro_batch_sizes": [micro_batch_size, micro_batch_size],
                "replica_num_microbatches": [grad_accumulation, grad_accumulation],
            }
            for mode in ("healthy", "nep")
        }

    return {
        "label": label,
        "comparison_artifact": str(comparison_path),
        "job_id": comparison_path.stem.removeprefix("comparison_"),
        "source_commit": comparison["source_commit"],
        "workload": comparison["workload"],
        "batch_configuration": batch,
        "topology": {
            mode: comparison[mode]["topology"] for mode in ("healthy", "nep")
        },
        "timing": timing,
        "profile": profile,
    }


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a legible system font with a Pillow fallback."""
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    value: str,
    font: ImageFont.ImageFont,
    fill: str = "#202124",
    anchor: str | None = None,
) -> None:
    """Draw text while keeping Pillow's typing details local."""
    draw.text(position, value, font=font, fill=fill, anchor=anchor)


def draw_case_panel(
    draw: ImageDraw.ImageDraw,
    case: dict[str, Any],
    left: int,
    top: int,
    width: int,
) -> None:
    """Draw one healthy/NEP stacked trace panel."""
    title_font = load_font(27, bold=True)
    body_font = load_font(20)
    small_font = load_font(17)
    tiny_font = load_font(15)
    timing = case["timing"]
    profile = case["profile"]

    draw_text(draw, (left, top), case["label"], title_font)
    draw_text(
        draw,
        (left, top + 40),
        (
            f"job {case['job_id']} | measured H {timing['healthy_mean_ms']:.1f} ms, "
            f"NEP {timing['nep_mean_ms']:.1f} ms | parity {timing['parity_percent']:.2f}%"
        ),
        small_font,
    )

    bar_left = left + 92
    bar_width = width - 112
    bar_height = 44
    max_span = max(
        float(profile["healthy"]["span_mean_ms"]),
        float(profile["nep"]["span_mean_ms"]),
    )
    for index, mode in enumerate(("healthy", "nep")):
        y = top + 92 + index * 76
        draw_text(draw, (left, y + bar_height / 2), mode.upper(), body_font, anchor="lm")
        x = float(bar_left)
        stack = profile[mode]["stack_mean_ms"]
        for category in STACK_ORDER:
            value = float(stack[category])
            segment_width = bar_width * value / max_span
            if segment_width <= 0:
                continue
            draw.rectangle(
                (int(x), y, int(x + segment_width), y + bar_height),
                fill=STACK_COLORS[category],
            )
            if segment_width >= 52:
                text_color = "#FFFFFF" if category not in {"nep_staging_only", "gpu_idle"} else "#202124"
                draw_text(
                    draw,
                    (x + segment_width / 2, y + bar_height / 2),
                    f"{value:.1f}",
                    tiny_font,
                    fill=text_color,
                    anchor="mm",
                )
            x += segment_width
        draw.rectangle(
            (bar_left, y, bar_left + bar_width, y + bar_height),
            outline="#303030",
            width=1,
        )
        draw_text(
            draw,
            (bar_left + bar_width, y + bar_height + 18),
            f"profile span {profile[mode]['span_mean_ms']:.1f} ms",
            tiny_font,
            anchor="ra",
        )

    draw_text(
        draw,
        (left, top + 254),
        (
            f"Profiled span delta: +{profile['delta_span_ms']:.1f} ms "
            f"(trace parity {profile['trace_parity_percent']:.2f}%)"
        ),
        small_font,
    )
    draw_text(
        draw,
        (left, top + 286),
        "Owner-rank NEP operation residency: total / hidden by any other GPU work / exposed",
        small_font,
    )
    y = top + 320
    operation_labels = {
        "gather": "Gather",
        "expert_edp": "Expert EDP",
        "scatter": "Scatter",
        "nep_staging": "Staging",
    }
    for category in RESHARD_CATEGORIES:
        metrics = profile["nep"]["operations_mean"][category]
        draw_text(
            draw,
            (left + 16, y),
            (
                f"{operation_labels[category]:<10} "
                f"{metrics['residency_ms']:7.2f} / "
                f"{metrics['any_overlap_ms']:7.2f} / "
                f"{metrics['exposed_ms']:7.2f} ms "
                f"({metrics['any_overlap_percent']:5.1f}% hidden)"
            ),
            small_font,
        )
        y += 29

    healthy_edp = profile["healthy"]["operations_mean"]["expert_edp"]
    nep_edp = profile["nep"]["operations_mean"]["expert_edp"]
    draw_text(
        draw,
        (left + 16, y + 4),
        (
            "EDP exposed H -> NEP: "
            f"{healthy_edp['exposed_ms']:.2f} -> {nep_edp['exposed_ms']:.2f} ms"
        ),
        small_font,
    )


def render_chart(cases: list[dict[str, Any]], output: Path) -> None:
    """Render two side-by-side stacked owner-rank profile panels."""
    width = 1900
    height = 880
    margin = 64
    gap = 56
    panel_width = (width - 2 * margin - gap) // 2
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = load_font(32, bold=True)
    subtitle_font = load_font(19)
    legend_font = load_font(16)

    draw_text(
        draw,
        (width / 2, 38),
        "Why the same a500 NEP topology reaches different owner parity",
        title_font,
        anchor="ma",
    )
    draw_text(
        draw,
        (width / 2, 80),
        (
            "GB200 | 27 layers | H=1024 | MoE F=640 | 128 experts | top-k 6 | "
            "EP8+8 healthy vs EP8+6 NEP | MBS3, GA1 vs GA3 | rank 0, two profiler steps"
        ),
        subtitle_font,
        anchor="ma",
    )

    for index, case in enumerate(cases):
        left = margin + index * (panel_width + gap)
        draw_case_panel(draw, case, left, 126, panel_width)

    legend_y = 694
    x = margin
    for category in STACK_ORDER:
        draw.rectangle((x, legend_y, x + 20, legend_y + 20), fill=STACK_COLORS[category])
        x += 28
        draw_text(draw, (x, legend_y + 10), STACK_LABELS[category], legend_font, anchor="lm")
        x += int(draw.textlength(STACK_LABELS[category], font=legend_font)) + 28
        if x > width - 250:
            legend_y += 34
            x = margin

    note_font = load_font(17)
    draw_text(
        draw,
        (margin, 796),
        (
            "Bars are exact mutually exclusive GPU states from the profiled owner timeline; each bar "
            "sums to its profile span. 'Hidden' means temporal overlap with any other GPU work."
        ),
        note_font,
    )
    draw_text(
        draw,
        (margin, 824),
        (
            "Measured parity uses post-profiler iterations 8-10. Profiler host overhead inflates the "
            "trace spans, especially NEP, so trace-span parity is diagnostic and is not substituted "
            "for measured parity."
        ),
        note_font,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)



def render_measured_overhead_chart(cases: list[dict[str, Any]], output: Path) -> None:
    """Render measured slowdown as trace-informed, stacked overhead components."""
    width = 1500
    height = 1040
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = load_font(32, bold=True)
    subtitle_font = load_font(19)
    axis_font = load_font(17)
    label_font = load_font(20, bold=True)
    detail_font = load_font(17)
    segment_font = load_font(16, bold=True)

    draw_text(
        draw,
        (width / 2, 35),
        "a500 NEP overhead: fixed communication cost is amortized by more work",
        title_font,
        anchor="ma",
    )
    draw_text(
        draw,
        (width / 2, 78),
        (
            "Same GB200 model and EP8+8 -> EP8+6 topology; bars sum to the measured "
            "post-profiler slowdown relative to healthy"
        ),
        subtitle_font,
        anchor="ma",
    )

    plot_left = 150
    plot_right = width - 90
    plot_top = 155
    plot_bottom = 690
    totals = [
        100.0 * case["timing"]["gap_ms"] / case["timing"]["healthy_mean_ms"]
        for case in cases
    ]
    axis_max = 2.0 * (int(max(totals) / 2.0) + 2)
    for tick in range(0, int(axis_max) + 1, 2):
        y = plot_bottom - (plot_bottom - plot_top) * tick / axis_max
        draw.line((plot_left, y, plot_right, y), fill="#DEDEDE", width=1)
        draw_text(draw, (plot_left - 14, y), f"{tick}%", axis_font, anchor="rm")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#303030", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#303030", width=2)
    draw_text(
        draw,
        (plot_left, plot_top - 30),
        "NEP slowdown relative to healthy",
        axis_font,
    )

    centers = (450, 1050)
    bar_width = 270
    for center, case, total in zip(centers, cases, totals):
        values = case["profile"]["measured_slowdown_attribution_percent"]
        milliseconds = case["profile"]["measured_gap_attribution_ms"]
        if values["unattributed_residual"] < -1e-6:
            raise ValueError(
                f"{case['label']}: traced exposed components exceed measured gap"
            )

        y_bottom = float(plot_bottom)
        for category in OVERHEAD_ORDER:
            value = max(0.0, float(values[category]))
            segment_height = (plot_bottom - plot_top) * value / axis_max
            y_top = y_bottom - segment_height
            draw.rectangle(
                (center - bar_width / 2, y_top, center + bar_width / 2, y_bottom),
                fill=OVERHEAD_COLORS[category],
                outline="#FFFFFF",
                width=1,
            )
            if segment_height >= 32:
                draw_text(
                    draw,
                    (center, (y_top + y_bottom) / 2 - 9),
                    f"{value:.2f}%",
                    segment_font,
                    fill="#FFFFFF" if category != "nep_staging" else "#202124",
                    anchor="mm",
                )
                draw_text(
                    draw,
                    (center, (y_top + y_bottom) / 2 + 10),
                    f"{milliseconds[category]:.1f} ms",
                    axis_font,
                    fill="#FFFFFF" if category != "nep_staging" else "#202124",
                    anchor="mm",
                )
            y_bottom = y_top

        draw_text(
            draw,
            (center, y_bottom - 20),
            f"{total:.2f}% slowdown | {case['timing']['parity_percent']:.2f}% parity",
            label_font,
            anchor="ms",
        )
        draw_text(draw, (center, plot_bottom + 36), case["label"], label_font, anchor="ma")
        draw_text(
            draw,
            (center, plot_bottom + 68),
            (
                f"H {case['timing']['healthy_mean_ms']:.1f} ms -> "
                f"NEP {case['timing']['nep_mean_ms']:.1f} ms "
                f"(+{case['timing']['gap_ms']:.1f} ms)"
            ),
            detail_font,
            anchor="ma",
        )

        healthy_edp = case["profile"]["healthy"]["operations_mean"]["expert_edp"]
        nep_edp = case["profile"]["nep"]["operations_mean"]["expert_edp"]
        direct_ms = sum(
            milliseconds[category]
            for category in OVERHEAD_ORDER
            if category != "unattributed_residual"
        )
        draw_text(
            draw,
            (center, plot_bottom + 108),
            f"Direct traced exposed components: {direct_ms:.1f} ms",
            detail_font,
            anchor="ma",
        )
        draw_text(
            draw,
            (center, plot_bottom + 136),
            (
                f"NEP EDP: {nep_edp['residency_ms']:.1f} ms resident, "
                f"{nep_edp['any_overlap_ms']:.1f} ms hidden, "
                f"{nep_edp['exposed_ms']:.1f} ms exposed"
            ),
            detail_font,
            anchor="ma",
        )
        draw_text(
            draw,
            (center, plot_bottom + 164),
            (
                f"Healthy EDP exposed {healthy_edp['exposed_ms']:.1f} ms; "
                f"incremental exposure {milliseconds['incremental_expert_edp']:.1f} ms"
            ),
            detail_font,
            anchor="ma",
        )

    legend_y = 900
    x = 72
    for category in OVERHEAD_ORDER:
        draw.rectangle((x, legend_y, x + 20, legend_y + 20), fill=OVERHEAD_COLORS[category])
        x += 29
        draw_text(draw, (x, legend_y + 10), OVERHEAD_LABELS[category], axis_font, anchor="lm")
        x += int(draw.textlength(OVERHEAD_LABELS[category], font=axis_font)) + 32
        if x > width - 300:
            legend_y += 34
            x = 72

    draw_text(
        draw,
        (72, 985),
        (
            "Trace-attributed components are owner-rank exposed GPU intervals averaged over "
            "ProfilerStep#5-6. Incremental EDP and default/other sync subtract healthy exposure."
        ),
        axis_font,
    )
    draw_text(
        draw,
        (72, 1012),
        (
            "The residual is measured gap minus those direct components; it is intentionally "
            "not assigned to a cause because profile host overhead differs from unprofiled timing."
        ),
        axis_font,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
def main() -> None:
    """Analyze two comparison artifacts and render their overhead chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-comparison", type=Path, required=True)
    parser.add_argument("--first-label", required=True)
    parser.add_argument("--second-comparison", type=Path, required=True)
    parser.add_argument("--second-label", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-chart", type=Path, required=True)
    parser.add_argument("--output-overhead-chart", type=Path, required=True)
    parser.add_argument("--streaming-traces", action="store_true")
    args = parser.parse_args()

    cases = [
        case_summary(
            args.first_comparison,
            args.first_label,
            args.rank,
            streaming_traces=args.streaming_traces,
        ),
        case_summary(
            args.second_comparison,
            args.second_label,
            args.rank,
            streaming_traces=args.streaming_traces,
        ),
    ]
    if len({case["source_commit"] for case in cases}) != 1:
        raise ValueError("comparison artifacts use different source commits")
    if len({case["workload"]["name"] for case in cases}) != 1:
        raise ValueError("comparison artifacts use different workloads")

    output = {
        "definition": {
            "owner_rank": args.rank,
            "profile_stack": (
                "Mutually exclusive GPU states over each ProfilerStep; categories sum exactly "
                "to the profile span."
            ),
            "operation_exposed": (
                "Operation residency not temporally overlapped by any other GPU activity on "
                "the owner rank."
            ),
            "timing": "Post-profiler owner iteration means from each comparison artifact.",
        },
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    render_chart(cases, args.output_chart)
    render_measured_overhead_chart(cases, args.output_overhead_chart)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_chart}")
    print(f"wrote {args.output_overhead_chart}")


if __name__ == "__main__":
    main()
