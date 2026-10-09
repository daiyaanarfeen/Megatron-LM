#!/usr/bin/env python3
"""Plot exact owner-timeline deltas for the observed NEP failure categories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COMPONENTS = (
    ("compute_only", "Non-NCCL active", "#56B4E9"),
    ("native_comm_only", "Native comm only", "#0072B2"),
    ("expert_edp_only", "Expert EDP only", "#CC79A7"),
    ("gather_only", "NEP Gather only", "#E69F00"),
    ("scatter_only", "NEP Scatter only", "#D55E00"),
    ("nep_staging_only", "NEP staging only", "#F0E442"),
    ("overlap_state", "Compute/comm or comm/comm overlap", "#009E73"),
    ("gpu_idle", "GPU idle", "#999999"),
)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a readable system font with a Pillow fallback."""
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
    """Draw text while isolating Pillow typing details."""
    draw.text(position, value, font=font, fill=fill, anchor=anchor)


def component_deltas(case: dict[str, Any]) -> dict[str, float]:
    """Return an exact signed decomposition of NEP minus healthy profile span."""
    healthy = case["profile"]["healthy"]["stack_mean_ms"]
    nep = case["profile"]["nep"]["stack_mean_ms"]
    output = {}
    for key, _, _ in COMPONENTS:
        if key == "overlap_state":
            output[key] = (
                nep["compute_comm_overlap"]
                - healthy["compute_comm_overlap"]
                + nep["comm_comm_overlap"]
                - healthy["comm_comm_overlap"]
            )
        else:
            output[key] = nep[key] - healthy[key]
    expected = float(case["profile"]["delta_span_ms"])
    if not math.isclose(sum(output.values()), expected, abs_tol=1.0e-6):
        raise ValueError(
            f"{case['label']}: stack delta {sum(output.values())} does not close to {expected}"
        )
    return output


def scaled_components(case: dict[str, Any], percent: bool) -> dict[str, float]:
    """Return component deltas in milliseconds or percent of healthy span."""
    values = component_deltas(case)
    if not percent:
        return values
    denominator = float(case["profile"]["healthy"]["span_mean_ms"])
    return {key: 100.0 * value / denominator for key, value in values.items()}


def draw_delta_panel(
    draw: ImageDraw.ImageDraw,
    cases: list[dict[str, Any]],
    bounds: tuple[int, int, int, int],
    percent: bool,
) -> None:
    """Draw one signed stacked delta panel."""
    left, top, right, bottom = bounds
    title_font = load_font(23, bold=True)
    axis_font = load_font(16)
    label_font = load_font(17, bold=True)
    segment_font = load_font(14, bold=True)
    values = [scaled_components(case, percent) for case in cases]
    positive_limit = max(sum(max(0.0, value) for value in row.values()) for row in values)
    negative_limit = min(sum(min(0.0, value) for value in row.values()) for row in values)
    positive_limit = max(positive_limit * 1.12, 1.0)
    negative_limit = min(negative_limit * 1.18, 0.0)
    span = positive_limit - negative_limit

    def y_position(value: float) -> float:
        return top + (positive_limit - value) * (bottom - top) / span

    unit = "% of healthy profile span" if percent else "ms"
    draw_text(
        draw,
        ((left + right) / 2, top - 78),
        f"Exact signed profile delta ({unit})",
        title_font,
        anchor="ma",
    )

    for tick_index in range(7):
        value = negative_limit + span * tick_index / 6
        y = y_position(value)
        draw.line((left, y, right, y), fill="#DEDEDE", width=1)
        label = f"{value:.1f}%" if percent else f"{value:.0f}"
        draw_text(draw, (left - 12, y), label, axis_font, anchor="rm")
    zero_y = y_position(0.0)
    draw.line((left, zero_y, right, zero_y), fill="#202124", width=2)

    centers = (
        left + (right - left) * 0.31,
        left + (right - left) * 0.72,
    )
    bar_width = min(220, int((right - left) * 0.24))
    for center, case, row in zip(centers, cases, values):
        positive = 0.0
        negative = 0.0
        for key, _, color in COMPONENTS:
            value = float(row[key])
            if value >= 0:
                start, end = positive, positive + value
                positive = end
            else:
                start, end = negative, negative + value
                negative = end
            y1, y2 = sorted((y_position(start), y_position(end)))
            if y2 - y1 < 0.5:
                continue
            draw.rectangle(
                (center - bar_width / 2, y1, center + bar_width / 2, y2),
                fill=color,
                outline="#FFFFFF",
                width=1,
            )
            if y2 - y1 >= 25:
                label = f"{value:+.1f}%" if percent else f"{value:+.1f}"
                text_color = "#202124" if key == "nep_staging_only" else "#FFFFFF"
                draw_text(
                    draw,
                    (center, (y1 + y2) / 2),
                    label,
                    segment_font,
                    fill=text_color,
                    anchor="mm",
                )

        net = sum(row.values())
        net_label = f"net {net:+.2f}%" if percent else f"net {net:+.1f} ms"
        draw_text(
            draw,
            (center, y_position(positive) - 10),
            net_label,
            label_font,
            anchor="ms",
        )
        draw_text(draw, (center, bottom + 26), case["label"], label_font, anchor="ma")
        timing = case["timing"]
        draw_text(
            draw,
            (center, bottom + 54),
            (
                f"measured {timing['parity_percent']:.2f}% parity "
                f"(+{timing['gap_ms']:.1f} ms)"
            ),
            axis_font,
            anchor="ma",
        )


def render_performance_chart(
    cases: list[dict[str, Any]],
    output: Path,
    title: str,
    subtitle: str,
    findings: list[str],
) -> None:
    """Render absolute and normalized exact-closure stacks for one category."""
    width, height = 2100, 1280
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34, bold=True)
    subtitle_font = load_font(19)
    note_font = load_font(17)
    finding_font = load_font(19, bold=True)

    draw_text(draw, (width / 2, 34), title, title_font, anchor="ma")
    draw_text(draw, (width / 2, 82), subtitle, subtitle_font, anchor="ma")
    draw_delta_panel(draw, cases, (125, 245, 990, 825), percent=False)
    draw_delta_panel(draw, cases, (1180, 245, 2045, 825), percent=True)

    legend_y = 930
    x = 80
    for _, label, color in COMPONENTS:
        draw.rectangle((x, legend_y, x + 20, legend_y + 20), fill=color)
        x += 29
        draw_text(draw, (x, legend_y + 10), label, note_font, anchor="lm")
        x += int(draw.textlength(label, font=note_font)) + 35
        if x > width - 330:
            legend_y += 35
            x = 80

    finding_y = 1025
    for finding in findings:
        draw_text(draw, (90, finding_y), finding, finding_font)
        finding_y += 32
    draw_text(
        draw,
        (90, 1174),
        (
            "Each stack is the exact NEP-minus-healthy difference of mutually exclusive owner-GPU "
            "states; positive and negative segments sum exactly to the profiled span delta."
        ),
        note_font,
    )
    draw_text(
        draw,
        (90, 1205),
        (
            "Measured parity comes from clean post-profiler iterations. Profile deltas diagnose the "
            "mechanism but are not substituted for end-to-end timing because profiling perturbs hosts."
        ),
        note_font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def render_capacity_chart(output: Path) -> dict[str, Any]:
    """Render the unresolved Ultra capacity category without claiming parity."""
    width, height = 1800, 920
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34, bold=True)
    subtitle_font = load_font(20)
    label_font = load_font(19, bold=True)
    body_font = load_font(17)

    total_gib = 276.6
    rows = (
        {
            "label": "Initial NEP graph-capture failure",
            "occupied_gib": 216.5,
            "free_gib": 60.1,
            "detail": "job 2801307: capture subsequently grew until memory was exhausted",
        },
        {
            "label": "After lifetime fix and token cuts",
            "occupied_gib": 272.2,
            "free_gib": 4.4,
            "detail": "job 2831719: reached native gradient finalization",
        },
    )
    draw_text(
        draw,
        (width / 2, 34),
        "Category 3 - Nemotron Ultra capacity remained unresolved",
        title_font,
        anchor="ma",
    )
    draw_text(
        draw,
        (width / 2, 86),
        "EP64/EP56, 512 experts, Flex/HybridEP + distributed optimizer on 276.6 GiB GB300 GPUs",
        subtitle_font,
        anchor="ma",
    )

    bar_left, bar_right = 390, 1600
    bar_width = bar_right - bar_left
    for index, row in enumerate(rows):
        y = 230 + index * 205
        draw_text(draw, (bar_left, y - 44), row["label"], label_font)
        occupied_width = bar_width * row["occupied_gib"] / total_gib
        draw.rectangle((bar_left, y, bar_left + occupied_width, y + 74), fill="#0072B2")
        draw.rectangle((bar_left + occupied_width, y, bar_right, y + 74), fill="#D9D9D9")
        draw.rectangle((bar_left, y, bar_right, y + 74), outline="#202124", width=2)
        draw_text(
            draw,
            (bar_left + occupied_width / 2, y + 37),
            f"{row['occupied_gib']:.1f} GiB occupied",
            label_font,
            fill="#FFFFFF",
            anchor="mm",
        )
        if row["free_gib"] >= 15.0:
            free_position = (
                bar_left + occupied_width + (bar_width - occupied_width) / 2,
                y + 37,
            )
            free_anchor = "mm"
        else:
            free_position = (bar_right + 16, y + 37)
            free_anchor = "lm"
        draw_text(
            draw,
            free_position,
            f"{row['free_gib']:.1f} GiB free",
            body_font,
            anchor=free_anchor,
        )
        draw_text(draw, (bar_left, y + 100), row["detail"], body_font)

    final_y = 435
    request_gib = 6.89
    shortfall_gib = request_gib - rows[1]["free_gib"]
    request_width = bar_width * request_gib / total_gib
    request_start = bar_right - bar_width * rows[1]["free_gib"] / total_gib
    draw.line(
        (request_start, final_y - 22, request_start + request_width, final_y - 22),
        fill="#D55E00",
        width=7,
    )
    draw_text(
        draw,
        (bar_right, final_y - 34),
        f"6.89 GiB native coalescing request exceeds free memory by {shortfall_gib:.2f} GiB",
        body_font,
        fill="#A23617",
        anchor="rs",
    )

    draw_text(
        draw,
        (110, 690),
        "Partial fixes removed avoidable retained transfer staging and moved the failure later.",
        label_font,
    )
    draw_text(
        draw,
        (110, 727),
        (
            "They did not create enough headroom for the fixed-size native gradient-finalization "
            "temporary, even after severe sequence reductions."
        ),
        body_font,
    )
    draw_text(
        draw,
        (110, 778),
        "Result: no completed timing window, no measured parity, and therefore no valid before/after latency stack.",
        label_font,
        fill="#A23617",
    )
    draw_text(
        draw,
        (110, 824),
        (
            "Values are the approximate memory quantities reported by the failure logs/journal; "
            "occupied includes model, optimizer, activations, allocator state, and NEP state."
        ),
        body_font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return {
        "gpu_capacity_gib": total_gib,
        "initial": rows[0],
        "after_partial_fixes": rows[1],
        "final_native_request_gib": request_gib,
        "final_shortfall_gib": shortfall_gib,
        "parity": None,
        "status": "unresolved OOM; no completed timing window",
    }


def case_output(case: dict[str, Any]) -> dict[str, Any]:
    """Return compact machine-readable values represented in a performance plot."""
    components = component_deltas(case)
    healthy_span = float(case["profile"]["healthy"]["span_mean_ms"])
    return {
        "label": case["label"],
        "job_id": case["job_id"],
        "timing": case["timing"],
        "profile_healthy_span_ms": healthy_span,
        "profile_delta_ms": case["profile"]["delta_span_ms"],
        "profile_delta_percent": 100.0
        * float(case["profile"]["delta_span_ms"])
        / healthy_span,
        "component_delta_ms": components,
        "component_delta_percent": {
            key: 100.0 * value / healthy_span for key, value in components.items()
        },
        "nep_operation_overlap": {
            key: case["profile"]["nep"]["operations_mean"][key]
            for key in ("gather", "expert_edp", "scatter", "nep_staging")
        },
    }


def main() -> None:
    """Render the three recorded failure-category figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amortization-summary", type=Path, required=True)
    parser.add_argument("--readiness-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    amortization = json.loads(args.amortization_summary.read_text())["cases"]
    readiness = json.loads(args.readiness_summary.read_text())["cases"]
    if len(amortization) != 2 or len(readiness) != 2:
        raise ValueError("each trace summary must contain exactly two cases")
    amortization[0]["label"] = "Before: GA4"
    amortization[1]["label"] = "After: GA20"
    readiness[0]["label"] = "Before: reduced GA12"
    readiness[1]["label"] = "After: reduced GA9"

    render_performance_chart(
        amortization,
        args.output_dir / "category1_amortization_a8b.png",
        "Category 1 - More work amortizes a fixed/sublinear NEP critical path",
        (
            "a8b_120b_latentmoe_1t | GB300 | MBS2 | EP8+8 healthy vs EP8+6 NEP | "
            "only GA/GBS changes"
        ),
        [
            "Measured: healthy work grows 5.06x; absolute NEP gap stays 767.9 -> 819.3 ms.",
            "Outcome: owner parity rises 83.93% -> 96.12%; the communication was amortized, not removed.",
        ],
    )
    render_performance_chart(
        readiness,
        args.output_dir / "category2_readiness_a500.png",
        "Category 2 - Less reduced-replica work removes participant-readiness delay",
        (
            "a500m_2b_moe_140b | MBS1, full GA12 | EP8+8 healthy vs EP8+6 NEP | "
            "reduced GA12 -> GA9"
        ),
        [
            "Measured: NEP gap falls 144.1 -> 52.3 ms; owner parity rises 89.45% -> 95.72%.",
            "All-rank EDP participant wait falls 525.2 -> 15.5 ms while matched service stays ~2.69 ms.",
            "Caveat: before is GB300 and after is GB200; each parity is allocation-local.",
        ],
    )
    capacity = render_capacity_chart(args.output_dir / "category3_ultra_capacity_unresolved.png")

    output = {
        "definitions": {
            "profile_stack": (
                "Signed NEP-minus-healthy deltas of mutually exclusive owner-GPU states. "
                "Each stack closes exactly to the profiled span delta."
            ),
            "timing": "Owner-rank means from clean post-profiler iterations.",
            "parity": "healthy owner mean / NEP owner mean * 100",
        },
        "category1_amortization": [case_output(case) for case in amortization],
        "category2_readiness": [case_output(case) for case in readiness],
        "category2_all_rank_edp_wait_ms": {
            "before": 525.236,
            "after": 15.531,
            "matched_service_before": 2.694,
            "matched_service_after": 2.687,
            "source": "NEP_APPROACH_A_JOURNAL.md lines 5435-5454",
        },
        "category3_capacity": capacity,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "failure_category_plot_data.json"
    summary_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
