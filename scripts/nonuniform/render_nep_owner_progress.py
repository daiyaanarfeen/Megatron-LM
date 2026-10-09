#!/usr/bin/env python3
"""Plot observed latency, GPU completion phases, and cross-rank progress."""

import argparse
import json
import statistics
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PHASES = [
    ('through_backward', 'Through model backward completion', '#0072B2'),
    ('gradient_drain', 'Backward completion to GPU Adam', '#D55E00'),
    ('adam', 'GPU Adam span', '#009E73'),
    ('after_adam', 'After GPU Adam', '#999999'),
]


def mean(rows: Iterable[float]) -> float:
    """Average observed samples without excluding outliers."""
    return statistics.mean(rows)


def display_name(case: dict) -> str:
    """Use a compact display label without changing the analysis cache key."""
    return case.get('display_label', case['label'])


def case_data(case: dict, directory: Path) -> dict:
    """Read available per-rank analyses for a case."""
    slug = case['label'].replace('/', '_').replace(' ', '_')
    return {
        mode: {
            int(p.stem.split('-')[-1]): json.loads(p.read_text())
            for p in (directory / slug / mode).glob('rank-*.json')
        }
        for mode in ('healthy', 'nep')
    }


def field(data: dict, mode: str, rank: int, key: str) -> float:
    """Average one GPU-progress phase across profiled steps."""
    return mean(s['phases_ms'][key] for s in data[mode][rank]['steps'])


def save(fig: plt.Figure, path: Path) -> None:
    """Write and close a plot figure."""
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def dependency_figure(cases: list[dict], data: list[dict], output: Path, category: str) -> None:
    """Show recorded GPU gradient work and CPU waits after backward completion."""
    lanes = [
        ('expert_grad_rs', 'Expert gradient RS', '#D55E00'),
        ('dense_dp_sync', 'Dense-DP collectives', '#0072B2'),
        ('gradient_accumulation', 'Gradient accumulation', '#009E73'),
    ]
    fig, axes = plt.subplots(
        2, len(cases), figsize=(7 * len(cases), 7), squeeze=False, layout='constrained'
    )
    for column, (case, records) in enumerate(zip(cases, data)):
        for row, mode in enumerate(('healthy', 'nep')):
            ax = axes[row, column]
            step = records[mode][0]['steps'][0]
            origin, end = step['backward_end_ms'], step['gpu_adam_end_ms']
            for i, (category_name, label, color) in enumerate(lanes):
                intervals = [
                    (max(a, origin) - origin, min(b, end) - max(a, origin))
                    for cat, a, b, stream in step['timeline']
                    if cat == category_name and b > origin and a < end
                ]
                ax.broken_barh(intervals, (i - 0.3, 0.6), facecolors=color)
            syncs = [
                (
                    max(s['start_ms'], origin) - origin,
                    min(s['end_ms'], end) - max(s['start_ms'], origin),
                )
                for s in step['syncs']
                if s['end_ms'] > origin
                and s['start_ms'] < end
                and s['name'] == 'cudaStreamSynchronize'
            ]
            ax.broken_barh(syncs, (2.7, 0.6), facecolors='#777777')
            ax.broken_barh(
                [(step['gpu_adam_start_ms'] - origin, end - step['gpu_adam_start_ms'])],
                (3.7, 0.6),
                facecolors='#CC79A7',
            )
            ax.scatter(step['gpu_adam_start_ms'] - origin, 4, marker='|', color='black')
            ax.set_yticks(range(5), [x[1] for x in lanes] + ['CPU stream sync', 'GPU Adam'])
            ax.set_xlim(0, max(20, end - origin + 5))
            ax.invert_yaxis()
            ax.set_title(f"{display_name(case)}\n{mode}, owner rank 0, {step['step']}")
            ax.set_xlabel('Milliseconds after model-backward GPU completion')
    fig.suptitle(
        'Gradient drain: GPU Adam remains behind outstanding gradient work\nCPU synchronization is concurrent with GPU work, not an additive overhead. Each panel uses its own local origin.',
        fontsize=12,
    )
    save(fig, output / f'{category}_4_gradient_dependency.png')


def render(cases: list[dict], directory: Path, output: Path, category: str) -> None:
    """Render timing distributions and exact phase-location comparisons."""
    data = [case_data(c, directory) for c in cases]
    cols = min(2, len(cases))
    rows = (len(cases) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(14, 4.8 * rows), squeeze=False, layout='constrained'
    )
    for ax, case in zip(axes.flat, cases):
        comp = json.loads(Path(case['comparison_artifact']).read_text())
        hs, ns = comp['healthy']['elapsed_ms'], comp['nep']['elapsed_ms']
        for i, (label, values, color) in enumerate(
            [('Healthy', hs, '#0072B2'), ('NEP', ns, '#D55E00')]
        ):
            ax.bar(i, mean(values), color=color, width=0.55)
            ax.errorbar(
                i,
                mean(values),
                yerr=statistics.stdev(values) if len(values) > 1 else 0,
                fmt='none',
                color='black',
                capsize=5,
            )
            ax.scatter(
                np.linspace(i - 0.15, i + 0.15, len(values)), values, s=12, c='black', alpha=0.5
            )
        ax.set_xticks([0, 1], ['Healthy', 'NEP'])
        ax.set_ylabel('Logged iteration time (ms)')
        ax.set_title(
            f"{display_name(case)}\nMean parity {100*mean(hs)/mean(ns):.2f}%; median {100*statistics.median(hs)/statistics.median(ns):.2f}%"
        )
        batch = case['batch_configuration']
        h = batch['healthy']
        n = batch['nep']
        local = {
            mode: [
                units * mbs * ga
                for units, mbs, ga in zip(
                    case['topology'][mode],
                    batch[mode]['replica_micro_batch_sizes'],
                    batch[mode]['replica_num_microbatches'],
                )
            ]
            for mode in ('healthy', 'nep')
        }
        ax.set_xlabel(
            f"H GBS={h['global_batch_size']}, local BS={local['healthy']}; NEP GBS={n['global_batch_size']}, local BS={local['nep']}\nMBS H={h['replica_micro_batch_sizes']}, N={n['replica_micro_batch_sizes']}; GA H={h['replica_num_microbatches']}, N={n['replica_num_microbatches']}"
        )
        ax.spines[['top', 'right']].set_visible(False)
    for ax in list(axes.flat)[len(cases) :]:
        ax.set_visible(False)
    w = cases[0]['workload']
    details = (
        'Model H=4608, expert latent=1152, F=3072, 512 experts'
        if category == 'category1'
        else 'H=1024, expert F=640, 128 experts'
    )
    counts = {
        m: len(
            [
                c
                for c in data[0][m][0]['steps'][0]['collectives']
                if c['category'] == 'expert_grad_rs'
            ]
        )
        for m in ('healthy', 'nep')
    }
    full, reduced = data[0]['nep'][0]['full_ranks'], data[0]['nep'][0]['reduced_ranks']
    topology = f"Healthy EP{full}/{full}; NEP EP{full}/{reduced}"
    dense = {
        mode: sum(
            c['op'] == 'reduce_scatter_tensor_coalesced'
            and c['category'] in ('dense_grad_rs', 'dense_dp_sync')
            for c in data[0][mode][0]['steps'][0]['collectives']
        )
        for mode in ('healthy', 'nep')
    }
    title = f"{w['name']} | {topology}, TP{w['tensor_parallel']} CP{w['context_parallel']} ETP1 | sequence 8192, top-k 6\nMamba + GQA + MoE; {details} | Flex/HybridEP (dynamic receive) + DistOpt\nObserved grad RS calls: dense H={dense['healthy']}, N={dense['nep']}; expert H={counts['healthy']}, N={counts['nep']}. Bars: mean +/- sample SD; dots: iterations."
    fig.suptitle(title, fontsize=12)
    save(fig, output / f'{category}_1_latency.png')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout='constrained')
    for i, (case, d) in enumerate(zip(cases, data)):
        pos = neg = 0
        for key, label, color in PHASES:
            delta = field(d, 'nep', 0, key) - field(d, 'healthy', 0, key)
            base = pos if delta >= 0 else neg
            axes[0].bar(i, delta, bottom=base, color=color, label=label if i == 0 else None)
            if delta >= 0:
                pos += delta
            else:
                neg += delta
        total = sum(field(d, 'nep', 0, k) - field(d, 'healthy', 0, k) for k, _, _ in PHASES)
        axes[0].scatter(i, total, color='black', marker='D', zorder=5)
        axes[0].annotate(
            f'{total:.1f} ms', (i, total), xytext=(8, 5), textcoords='offset points', fontsize=9
        )
        owner_count = d['nep'][0]['reduced_ranks']
        ranks = [r for r in range(owner_count) if r in d['healthy'] and r in d['nep']]
        deltas = [
            sum(field(d, 'nep', r, k) - field(d, 'healthy', r, k) for k, _, _ in PHASES)
            for r in ranks
        ]
        axes[1].scatter(
            np.linspace(i - 0.16, i + 0.16, len(deltas)), deltas, s=28, label=display_name(case)
        )
    for ax in axes:
        ax.axhline(0, color='black', linewidth=0.7)
        ax.set_xticks(range(len(cases)), [display_name(c) for c in cases], rotation=0)
        ax.set_ylabel('NEP minus healthy, profiled milliseconds')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_title('Rank 0: disjoint GPU-completion phases\nDiamond = measured profile-step gap')
    axes[1].set_title(
        'All analyzed full-replica owners\nEach point is one rank, averaged over profiled steps'
    )
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
    fig.suptitle(
        f"{w['name']}: where the profile gap occurs\nThrough-backward includes forward, communication and host delays; it is not pure compute.",
        fontsize=12,
    )
    save(fig, output / f'{category}_2_progress_gap.png')

    fig, axes = plt.subplots(
        rows, cols, figsize=(15, 5 * rows), squeeze=False, layout='constrained'
    )
    for ax, case, d in zip(axes.flat, cases, data):
        peer = d['nep'][0]['full_ranks']
        entries = [(m, r) for m in ('healthy', 'nep') for r in (0, peer)]
        for i, (mode, rank) in enumerate(entries):
            left = 0
            for key, label, color in PHASES:
                value = field(d, mode, rank, key)
                ax.barh(i, value, left=left, color=color, label=label if i == 0 else None)
                if key == 'gradient_drain':
                    ax.text(
                        left + value / 2,
                        i,
                        f'{value:.1f}',
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='black',
                    )
                left += value
            cpu_end = mean(s['cpu_adam_end_ms'] for s in d[mode][rank]['steps'])
            ax.scatter(cpu_end, i, color='black', marker='|', s=100, zorder=4)
        ax.set_yticks(range(len(entries)), [f'{m} rank {r}' for m, r in entries])
        ax.invert_yaxis()
        ax.set_xlabel(
            'Milliseconds from each rank\'s own profiler-step start (no clock subtraction)'
        )
        ax.set_title(
            display_name(case)
            + '\nBlack mark: CPU finishes submitting Adam; orange: GPU gradient drain'
        )
        ax.spines[['top', 'right']].set_visible(False)
    for ax in list(axes.flat)[len(cases) :]:
        ax.set_visible(False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
    fig.suptitle(
        f'{w["name"]}: owner and EDP-peer progress\nGPU optimizer execution follows gradient readiness even when its CPU submission is much earlier.',
        fontsize=12,
    )
    save(fig, output / f'{category}_3_owner_peer.png')

    if category == 'category2':
        dependency_figure(cases, data, output, category)

    if category == 'confirmation':
        fig, axes = plt.subplots(
            rows, cols, figsize=(14, 3.5 * rows), squeeze=False, layout='constrained'
        )
        for ax, case in zip(axes.flat, cases):
            comparison = json.loads(Path(case['comparison_artifact']).read_text())
            for mode, color in (('healthy', '#0072B2'), ('nep', '#D55E00')):
                ax.plot(
                    comparison[mode]['timing_iterations'],
                    comparison[mode]['elapsed_ms'],
                    marker='.',
                    color=color,
                    label=mode,
                )
            ax.set_title(display_name(case))
            ax.set_xlabel('Training iteration')
            ax.set_xticks([11, 15, 20, 25, 30])
            ax.set_ylabel('Logged time (ms)')
            ax.legend()
            ax.spines[['top', 'right']].set_visible(False)
        for ax in list(axes.flat)[len(cases) :]:
            ax.set_visible(False)
        fig.suptitle(
            'Confirmation: retain run order and within-run drift\nTiming iterations 11--30; all-rank profiles 32--33. Each A/B pair shares nodes; job IDs identify different allocations.',
            fontsize=12,
        )
        save(fig, output / 'confirmation_4_timing_drift.png')

    if category == 'category1':
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
        colors = ['#0072B2', '#009E73', '#D55E00', '#999999']
        labels = [
            'Through first backward',
            'Middle microbatches',
            'Last microbatch',
            'Post-backward',
        ]
        for index, (case, d) in enumerate(zip(cases, data)):
            ends = {}
            for mode in ('healthy', 'nep'):
                steps = d[mode][0]['steps']
                ends[mode] = [
                    mean(s['backward_microbatches_ms'][i][1] for s in steps)
                    for i in range(len(steps[0]['backward_microbatches_ms']))
                ]
            cumulative = np.array(ends['nep']) - np.array(ends['healthy'])
            axes[0].plot(range(1, len(cumulative) + 1), cumulative, marker='.', label=case['label'])
            total = mean(s['span_ms'] for s in d['nep'][0]['steps']) - mean(
                s['span_ms'] for s in d['healthy'][0]['steps']
            )
            values = [
                cumulative[0],
                cumulative[-2] - cumulative[0],
                cumulative[-1] - cumulative[-2],
                total - cumulative[-1],
            ]
            bottom = 0
            for value, label, color in zip(values, labels, colors):
                axes[1].bar(
                    index, value, bottom=bottom, color=color, label=label if index == 0 else None
                )
                axes[1].text(
                    index, bottom + value / 2, f'{value:.0f}', ha='center', va='center', fontsize=9
                )
                bottom += value
        axes[0].set_xlabel('Completed backward microbatch')
        axes[0].set_ylabel('Cumulative NEP minus healthy (ms)')
        axes[0].legend()
        axes[0].set_title('Rank 0 GPU progress gap')
        axes[1].set_xticks(range(len(cases)), [c['label'] for c in cases])
        axes[1].set_ylabel('Disjoint profile-step delta (ms)')
        axes[1].set_title('Timing location, not additive operation costs')
        fig.legend(*axes[1].get_legend_handles_labels(), loc='outside lower center', ncol=2)
        fig.suptitle(
            'a8b: similar absolute overhead, but its timing within the iteration changes\nGA4 and GA20 are separate GB300 allocations; profile gaps are not logged timing gaps.',
            fontsize=12,
        )
        save(fig, output / 'category1_4_microbatch_progress.png')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for cat in ('category1', 'category2'):
        p.add_argument(f'--{cat}-summary', type=Path, required=True)
        p.add_argument(f'--{cat}-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--confirmation-summary', type=Path)
    p.add_argument('--confirmation-dir', type=Path)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for cat in ('category1', 'category2'):
        cases = json.loads(getattr(a, f'{cat}_summary').read_text())['cases']
        render(cases, getattr(a, f'{cat}_dir'), a.output_dir, cat)
    if a.confirmation_summary:
        if not a.confirmation_dir:
            p.error('--confirmation-dir is required with --confirmation-summary')
        cases = json.loads(a.confirmation_summary.read_text())['cases']
        render(cases, a.confirmation_dir, a.output_dir, 'confirmation')


if __name__ == '__main__':
    main()
