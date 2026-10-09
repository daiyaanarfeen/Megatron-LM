#!/usr/bin/env python3
"""Render audited existing-data comparisons without counterfactual attribution."""

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHASES = [
    ('through_backward', 'Start through model backward', '#0072B2'),
    ('gradient_drain', 'Backward end to GPU Adam', '#D55E00'),
    ('adam', 'GPU Adam', '#009E73'),
    ('after_adam', 'After GPU Adam', '#777777'),
]
OPS = [
    ('grad_gather', 'Gradient Gather'),
    ('expert_grad_rs', 'Expert gradient reduce-scatter'),
    ('expert_param_ag', 'Expert parameter all-gather'),
    ('param_redistribution', 'NEP parameter redistribution'),
    ('model_ep_poll', 'Model-EP polling'),
    ('model_ep_other', 'Other identified HybridEP kernels'),
    ('dense_dp_sync', 'Dense-DP sync (includes small reductions)'),
    ('dense_param_ag', 'Dense parameter all-gather'),
    ('other_collective', 'Other NCCL collectives'),
]


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def profile(path: Path, rank: int = 0) -> dict:
    rows = read(path / f'rank-{rank}.json')['steps']
    assert {r['step'] for r in rows} == {'ProfilerStep#5', 'ProfilerStep#6'}
    for row in rows:
        assert abs(sum(row['phases_ms'].values()) - row['span_ms']) < 1e-5
        assert abs(sum(row['step_states_ms'].values()) - row['span_ms']) < 1e-5
        for op in row['operations'].values():
            assert -1e-5 <= op['exposed_ms'] <= op['residency_ms'] + 1e-5
    return {
        'span_ms': mean(r['span_ms'] for r in rows),
        'phases_ms': {k: mean(r['phases_ms'][k] for r in rows) for k, _, _ in PHASES},
        'operations': {
            k: {
                field: mean(r['operations'].get(k, {}).get(field, 0) for r in rows)
                for field in ('residency_ms', 'exposed_ms')
            }
            for k, _ in OPS
        },
        'source': str(path / f'rank-{rank}.json'),
    }


def stacks(ax: plt.Axes, values: list[dict], labels: list[str]) -> None:
    bottom = [0.0] * len(values)
    for key, label, color in PHASES:
        heights = [v['phases_ms'][key] for v in values]
        ax.bar(range(len(values)), heights, bottom=bottom, color=color, label=label)
        bottom = [a + b for a, b in zip(bottom, heights)]
    for i, value in enumerate(bottom):
        ax.text(i, value, f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(values)), labels)
    ax.set_ylabel('Profiled milliseconds; rank-local step origin')
    ax.set_ylim(0, max(bottom) * 1.27)
    ax.legend(fontsize=8, loc='upper left')


def overview(output: Path, name: str, title: str, cases: list[dict], scopes: list | None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), layout='constrained')
    ax = axes[0, 0]
    for i, case in enumerate(cases):
        for offset, mode, color in [(-0.18, 'healthy', '#0072B2'), (0.18, 'nep', '#D55E00')]:
            vals = case['comparison'][mode]['elapsed_ms']
            ax.bar(i + offset, mean(vals), width=0.35, color=color, label=mode if i == 0 else None)
            ax.scatter([i + offset] * len(vals), vals, color='black', s=14, zorder=4)
        pct = case['comparison']['comparison']['owner_iteration_time_parity_percent']
        ax.text(
            i,
            max(case['comparison'][m]['mean_elapsed_ms'] for m in ('healthy', 'nep')) * 1.04,
            f'{pct:.2f}% mean parity',
            ha='center',
        )
    ax.set_xticks([0, 1], [c['label'] for c in cases])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_title('Original logged means and every sample (iterations 8--10)')
    ax.set_ylabel('Iteration milliseconds; dots are individual samples')
    ax.legend()
    values = [c['profile'][m] for c in cases for m in ('healthy', 'nep')]
    labels = [f'{c["short"]}\n{m}' for c in cases for m in ('healthy', 'nep')]
    stacks(axes[0, 1], values, labels)
    axes[0, 1].set_title('Owner rank 0: exact phase partition (profile steps 5--6)')
    ax = axes[1, 0]
    for i, case in enumerate(cases):
        h, n = (case['profile'][m] for m in ('healthy', 'nep'))
        positive = negative = 0
        for key, label, color in PHASES:
            delta = n['phases_ms'][key] - h['phases_ms'][key]
            ax.bar(
                i,
                delta,
                bottom=positive if delta >= 0 else negative,
                color=color,
                width=0.6,
                label=label if i == 0 else None,
            )
            if delta >= 0:
                positive += delta
            else:
                negative += delta
        gap = n['span_ms'] - h['span_ms']
        assert abs(positive + negative - gap) < 1e-5
        ax.scatter(i, gap, color='black', marker='D', zorder=5)
        ax.annotate(f'Net {gap:.1f} ms', (i, gap), xytext=(12, 8), textcoords='offset points')
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xticks([0, 1], [c['short'] for c in cases])
    ax.set_ylabel('NEP minus healthy, profiled milliseconds')
    ax.set_title('Where the measured gap occurs; not independent causal costs')
    ax.margins(y=0.25)
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    if scopes is not None:
        for i, scope in enumerate(scopes):
            for offset, key, label, color in [
                (
                    -0.18,
                    'nep_distopt_param_scatter',
                    'Parameter redistribution (4500 copies)',
                    '#009E73',
                ),
                (
                    0.18,
                    'nep_distopt_stage_owner_grads',
                    'Gradient staging (2580 copies)',
                    '#CC79A7',
                ),
            ]:
                value = sum(scope['scope_states_mean_ms'][key].values())
                ax.bar(i + offset, value, width=0.35, color=color, label=label if i == 0 else None)
                ax.text(i + offset, value, f'{value:.1f}', ha='center', va='bottom')
        ax.set_xticks([0, 1], [c['short'] for c in cases])
        ax.set_ylim(0, 270)
        ax.legend(fontsize=8)
        ax.set_ylabel('GPU scope span ms, including gaps; not extra additive cost')
        ax.set_title('Fixed-count layout work persists as GA grows')
    else:
        values = [c['peers'][r] for c in cases for r in ('0', '8')]
        labels = [f'{c["short"]}\nNEP r{r}' for c in cases for r in ('0', '8')]
        stacks(ax, values, labels)
        ax.set_title('Reduced-replica work reduction shifts gradient waiting')
    fig.suptitle(title, fontsize=12)
    fig.savefig(output / f'{name}_overview.png', dpi=160)
    plt.close(fig)


def communications(output: Path, name: str, title: str, cases: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), layout='constrained')
    for ax, case in zip(axes, cases):
        for offset, mode, color in [(-0.18, 'healthy', '#0072B2'), (0.18, 'nep', '#D55E00')]:
            for i, (key, label) in enumerate(OPS):
                op = case['profile'][mode]['operations'][key]
                total, exposed = op['residency_ms'], op['exposed_ms']
                ax.barh(
                    i + offset,
                    exposed,
                    height=0.32,
                    color=color,
                    label=f'{mode}: exposed' if i == 0 else None,
                )
                ax.barh(
                    i + offset,
                    max(0, total - exposed),
                    left=exposed,
                    height=0.32,
                    color=color,
                    alpha=0.32,
                    hatch='//',
                    label=f'{mode}: overlaps another category' if i == 0 else None,
                )
                ax.text(
                    total + 2, i + offset, f'{exposed:.1f} / {total:.1f}', va='center', fontsize=8
                )
        ax.set_yticks(range(len(OPS)), [label for _, label in OPS])
        ax.invert_yaxis()
        ax.set_xlim(0, ax.get_xlim()[1] * 1.25)
        ax.set_xlabel('Category activity union, ms; labels = exposed / total')
        ax.set_title(case['label'] + '\nOwner rank 0, profile steps 5--6')
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.suptitle(
        title + '\nExposure = no other classified GPU activity on this rank. '
        'Waiting is included; overlap is not necessarily useful compute.\n'
        'Overlapped portions of different rows are NOT additive. '
        'Parameter redistribution is NOT gradient Scatter.',
        fontsize=11,
    )
    fig.savefig(output / f'{name}_communications.png', dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10})
    progress = args.root / 'analysis/nep_owner_progress_20260909'
    manifest = read(
        args.root / 'analysis/nep_failure_categories/amortization_a8b_ga4_ga20/trace_summary.json'
    )
    evidence = {}
    for name in ('a8b', 'a500'):
        cases = []
        for i in range(2):
            if name == 'a8b':
                src = manifest['cases'][i]
                comparison = Path(src['comparison_artifact'])
                directory = progress / 'category1' / src['label'].replace(' ', '_')
                short = ['GA4', 'GA20'][i]
                label = short + ['; GBS H128 / N112', '; GBS H640 / N560'][i]
            else:
                ga = ['ga12_12', 'ga12_9'][i]
                comparison = (
                    args.root
                    / f'slurm_runs/a500_readiness_controlled/{ga}/ep8/scaling_ladder__a500m_2b_moe_140b/comparison_2977274.json'
                )
                directory = (
                    progress / 'category2_v3' / ['GA12_12_proportional', 'GA12_9_reduced'][i]
                )
                short = ['NEP GA12/12', 'NEP GA12/9'][i]
                label = short + ['; GBS H192 / N168', '; GBS H192 / N150'][i]
            case = {
                'short': short,
                'label': label,
                'comparison': read(comparison),
                'comparison_path': str(comparison),
                'profile': {m: profile(directory / m) for m in ('healthy', 'nep')},
            }
            if name == 'a500':
                case['peers'] = {str(r): profile(directory / 'nep', r) for r in (0, 8)}
            cases.append(case)
        title = (
            'a8b latent MoE | H4608, expert input1152/F3072, 512 experts, top-k6; seq8192, TP2 CP1, EP16/12, MBS2\n'
            'Flex/HybridEP + DistOpt; expert gradient RS calls H6/N8. GA4: one NVLink domain; GA20: two domains.\n'
            'Observed amortization, NOT an isolated GA intervention. Logged timing and profile windows are separate.'
            if name == 'a8b'
            else 'a500 hybrid MoE | H1024/F640, 128 experts, top-k6; seq8192, TP1 CP1, EP8/6, MBS1; healthy GA12/12\n'
            'Flex/HybridEP + DistOpt; expert gradient RS calls H4/N8. Both cases: same four GB300 nodes, one NVLink domain.\n'
            'Historical mean parity is shown faithfully; its 97.45% endpoint was not reproduced reliably.'
        )
        scopes = read(progress / 'category1_scope_progress.json') if name == 'a8b' else None
        overview(args.output, name, title, cases, scopes)
        communications(args.output, name, title.split('\n')[0], cases)
        evidence[name] = cases
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    reproduction = []
    for i, ga in enumerate(('ga12_12', 'ga12_9')):
        original = evidence['a500'][i]['comparison']
        repeat = read(
            args.root
            / f'slurm_runs/a500_exact_placement_reproduction/{ga}/ep8/scaling_ladder__a500m_2b_moe_140b/comparison_2979126.json'
        )
        for j, (name, comparison) in enumerate(
            [('Original', original), ('Exact-node repeat', repeat)]
        ):
            for offset, mode, color in [(-0.15, 'healthy', '#0072B2'), (0.15, 'nep', '#D55E00')]:
                vals = comparison[mode]['elapsed_ms']
                axes[i].scatter(
                    [j + offset] * len(vals), vals, color=color, label=mode if j == 0 else None
                )
                axes[i].plot([j + offset - 0.09, j + offset + 0.09], [mean(vals)] * 2, color=color)
            pct = comparison['comparison']['owner_iteration_time_parity_percent']
            axes[i].text(j, 1550, f'{pct:.2f}% mean parity', ha='center')
        axes[i].set_xticks([0, 1], ['Original job2977274', 'Exact-node job2979126'])
        axes[i].set_title(evidence['a500'][i]['short'])
        axes[i].set_ylim(1100, 1600)
        axes[i].set_ylabel('Logged iteration ms; all samples8--10, line = mean')
        axes[i].legend()
        reproduction.append({'setting': ga, 'original': original, 'repeat': repeat})
    fig.suptitle(
        'a500 reproducibility | same model/batch settings and original GB300 nodes theia0028--0031, segment4\n'
        'Profiles enabled at steps5--6; logged samples8--10 shown separately. '
        'Below-proportional median parity: original94.41%, repeat94.25%.',
        fontsize=11,
    )
    fig.savefig(args.output / 'a500_reproducibility.png', dpi=160)
    plt.close(fig)
    evidence['a500_reproduction'] = reproduction
    (args.output / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print('Validated phase/state closure and exposure bounds; rendered five figures.')


if __name__ == '__main__':
    main()
