#!/usr/bin/env python3
"""Render two bounded closeout figures; temporal accounting is not causal attribution."""

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def phase_delta(case: Path) -> dict:
    h, n = (read(case / mode / 'rank-0.json')['steps'] for mode in ('healthy', 'nep'))
    result = {
        k: mean(s['phases_ms'][k] for s in n) - mean(s['phases_ms'][k] for s in h)
        for k in h[0]['phases_ms']
    }
    assert (
        abs(sum(result.values()) - (mean(s['span_ms'] for s in n) - mean(s['span_ms'] for s in h)))
        < 1e-5
    )
    return result


def phase_bars(ax: plt.Axes, cases: list[Path], labels: list[str]) -> list[dict]:
    data = [phase_delta(p) for p in cases]
    styles = [
        ('through_backward', 'Through backward (not pure compute)', '#0072B2'),
        ('gradient_drain', 'Backward end to GPU Adam', '#D55E00'),
        ('adam', 'GPU Adam span', '#009E73'),
        ('after_adam', 'After GPU Adam', '#888888'),
    ]
    for i, row in enumerate(data):
        positive = negative = 0
        for key, label, color in styles:
            value = row[key]
            ax.bar(
                i,
                value,
                bottom=positive if value >= 0 else negative,
                color=color,
                label=label if i == 0 else None,
                width=0.6,
            )
            if value >= 0:
                positive += value
            else:
                negative += value
        total = sum(row.values())
        ax.scatter(i, total, color='black', marker='D', zorder=5)
        ax.annotate(f'{total:.1f} ms', (i, total), xytext=(12, 4), textcoords='offset points')
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel('NEP - healthy, profiled ms')
    ax.set_title('Same-profile gap: exact temporal decomposition')
    low, high = ax.get_ylim()
    ax.set_ylim(low, high * 1.55)
    ax.legend(fontsize=8)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--recurring', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10})
    c1 = [args.progress / 'category1' / x for x in ('MBS2_x_GA4_(before)', 'MBS2_x_GA20_(after)')]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout='constrained')
    manifest = read(
        Path(
            '/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/nep_failure_categories/amortization_a8b_ga4_ga20/trace_summary.json'
        )
    )
    for i, case in enumerate(manifest['cases']):
        for offset, mode, color in [(-0.18, 'healthy', '#0072B2'), (0.18, 'nep', '#D55E00')]:
            value = case['timing'][f'{mode}_mean_ms'] / 1000
            axes[0].bar(i + offset, value, width=0.35, color=color, label=mode if i == 0 else None)
        axes[0].text(
            i,
            case['timing']['nep_mean_ms'] / 1000 + 0.7,
            f"{case['timing']['parity_percent']:.1f}% parity",
            ha='center',
        )
    axes[0].set_xticks([0, 1], ['GA4; GBS H128/N112', 'GA20; GBS H640/N560'])
    axes[0].set_ylim(0, 25)
    axes[0].set_ylabel('Logged iteration seconds')
    axes[0].set_title('Accepted: denominator grows, penalty persists')
    axes[0].legend()
    delta1 = phase_bars(axes[1], c1, ['GA4', 'GA20'])
    scopes = read(args.progress / 'category1_scope_progress.json')
    for i, scope in enumerate(scopes):
        values = scope['scope_states_mean_ms']
        for offset, key, color, label in [
            (-0.18, 'nep_distopt_param_scatter', '#009E73', 'Parameter redistribution'),
            (0.18, 'nep_distopt_stage_owner_grads', '#CC79A7', 'Gradient staging'),
        ]:
            axes[2].bar(
                i + offset,
                sum(values[key].values()),
                width=0.35,
                color=color,
                label=label if i == 0 else None,
            )
    axes[2].set_xticks([0, 1], [x['label'] for x in scopes])
    axes[2].set_title('Fixed layout scopes: 4500 + 2580 copies/iter')
    axes[2].set_ylabel('Recorded scope ms; contained in middle panel')
    axes[2].legend(fontsize=8)
    fig.suptitle(
        'a8b latent MoE: amortization accepted; complete causal attribution unresolved\n'
        'H4608 / expert input1152 / F3072; 512 experts, top-k6; EP16/12, TP2 CP1, MBS2, seq8192\n'
        'Flex/HybridEP dynamic sizing + DistOpt; expert RS calls H6/N8; distinct GB300 allocations',
        fontsize=12,
    )
    fig.savefig(args.output / 'a8b_closeout.png', dpi=160)
    plt.close(fig)
    names = ['GA12_12_proportional', 'GA12_9_reduced']
    c2 = [args.progress / 'category2_v3' / x for x in names]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout='constrained')
    delta2 = phase_bars(axes[0], c2, ['GA12/12', 'GA12/9'])
    entries = [
        (names[0], 'healthy', 8),
        (names[0], 'healthy', 11),
        (names[0], 'nep', 8),
        (names[0], 'nep', 11),
        (names[1], 'nep', 8),
        (names[1], 'nep', 11),
    ]
    labels = ['H r8', 'H r11', 'N12 r8', 'N12 r11', 'N9 r8', 'N9 r11']
    recurring = []
    for i, (case, mode, rank) in enumerate(entries):
        w = read(args.recurring / case / mode / f'rank-{rank}.json')['windows']
        total = mean(s['span_ms'] for s in w)
        idle = mean(s['states_ms'].get('idle', 0) for s in w)
        poll = mean(s['states_ms'].get('model_ep_poll', 0) for s in w)
        values = [idle, poll, total - idle - poll]
        bottom = 0
        for value, label, color in zip(
            values,
            ['GPU idle', 'Model-EP polling only', 'All other GPU states'],
            ['#999999', '#D55E00', '#0072B2'],
        ):
            axes[1].bar(i, value, bottom=bottom, color=color, label=label if i == 0 else None)
            bottom += value
        host = read(args.recurring / case / mode / f'host-{rank}.json')['windows']
        late = mean(s['idle_before_next_launch_ms'] for s in host)
        grouped = mean(s['idle_before_next_launch_in_grouped_cpu_ms'] for s in host)
        axes[2].bar(i, late, color='#999999')
        axes[2].bar(i, grouped, color='#009E73')
        recurring.append(
            {
                'case': case,
                'mode': mode,
                'rank': rank,
                'window_ms': total,
                'idle_ms': idle,
                'poll_only_ms': poll,
                'idle_before_launch_ms': late,
                'grouped_subset_ms': grouped,
            }
        )
    axes[1].set_title('Equal work count: microbatches 2--8 on each rank')
    axes[1].set_ylabel('Local seven-microbatch window ms')
    axes[1].legend(fontsize=8)
    axes[2].set_title(
        'Idle before next GPU launch begins\nGreen = subset inside grouped-linear CPU calls'
    )
    axes[2].set_ylabel('ms; subset of idle in middle panel, not additive')
    for ax in axes[1:]:
        ax.set_xticks(range(len(labels)), labels, rotation=30)
    fig.suptitle(
        'a500: historical gradient-wait shift observed; general failure/fix claim withdrawn\n'
        'H1024 F640; 128 experts, top-k6; EP8/6, TP1 CP1, MBS1, seq8192; H GA12/12, N GA12/12 or12/9\n'
        'GBS H192/N168 or150; expert RS H4/N8; dynamic HybridEP + DistOpt; same GB300 allocation',
        fontsize=12,
    )
    fig.savefig(args.output / 'a500_closeout.png', dpi=160)
    plt.close(fig)
    (args.output / 'evidence.json').write_text(
        json.dumps(
            {
                'a8b_profile_phase_deltas': delta1,
                'a500_profile_phase_deltas': delta2,
                'a500_recurring_windows': recurring,
            },
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    main()
