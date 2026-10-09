"""Matched-window expert-count results, retaining per-count transport/payload audits."""

import argparse
import json
import statistics as st
from pathlib import Path

from analyze_nep_actual_ablation import summarize

PHASES = (
    'first_forward_ms',
    'other_forward_ms',
    'nonfinal_backward_ms',
    'final_backward_ms',
    'outside_fb_ms',
)


def delivery_waits(run, ranks):
    """Device demand-to-ready intervals, not NCCL transfer service times."""
    rows = []
    for rank in ranks:
        events = json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())['traceEvents']
        for step in range(25, 31):
            selected = [
                e
                for e in events
                if e.get('cat') == 'gpu_operation_range' and e['args']['iteration'] == step
            ]
            demands = {
                e['name'].removeprefix('parameter.demand.'): e
                for e in selected
                if e['name'].startswith('parameter.demand.expert.')
            }
            ready = {
                e['name'].removeprefix('parameter.ready.'): e
                for e in selected
                if e['name'].startswith('parameter.ready.expert.')
            }
            assert len(demands) == 5 and demands.keys() == ready.keys(), (run, rank, step)
            ordered = sorted(demands, key=lambda name: demands[name]['ts'])
            waits = [(ready[key]['ts'] - demands[key]['ts']) / 1000 for key in ordered]
            assert all(x >= 0 for x in waits)
            rows.append(
                dict(
                    first_bucket_ms=waits[0],
                    later_buckets_ms=sum(waits[1:]),
                    total_ms=sum(waits),
                    rank=rank,
                    iteration=step,
                )
            )
    return dict(
        means={
            key: st.mean(row[key] for row in rows)
            for key in ('first_bucket_ms', 'later_buckets_ms', 'total_ms')
        },
        records=rows,
    )


def plot_results(results, output, ga=8):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    counts = sorted(map(int, results))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    x = list(range(len(counts)))
    for mode, color, shift, label in [
        ('profiled', '#167b8c', -0.18, 'All-rank profile'),
        ('companion', '#bf562f', 0.18, 'Timing companion'),
    ]:
        ys = [results[e][mode]['parity'] for e in counts]
        lo = [y - min(r['parity'] for r in results[e][mode]['repeats']) for e, y in zip(counts, ys)]
        hi = [max(r['parity'] for r in results[e][mode]['repeats']) - y for e, y in zip(counts, ys)]
        bars = axes[0].bar(
            [i + shift for i in x],
            ys,
            width=0.34,
            color=color,
            label=label,
            yerr=[lo, hi],
            capsize=4,
        )
        axes[0].bar_label(bars, labels=[f'{y:.2f}%' for y in ys], padding=8, fontsize=9)
    axes[0].set_ylim(
        min(results[e][m]['parity'] for e in counts for m in ('profiled', 'companion')) - 3, 101
    )
    axes[0].axhline(100, color='#555555', linewidth=0.8, linestyle='--')
    axes[0].set_ylabel('Owner iteration-time parity (%)')
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, 1.14), ncol=2, frameon=False)
    positive, negative = [0.0] * len(x), [0.0] * len(x)
    for phase, label, color in zip(
        PHASES,
        ('First forward', 'Other forwards', 'Non-final backwards', 'Final backward', 'Outside F/B'),
        ('#176b8a', '#61adbb', '#b385bd', '#cf7b3d', '#969696'),
    ):
        ys = [results[e]['profiled']['phases']['owners'][phase]['gap_ms'] for e in counts]
        axes[1].bar(
            x,
            ys,
            bottom=[positive[i] if y >= 0 else negative[i] for i, y in enumerate(ys)],
            label=label,
            color=color,
            width=0.65,
        )
        for i, y in enumerate(ys):
            if y >= 0:
                positive[i] += y
            else:
                negative[i] += y
    axes[1].axhline(0, color='#555555', linewidth=0.8)
    axes[1].set_ylabel('Profiled NEP minus healthy phase time (ms)')
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xticks(x, counts)
        ax.set_xlabel('Total experts (top-k = 8)')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(
        'Expert-count ablation: 15-MoE a8b-derived hybrid/MTP, H4608 / latent1152 / F3072\n'
        f'S8448, MBS2, GA{ga}; healthy EP16/16 vs NEP EP16/12; TP2, CP1, ETP1',
        fontsize=12,
    )
    fig.text(
        0.5,
        0.015,
        f'GB300; GBS{32 * ga}/{28 * ga}; proportional local BS: '
        f'healthy{16 * ga}/{16 * ga}, NEP{16 * ga}/{12 * ga}; five expert and three dense buckets.\n'
        'DistOpt + Flex/HybridEP + graphs. Matched iterations25-30 after24 warmup iterations; two reversed repeats.\n'
        'Whiskers show repeat range. Phase spans include compute and communication.',
        ha='center',
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.91))
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'expert_count_ablation.{extension}', dpi=180)
    plt.close(fig)


def main(root, repeats=2):
    output = root / 'expert_count'
    output.mkdir(exist_ok=True)
    results = {}
    for experts in (128, 256, 384):
        subset = output / str(experts)
        subset.mkdir(exist_ok=True)
        expected = [
            root / f'controlled_actual_{cell}{experts}_r{rep}{suffix}_{topology}_moe'
            for cell, topology in [('H', 'healthy'), ('N', 'nep')]
            for rep in range(repeats)
            for suffix in ('', '_off')
        ]
        if not all((run / 'actual-validation.json').exists() for run in expected):
            print('INCOMPLETE', experts, flush=True)
            continue
        for run in expected:
            link = subset / run.name
            if not link.exists():
                link.symlink_to(run, target_is_directory=True)
        # Deliberately scope invariant-weight and native-payload checks to one model.
        summarize(subset)
        source = json.loads((subset / 'actual-summary.json').read_text())['cases']
        modes = {}
        for mode, suffix in [('profiled', ''), ('companion', '_off')]:
            h, n = [
                [
                    source[f'controlled_actual_{cell}{experts}_r{rep}{suffix}_{topology}_moe']
                    for rep in range(repeats)
                ]
                for cell, topology in [('H', 'healthy'), ('N', 'nep')]
            ]
            healthy, nep = [st.mean(row['timing']['mean'] for row in rows) for rows in (h, n)]
            entry = dict(
                healthy_ms=healthy,
                nep_ms=nep,
                gap_ms=nep - healthy,
                parity=100 * healthy / nep,
                repeats=[
                    dict(
                        healthy_ms=a['timing']['mean'],
                        nep_ms=b['timing']['mean'],
                        parity=100 * a['timing']['mean'] / b['timing']['mean'],
                    )
                    for a, b in zip(h, n)
                ],
            )
            if mode == 'profiled':
                entry['phases'] = {
                    role: {
                        phase: dict(
                            healthy_ms=st.mean(r['cohorts'][role][phase]['mean'] for r in h),
                            nep_ms=st.mean(r['cohorts'][role][phase]['mean'] for r in n),
                        )
                        for phase in PHASES
                    }
                    for role in ('owners', 'followers', 'second')
                }
                for phases in entry['phases'].values():
                    for values in phases.values():
                        values['gap_ms'] = values['nep_ms'] - values['healthy_ms']
                phase_gap = sum(v['gap_ms'] for v in entry['phases']['owners'].values())
                entry['phase_accounting_residual_ms'] = entry['gap_ms'] - phase_gap
                assert abs(entry['phase_accounting_residual_ms']) < 1.0
                entry['expert_delivery_waits'] = {}
                for cell, topology in [('H', 'healthy'), ('N', 'nep')]:
                    entry['expert_delivery_waits'][topology] = [
                        delivery_waits(
                            root / f'controlled_actual_{cell}{experts}_r{rep}_{topology}_moe',
                            range(12),
                        )
                        for rep in range(repeats)
                    ]
            modes[mode] = entry
        modes['profile_differential_ms'] = (
            modes['profiled']['gap_ms'] - modes['companion']['gap_ms']
        )
        results[experts] = modes
        print(
            experts,
            {mode: round(modes[mode]['parity'], 3) for mode in ('profiled', 'companion')},
            flush=True,
        )
    (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    if len(results) == 3 and repeats == 2:
        specs = [json.loads((run / 'actual-spec.json').read_text()) for run in expected]
        gas = {spec['ga'] for spec in specs}
        assert len(gas) == 1 and all(spec['reduced_ga'] == spec['ga'] for spec in specs)
        plot_results(results, output, ga=gas.pop())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    main(args.root.resolve(), args.repeats)
