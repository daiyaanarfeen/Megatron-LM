"""Locate expert-width forward overhead using existing CPU/CUDA-event ranges.

Ranges are not kernel busy time. Native owner all-gather launch/completion and
non-expert GPU kernels are not instrumented by this capture protocol.
"""

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from analyze_cuda_event_mbs_pair import ROOT, analyze

JOB = '3063815'
TAGS = ('anchor_start', 'expert_f3072', 'expert_f3072_seq2048', 'anchor_end')
SLUG = 'scaling_ladder__a770m_4b_moe_215b'
RUNTIME = Path('/home/darfeen/Megatron-LM/.worktrees/nep-sync-cpu-20260914')
OUT = ROOT / JOB / 'width_delivery_analysis'


def mean(values):
    return statistics.mean(values)


def stats(values):
    return dict(mean=mean(values), sd=statistics.stdev(values), samples=values)


def end(event):
    return event['ts'] + event['dur']


def read_rank(path: Path, mode: str) -> list[dict]:
    """Audit raw profiles, then partition forward into expert ranges and gaps."""
    audited = analyze(path, 40, 50, mode)
    data = json.loads(path.read_text())
    result = []
    for step in range(40, 50):
        selected = [e for e in data['traceEvents'] if e.get('args', {}).get('iteration') == step]

        def events(kind, name):
            return sorted((e for e in selected if e['cat'] == kind + '_operation_range'
                           and e['name'] == name), key=lambda e: e['ts'])

        gf, cf = events('gpu', 'expert.forward'), events('cpu', 'expert.forward')
        gb = events('gpu', 'expert.backward')
        gp, cp = events('gpu', 'nep.redistribute_params'), events('cpu', 'nep.redistribute_params')
        assert len(gf) == len(cf) == len(gb) == 90
        assert len(gp) == len(cp) == (8 if mode == 'nep' else 0)
        window = next(e for e in data['traceEvents'] if e['cat'] == 'iteration'
                      and e['name'] == f'iteration_{step}')
        layers = []
        for mb in range(3):
            layer_rows = []
            fs = gf[30 * mb:30 * (mb + 1)]
            for layer in range(15):
                a, b = fs[2 * layer:2 * layer + 2]
                gap = 0 if layer == 0 else (a['ts'] - end(fs[2 * layer - 1])) / 1000
                row = dict(grouped_ranges=(a['dur'] + b['dur']) / 1000,
                           between_fc=(b['ts'] - end(a)) / 1000, before_layer=gap)
                assert min(row.values()) >= 0
                row['increment'] = sum(row.values())
                layer_rows.append(row)
            assert abs(sum(r['increment'] for r in layer_rows) - (end(fs[-1]) - fs[0]['ts']) / 1000) < 1e-6
            layers.append(layer_rows)
        params = []
        for ordinal, (g, c) in enumerate(zip(gp, cp)):
            upcoming = next((i for i, e in enumerate(cf[:30]) if e['ts'] >= end(c)), None)
            assert upcoming is not None
            previous = gf[upcoming - 1] if upcoming else None
            overlaps = sum(max(0, min(end(g), end(e)) - max(g['ts'], e['ts'])) for e in gf[:30]) / 1000
            params.append(dict(
                ordinal=ordinal, next_expert_layer=upcoming // 2, next_fc=upcoming % 2 + 1,
                gpu_range_ms=g['dur'] / 1000, cpu_ms=c['dur'] / 1000,
                expert_range_overlap_ms=overlaps,
                same_stream=g['args']['stream'] == gf[upcoming]['args']['stream'],
                cpu_submit_after_previous_expert_ms=(c['ts'] - end(previous)) / 1000 if previous else None,
                gpu_start_after_previous_expert_ms=(g['ts'] - end(previous)) / 1000 if previous else None,
                gpu_end_before_next_expert_ms=(gf[upcoming]['ts'] - end(g)) / 1000,
                cpu_submit_before_next_expert_ms=(gf[upcoming]['ts'] - c['ts']) / 1000,
                clock_error_ms=data['clock_error_bound_ns'] / 1e6,
            ))
        scalars = dict(
            first_forward_ms=sum(r['increment'] for r in layers[0]),
            ordinary_forward_ms=mean(sum(r['increment'] for r in layers[mb]) for mb in (1, 2)),
            first_forward_grouped_ms=sum(r['grouped_ranges'] for r in layers[0]),
            ordinary_forward_grouped_ms=mean(sum(r['grouped_ranges'] for r in layers[mb]) for mb in (1, 2)),
            ordinary_backward_grouped_ms=mean(sum(e['dur'] for e in gb[mb * 30:(mb + 1) * 30]) / 1000 for mb in (0, 1)),
            param_gpu_range_sum_ms=sum(e['dur'] for e in gp) / 1000,
            param_cpu_sum_ms=sum(e['dur'] for e in cp) / 1000,
            before_first_expert_ms=(gf[0]['ts'] - window['ts']) / 1000,
            iteration_ms=window['dur'] / 1000,
        )
        assert abs(scalars['first_forward_ms'] - audited['iterations'][step - 40]['temporal_spans_ms'][
            'mb0_first_forward -> mb0_last_forward']) < 1e-6
        result.append(dict(iteration=step, layers=layers, params=params, scalars=scalars))
    return result


def cohort(ranks: list[list[dict]], ids: range) -> dict:
    """Average ranks inside each iteration before computing SD across iterations."""
    def series(fn):
        return stats([mean(fn(ranks[r][i]) for r in ids) for i in range(10)])

    layers = []
    for layer in range(15):
        row = {}
        for key in ranks[0][0]['layers'][0][0]:
            row['first_' + key] = series(lambda x: x['layers'][0][layer][key])
            row['ordinary_' + key] = series(lambda x: mean(x['layers'][mb][layer][key] for mb in (1, 2)))
        layers.append(row)
    scalar = {k: series(lambda x: x['scalars'][k]) for k in ranks[0][0]['scalars']}
    params = []
    if ranks[ids[0]][0]['params']:
        for ordinal in range(8):
            entries = [ranks[r][i]['params'][ordinal] for r in ids for i in range(10)]
            row = dict(ordinal=ordinal, next_expert_layers=dict(Counter(e['next_expert_layer'] for e in entries)),
                       next_fc=dict(Counter(e['next_fc'] for e in entries)),
                       same_stream_fraction=mean(e['same_stream'] for e in entries))
            for k in ('gpu_range_ms', 'cpu_ms', 'expert_range_overlap_ms',
                      'cpu_submit_after_previous_expert_ms', 'gpu_start_after_previous_expert_ms',
                      'gpu_end_before_next_expert_ms', 'cpu_submit_before_next_expert_ms'):
                values = [e[k] for e in entries if e[k] is not None]
                row[k] = dict(mean=mean(values), min=min(values), max=max(values)) if values else None
            with_previous = [e for e in entries if e['cpu_submit_after_previous_expert_ms'] is not None]
            row['cpu_submit_after_previous_complete_fraction'] = (
                mean(e['cpu_submit_after_previous_expert_ms'] > 3 * e['clock_error_ms'] for e in with_previous)
                if with_previous else None)
            params.append(row)
    return dict(scalars=scalar, layers=layers, params=params)


def main() -> None:
    """Persist per-layer and parameter-call evidence without modifying runtime."""
    OUT.mkdir(exist_ok=True)
    runtime_hash = hashlib.sha256((RUNTIME / 'megatron/core/distributed/nonuniform_ep.py').read_bytes()).hexdigest()
    result = {}
    for tag in TAGS:
        entry = {}
        for mode, size in (('healthy', 16), ('nep', 14)):
            run = ROOT / JOB / tag / 'cupti/ep8' / SLUG / mode / JOB
            manifest = json.loads((run / 'launch_manifest/rank-0.json').read_text())
            assert manifest['runtime_sha256'] == runtime_hash
            options = manifest['options']
            pattern = options[options.index('--hybrid-layer-pattern') + 1]
            entry['moe_model_layer_indices_zero_based'] = [i for i, ch in enumerate(pattern) if ch == 'E']
            assert len(entry['moe_model_layer_indices_zero_based']) == 15
            ranks = [read_rank(run / f'cupti_profile/rank-{rank}.json', mode) for rank in range(size)]
            entry[mode] = {role: cohort(ranks, ids) for role, ids in (
                ('full_owners', range(6)), ('other_owners', range(8, 14)), ('full_followers', range(6, 8)))}
            print(tag, mode, {role: {k: round(v['mean'], 3) for k, v in data['scalars'].items()}
                             for role, data in entry[mode].items()}, flush=True)
        contrasts = []
        h, n = entry['healthy']['full_owners'], entry['nep']['full_owners']
        for layer, (hl, nl) in enumerate(zip(h['layers'], n['layers'])):
            row = dict(expert_layer=layer, model_layer=entry['moe_model_layer_indices_zero_based'][layer])
            for key in ('grouped_ranges', 'between_fc', 'before_layer', 'increment'):
                row['first_delta_' + key] = nl['first_' + key]['mean'] - hl['first_' + key]['mean']
                row['ordinary_delta_' + key] = nl['ordinary_' + key]['mean'] - hl['ordinary_' + key]['mean']
                row['sync_specific_' + key] = row['first_delta_' + key] - row['ordinary_delta_' + key]
            contrasts.append(row)
        delta = n['scalars']['first_forward_ms']['mean'] - h['scalars']['first_forward_ms']['mean']
        assert abs(sum(r['first_delta_increment'] for r in contrasts) - delta) < 1e-6
        entry['full_owner_layer_contrasts'] = contrasts
        entry['runtime_sha256'] = runtime_hash
        result[tag] = entry
        print(tag, 'per-layer first-forward delta', [(r['expert_layer'], round(r['first_delta_increment'], 3))
                                                     for r in contrasts], flush=True)
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (OUT / 'layer_contrasts.csv').open('w') as f:
        keys = list(result[TAGS[0]]['full_owner_layer_contrasts'][0])
        writer = csv.DictWriter(f, fieldnames=['case'] + keys)
        writer.writeheader()
        for tag, entry in result.items():
            for row in entry['full_owner_layer_contrasts']:
                writer.writerow(dict(case=tag, **row))
    print(OUT, flush=True)


if __name__ == '__main__':
    main()
