"""Cross-replica progress, retaining both local elapsed and wall-clock deltas.

Wall-clock deltas assume host clock alignment, which these profiles alone cannot
prove. Local elapsed progress and its cross-rank differences cancel fixed clock
offsets. Event envelopes are not pure kernel service or collective readiness.
"""

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline')
SLUG = 'scaling_ladder__a770m_4b_moe_215b'


def stats(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values), samples=values)


def read_rank(path):
    data = json.loads(path.read_text())
    assert data['clock_error_bound_ns'] < 200000
    rows = []
    for step in range(40, 50):
        events = data['traceEvents']
        window = next(e for e in events if e['cat'] == 'iteration' and e['name'] == f'iteration_{step}')
        points = dict(iteration_start=window['ts'], iteration_end=window['ts'] + window['dur'])
        for kind in ('gpu', 'cpu'):
            selected = [e for e in events if e['cat'] == f'{kind}_operation_range'
                        and e.get('args', {}).get('iteration') == step]
            for direction in ('forward', 'backward'):
                es = sorted((e for e in selected if e['name'] == f'expert.{direction}'), key=lambda e: e['ts'])
                assert len(es) == 90
                for mb in range(3):
                    chunk = es[mb * 30:(mb + 1) * 30]
                    points[f'{kind}_mb{mb}_{direction}_start'] = chunk[0]['ts']
                    points[f'{kind}_mb{mb}_{direction}_end'] = max(e['ts'] + e['dur'] for e in chunk)
            for label, name in [('grad_stage', 'nep.stage_grads'),
                                ('param_redistribute', 'nep.redistribute_params'),
                                ('optimizer', 'optimizer.step')]:
                es = [e for e in selected if e['name'] == name]
                if es:
                    points[f'{kind}_{label}_start'] = min(e['ts'] for e in es)
                    points[f'{kind}_{label}_end'] = max(e['ts'] + e['dur'] for e in es)
                    if label == 'grad_stage':
                        assert len(es) == 8
                        for index, event in enumerate(sorted(es, key=lambda e: e['ts'])):
                            points[f'{kind}_grad_stage_{index}_start'] = event['ts']
                            points[f'{kind}_grad_stage_{index}_end'] = event['ts'] + event['dur']
        rows.append(dict(local_ms={k: (v - window['ts']) / 1000 for k, v in points.items()},
                         epoch_ms={k: data['origin_ns'] / 1e6 + v / 1000 for k, v in points.items()}))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('--tags', nargs='+', required=True)
    args = parser.parse_args()
    root = ROOT / args.job
    result = dict(notes=__doc__, cases={})
    for tag in args.tags:
        pair = {}
        for case, size in [('healthy', 16), ('nep', 14)]:
            run = root / tag / 'cupti/ep8' / SLUG / case / args.job / 'cupti_profile'
            ranks = [read_rank(run / f'rank-{r}.json') for r in range(size)]
            groups = {'full_owners': range(6), 'other_owners': range(8, 14),
                      'full_replica': range(8), 'other_replica': range(8, size)}
            cohorts = {}
            for label, rs in groups.items():
                kinds = {}
                for kind in ('local_ms', 'epoch_ms'):
                    keys = set.intersection(*(set(ranks[r][0][kind]) for r in rs))
                    kinds[kind] = {key: stats([statistics.mean(ranks[r][i][kind][key] for r in rs)
                                              for i in range(10)]) for key in keys}
                cohorts[label] = kinds
            deltas = {}
            for kind in ('local_ms', 'epoch_ms'):
                a, b = cohorts['full_owners'][kind], cohorts['other_owners'][kind]
                deltas[kind] = {key: stats([y - x for x, y in zip(a[key]['samples'], b[key]['samples'])])
                                for key in a.keys() & b.keys()}
            pair[case] = dict(cohorts=cohorts, other_minus_full=deltas)
            selected = ['iteration_start', 'gpu_mb0_forward_end', 'gpu_mb1_forward_start',
                        'gpu_mb1_backward_end', 'gpu_mb2_backward_start', 'gpu_mb2_backward_end',
                        'gpu_grad_stage_start', 'gpu_grad_stage_end', 'gpu_optimizer_start']
            print(tag, case, 'other-full local progress',
                  {key: round(deltas['local_ms'][key]['mean'], 3) for key in selected if key in deltas['local_ms']}, flush=True)
            print('wall-clock iteration-start difference ms', deltas['epoch_ms']['iteration_start']['mean'], flush=True)
        result['cases'][tag] = pair
    out = root / 'ordinary_progress_analysis'
    out.mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(out / 'summary.json')


if __name__ == '__main__':
    main()
