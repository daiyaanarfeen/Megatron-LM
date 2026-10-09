"""Sequence-only A/B using matched iteration timings and existing event analysis."""

import argparse
import json
import statistics

from analyze_nep_collective_probe import inspect_rank
from analyze_nep_ordinary_experts_20260915 import phase_metrics, summary
from analyze_nep_sync_workload import ROOT, SLUG, compare, read_case
from analyze_nep_tail_control import milestones


def normalized_options(options):
    result = options.copy()
    result[result.index('--seq-length') + 1] = 'SEQUENCE'
    return ['ARTIFACT' if value.startswith('/lustre/') else value for value in result]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    root = ROOT / args.job
    result = dict(job=args.job, points={})
    reference = {}
    for sequence in (2048, 8192):
        tag = f'seq{sequence}'
        pair = {case: read_case(root, tag, case) for case in ('healthy', 'nep')}
        point = dict(phases=pair, comparison=compare(pair), cases={})
        for case, size in (('healthy', 16), ('nep', 14)):
            run = root / tag / 'cupti/ep8' / SLUG / case / args.job
            phases, comm = {}, {}
            for rank in range(size):
                manifest = json.loads((run / 'launch_manifest' / f'rank-{rank}.json').read_text())
                opts = manifest['options']
                assert opts[opts.index('--seq-length') + 1] == str(sequence)
                assert manifest['environment']['REPLICA_MICRO_BATCH_SIZES'] == '2 2'
                key = case, rank
                signature = (normalized_options(opts), manifest['runtime_sha256'])
                if key in reference:
                    assert signature == reference[key], (key, 'unrelated argument/runtime change')
                else:
                    reference[key] = signature
                path = run / 'cupti_profile' / f'rank-{rank}.json'
                phases[rank] = phase_metrics(path)
                comm[rank] = inspect_rank(path)
            entry = dict(milestones=milestones(run, 'full', size), cohorts={})
            for name, ranks in (('full', range(6)), ('reduced', range(8, 14))):
                phase_summary = {
                    key: summary([statistics.mean(phases[r][i][key] for r in ranks)
                                  for i in range(10)])
                    for key in phases[0][0]
                }
                categories = set().union(*(set(row['categories']) for r in ranks
                                           for row in comm[r]['iterations']))
                comm_summary = {}
                for category in categories:
                    example = next(row['categories'][category] for r in ranks
                                   for row in comm[r]['iterations'] if category in row['categories'])
                    comm_summary[category] = {
                        key: summary([statistics.mean(
                            comm[r]['iterations'][i]['categories'].get(category, {}).get(key, 0)
                            for r in ranks) for i in range(10)]) for key in example
                    }
                entry['cohorts'][name] = dict(phases=phase_summary, collectives=comm_summary)
            point['cases'][case] = entry
        result['points'][tag] = point
        print(tag, json.dumps(point['comparison'], indent=2), flush=True)
        for case, entry in point['cases'].items():
            print(case, 'tail', {k: round(v['mean'], 3)
                                for k, v in entry['milestones']['stats'].items()}, flush=True)
            for cohort, values in entry['cohorts'].items():
                print(case, cohort, 'ordinary', {k: round(v['mean'], 3)
                    for k, v in values['phases'].items() if k.startswith(('forward1_', 'backward1_'))}, flush=True)
    for case in ('healthy', 'nep'):
        assert result['points']['seq2048']['phases'][case]['bucket_log'] == result['points']['seq8192']['phases'][case]['bucket_log']
        for cohort in ('full', 'reduced'):
            a = result['points']['seq2048']['cases'][case]['cohorts'][cohort]['collectives']
            b = result['points']['seq8192']['cases'][case]['cohorts'][cohort]['collectives']
            for category in ('expert_all_gather', 'expert_reduce_scatter', 'nep_grad_gather', 'nep_param_broadcast'):
                if category not in a:
                    assert category not in b
                    continue
                for key in ('count', 'input_bytes', 'output_bytes'):
                    assert a[category][key]['mean'] == b[category][key]['mean'], (case, cohort, category, key)
    output = root / 'sequence_control_analysis'
    output.mkdir(exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(output / 'summary.json')


if __name__ == '__main__':
    main()
