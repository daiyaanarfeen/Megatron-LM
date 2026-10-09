"""Validate paired local expert timing and diagnostic kernel identities."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def analyze(root):
    data = json.loads((root / 'results.json').read_text())
    summary = {'device': data['device'], 'cases': [], 'comparisons': []}
    for case in data['cases']:
        events = json.loads(
            (root / f"seq{case['sequence']}_experts{case['experts']}.json").read_text()
        )['traceEvents']
        item = {'sequence': case['sequence'], 'experts': case['experts'], 'modes': {}}
        fingerprints = []
        assert case['correctness'] == 'bitwise equal'
        for mode, name in [(False, 'normal'), (True, 'prequeued')]:
            rows = [r for r in case['rows'] if r['prequeued'] == mode]
            assert len(rows) == 20
            if mode:
                assert all(r['gpu_start_delay_ms'] - r['submission_ms'] > 20 for r in rows)
            result = {
                key: statistics.mean(r[key] for r in rows)
                for key in ['forward_ms', 'backward_ms', 'total_ms', 'submission_ms']
            }
            result['total_sample_sd_ms'] = statistics.stdev(r['total_ms'] for r in rows)
            result['kernel_profile_total_ms'] = statistics.mean(
                r['total_ms'] for r in case['diagnostic_profile_rows'] if r['prequeued'] == mode
            )
            result['kernel_profile_distortion_pct'] = 100 * (
                result['kernel_profile_total_ms'] / result['total_ms'] - 1
            )
            traces = []
            for scope in events:
                if scope.get('cat') != 'user_annotation' or scope['name'] != name:
                    continue
                es = [e for e in events if scope['ts'] <= e['ts'] < scope['ts'] + scope['dur']]
                kernels = sorted(
                    (
                        e
                        for e in es
                        if e.get('cat') == 'kernel'
                        and 'FillFunctor' not in e['name']
                        and 'spin_kernel' not in e['name']
                    ),
                    key=lambda e: e['ts'],
                )
                fingerprints.append(
                    Counter(
                        (e['name'], tuple(e['args']['grid']), tuple(e['args']['block']))
                        for e in kernels
                    )
                )
                last, union = 0, 0
                for e in kernels:
                    end = e['ts'] + e['dur']
                    union += max(0, end - max(last, e['ts']))
                    last = max(last, end)
                span = last - kernels[0]['ts']
                traces.append(
                    dict(
                        count=len(kernels),
                        kernel_union_ms=union / 1000,
                        kernel_span_ms=span / 1000,
                        no_kernel_gap_ms=(span - union) / 1000,
                        gemm_sum_ms=sum(e['dur'] for e in kernels if 'nvjet' in e['name']) / 1000,
                        grouped_backward_cpu_ms=sum(
                            e['dur']
                            for e in es
                            if e.get('cat') == 'cpu_op' and e['name'] == '_GroupedLinearBackward'
                        )
                        / 1000,
                    )
                )
            assert len(traces) == 2
            result['diagnostic'] = {k: statistics.mean(t[k] for t in traces) for k in traces[0]}
            item['modes'][name] = result
        assert all(fp == fingerprints[0] for fp in fingerprints)
        item['identical_kernel_names_and_launch_geometry'] = True
        summary['cases'].append(item)
    for seq in (8192, 16384):
        shapes = {c['experts']: c['modes'] for c in summary['cases'] if c['sequence'] == seq}
        comparison = {'sequence': seq}
        for phase in ['backward_ms', 'total_ms']:
            gaps = {}
            for mode in ['normal', 'prequeued']:
                reduced = (shapes[42][mode][phase] + 2 * shapes[43][mode][phase]) / 3
                gaps[mode] = reduced - shapes[32][mode][phase]
            comparison[phase] = dict(
                gaps,
                headroom_sensitive_ms=gaps['normal'] - gaps['prequeued'],
                removed_fraction=1 - gaps['prequeued'] / gaps['normal'],
            )
        summary['comparisons'].append(comparison)
    (root / 'analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['comparisons'], indent=2))
    print('PASS: all six shapes, bitwise equality, host fully prequeued, identical kernels/grids')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
