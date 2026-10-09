"""Compare matched recording windows and validate device-only trace contents."""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from analyze_workload_parity_pair import iter_trace_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    results = {}
    for mode in ('cpu_cuda', 'cuda'):
        cases = results[mode] = {}
        for case, ranks in (('healthy', 16), ('nep', 14)):
            logs = list((args.root / mode).glob(f'ep8/*/{case}/*/driver.log'))
            assert len(logs) == 1, logs
            log = logs[0]
            elapsed = {
                int(i): float(t)
                for i, t in re.findall(
                    r'iteration\s+(\d+)/\s*24.*?elapsed time per iteration \(ms\):\s*([\d.]+)',
                    log.read_text(),
                )
            }
            assert len(elapsed) == 24, (case, len(elapsed))
            row = cases[case] = {}
            for window, indices in (('recorded', range(10, 14)), ('post', range(15, 25))):
                values = [elapsed[i] for i in indices]
                row[window] = {
                    'iterations': list(indices),
                    'ms': values,
                    'mean_ms': statistics.mean(values),
                    'stdev_ms': statistics.stdev(values),
                }
            row['recorded_minus_post_ms'] = row['recorded']['mean_ms'] - row['post']['mean_ms']
            traces = list(log.parent.glob('torch_profile/rank-*.json.gz'))
            assert len(traces) == ranks, (case, len(traces))
            row['traces'] = {}
            for path in sorted(traces):
                cats = Counter()
                nccl = metadata = 0
                for event in iter_trace_events(path):
                    cats[event.get('cat', '')] += 1
                    if event.get('cat') == 'kernel' and 'nccl' in event.get('name', '').lower():
                        nccl += 1
                        metadata += 'Process Group Description' in event.get('args', {})
                assert cats['kernel'] > 0, path
                assert (cats['cpu_op'] > 0) == (mode == 'cpu_cuda'), (path, cats)
                rank = path.name.split('.')[0]
                row['traces'][rank] = {
                    'counts': dict(cats), 'nccl_kernels': nccl, 'nccl_with_pg_metadata': metadata
                }
            print(mode, case, row['recorded'], row['post'], flush=True)
        cases['parity_percent'] = {
            window: 100 * cases['healthy'][window]['mean_ms'] / cases['nep'][window]['mean_ms']
            for window in ('recorded', 'post')
        }
    output = args.root / 'profile_activity_analysis.json'
    output.write_text(json.dumps(results, indent=2) + '\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
