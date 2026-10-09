"""Compare matched profiled steps and preserve rank/phase data for ordering A/B."""

import argparse
import json
import re
import statistics
from pathlib import Path

from analyze_a8b_graph_attribution import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    args = parser.parse_args()
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline') / args.job
    result = {}
    for order in (1, 0):
        for case, size in (('healthy', 32), ('nep', 28)):
            key = f'launch_order{order}_{case}'
            run = root / key / 'nsys/ep16/a8b_120b_latentmoe_1t' / case / args.job
            if not (run / 'baseline_uniform.json').exists():
                continue
            ranks = {}
            for rank in range(size):
                path = run / 'cupti_profile' / f'rank-{rank}.json'
                data = json.loads(path.read_text())
                records = [e for e in data['traceEvents'] if e.get('args', {}).get('iteration') in (25, 26, 27)]
                gpu = [e for e in records if e['cat'] == 'gpu_operation_range']
                host = [e for e in records if e['cat'] == 'cpu_operation_range']
                assert len(gpu) == len(host), (key, rank)
                names = {e['name'] for e in gpu}
                assert {'collective.all_gather', 'collective.reduce_scatter'} <= names
                if case == 'nep':
                    assert 'nep.redistribute_params' in names
                for event in gpu:
                    if event['name'].startswith('collective.'):
                        meta = event['args']['collective']
                        assert meta['ranks'] and meta['operations']
                        assert 'completion_stream' in event['args']
                assert (run / 'nsys_profile' / f'rank-{rank}.sqlite').is_file()
                ranks[str(rank)] = analyze(path, range(25, 28))
            roles = {}
            for role, members in (('owners', range(12)), ('followers', range(12, 16)), ('second', range(16, size))):
                per_step = []
                for step in range(3):
                    rows = [ranks[str(r)][step] for r in members]
                    per_step.append({k: statistics.mean(row[k] for row in rows)
                                     for k, value in rows[0].items() if isinstance(value, (int, float))})
                roles[role] = dict(steps=per_step, mean={k: statistics.mean(row[k] for row in per_step)
                                                      for k in per_step[0]})
            log = (run / 'driver.log').read_text()
            numerics = re.findall(r'iteration\s+(\d+)/.*?lm loss: ([\d.E+-]+).*?grad norm: ([\d.E+-]+)', log)
            result[key] = dict(run=str(run), roles=roles, ranks=ranks, loss_and_grad_norm=numerics)
            print(key, json.dumps(roles['owners'], indent=2))
    (root / 'launch_order_comparison.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
