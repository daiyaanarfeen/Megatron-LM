"""Compare all replica compute and disjoint NEP-scoped host-idle/staging time."""

import json
from statistics import mean

from analyze_nep_final_ga_sums import CASES, OUT, ROOT, scan


def main():
    output = {}
    for job, model, layers in CASES:
        output[job] = {}
        for variant, count in (('healthy', 16), ('nep', 14)):
            ranks = {}
            for rank in range(count):
                cache = OUT / f'compute_host_{job}_{variant}_{rank}.json'
                if cache.exists():
                    steps = json.loads(cache.read_text())
                else:
                    path = ROOT / f'slurm_runs/soft_clique_baseline/{job}/ep8/scaling_ladder__{model}/{variant}/{job}/torch_profile/rank-{rank}.json.gz'
                    steps = scan(path, layers, require_owner_comms=False, include_host_staging=True)
                    cache.write_text(json.dumps(steps, indent=2) + '\n')
                ranks[str(rank)] = steps
                print(job, variant, rank, 'validated', flush=True)
            roles = {'healthy': list(range(16))} if variant == 'healthy' else {
                'full': list(range(8)), 'reduced': list(range(8, 14)), 'owners': list(range(6))}
            summary = {}
            for role, ids in roles.items():
                steps = [s for rank in ids for s in ranks[str(rank)]]
                summary[role] = {'final_ga_compute_ms': mean(s['sum_ms']['backward_compute'] for s in steps),
                                 'host_staging_ms': {k: mean(s['host_staging_ms'][k] for s in steps)
                                                     for k in steps[0]['host_staging_ms']}}
            output[job][variant] = {'summary': summary, 'ranks': ranks}
            print(job, variant, summary, flush=True)
            (OUT / 'compute_replica_host_staging.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
