"""Audit expert gradient payload and peer kernel residency in existing profiles."""

import gzip
import json
from statistics import mean

from analyze_nep_final_ga_sums import CASES, OUT, ROOT


def main():
    result = {}
    for job, model, _ in CASES:
        result[job] = {}
        for rank in (0, 8):
            path = ROOT / f'slurm_runs/soft_clique_baseline/{job}/ep8/scaling_ladder__{model}/nep/{job}/torch_profile/rank-{rank}.json.gz'
            with gzip.open(path, 'rt') as handle:
                events = json.load(handle)['traceEvents']
            meta = {e.get('args', {}).get('External id'): e['args'] for e in events
                    if e.get('name') == 'record_param_comms' and e.get('args', {}).get('Collective name') != 'wait'}
            steps = sorted([e for e in events if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('ProfilerStep#')], key=lambda e: e['ts'])
            kernels = []
            for e in events:
                if e.get('cat') != 'kernel' or 'nccl' not in e['name'].lower():
                    continue
                a = dict(e.get('args', {}))
                a.update(meta.get(a.get('External id'), {}))
                if a.get('Process Group Description') != 'ep_dp' or 'reduce_scatter' not in str(a.get('Collective name')):
                    continue
                kernels.append({'start_us': e['ts'], 'end_us': e['ts'] + e['dur'],
                                'duration_ms': e['dur'] / 1000, 'input_elements': a.get('In msg nelems'),
                                'dtype': a.get('dtype'), 'ranks': a.get('Process Group Ranks')})
            by_step = [[k for k in kernels if s['ts'] <= k['start_us'] < s['ts'] + s['dur']] for s in steps]
            assert [len(v) for v in by_step] == [8] * 4, (job, rank, [len(v) for v in by_step])
            result[job][str(rank)] = by_step
            print(job, rank, 'sum ms', mean(sum(k['duration_ms'] for k in v) for v in by_step),
                  'payloads', [k['input_elements'] for k in by_step[0]], 'dtype', by_step[0][0]['dtype'], flush=True)
    (OUT / 'edp_peer_audit.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
