"""First-forward kernel residency comparison; sums are not additive wall time."""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


def main():
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline') / sys.argv[1]
    result = {}
    for tag in ('cleanup', 'parallel'):
        run = root / f'cleanup_ab_{tag}/nsys/ep16/a8b_120b_latentmoe_1t/nep' / sys.argv[1]
        result[tag] = {}
        for rank in (0, 11, 15, 16, 27):
            data = json.loads((run / 'cupti_profile' / f'rank-{rank}.json').read_text())
            es = data['traceEvents']
            con = sqlite3.connect(f'file:{run}/nsys_profile/rank-{rank}.sqlite?mode=ro', uri=True)
            strings = dict(con.execute('select id,value from StringIds'))
            cpu = [e for e in es if e['cat'] == 'cpu_operation_range' and e['name'] == 'microbatch.forward']
            nv = [(a, b) for a, b, name, ident in con.execute('select start,end,text,textId from NVTX_EVENTS where end is not null order by start') if (name or strings.get(ident)) == 'microbatch.forward']
            assert len(cpu) == len(nv)
            lo = max(a/1e6-e['ts']/1000 for (a, b), e in zip(nv, cpu))
            hi = min(b/1e6-(e['ts']+e['dur'])/1000 for (a, b), e in zip(nv, cpu))
            assert hi >= lo
            offset = (lo+hi)/2
            kernels = list(con.execute('select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL'))
            totals = defaultdict(float)
            for step in range(20, 28):
                f = min((e for e in es if e['cat'] == 'gpu_operation_range' and e['name'] == 'microbatch.forward' and e['args']['iteration'] == step), key=lambda e: e['ts'])
                begin, end = f['ts']/1000+offset, (f['ts']+f['dur'])/1000+offset
                for a, b, name in kernels:
                    duration = min(end, b/1e6)-max(begin, a/1e6)
                    if duration > 0:
                        totals[strings[name]] += duration/8
            result[tag][rank] = dict(totals)
            con.close()
    for rank in (0, 11, 15, 16, 27):
        a, b = result['cleanup'][rank], result['parallel'][rank]
        names = sorted(a.keys() | b.keys(), key=lambda name: b.get(name, 0)-a.get(name, 0), reverse=True)
        print('rank', rank, [(name, round(a.get(name, 0), 3), round(b.get(name, 0), 3)) for name in names[:8]])
    (root / 'forward_kernel_residencies.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
