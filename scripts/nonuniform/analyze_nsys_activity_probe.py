"""Validate all-rank Nsight capture and keep recorded/post timings separate."""

import argparse
import json
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_exact_uniform_nep_slowdown import intersection_duration, union_duration


def analyze_trace(path):
    c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    diagnostics = [x[0] for x in c.execute('SELECT text FROM DIAGNOSTIC_EVENT')]
    assert any('Hardware tracing used' in d for d in diagnostics), path
    names = dict(c.execute('SELECT id,value FROM StringIds'))
    ranges = c.execute('SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL').fetchall()
    ranges = [(a, b, text or names.get(tid, '')) for a, b, text, tid in ranges]
    ranges = [(a, b, text) for a, b, text in ranges if text.startswith('iteration_')]
    ranges.sort()
    ranges = [(a,b,text) for a,b,text in ranges if text != 'iteration_9']
    assert [text for _, _, text in ranges] == [f'iteration_{i}' for i in range(10, 14)], (path, ranges)
    kernels = c.execute('SELECT start,end,streamId,demangledName,graphId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start').fetchall()
    streams = defaultdict(list)
    for a, b, stream, name, graph in kernels:
        streams[stream].append((a, b, names[name], graph))
    expert_streams = {s for s, events in streams.items() if all('nvjet' in name for _, _, name, _ in events)}
    assert len(expert_streams) == 4, (path, expert_streams)
    rows = []
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    memory = []
    for table in ['CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET']:
        if table in tables:
            memory.extend(c.execute(f'SELECT start,end FROM {table}').fetchall())
    runtime = c.execute('SELECT start,end,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME').fetchall()
    for start, end, label in ranges:
        spans = defaultdict(list)
        counts = Counter()
        expert = []
        hist = Counter()
        for a, b, stream, nameid, graph in kernels:
            if b <= start or a >= end:
                continue
            name = names[nameid]
            category = 'nccl' if 'nccl' in name.lower() else 'hybrid_ep' if 'hybrid_ep::' in name else 'other_kernels'
            interval = (max(a, start) / 1e6, min(b, end) / 1e6)
            spans[category].append(interval)
            counts[category] += 1
            hist[name] += 1
            if stream in expert_streams:
                expert.append(interval)
        spans['memory'] = [(max(a,start)/1e6,min(b,end)/1e6) for a,b in memory if b>start and a<end]
        all_spans = [v for values in spans.values() for v in values]
        categories = {}
        for category, values in spans.items():
            others = [v for k, vv in spans.items() if k != category for v in vv]
            total = union_duration(values)
            categories[category] = {'union_ms': total, 'exposed_ms': total - intersection_duration(values, others)}
        sync = defaultdict(float)
        for a, b, nameid in runtime:
            if b>start and a<end and 'Synchronize' in names[nameid]:
                sync[names[nameid]] += (min(b,end)-max(a,start))/1e6
        rows.append({'iteration':int(label.split('_')[1]), 'nvtx_span_ms':(end-start)/1e6,
                     'gpu_busy_ms':union_duration(all_spans), 'expert_gemm_ms':union_duration(expert),
                     'expert_kernel_count':len(expert), 'kernel_counts':dict(counts),
                     'categories':categories, 'runtime_sync_sum_ms':dict(sync), 'kernel_histogram':dict(hist)})
    c.close()
    return {'path':str(path), 'diagnostics':diagnostics, 'total_kernels':len(kernels),
            'expert_streams':sorted(expert_streams), 'iterations':rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('job')
    args = p.parse_args()
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline') / args.job / 'nsys'
    work = root / 'ep8/scaling_ladder__a770m_4b_moe_215b'
    result = {'job':args.job, 'timing':{}, 'ranks':{}}
    for case, size in [('healthy',16),('nep',14)]:
        run = work / case / args.job
        log = (run / 'driver.log').read_text()
        timings = {int(i):float(t) for i,t in re.findall(r'iteration\s+(\d+)/\s*24.*?elapsed time per iteration \(ms\):\s*([\d.]+)',log)}
        result['timing'][case] = {}
        for label, indices in [('pre',range(6,9)),('recorded',range(10,14)),('post',range(15,25))]:
            values = [timings[i] for i in indices]
            result['timing'][case][label] = {'ms':values,'mean_ms':statistics.mean(values),'std_ms':statistics.stdev(values)}
        paths = list((run / 'nsys_profile').glob('rank-*.sqlite'))
        assert len(paths) == size, (case, len(paths))
        for rank in range(size):
            path = run / 'nsys_profile' / f'rank-{rank}.sqlite'
            row = analyze_trace(path)
            experts = 16 if case=='healthy' or rank<8 else 22 if rank<10 else 21
            assert all(i['expert_kernel_count']==15*3*experts*6 for i in row['iterations']), (path, [i['expert_kernel_count'] for i in row['iterations']])
            result['ranks'][f'{case}_{rank}'] = row
            print(case,rank,row['total_kernels'],flush=True)
    result['parity'] = {window:result['timing']['healthy'][window]['mean_ms']/result['timing']['nep'][window]['mean_ms'] for window in ['pre','recorded','post']}
    (root / 'nsys_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'timing':result['timing'],'parity':result['parity']},indent=2))


if __name__ == '__main__':
    main()
