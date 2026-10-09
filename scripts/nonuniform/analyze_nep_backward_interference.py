"""Compare native graph kernels in final and penultimate backwards, offline."""

import argparse
import bisect
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def union_ms(intervals):
    end, total = float('-inf'), 0
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total / 1e6


def category(name):
    if 'nccl' in name.lower():
        return 'model_TP'
    if name in ('device_sync_kernel', 'ag_nvl_kernel', 'dispatch_kernel', 'combine_kernel'):
        return 'model_EP'
    if any(x in name.lower() for x in ('gemm', 'gemv', 'matmul', 'mma', 'nvjet')):
        return 'GEMM'
    return 'other_native_compute'


def analyze(run, rank):
    trace = json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())
    events = trace['traceEvents']
    cpu = [e for e in events if e.get('cat') == 'cpu_operation_range']
    gpu = [e for e in events if e.get('cat') == 'gpu_operation_range']
    assert [e['name'] for e in cpu] == [e['name'] for e in gpu]
    graph_pairs = [(c, g) for c, g in zip(cpu, gpu) if c['name'].startswith('graph.')]
    con = sqlite3.connect(f'file:{run}/nsys_profile/rank-{rank}.sqlite?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    strings = dict(con.execute('select id,value from StringIds'))
    nv, nv_forward = [], []
    for row in con.execute('select * from NVTX_EVENTS where end is not null order by start'):
        name = row['text'] or strings.get(row['textId'], '')
        if name.startswith('graph.'):
            nv.append(dict(start=row['start'], end=row['end'], name=name))
        elif name == 'microbatch.forward':
            nv_forward.append(dict(start=row['start'], end=row['end'], name=name))
    assert len(nv) == len(graph_pairs)
    cpu_forward = [c for c in cpu if c['name'] == 'microbatch.forward']
    assert len(nv_forward) == len(cpu_forward)
    starts = [n['start'] for n in nv]
    launches = {}
    for row in con.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME order by start'):
        if 'cudaGraphLaunch' not in strings[row['nameId']]:
            continue
        index = bisect.bisect_right(starts, row['start']) - 1
        assert index >= 0 and row['end'] <= nv[index]['end']
        launches[row['correlationId']] = (index, row['end'])
    native = defaultdict(list)
    streams = {}
    for row in con.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId != 0'):
        index, submitted = launches[row['correlationId']]
        native[index].append((row['start'], row['end'], strings[row['shortName']], submitted))
        streams[index, row['start']] = row['streamId']
    con.close()
    rows = []
    for step in range(25, 31):
        aligned = [
            (n, c) for n, c in zip(nv_forward, cpu_forward) if c['args']['iteration'] == step
        ]
        lower = max(n['start'] / 1e6 - c['ts'] / 1000 for n, c in aligned)
        upper = min(n['end'] / 1e6 - (c['ts'] + c['dur']) / 1000 for n, c in aligned)
        assert lower <= upper, (rank, step, lower, upper)
        offset = (lower + upper) / 2
        backs = [
            e for e in cpu if e['name'] == 'microbatch.backward' and e['args']['iteration'] == step
        ]
        assert len(backs) == 8
        for phase, back in [('penultimate', backs[-2]), ('final', backs[-1])]:
            selected = [
                i
                for i, (c, _) in enumerate(graph_pairs)
                if back['ts'] <= c['ts'] < back['ts'] + back['dur']
            ]
            assert selected
            for block, index in enumerate(selected):
                c, g = graph_pairs[index]
                ks = native[index]
                assert ks
                intervals, sums, counts = defaultdict(list), defaultdict(float), Counter()
                names = defaultdict(float)
                for a, b, name, _ in ks:
                    label = category(name)
                    intervals[label].append((a, b))
                    sums[label] += (b - a) / 1e6
                    counts[name] += 1
                    names[name] += (b - a) / 1e6
                norm_gaps = []
                for a, _, name, _ in ks:
                    if name != 'rmsnorm_bwd_general_kernel':
                        continue
                    previous = [(x, y) for x, y, _, _ in ks if x < a]
                    same_stream = [
                        (x, y) for x, y in previous if streams[index, x] == streams[index, a]
                    ]
                    if not same_stream:
                        continue
                    norm_gaps.append(
                        dict(
                            same_stream_ms=max(0, a - max(y for _, y in same_stream)) / 1e6,
                            no_native_ms=max(0, a - max(y for _, y in previous)) / 1e6,
                        )
                    )
                rows.append(
                    dict(
                        rank=rank,
                        iteration=step,
                        phase=phase,
                        block=block,
                        name=c['name'],
                        wrapper_ms=g['dur'] / 1000,
                        kernel_span_ms=(max(k[1] for k in ks) - min(k[0] for k in ks)) / 1e6,
                        native_union_ms=union_ms([(a, b) for a, b, _, _ in ks]),
                        host_lead_ms=(min(k[0] for k in ks) - max(k[3] for k in ks)) / 1e6,
                        category_union_ms={k: union_ms(v) for k, v in intervals.items()},
                        category_sum_ms=dict(sums),
                        kernel_counts=dict(counts),
                        kernel_sum_ms=dict(names),
                        device_sync=[
                            dict(
                                start_ms=a / 1e6 - offset + trace['origin_ns'] / 1e6,
                                end_ms=b / 1e6 - offset + trace['origin_ns'] / 1e6,
                                duration_ms=(b - a) / 1e6,
                            )
                            for a, b, name, _ in sorted(ks)
                            if name == 'device_sync_kernel'
                        ],
                        clock_error_ms=trace['clock_error_bound_ns'] / 1e6 + upper - lower,
                        norm_same_stream_gap_ms=sum(x['same_stream_ms'] for x in norm_gaps),
                        norm_no_native_gap_ms=sum(x['no_native_ms'] for x in norm_gaps),
                    )
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--healthy', default='H')
    parser.add_argument('--nep', default='A')
    parser.add_argument('--ranks', nargs='+', type=int, default=[0, 8])
    parser.add_argument('--repeats', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {}
    for rep in args.repeats:
        for topology, cell in [('healthy', args.healthy), ('nep', args.nep)]:
            all_rows = []
            run = args.root / f'controlled_actual_{cell}_r{rep}_{topology}_moe'
            for rank in args.ranks:
                cache = args.output / f'{topology}-r{rep}-rank{rank}.json'
                if not cache.exists():
                    cache.write_text(json.dumps(analyze(run, rank)) + '\n')
                all_rows.extend(json.loads(cache.read_text()))
                print('ANALYZED', topology, rep, rank, flush=True)
            summary = {}
            for phase in ('penultimate', 'final'):
                groups = defaultdict(list)
                for row in all_rows:
                    if row['phase'] == phase:
                        groups[row['rank'], row['iteration']].append(row)
                numeric = (
                    'wrapper_ms',
                    'kernel_span_ms',
                    'native_union_ms',
                    'norm_same_stream_gap_ms',
                    'norm_no_native_gap_ms',
                )
                phase_summary = {
                    key: statistics.mean(sum(r[key] for r in rows) for rows in groups.values())
                    for key in numeric
                }
                for field in ('category_sum_ms', 'category_union_ms', 'kernel_sum_ms'):
                    keys = set().union(*(row[field] for rows in groups.values() for row in rows))
                    phase_summary[field] = {
                        key: statistics.mean(
                            sum(r[field].get(key, 0) for r in rows) for rows in groups.values()
                        )
                        for key in sorted(keys)
                    }
                phase_summary['min_host_lead_ms'] = min(
                    r['host_lead_ms'] for rows in groups.values() for r in rows
                )
                summary[phase] = phase_summary
            result[f'{topology}_r{rep}'] = summary
            (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
            print(
                topology,
                rep,
                json.dumps(
                    {
                        k: {a: b for a, b in v.items() if a != 'kernel_sum_ms'}
                        for k, v in summary.items()
                    }
                ),
                flush=True,
            )


if __name__ == '__main__':
    main()
