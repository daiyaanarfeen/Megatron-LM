"""Check actual AG/redistribution overlap for the isolated launch-order A/B."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_nep_first_forward_delivery import read_rank
from summarize_nep_first_forward_delivery import expert_ag, phase_summary, redistribution, shifts


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def duration(intervals):
    return sum(b - a for a, b in union(intervals))


def broadcast_peers(data, run):
    shift = shifts(data)
    matched = defaultdict(dict)
    for rank, trace in data.items():
        raw = json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())
        for row in trace['rows']:
            step = row['iteration']
            events = [e for e in raw['traceEvents'] if e['cat'] == 'gpu_operation_range'
                      and e['args']['iteration'] == step and e['name'] == 'collective.broadcast'
                      and e['args']['collective']['group_desc'] == 'nep_owner_transfer']
            kernels = sorted([k for k in row['kernels'] if k['label'] == 'NEP_broadcast'],
                             key=lambda k: k['launch'])
            assert len(events) == len(kernels)
            counts = Counter()
            for event, kernel in zip(events, kernels):
                meta = event['args']['collective']
                gid = meta['group_id']
                matched[step, gid, counts[gid]][rank] = dict(kernel, members=meta['ranks'])
                counts[gid] += 1
    totals = defaultdict(float)
    for group in matched.values():
        assert set(group) == set(next(iter(group.values()))['members'])
        latest = max(k['start'] + shift[r] for r, k in group.items())
        for rank, k in group.items():
            if rank >= 12:
                continue
            start, end = k['start'] + shift[rank], k['end'] + shift[rank]
            cut = min(end, max(start, latest))
            totals['residency_ms'] += (end - start) / 96
            totals['before_last_peer_kernel_start_ms'] += (cut - start) / 96
            totals['after_last_peer_kernel_start_ms'] += (end - cut) / 96
    return dict(totals)


def analyze(data):
    result = phase_summary(data)
    samples = []
    for rank in range(12):
        for row in data[rank]['rows']:
            ag, red = expert_ag(row), redistribution(row)
            assert len(ag) == len(red) == 8
            kernels = row['kernels']
            ak = sorted([k for k in kernels if k['label'] == 'expert_AG'], key=lambda k: k['start'])
            assert len(ak) == 8
            assert all(a['end'] <= b['start'] + 0.001 for a, b in zip(ak, ak[1:]))
            assert len({k['stream'] for k in ak}) == 1
            delivery_kernels = [
                [k for k in kernels if k['label'] in ('NEP_broadcast', 'NEP_staging')
                 and r['cpu_start'] <= k['launch'] <= r['cpu_end']]
                for r in red
            ]
            assert all(delivery_kernels)
            actual_delivery = [(min(k['start'] for k in ks), max(k['end'] for k in ks))
                               for ks in delivery_kernels]
            assert all(a[1] <= b[0] + 0.001 for a, b in zip(actual_delivery, actual_delivery[1:]))
            assert all(r[0] >= k['end'] - 0.001 for r, k in zip(actual_delivery, ak))
            transfers = union((k['start'], k['end']) for k in kernels
                              if k['label'] in ('NEP_broadcast', 'NEP_staging'))
            overlap = [(max(k['start'], a), min(k['end'], b))
                       for k in ak for a, b in transfers]
            samples.append(dict(
                rank=rank, iteration=row['iteration'],
                first_forward_ms=row['end'] - row['start'],
                AG_kernel_union_ms=duration((k['start'], k['end']) for k in ak),
                redistribution_kernel_union_ms=duration(transfers),
                AG_redistribution_kernel_overlap_ms=duration(overlap),
                next_AG_kernel_before_prior_delivery=sum(
                    k['start'] < r['end'] - 0.03 for k, r in zip(ak[1:], red[:-1])),
                next_AG_ready_before_prior_delivery=sum(
                    a['start'] < r['end'] - 0.03 for a, r in zip(ag[1:], red[:-1])),
            ))
    result['owner_means'] = {
        key: statistics.mean(s[key] for s in samples)
        for key in samples[0] if key not in ('rank', 'iteration')
    }
    result['samples'] = samples
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('--labels', nargs='+', default=['healthy_margin', 'cleanup_margin', 'ag_overlap_margin'])
    args = parser.parse_args()
    root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline') / args.job
    cases = json.loads((root / 'cleanup_phase_analysis.json').read_text())
    out = root / 'ag_redistribution_analysis'
    out.mkdir(exist_ok=True)
    results = {}
    for label in args.labels:
        run = Path(cases['cleanup_ab_' + label]['run'])
        data = {}
        for rank in range(16):
            path = out / f'delivery-{label}-rank-{rank}.json'
            if not path.exists() or json.loads(path.read_text()).get('schema_version') != 4:
                path.write_text(json.dumps(read_rank(run, rank, per_iteration_alignment=True)) + '\n')
            data[rank] = json.loads(path.read_text())
            print('read', label, rank, flush=True)
        results[label] = phase_summary(data) if label == 'healthy_margin' else analyze(data)
        if label != 'healthy_margin':
            results[label]['broadcast_peer_kernel_start'] = broadcast_peers(data, run)
        print(label, {k: v for k, v in results[label].items() if k != 'samples'}, flush=True)
        (out / 'overlap_results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
