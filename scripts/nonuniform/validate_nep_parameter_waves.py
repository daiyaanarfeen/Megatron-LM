"""Validate measured collective payloads and cached parameter-wave order across ranks."""

import argparse
import json
from collections import Counter
from pathlib import Path


def inspect(run, steps, wave_order):
    result, memberships, rank_events = {}, {}, {}
    paths = sorted((run / 'cupti_profile').glob('rank-*.json'))
    for path in paths:
        rank = int(path.stem.split('-')[1])
        events = json.loads(path.read_text())['traceEvents']
        collective_counts = Counter()
        selected = []
        for event in events:
            args = event.get('args', {})
            if event.get('cat') != 'gpu_operation_range' or args.get('iteration') not in steps:
                continue
            metadata = args.get('collective')
            if metadata is None:
                continue
            key = (args['iteration'], event['name'], metadata['group_desc'],
                   tuple(metadata['ranks']), json.dumps(metadata['operations'], sort_keys=True))
            collective_counts[key] += 1
            if event['name'] == 'collective.broadcast' and metadata['group_desc'] == 'nep_owner_transfer':
                source = metadata['operations'][0]['source_rank']
                members = frozenset(metadata['ranks'])
                assert source not in memberships or memberships[source] == members
                memberships[source] = members
                selected.append((args['iteration'], source))
        result[rank] = collective_counts
        rank_events[rank] = selected
    waves = []
    for owner, members in sorted(memberships.items()):
        for owners, participants in waves:
            if participants.isdisjoint(members):
                owners.append(owner)
                participants.update(members)
                break
        else:
            waves.append(([owner], set(members)))
    order = [owner for owners, _ in waves for owner in owners] if wave_order else sorted(memberships)
    for rank, events in rank_events.items():
        pattern = [owner for owner in order if rank in memberships[owner]]
        for step in steps:
            sources = [owner for iteration, owner in events if iteration == step]
            if not pattern:
                assert not sources
                continue
            assert sources and len(sources) % len(pattern) == 0
            assert sources == pattern * (len(sources) // len(pattern)), (rank, step, sources, pattern)
    return result, [owners for owners, _ in waves]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--steps', nargs=2, type=int, default=[20, 27])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    steps = range(args.steps[0], args.steps[1] + 1)
    before, waves = inspect(args.reference, steps, False)
    after, candidate_waves = inspect(args.candidate, steps, True)
    assert before.keys() == after.keys()
    assert waves == candidate_waves
    for rank in before:
        assert before[rank] == after[rank], (rank, before[rank] - after[rank], after[rank] - before[rank])
    # Optional hard-run Nsight caches contain kernels associated with the exact
    # first-forward CPU/graph launches rather than a broad GPU time window.
    kernel_samples = 0
    cache = args.output.parent / 'ag_redistribution_analysis'
    for path in cache.glob('delivery-param_waves_margin-rank-*.json'):
        trace = json.loads(path.read_text())
        assert trace['schema_version'] == 4
        for row in trace['rows']:
            staging = [k for k in row['kernels'] if k['label'] == 'NEP_staging']
            assert len({k['stream'] for k in staging}) == 1, path
            # Nsight cuda-sw ranges for consecutive copy kernels on the SAME
            # stream overlap by 1-3us in both versions. Check stream ordering
            # for staging and actual non-concurrency of the NCCL broadcasts.
            kernels = sorted((k for k in row['kernels'] if k['label'] == 'NEP_broadcast'),
                             key=lambda k: k['start'])
            assert all(a['end'] <= b['start'] + .001 for a, b in zip(kernels, kernels[1:])), path
            kernel_samples += 1
    result = dict(ranks=len(before), steps=list(steps), waves=waves,
                  collective_calls=sum(sum(counts.values()) for counts in before.values()),
                  payloads_and_counts_unchanged=True, broadcast_order_valid=True,
                  serial_redistribution_kernel_samples=kernel_samples)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
