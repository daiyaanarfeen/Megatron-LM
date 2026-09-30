"""Fail closed on initialization, routing, and bucket-control mismatches."""

import json
import sys
from pathlib import Path


def validate(
    left: Path, right: Path, full_ep: int, reduced_ep: int, require_routing: bool = True
) -> dict:
    """Compare paired placement manifests before accepting timing results."""
    world = full_ep + reduced_ep
    manifests = []
    for run in (left, right):
        paths = sorted((run / 'controls').glob('initialization-*.json'))
        assert len(paths) == world, (run, 'missing initialization records', len(paths))
        ranks = {int(path.stem.rsplit('-', 1)[1]): json.loads(path.read_text()) for path in paths}
        replicas = []
        for members in (range(full_ep), range(full_ep, world)):
            experts = {}
            for rank in members:
                local = {
                    name: value
                    for name, value in ranks[rank]['parameters'].items()
                    if '.experts.' in name
                }
                assert not set(experts).intersection(local), (
                    run,
                    rank,
                    'duplicated logical expert',
                )
                experts.update(local)
            replicas.append(experts)
        assert replicas[0] and replicas[0] == replicas[1], (run, 'replica weights differ')
        manifests.append((ranks, replicas[0]))
    assert manifests[0][1] == manifests[1][1], 'Logical expert weights differ across placements'
    peer_counts = {}
    for rank in range(world):
        a, b = [manifest[0][rank] for manifest in manifests]
        for key in ('dense_buckets', 'expert_buckets'):
            assert a[key] == b[key], (rank, key)
        dense = [
            {name: value for name, value in m['parameters'].items() if '.experts.' not in name}
            for m in (a, b)
        ]
        assert dense[0] == dense[1], (rank, 'non-expert parameters differ')
        paths = [sorted((run / 'controls').glob(f'routing-{rank}-*.json')) for run in (left, right)]
        if not require_routing:
            assert not paths[0] and not paths[1], 'Token dispatch occurred in parameter-only replay'
            continue
        assert paths[0] and [p.name for p in paths[0]] == [p.name for p in paths[1]], (
            rank,
            'routing shapes',
        )
        for p, q in zip(*paths):
            x, y = json.loads(p.read_text()), json.loads(q.read_text())
            for key in (
                'route_sha256',
                'expert_counts',
                'peer_assignments',
                'tokens',
                'experts',
                'topk',
                'seed',
            ):
                assert x[key] == y[key], (rank, key)
            assert len(set(x['expert_counts'])) == 1
            peer_counts[p.name] = [x['peer_unique_tokens'], y['peer_unique_tokens']]
    result = dict(
        status='PASS',
        full_ep=full_ep,
        reduced_ep=reduced_ep,
        logical_expert_parameters=len(manifests[0][1]),
        token_peer_counts=peer_counts,
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    validate(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
