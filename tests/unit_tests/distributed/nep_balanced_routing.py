"""Startup-only synthetic routing controls for placement experiments."""

import hashlib
import random


def balanced_routes(tokens: int, experts: int, topk: int, seed: int) -> list[list[int]]:
    """Randomize a regular token/expert graph without changing either degree.

    Degree-preserving edge swaps remove consecutive-ID co-selection. This is
    not a claim of exact uniform sampling over all possible regular graphs.
    """
    if tokens < 1 or not 0 < topk <= experts or tokens * topk % experts:
        raise ValueError('Expected positive dimensions and divisible assignment count')
    rows = [[(token * topk + k) % experts for k in range(topk)] for token in range(tokens)]
    members = [set(row) for row in rows]
    rng = random.Random(seed)
    edges = tokens * topk
    for _ in range(20 * edges):
        a, i = divmod(rng.randrange(edges), topk)
        b, j = divmod(rng.randrange(edges), topk)
        x, y = rows[a][i], rows[b][j]
        if a == b or y in members[a] or x in members[b]:
            continue
        rows[a][i], rows[b][j] = y, x
        members[a].remove(x)
        members[a].add(y)
        members[b].remove(y)
        members[b].add(x)
    return rows


def identity_seed(identity: str, seed: int = 1234) -> int:
    """Give a logical tensor a stable seed, independent of its physical rank."""
    return int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:8], 'little') % (
        2**63
    )


def route_summary(rows: list[list[int]], placement: list[list[int]]) -> dict:
    """Separate expert assignments from distinct token/destination incidences."""
    mapping = {expert: rank for rank, ids in enumerate(placement) for expert in ids}
    expert_counts = [0] * len(mapping)
    peer_assignments = [0] * len(placement)
    peer_tokens = [0] * len(placement)
    adjacent_pairs = 0
    for row in rows:
        assert len(row) == len(set(row))
        for expert in row:
            expert_counts[expert] += 1
            peer_assignments[mapping[expert]] += 1
        for peer in {mapping[expert] for expert in row}:
            peer_tokens[peer] += 1
        adjacent_pairs += sum((expert + 1) % len(mapping) in row for expert in row)
    return dict(
        expert_counts=expert_counts,
        peer_assignments=peer_assignments,
        peer_unique_tokens=peer_tokens,
        total_token_peer_pairs=sum(peer_tokens),
        adjacent_expert_pairs=adjacent_pairs,
        route_sha256=hashlib.sha256(repr(rows).encode()).hexdigest(),
    )
