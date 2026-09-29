"""Placement ablation must preserve expert work and aggregate useful payload."""

import pytest

from megatron.core.distributed.nonuniform_common import (
    compute_nonuniform_ep_expert_placement,
    compute_nonuniform_ep_owner_expert_slots,
)


@pytest.mark.parametrize('experts,ep,owners', [(512, 16, 12), (512, 8, 6), (23, 4, 3), (13, 4, 2)])
def test_placement_payload_invariants(monkeypatch, experts, ep, owners):
    monkeypatch.setenv('NEP_ABLATION_PLACEMENT', 'compact')
    compact = compute_nonuniform_ep_expert_placement(experts, ep, owners, 1)
    monkeypatch.setenv('NEP_ABLATION_PLACEMENT', 'spread')
    spread = compute_nonuniform_ep_expert_placement(experts, ep, owners, 1)
    slots = compute_nonuniform_ep_owner_expert_slots(experts, owners)
    assert compact[:owners] == spread[:owners]
    assert list(map(len, compact)) == list(map(len, spread))
    assert sorted(e for ids in spread for e in ids) == list(range(experts))
    for owner in range(owners):
        remote = set(slots[owner]) - {None} - set(compact[owner])
        assert sum(len(remote.intersection(ids)) for ids in compact[owners:]) == len(remote)
        assert sum(len(remote.intersection(ids)) for ids in spread[owners:]) == len(remote)


def test_reject_unknown_placement(monkeypatch):
    monkeypatch.setenv('NEP_ABLATION_PLACEMENT', 'unknown')
    with pytest.raises(ValueError, match='Unknown NEP placement ablation'):
        compute_nonuniform_ep_expert_placement(512, 8, 6, 1)
