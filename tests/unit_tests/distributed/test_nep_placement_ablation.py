"""Placement controls preserve model identity, local work and useful bytes."""

import pytest

from megatron.core.distributed.nonuniform_common import (
    compute_nonuniform_ep_expert_placement,
    compute_nonuniform_ep_owner_expert_slots,
)


@pytest.mark.parametrize(
    'experts,ep,owners',
    [(512, 16, 12), (512, 8, 6), (128, 8, 4), (17, 4, 2), (17, 4, 3), (128, 16, 14), (512, 12, 12)],
)
def test_spread_preserves_work_and_locality(monkeypatch, experts, ep, owners):
    monkeypatch.setenv('NEP_ABLATION_PLACEMENT', 'current')
    current = compute_nonuniform_ep_expert_placement(experts, ep, owners, 1)
    monkeypatch.setenv('NEP_ABLATION_PLACEMENT', 'spread')
    spread = compute_nonuniform_ep_expert_placement(experts, ep, owners, 1)
    assert sorted(e for row in spread for e in row) == list(range(experts))
    assert list(map(len, current)) == list(map(len, spread))
    assert current[:owners] == spread[:owners]
    assert sorted(e for row in current[owners:] for e in row) == sorted(
        e for row in spread[owners:] for e in row
    )
    if (experts, ep, owners) in [(512, 16, 12), (512, 8, 6), (17, 4, 2)]:
        slots = compute_nonuniform_ep_owner_expert_slots(experts, owners)
        degree = lambda placement: sum(
            bool(set(row).intersection(owner)) for row in placement[owners:] for owner in slots
        )
        assert degree(spread) > degree(current)


def test_unknown_placement_rejected(monkeypatch):
    monkeypatch.setenv('NEP_ABLATION_PLACEMENT', 'typo')
    with pytest.raises(ValueError, match='Unknown NEP ablation placement'):
        compute_nonuniform_ep_expert_placement(17, 4, 2)
