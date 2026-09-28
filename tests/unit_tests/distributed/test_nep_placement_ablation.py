"""Placement controls preserve model identity, local work and useful bytes."""

from types import SimpleNamespace

import pytest

from megatron.core.distributed.nonuniform_common import (
    compute_nonuniform_ep_expert_placement,
    compute_nonuniform_ep_owner_expert_slots,
)
from megatron.core.distributed.nonuniform_ep import NonuniformEPDistributedDataParallel


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


def test_deferred_gradient_callback_only_tracks_readiness():
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    ddp._nep_ablation_defer_grad = True
    # No collective or scheduler state access is allowed before final sync.
    ddp._launch_and_release_nep_two_level_gather((object(),), 'layer')


def test_deferred_gradient_drain_preserves_order(monkeypatch):
    ddp = NonuniformEPDistributedDataParallel.__new__(NonuniformEPDistributedDataParallel)
    ddp._nep_ablation_defer_grad = True
    state = {'task_next_index': 0, 'task_sequence': []}
    groups = [
        SimpleNamespace(
            is_first_batch=False,
            _nep_nccl_scheduler_state=state,
            _nep_dispatch_boundary_module_label=f'layer{i}',
        )
        for i in range(3)
    ]
    state['task_sequence'] = [{'group': group} for group in groups]
    ddp.expert_parallel_bucket_groups = groups
    calls = []

    def launch(batch, label):
        assert batch[0] is groups[state['task_next_index']]
        calls.append(label)
        state['task_next_index'] += 1
        return True

    ddp._launch_nep_dispatch_boundary_tasks = launch
    ddp._wait_for_nep_dispatch_launch = lambda final=False: calls.append(('drain', final))
    monkeypatch.setattr(
        NonuniformEPDistributedDataParallel.__mro__[1],
        'finish_grad_sync',
        lambda self, force_all_reduce=False: calls.append('native_finish'),
    )
    ddp.finish_grad_sync()
    assert calls == [
        'layer0',
        ('drain', False),
        'layer1',
        ('drain', False),
        'layer2',
        ('drain', False),
        ('drain', True),
        'native_finish',
    ]
