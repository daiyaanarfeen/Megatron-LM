"""Compare actual owner-grad staging methods, including rebinding and padding."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from nep_param_transfer_standalone import load_methods


def load_stage(root):
    namespace = load_methods(root)
    source = ast.parse((root / 'megatron/core/distributed/nonuniform_ep.py').read_text())
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == 'NonuniformEPNCCLParamAndGradBucketGroup'
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == '_copy_nep_nccl_contexts_to_distopt_grads'
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(root), 'exec'), namespace)
    return namespace[method.name], namespace['compute_nonuniform_ep_owner_expert_slots']


@torch.no_grad()
def run(reference, candidate, device, dtype):
    old, placement = load_stage(reference)
    new, _ = load_stage(candidate)
    checks = 0
    for full, reduced in ((8, 4), (8, 6), (16, 12), (32, 28)):
        slots = placement(128, reduced)
        for owner in range(reduced):
            real = [expert for expert in slots[owner] if expert is not None]
            offsets = {expert: i for i, expert in enumerate(slots[owner]) if expert is not None}
            bundle = {'proxy_by_key': {}}
            contexts = []
            for index, widths in enumerate(((6, 10), (12, 4))):
                group = SimpleNamespace(
                    _nep_distopt_owner_bundle=bundle,
                    _nep_runtime_config={'ep_rank': owner},
                    ddp_config=SimpleNamespace(use_distributed_optimizer=True),
                    _nep_nccl_group_index=index,
                    _nep_nccl_expert_stride=sum(widths),
                    _nep_nccl_slot_offsets=(0, widths[0]),
                    _nep_nccl_slot_numels=widths,
                    _nep_nccl_owner_expert_ids=lambda rank: real,
                    _nep_nccl_owner_slot_for_expert=lambda expert: (owner, offsets[expert]),
                    _nep_nccl_scheduler_state={'buffer_slot_events': {}},
                )
                for expert in real:
                    for slot, width in enumerate(widths):
                        bundle['proxy_by_key'][index, expert, slot] = SimpleNamespace(
                            main_grad=torch.empty((2, width // 2), device=device, dtype=dtype)
                        )
                contexts.append(
                    dict(group=group, owner_ep_rank=owner, chunk=None, buffer_slot_key=index)
                )
            for iteration in range(2):
                # Rebind gradient views AND source storage; cached tensor views
                # would fail the second iteration. Padding has no destination.
                guards = []
                for proxy in bundle['proxy_by_key'].values():
                    shape = proxy.main_grad.shape
                    storage = torch.full(
                        (proxy.main_grad.numel() + 2,), -17, device=device, dtype=dtype
                    )
                    proxy.main_grad = storage[1:-1].view(shape)
                    guards.append(storage)
                for context in contexts:
                    stride = context['group']._nep_nccl_expert_stride
                    chunk = (
                        torch.arange(len(slots[owner]) * stride, device=device, dtype=dtype)
                        + iteration * 100
                    )
                    for i, expert in enumerate(slots[owner]):
                        if expert is None:
                            chunk[i * stride : (i + 1) * stride].fill_(float('nan'))
                    context['chunk'] = chunk
                snapshots = [context['chunk'].clone() for context in contexts]
                new(contexts[0]['group'], contexts)
                expected = [proxy.main_grad.clone() for proxy in bundle['proxy_by_key'].values()]
                for proxy in bundle['proxy_by_key'].values():
                    proxy.main_grad.fill_(-17)
                old(contexts[0]['group'], contexts)
                for proxy, value in zip(bundle['proxy_by_key'].values(), expected):
                    assert torch.equal(proxy.main_grad, value), (full, reduced, owner, iteration)
                for storage in guards:
                    assert storage[0].item() == storage[-1].item() == -17
                for context, snapshot in zip(contexts, snapshots):
                    torch.testing.assert_close(
                        context['chunk'], snapshot, rtol=0, atol=0, equal_nan=True
                    )
                    events = context['group']._nep_nccl_scheduler_state['buffer_slot_events'][
                        context['buffer_slot_key']
                    ]
                    assert len(events) == 2 * (iteration + 1)
                checks += 1
    print('NEP_GRAD_STAGE_PASS', device, dtype, 'checks', checks, flush=True)


if __name__ == '__main__':
    reference, candidate = map(Path, sys.argv[1:3])
    for dtype in (torch.float32, torch.bfloat16):
        with (
            patch.object(torch.cuda, 'Event', lambda: SimpleNamespace(record=lambda stream: None)),
            patch.object(torch.cuda, 'current_stream', lambda: None),
        ):
            run(reference, candidate, 'cpu', dtype)
        run(reference, candidate, 'cuda', dtype)
