"""Exercise actual transfer methods without initializing Megatron's global PGs."""

import ast
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import torch


def load_methods(root):
    namespace = dict(torch=torch, math=math, List=List, Optional=Optional, Sequence=Sequence)
    common = ast.parse((root / 'megatron/core/distributed/nonuniform_common.py').read_text())
    names = {
        'compute_nonuniform_ep_owner_expert_slots',
        'compute_nonuniform_ep_expert_placement',
        'build_expert_to_ep_rank_map',
    }
    nodes = [n for n in common.body if isinstance(n, ast.FunctionDef) and n.name in names]
    source = ast.parse((root / 'megatron/core/distributed/nonuniform_ep.py').read_text())
    cls = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == 'NonuniformEPNCCLParamAndGradBucketGroup'
    )
    names = {
        '_get_nep_distopt_param_transfer_plan',
        '_scatter_nep_distopt_owner_params_to_physical_holders',
    }
    nodes += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root), 'exec'), namespace)
    return namespace


def run(root, device):
    ns = load_methods(root)
    scatter = ns['_scatter_nep_distopt_owner_params_to_physical_holders']
    total_copies = 0
    singleton_copies = 0
    broadcasts = []
    for full, reduced in ((8, 4), (8, 6), (16, 12), (32, 28)):
        slots = ns['compute_nonuniform_ep_owner_expert_slots'](128, reduced)
        owner_of = {
            e: (owner, slot)
            for owner, row in enumerate(slots)
            for slot, e in enumerate(row)
            if e is not None
        }
        for degree in (full, reduced):
            placement = ns['compute_nonuniform_ep_expert_placement'](128, degree, reduced)
            transfers = {
                owner: tuple(
                    sorted(
                        {owner}
                        | {
                            rank
                            for rank, row in enumerate(placement)
                            if set(row).intersection(e for e in slots[owner] if e is not None)
                        }
                    )
                )
                for owner in range(reduced)
            }
            wire = {}
            current_rank = [0]

            def broadcast(tensor, src, group):
                broadcasts.append((full, reduced, degree, current_rank[0], src))
                if current_rank[0] == src:
                    wire[src] = tensor.clone()
                else:
                    tensor.copy_(wire[src])

            ns['dist'] = SimpleNamespace(get_global_rank=lambda pg, rank: rank, broadcast=broadcast)
            groups = []
            for rank, experts in enumerate(placement):
                entries = []
                for expert in experts:
                    for slot, shape in enumerate(((2, 3), (1, 5))):
                        param = torch.nn.Parameter(torch.full(shape, -1.0, device=device))
                        param.grad = torch.ones_like(param)
                        entries.append(
                            dict(
                                bucket=SimpleNamespace(params_list=[param]),
                                expert_id=expert,
                                slot_index=slot,
                                slot_offset=(0, 6)[slot],
                                numel=param.numel(),
                            )
                        )
                proxies = (
                    {
                        (0, e, slot): torch.nn.Parameter(
                            torch.full(shape, float(e * 2 + slot), device=device)
                        )
                        for e in slots[rank]
                        if e is not None
                        for slot, shape in enumerate(((2, 3), (1, 5)))
                    }
                    if rank < reduced
                    else {}
                )
                group = SimpleNamespace(
                    _nep_nccl_group_index=0,
                    _nep_runtime_config=dict(ep_rank=rank, ep_group=None),
                    _nep_nccl_slot_offsets=(0, 6),
                    _nep_nccl_slot_numels=(6, 5),
                )
                group._get_nep_nccl_owner_layout = lambda: dict(
                    min_ep_size=reduced, owner_numel=len(slots[0]) * 11, expert_stride=11
                )
                group._get_nep_nccl_transfer_group_info = lambda owner: (
                    owner if len(transfers[owner]) > 1 else None,
                    None,
                    len(transfers[owner]),
                    transfers[owner],
                )
                group._nep_nccl_owner_entries = lambda owner, entries=entries: [
                    e for e in entries if owner_of[e['expert_id']][0] == owner
                ]
                group._nep_nccl_owner_expert_ids = lambda owner: [
                    e for e in slots[owner] if e is not None
                ]
                group._nep_nccl_owner_slot_for_expert = lambda expert: owner_of[expert]
                group._nep_nccl_entry_owner_start = (
                    lambda entry, owner: owner_of[entry['expert_id']][1] * 11 + entry['slot_offset']
                )
                group._nep_distopt_owner_bundle = dict(
                    groups=(group,),
                    proxy_by_key=proxies,
                    param_transfer_slot=0,
                    param_transfer_numel_by_slot={0: len(slots[0]) * 11},
                    param_transfer_buffers={},
                )
                if '_get_nep_distopt_param_transfer_plan' in ns:
                    group._get_nep_distopt_param_transfer_plan = lambda group=group: ns[
                        '_get_nep_distopt_param_transfer_plan'
                    ](group)
                groups.append((group, entries))
            for iteration in range(2):
                wire.clear()
                for rank, (group, entries) in enumerate(groups):
                    current_rank[0] = rank
                    if iteration:
                        group._nep_distopt_owner_bundle['param_transfer_buffers'].clear()
                        for key, proxy in tuple(
                            group._nep_distopt_owner_bundle['proxy_by_key'].items()
                        ):
                            group._nep_distopt_owner_bundle['proxy_by_key'][key] = (
                                torch.nn.Parameter(proxy.detach().clone() + 1000)
                            )
                        for entry in entries:
                            old = entry['bucket'].params_list[0]
                            new = torch.nn.Parameter(torch.empty_like(old))
                            new.grad = torch.ones_like(new)
                            entry['bucket'].params_list[0] = new
                    identities = [
                        (
                            e['bucket'].params_list[0],
                            e['bucket'].params_list[0].data_ptr(),
                            e['bucket'].params_list[0].grad.data_ptr(),
                        )
                        for e in entries
                    ]
                    copies = []
                    original = torch.Tensor.copy_

                    def copy(dest, source, *args, **kwargs):
                        copies.append(dest.numel())
                        return original(dest, source, *args, **kwargs)

                    torch.Tensor.copy_ = copy
                    try:
                        scatter(group)
                    finally:
                        torch.Tensor.copy_ = original
                    total_copies += len(copies)
                    if rank < reduced and len(transfers[rank]) == 1:
                        singleton_copies += len(copies)
                        expected = len(entries) * (
                            1 if '_get_nep_distopt_param_transfer_plan' in ns else 2
                        )
                        assert len(copies) == expected, (rank, len(copies), expected)
                        if '_get_nep_distopt_param_transfer_plan' in ns:
                            assert not group._nep_distopt_owner_bundle['param_transfer_buffers']
                    for entry, (param, ptr, grad_ptr) in zip(entries, identities):
                        assert entry['bucket'].params_list[0] is param
                        assert param.data_ptr() == ptr and param.grad.data_ptr() == grad_ptr
                        assert torch.equal(
                            param,
                            torch.full_like(
                                param,
                                entry['expert_id'] * 2 + entry['slot_index'] + iteration * 1000,
                            ),
                        )
                        assert torch.equal(param.grad, torch.ones_like(param.grad))
    if device == 'cuda':
        torch.cuda.synchronize()
    print(
        f'PASS root={root} device={device} copies={total_copies} singleton_copies={singleton_copies} broadcasts={len(broadcasts)}'
    )
    return broadcasts


if __name__ == '__main__':
    baseline, candidate = map(Path, sys.argv[1:3])
    for device in ('cpu', 'cuda'):
        assert run(baseline, device) == run(candidate, device)
