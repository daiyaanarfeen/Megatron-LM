"""Check NEP sync results against trusted methods on identical live inputs.

Shadow execution is correctness-only, never used for performance measurements.
The existing low-overhead profiler remains enabled independently.
"""

import ast
import atexit
import functools
import json
import os
import sys
from pathlib import Path

import torch


def install_shadow() -> None:
    from megatron.core.distributed import nonuniform_ep as nep

    cls = nep.NonuniformEPNCCLParamAndGradBucketGroup
    reference = Path(os.environ['NEP_SYNC_REFERENCE'])
    source = ast.parse((reference / 'megatron/core/distributed/nonuniform_ep.py').read_text())
    old_class = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == cls.__name__
    )
    names = (
        '_scatter_nep_distopt_owner_params_to_physical_holders',
        '_copy_nep_nccl_contexts_to_distopt_grads',
    )
    namespace = dict(vars(nep))
    methods = [
        node for node in old_class.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(methods) == len(names)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(reference), 'exec'), namespace)
    counts = {name: 0 for name in names}
    seen = {}

    def wrap(name):
        candidate, original = getattr(cls, name), namespace[name]

        @functools.wraps(candidate)
        @torch.no_grad()
        def checked(group, *args, **kwargs):
            key = (id(group), name)
            if seen.get(key, 0) >= 2:
                return candidate(group, *args, **kwargs)
            seen[key] = seen.get(key, 0) + 1
            bundle = group._nep_distopt_owner_bundle
            if name == names[0]:
                tensors = [
                    entry['bucket'].params_list[0]
                    for bg in bundle['groups']
                    for entry in bg._nep_nccl_entries
                ]
            else:
                tensors = [proxy.main_grad for proxy in bundle['proxy_by_key'].values()]
            identities = [(tensor.data_ptr(), tensor.shape, tensor.stride()) for tensor in tensors]
            # Both implementations read the same untouched owner weights or Gather
            # output. Their only data mutation is overwriting these destinations.
            candidate(group, *args, **kwargs)
            expected = [tensor.detach().clone() for tensor in tensors]
            original(group, *args, **kwargs)
            for tensor, value, identity in zip(tensors, expected, identities):
                assert (tensor.data_ptr(), tensor.shape, tensor.stride()) == identity
                assert torch.equal(tensor, value), (name, torch.distributed.get_rank())
            counts[name] += 1
            print('NEP_SYNC_SHADOW_PASS', name, 'tensors', len(tensors), flush=True)

        setattr(cls, name, checked)

    for name in names:
        wrap(name)
    atexit.register(lambda: print('NEP_SYNC_SHADOW_COUNTS', json.dumps(counts), flush=True))


def main() -> None:
    sys.path.insert(0, '/home/darfeen/Megatron-LM/scripts/nonuniform')
    sys.path.insert(0, os.environ['CODE_REPO'])
    if os.environ.get('NEP_SYNC_SHADOW') == '1':
        install_shadow()
    from pretrain_profile_activity_probe import main as pretrain

    pretrain()


if __name__ == '__main__':
    main()
