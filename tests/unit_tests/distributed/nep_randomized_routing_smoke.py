"""Validate cached routing, straight-through gradients and graph replay on CUDA."""

import json
import os
from pathlib import Path

import torch
from nep_randomized_controls import install

from megatron.core.distributed import nonuniform_common as common


def main() -> None:
    torch.cuda.set_device(0)
    root = Path(os.environ['GRAPH_OUTPUT'])
    install()
    common.set_nonuniform_ep_runtime_config(dict(expert_placement=[list(range(512))]))
    logits = torch.zeros(1024, 512, device='cuda', requires_grad=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        output = common.apply_exact_uniform_routing_logits(logits, 6)
        mask = output > 0
        assert torch.equal(mask.sum(0), torch.full((512,), 12, device='cuda'))
        assert torch.equal(mask.sum(1), torch.full((1024,), 6, device='cuda'))
        output.sum().backward()
        assert torch.equal(logits.grad, torch.ones_like(logits))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                common.apply_exact_uniform_routing_logits(logits, 6)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = common.apply_exact_uniform_routing_logits(logits, 6)
        for _ in range(5):
            graph.replay()
            assert torch.equal(captured, output)
    profile.export_chrome_trace(str(root / 'routing-cuda.json'))
    manifest = json.loads((root / 'controls/routing-0-1024.json').read_text())
    assert set(manifest['expert_counts']) == {12}
    print(
        'RANDOMIZED_ROUTING_CUDA_PASS: exact counts, STE gradients, cache and graph replay',
        flush=True,
    )


if __name__ == '__main__':
    main()
