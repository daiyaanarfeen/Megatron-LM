"""Check targeted expert ranges and unchanged grouped-linear gradients."""

import json
import os
from pathlib import Path

import torch
from transformer_engine.pytorch import GroupedLinear

from cuda_event_capture_adapter import CudaEventCapture

torch.manual_seed(123)
layer = GroupedLinear(num_gemms=2, in_features=128, out_features=128,
                      params_dtype=torch.bfloat16, device='cuda')
x = torch.randn(32, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)


def step():
    layer.zero_grad(set_to_none=True)
    x.grad = None
    output = layer(x, [16, 16])
    if isinstance(output, tuple):
        output = output[0]
    output.float().square().sum().backward()
    return output.detach().clone(), x.grad.clone(), [p.grad.clone() for p in layer.parameters()]


baseline = step()
CudaEventCapture.install_stage_labels()
capture = CudaEventCapture(schedule=lambda step: torch.profiler.ProfilerAction.RECORD)
capture.start()
observed = step()
capture.stop()
torch.testing.assert_close(observed[0], baseline[0], rtol=0, atol=0)
torch.testing.assert_close(observed[1], baseline[1], rtol=0, atol=0)
for actual, expected in zip(observed[2], baseline[2]):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
data = json.loads((Path(os.environ['CUPTI_OUTPUT_DIR'])/'rank-0.json').read_text())
events = [e for e in data['traceEvents'] if e['cat']=='gpu_operation_range']
assert [e['name'] for e in events] == ['expert.forward', 'expert.backward'], events
assert all(e['dur'] > 0 for e in events)
print('PASS: forward/backward CUDA ranges; bitwise-identical outputs and gradients', flush=True)
