"""Two-rank correctness and nonblocking-completion-observer smoke."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from cuda_event_capture_adapter import CudaEventCapture

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 2
assert os.environ['TORCH_NCCL_BLOCKING_WAIT'] == '0'
CudaEventCapture.install_stage_labels()
import megatron.core.distributed.param_and_grad_buffer as buffers


class Probe:
    @staticmethod
    def compute():
        torch.cuda._sleep(1000000)


CudaEventCapture.wrap(Probe, 'compute', 'smoke.independent_compute')
capture = CudaEventCapture(schedule=lambda step: torch.profiler.ProfilerAction.RECORD
                           if 1 <= step < 4 else torch.profiler.ProfilerAction.NONE)
source = torch.full((262144,), rank + 1, device='cuda', dtype=torch.float32)
output = torch.empty(world * source.numel(), device='cuda')
rs_input = torch.full_like(output, rank + 1)
rs_output = torch.empty_like(source)
a2a_output = torch.empty_like(source)
broadcast = torch.full_like(source, rank + 1)
capture.start()
for step in range(5):
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 1 and 1 <= step < 4:
        torch.cuda._sleep(100000000)
    with buffers._coalescing_manager(group=dist.group.WORLD, async_ops=True) as ag:
        buffers.dist_all_gather_func(output, source, group=dist.group.WORLD, async_op=True)
    Probe.compute()
    ag.wait()
    with buffers._coalescing_manager(group=dist.group.WORLD, async_ops=True) as rs:
        buffers.dist_reduce_scatter_func(rs_output, rs_input, group=dist.group.WORLD, async_op=True)
    rs.wait()
    work = dist.all_to_all_single(a2a_output, source, group=dist.group.WORLD, async_op=True)
    work.wait()
    dist.broadcast(broadcast, src=0, group=dist.group.WORLD)
    torch.cuda.synchronize()
    assert torch.all(output[:source.numel()] == 1).item()
    assert torch.all(output[source.numel():] == 2).item()
    assert torch.all(rs_output == 3).item()
    assert torch.all(a2a_output[:source.numel() // 2] == 1).item()
    assert torch.all(a2a_output[source.numel() // 2:] == 2).item()
    assert torch.all(broadcast == 1).item()
    capture.step()
capture.stop()
path = Path(os.environ['CUPTI_OUTPUT_DIR']) / f'rank-{rank}.json'
data = json.loads(path.read_text())
gpu = [e for e in data['traceEvents'] if e['cat'] == 'gpu_operation_range']
for step in range(1, 4):
    events = [e for e in gpu if e['args']['iteration'] == step]
    names = [e['name'] for e in events]
    assert all(names.count(name) == 1 for name in ('collective.all_gather',
               'collective.reduce_scatter', 'collective.all_to_all', 'collective.broadcast'))
    assert all(e['dur'] > 0 for e in events)
    if rank == 0:
        ag = next(e for e in events if e['name'] == 'collective.all_gather')
        compute = next(e for e in events if e['name'] == 'smoke.independent_compute')
        assert compute['ts'] + compute['dur'] < ag['ts'] + ag['dur'], (compute, ag)
        assert ag['dur'] > 10000, ag
        print('ASYNC_OBSERVER_PASS', step, 'compute_end_before_AG_end_ms',
              (ag['ts'] + ag['dur'] - compute['ts'] - compute['dur']) / 1000, flush=True)
print('COLLECTIVE_SMOKE_PASS', rank, flush=True)
dist.destroy_process_group()
