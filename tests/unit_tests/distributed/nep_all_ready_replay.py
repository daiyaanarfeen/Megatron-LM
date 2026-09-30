"""Replay real NEP parameter plans with owner inputs ready and no token dispatch."""

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def transfer_iteration(groups) -> None:
    """Use the production packing, exact-destination A2A and unpacking path."""
    for group in groups:
        group._scatter_nep_distopt_owner_params_to_physical_holders()


def run(ddp) -> None:
    """Profile a bounded replay and exit before optimizer construction/training."""
    from cuda_event_capture_adapter import CudaEventCapture
    from nsys_capture_adapter import NsightCapture

    groups = [
        group
        for group in reversed(ddp.expert_parallel_bucket_groups)
        if group._nep_distopt_param_sync_representative
    ]
    assert groups
    rank = dist.get_rank()
    root = Path(os.environ['GRAPH_OUTPUT'])
    expected = {
        name: hashlib.sha256(
            param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        for name, param in ddp.module.named_parameters()
    }
    for _ in range(3):
        dist.barrier()
        torch.cuda.synchronize()
        transfer_iteration(groups)
        torch.cuda.synchronize()

    CudaEventCapture.install_stage_labels()
    CudaEventCapture.wrap(sys.modules[__name__], 'transfer_iteration', 'replay.parameter_transfers')
    schedule = lambda step: torch.profiler.ProfilerAction.RECORD
    events = CudaEventCapture(schedule=schedule)
    nsight = NsightCapture(schedule=schedule)
    events.start()
    nsight.start()
    timings = []
    for iteration in range(8):
        # Arrival alignment is outside the transfer range. No AG or compute is
        # running: the startup synchronization populated every owner input.
        dist.barrier()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        transfer_iteration(groups)
        end.record()
        end.synchronize()
        timings.append(dict(iteration=iteration, elapsed_ms=start.elapsed_time(end)))
        events.step()
        nsight.step()
    events.stop()
    nsight.stop()
    for name, param in ddp.module.named_parameters():
        actual = hashlib.sha256(
            param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        assert actual == expected[name], (rank, name, 'parameter replay changed values')
    (root / f'replay-{rank}.json').write_text(
        json.dumps(
            dict(rank=rank, timings=timings, groups=len(groups), verified_parameters=len(expected)),
            indent=2,
        )
        + '\n'
    )
    print(f'ALL_READY_REPLAY_PASS rank={rank}', flush=True)
    dist.destroy_process_group()
    raise SystemExit(0)
