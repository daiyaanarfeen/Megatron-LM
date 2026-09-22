"""Two independent owners, shared followers, and bounded parameter-buffer reuse."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from cuda_event_capture_adapter import CudaEventCapture

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.nonuniform_ep import NonuniformEPNCCLParamAndGradBucketGroup


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    transfers = {
        source: dist.new_group([source, 1, 3], group_desc='nep_owner_transfer') for source in (0, 2)
    }
    config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_param_gather=True
    )
    streams, buffers = {}, {}
    coordinator = torch.cuda.Stream()
    n = 1024 * 1024
    groups = []
    for index in range(3):
        group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
            NonuniformEPNCCLParamAndGradBucketGroup
        )
        group.ddp_config = config
        group._nep_distopt_param_sync_representative = True
        group.param_gather_dispatched = False
        group.next_param_gather_bucket_group = None
        physical, proxies, plan = {}, {}, []
        for source in (0, 2):
            if rank not in (source, 1, 3):
                continue
            param = torch.nn.Parameter(torch.empty(n, device='cuda', dtype=torch.bfloat16))
            physical[source] = param
            key = (index, source)
            proxies[key] = torch.nn.Parameter(torch.empty_like(param))
            plan.append(
                (
                    transfers[source],
                    3,
                    source,
                    n,
                    ((key, 0, n),) if rank == source else (),
                    ((SimpleNamespace(params_list=[param]), 0, n, key),),
                )
            )
        group._get_nep_distopt_param_transfer_plan = lambda plan=plan: plan
        group._nep_distopt_owner_bundle = dict(
            native_group=None,
            proxy_by_key=proxies,
            param_transfer_buffers=buffers,
            param_owner_streams=streams,
            param_transfer_slot=index % 2,
            param_transfer_numel_by_slot={0: n, 1: n},
            param_transfer_stream=coordinator,
            param_sync_event=torch.cuda.Event(),
            param_sync_completed=False,
        )
        groups.append((group, physical, proxies))
    for left, right in zip(groups, groups[1:]):
        left[0].next_param_gather_bucket_group = right[0]
    CudaEventCapture.install_stage_labels()
    actions = torch.profiler.ProfilerAction
    capture = CudaEventCapture(
        schedule=lambda step: actions.RECORD if 1 <= step <= 7 else actions.NONE
    )
    capture.start()
    for iteration in range(9):
        config.overlap_param_gather = iteration != 6
        for index, (group, physical, proxies) in enumerate(groups):
            group.param_gather_dispatched = False
            for source, param in physical.items():
                param.data.fill_(-1)
                proxies[(index, source)].data.fill_(iteration * 10 + index * 2 + source)
        dist.barrier()
        if rank == 0:
            torch.cuda._sleep(10000000)
        groups[0][0].start_param_sync(force_sync=iteration == 3)
        for index, (group, physical, _) in enumerate(groups):
            if not config.overlap_param_gather:
                group.start_param_sync()
            group.finish_param_sync()
            for source, param in physical.items():
                assert torch.equal(
                    param, torch.full_like(param, iteration * 10 + index * 2 + source)
                )
        assert not buffers
        capture.step()
    capture.stop()
    if rank == 0:
        print(
            'PARALLEL_OWNER_SMOKE_PASS: shared followers, live values, 2-slot reuse, force sync and non-overlap',
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
