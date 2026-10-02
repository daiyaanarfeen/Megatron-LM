"""Real native AG/RS and owner/follower redistribution on four GPUs."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from cuda_event_capture_adapter import CudaEventCapture

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.nonuniform_ep import NonuniformEPNCCLParamAndGradBucketGroup
from megatron.core.distributed.param_and_grad_buffer import (
    _ParamAndGradBucket,
    _ParamAndGradBucketGroup,
)


class Probe:
    @staticmethod
    def independent():
        torch.cuda._sleep(1000000)


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    owner = rank % 2 == 0
    edp = dist.new_group([0, 2], group_desc='ep_dp')
    transfers = [dist.new_group(pair, group_desc='nep_owner_transfer') for pair in ([0, 1], [2, 3])]
    model_ep = [dist.new_group(pair, group_desc='ep') for pair in ([0, 1], [2, 3])]
    replica = rank // 2
    config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_param_gather=True, overlap_grad_reduce=True
    )
    stream = torch.cuda.Stream()
    lane_count = int(os.getenv('NEP_ABLATION_PARAM_LANES', '0'))
    lane_streams = (torch.cuda.Stream(), torch.cuda.Stream()) if lane_count else None
    cache = {}
    groups = []
    n = 16384
    for index in range(3):
        group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
            NonuniformEPNCCLParamAndGradBucketGroup
        )
        group.ddp_config = config
        group._nep_distopt_param_sync_representative = True
        group.param_gather_dispatched = False
        physical = torch.nn.Parameter(torch.empty(n, device='cuda', dtype=torch.bfloat16))
        param_data = torch.empty(2 * n, device='cuda', dtype=torch.bfloat16)
        proxies = [torch.nn.Parameter(param_data[i * n : (i + 1) * n]) for i in range(2)]
        native = None
        if owner:
            bucket = _ParamAndGradBucket(
                proxies,
                param_data,
                torch.zeros(2 * n, device='cuda'),
                0,
                2 * n,
                1.0,
                index,
                {p: (i * n, (i + 1) * n, index) for i, p in enumerate(proxies)},
                [],
            )
            native = _ParamAndGradBucketGroup([bucket], config, edp, 2)
        unpack = (
            (
                type('PhysicalBucket', (), {'params_list': [physical]})(),
                -1 if owner else 0,
                n,
                (index, rank % 2),
            ),
        )
        pack = (((index, 1), 0, n),) if owner else ()
        plan = (
            (
                transfers[replica],
                2,
                [0, n] if owner else [0, 0],
                [0, 0] if owner else [n, 0],
                n if owner else 0,
                0 if owner else n,
                pack,
                unpack,
            ),
        )
        group._get_nep_distopt_param_transfer_plan = lambda plan=plan: plan
        group._nep_distopt_owner_bundle = dict(
            native_group=native,
            proxy_by_key={(index, i): p for i, p in enumerate(proxies)},
            param_transfer_buffers=cache,
            param_transfer_slot=index % 2,
            param_transfer_numel_by_slot={0: 2 * n, 1: 2 * n},
            param_transfer_stream=stream,
            param_sync_event=torch.cuda.Event(),
            param_sync_completed=False,
            param_delivery_policy=os.getenv('NEP_ABLATION_PARAM_RELEASE', 'early'),
            param_prefetch_depth=int(os.getenv('NEP_ABLATION_PREFETCH_DEPTH', '1')),
        )
        group.next_param_gather_bucket_group = None
        if lane_streams:
            group._nep_distopt_owner_bundle.update(
                param_lane_streams=lane_streams,
                param_lane_count=lane_count,
                param_lane_ready=torch.cuda.Event(),
                param_transfer_owners=(index % 2,),
            )
            for slot in range(2):
                for lane in range(2):
                    key = (slot, physical.dtype, physical.device, lane)
                    if key not in cache:
                        cache[key] = torch.empty(
                            2 * n, dtype=physical.dtype, device=physical.device
                        )
        groups.append((group, physical, param_data, native))
    for left, right in zip(groups, groups[1:]):
        left[0].next_param_gather_bucket_group = right[0]

    scatter = groups[0][0]._scatter_nep_distopt_owner_params_to_physical_holders

    def delayed_scatter():
        # Make delivery remain pending long enough to prove the next AG progresses.
        torch.cuda._sleep(20000000)
        scatter()

    groups[0][0]._scatter_nep_distopt_owner_params_to_physical_holders = delayed_scatter

    CudaEventCapture.install_stage_labels()
    CudaEventCapture.wrap(Probe, 'independent', 'smoke.independent')
    actions = torch.profiler.ProfilerAction
    capture = CudaEventCapture(
        schedule=lambda step: actions.RECORD if 1 <= step <= 3 else actions.NONE
    )
    capture.start()
    for iteration in range(5):
        for index, (group, physical, param_data, native) in enumerate(groups):
            group.param_gather_dispatched = False
            physical.data.fill_(-1)
            param_data.fill_(-1)
            if owner:
                param_data[replica * n : (replica + 1) * n].fill_(
                    iteration * 10 + index * 2 + replica
                )
                native.param_gather_dispatched = False
                native.reset()
                native.buckets[0].grad_data.fill_(1)
        dist.barrier()
        if rank == 2:
            torch.cuda._sleep(100000000)
        force = iteration == 3
        groups[0][0].start_param_sync(force_sync=force)
        if not force:
            assert groups[0][0]._nep_distopt_owner_bundle['param_delivery_submitted'] == (
                os.getenv('NEP_ABLATION_PARAM_RELEASE', 'early') == 'early'
            )
        Probe.independent()
        for index, (group, physical, _, native) in enumerate(groups):
            group.finish_param_sync()
            if index == 0 and not force:
                assert groups[2][0].param_gather_dispatched == (
                    int(os.getenv('NEP_ABLATION_PREFETCH_DEPTH', '1')) == 2
                )
            group.finish_param_sync()
            expected = iteration * 10 + index * 2 + rank % 2
            assert torch.equal(physical, torch.full_like(physical, expected)), (
                rank,
                iteration,
                index,
            )
            # Independent native model-EP traffic while the next bucket is prefetched.
            source = torch.ones(4, device='cuda')
            dest = torch.empty_like(source)
            dist.all_to_all_single(dest, source, group=model_ep[replica])
            assert torch.equal(dest, source)
            if native is not None:
                native.start_grad_sync()
                native.finish_grad_sync()
        assert len(cache) == 4 if lane_streams else not cache
        capture.step()
    capture.stop()
    if owner:
        data = json.loads((Path(os.environ['CUPTI_OUTPUT_DIR']) / f'rank-{rank}.json').read_text())
        for step in (1, 2):
            events = [
                e
                for e in data['traceEvents']
                if e['cat'] == 'gpu_operation_range' and e['args']['iteration'] == step
            ]
            ag = next(e for e in events if e['name'] == 'collective.all_gather')
            compute = next(e for e in events if e['name'] == 'smoke.independent')
            if rank == 0:
                assert compute['ts'] + compute['dur'] < ag['ts'] + ag['dur'], (ag, compute)
            ags = [e for e in events if e['name'] == 'collective.all_gather']
            delivery = next(e for e in events if e['name'] == 'nep.redistribute_params')
            assert len(ags) == 3
            assert ags[1]['ts'] + ags[1]['dur'] < delivery['ts'] + delivery['dur'], (
                rank,
                step,
                ags[1],
                delivery,
            )
        print(
            'PREFETCH_SMOKE_PASS: fresh values, followers, reuse, force-sync, native EP/AG/RS and independent GPU progress',
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
