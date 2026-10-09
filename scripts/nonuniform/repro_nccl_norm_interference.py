"""Isolate native TE RMSNorm graph progress during an unrelated NCCL Gather."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    assert dist.get_world_size() == 2
    groups = {}
    for ctas in (32, 4):
        options = dist.ProcessGroupNCCL.Options()
        options.config.max_ctas = ctas
        options.config.cga_cluster_size = 4
        groups[ctas] = dist.new_group([0, 1], pg_options=options, group_desc=f'gather_ctas{ctas}')
    model = te.RMSNorm(4608, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(4096, 4608, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.ones_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model(x).backward(dy)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model(x).backward(dy)
    n = 4 * 1024 * 1024
    source = torch.ones(n if rank else 0, device='cuda', dtype=torch.bfloat16)
    dest = torch.empty(0 if rank else n, device='cuda', dtype=torch.bfloat16)
    comm = torch.cuda.Stream()
    results = []
    for ctas in (0, 32, 4, 32, 4, 0):
        for iteration in range(5):
            dist.barrier()
            torch.cuda.synchronize()
            begin, end, comm_end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            torch.cuda.nvtx.range_push(f'ctas{ctas}_iteration{iteration}')
            work = None
            if ctas:
                with torch.cuda.stream(comm):
                    if rank == 1:
                        torch.cuda._sleep(30000000)
                    work = dist.all_to_all_single(
                        dest, source,
                        output_split_sizes=[0, 0] if rank else [0, n],
                        input_split_sizes=[n, 0] if rank else [0, 0],
                        group=groups[ctas], async_op=True,
                    )
            begin.record()
            graph.replay()
            end.record()
            if work is not None:
                with torch.cuda.stream(comm):
                    work.wait()
                    comm_end.record()
            torch.cuda.synchronize()
            row = dict(rank=rank, ctas=ctas, iteration=iteration, graph_ms=begin.elapsed_time(end))
            if work is not None:
                row['comm_after_graph_ms'] = end.elapsed_time(comm_end)
                if rank == 0:
                    assert torch.equal(dest, torch.ones_like(dest))
            results.append(row)
            torch.cuda.nvtx.range_pop()
            print(json.dumps(row), flush=True)
    root = Path(os.environ['DIAGNOSTIC_OUTPUT'])
    root.mkdir(parents=True, exist_ok=True)
    (root / f'rank-{rank}.json').write_text(json.dumps(results, indent=2) + '\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
