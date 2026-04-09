#!/usr/bin/env python3
"""Benchmark a single approach for heterogeneous EP grad sync.

Usage:
    torchrun ... bench_single.py --approach {a,b} --hidden 256
"""
import argparse, os, time, torch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.moe.moe_layer import MoELayer

WARMUP, ITERS = 5, 30

class MoEModel(torch.nn.Module):
    def __init__(self, h, n_exp, ep, etp):
        super().__init__()
        cfg = TransformerConfig(
            num_layers=1, hidden_size=h, num_attention_heads=max(1, h//64),
            num_moe_experts=n_exp, moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2, moe_aux_loss_coeff=0.01,
            moe_token_dispatcher_type="alltoall",
            expert_model_parallel_size=ep, expert_tensor_parallel_size=etp,
            add_bias_linear=False, use_cpu_initialization=True)
        sub = get_gpt_layer_local_submodules(num_experts=n_exp, moe_grouped_gemm=False)
        self.layers = torch.nn.ModuleList([MoELayer(cfg, sub.mlp.submodules).cuda()])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--approach', choices=['a', 'b'], required=True)
    p.add_argument('--hidden', type=int, nargs='+', default=[64, 256, 1024, 4096])
    args = p.parse_args()

    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    ws = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend='nccl', world_size=ws, rank=rank,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

    topo = dict(tp=2, cp=1, k=[2,4], etp=2, num_moe_experts=8)
    use_pipe = (args.approach == 'b')

    parallel_state.destroy_model_parallel()
    parallel_state.initialize_heterogeneous_model_parallel(
        tensor_model_parallel_size=topo['tp'], context_parallel_size=topo['cp'],
        num_tp_cp_per_replica=topo['k'], expert_tensor_parallel_size=topo['etp'],
        num_moe_experts=topo['num_moe_experts'],
        hidden_size=max(args.hidden))

    het = parallel_state.get_heterogeneous_ep_config()
    nvshmem_on = het.get('nvshmem_initialized', False)

    if rank == 0:
        label = f"Approach {'B (NVSHMEM)' if use_pipe and nvshmem_on else 'B (NCCL fallback)' if use_pipe else 'A (NCCL)'}"
        print(f"\n{'='*60}")
        print(f"{label}  |  k={topo['k']}, ws={ws}")
        print(f"{'='*60}")

    for h in args.hidden:
        ddp_cfg = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, overlap_grad_reduce=False,
            average_in_collective=False,
            use_pipelined_ep_reshard=use_pipe, num_ep_reshard_pipeline_chunks=4)
        mod = MoEModel(h, topo['num_moe_experts'], het['local_ep_size'], topo['etp'])
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            ddp_cfg, module=mod)

        bufs = model.expert_parallel_buffers
        bgs = model.expert_parallel_bucket_groups
        buf_n = sum(b.grad_data.numel() for b in bufs)

        for _ in range(WARMUP):
            for b in bufs: b.grad_data.data.fill_(1.0)
            for bg in bgs: bg.finish_grad_sync()

        torch.cuda.synchronize(); torch.distributed.barrier()
        times = []
        for _ in range(ITERS):
            for b in bufs: b.grad_data.data.fill_(1.0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for bg in bgs: bg.finish_grad_sync()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times)/len(times)
        p50 = sorted(times)[len(times)//2]
        mn = min(times)
        if rank == 0:
            print(f"  h={h:>5}  buf={buf_n:>12}  avg={avg:.3f}ms  p50={p50:.3f}ms  min={mn:.3f}ms")
        del model, mod; torch.cuda.empty_cache()

    parallel_state.destroy_model_parallel()

if __name__ == '__main__':
    main()
