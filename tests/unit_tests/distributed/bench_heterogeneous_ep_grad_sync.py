#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Benchmark: Approach A (NCCL) vs Approach B (NVSHMEM) for het EP grad sync.

Usage (via torchrun):
    torchrun --nproc_per_node=4 --nnodes=N --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        tests/unit_tests/distributed/bench_heterogeneous_ep_grad_sync.py
"""

import os
import time

import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.moe.moe_layer import MoELayer

import argparse

WARMUP_ITERS = 5
BENCH_ITERS = 30

TOPOLOGIES = {
    '12gpu_k2_4': dict(tp=2, cp=1, k=[2, 4], etp=2, num_moe_experts=8, world_size=12),
    '28gpu_k4_4_6': dict(tp=2, cp=1, k=[4, 4, 6], etp=2, num_moe_experts=12, world_size=28),
}

DEFAULT_HIDDEN_SIZES = [64, 256, 1024, 2048, 4096, 8192]


class HeterogeneousMoEModel(torch.nn.Module):
    def __init__(self, hidden_size, num_moe_experts, ep_size, etp_size, num_layers=1):
        super().__init__()
        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=max(1, hidden_size // 64),
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_token_dispatcher_type="alltoall",
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size,
            add_bias_linear=False,
            use_cpu_initialization=True,
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=False,
        )
        self.layers = torch.nn.ModuleList(
            [MoELayer(config, submodules.mlp.submodules).cuda() for _ in range(num_layers)]
        )


def init_distributed():
    if not torch.distributed.is_initialized():
        rank = int(os.environ.get('RANK', '0'))
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl', world_size=world_size, rank=rank,
            init_method=f"tcp://{os.environ.get('MASTER_ADDR', 'localhost')}:"
                        f"{os.environ.get('MASTER_PORT', '29500')}",
            device_id=torch.device(f'cuda:{local_rank}'),
        )


def setup_groups(topo):
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_heterogeneous_model_parallel(
        tensor_model_parallel_size=topo['tp'],
        context_parallel_size=topo['cp'],
        num_tp_cp_per_replica=topo['k'],
        expert_tensor_parallel_size=topo['etp'],
        num_moe_experts=topo['num_moe_experts'],
        distributed_timeout_minutes=10,
    )


def create_model(topo, hidden_size, use_pipelined, num_chunks=4, bucket_size=None):
    het_cfg = parallel_state.get_heterogeneous_ep_config()
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        bucket_size=bucket_size,
        average_in_collective=False,
        use_pipelined_ep_reshard=use_pipelined,
        num_ep_reshard_pipeline_chunks=num_chunks,
    )
    module = HeterogeneousMoEModel(
        hidden_size=hidden_size,
        num_moe_experts=topo['num_moe_experts'],
        ep_size=het_cfg['local_ep_size'],
        etp_size=topo['etp'],
    )
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        ddp_config=ddp_config, module=module,
    )
    return model


def bench_grad_sync(model, warmup, iters):
    ep_bucket_groups = model.expert_parallel_bucket_groups
    ep_buffers = model.expert_parallel_buffers

    for _ in range(warmup):
        for buf in ep_buffers:
            buf.grad_data.data.fill_(1.0)
        for bg in ep_bucket_groups:
            bg.finish_grad_sync()

    torch.cuda.synchronize()
    torch.distributed.barrier()

    times = []
    for _ in range(iters):
        for buf in ep_buffers:
            buf.grad_data.data.fill_(1.0)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for bg in ep_bucket_groups:
            bg.finish_grad_sync()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000)

    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, nargs='+', default=None,
                        help='Hidden sizes to benchmark (default: all)')
    parser.add_argument('--bucket-sizes', type=int, nargs='+', default=None,
                        help='Bucket sizes for Approach A sweep (default: 1-bucket only)')
    # Parse only known args so torchrun args don't conflict.
    args, _ = parser.parse_known_args()

    init_distributed()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    topo_name = None
    topo = None
    for name, t in TOPOLOGIES.items():
        if t['world_size'] == world_size:
            topo_name = name
            topo = t
            break

    if topo is None:
        if rank == 0:
            print(f"No topology for world_size={world_size}")
        return

    hidden_sizes = args.hidden or DEFAULT_HIDDEN_SIZES
    slot_mb = int(os.environ.get('MEGATRON_NVSHMEM_SLOT_MB', '32'))
    # bucket_sizes: None = 1 big bucket (no overlap), or list of sizes to sweep.
    a_bucket_sizes = args.bucket_sizes or [None]

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Benchmark: {topo_name} (world_size={world_size}, k={topo['k']})")
        print(f"  slot_size={slot_mb}MB, warmup={WARMUP_ITERS}, iters={BENCH_ITERS}")
        print(f"  A bucket_sizes: {a_bucket_sizes}")
        print(f"{'='*70}")

    results = []

    # Single setup — both A and B share the same process groups.
    # NVSHMEM init happens once here; both models use it.
    setup_groups(topo)
    het_cfg = parallel_state.get_heterogeneous_ep_config()
    nvshmem_active = het_cfg.get('nvshmem_initialized', False)
    chunk_size = het_cfg.get('nvshmem_chunk_size', slot_mb * 1024 * 1024)

    for hidden_size in hidden_sizes:
        # Compute K for B display.
        slot_elems = chunk_size // 4  # float32
        max_ratio = het_cfg.get('max_ep_size', 1) // max(het_cfg.get('min_ep_size', 1), 1)
        eff_ar = (slot_elems // max(max_ratio, 1)) * max(max_ratio, 1)
        ratio = het_cfg.get('ratio', 1)
        per_member = eff_ar // max(ratio, 1)

        if rank == 0:
            print(f"\nhidden={hidden_size}, slot={slot_mb}MB")

        # ── Approach A: sweep bucket sizes ──
        best_a_avg = float('inf')
        best_a_label = ""
        for bs in a_bucket_sizes:
            model_a = create_model(topo, hidden_size, use_pipelined=False, bucket_size=bs)
            n_buckets = sum(len(bg.buckets) for bg in model_a.expert_parallel_bucket_groups)
            expert_buf_size = sum(buf.grad_data.numel() for buf in model_a.expert_parallel_buffers)
            times_a = bench_grad_sync(model_a, WARMUP_ITERS, BENCH_ITERS)
            del model_a
            torch.cuda.empty_cache()

            avg_a = sum(times_a) / len(times_a)
            p50_a = sorted(times_a)[len(times_a) // 2]
            min_a = min(times_a)
            bs_label = f"1-bucket" if bs is None else f"bs={bs}"

            if rank == 0:
                print(f"  A ({bs_label}, {n_buckets} buckets): "
                      f"avg={avg_a:.3f}ms  p50={p50_a:.3f}ms  min={min_a:.3f}ms")

            if avg_a < best_a_avg:
                best_a_avg = avg_a
                best_a_label = f"{bs_label}/{n_buckets}b"

        # ── Approach B ──
        model_b = create_model(topo, hidden_size, use_pipelined=True)
        expert_buf_size = sum(buf.grad_data.numel() for buf in model_b.expert_parallel_buffers)
        K = (expert_buf_size + per_member - 1) // per_member if per_member > 0 else 1
        times_b = bench_grad_sync(model_b, WARMUP_ITERS, BENCH_ITERS)
        del model_b
        torch.cuda.empty_cache()

        avg_b = sum(times_b) / len(times_b)
        p50_b = sorted(times_b)[len(times_b) // 2]
        min_b = min(times_b)
        speedup = best_a_avg / avg_b if avg_b > 0 else float('inf')

        if rank == 0:
            label = "NVSHMEM" if nvshmem_active else "NCCL fallback"
            print(f"  B ({label}, K={K}): "
                  f"avg={avg_b:.3f}ms  p50={p50_b:.3f}ms  min={min_b:.3f}ms  "
                  f"B/bestA={speedup:.2f}x")

        results.append(dict(
            hidden=hidden_size, buf_numel=expert_buf_size, K=K,
            best_a=best_a_avg, best_a_label=best_a_label,
            avg_b=avg_b, speedup=speedup,
        ))

    if rank == 0:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'hidden':>8} {'buf_numel':>12} {'K':>5} {'bestA_ms':>10} {'bestA_cfg':>14} "
              f"{'B_avg_ms':>10} {'B/bestA':>8}")
        for r in results:
            print(f"{r['hidden']:>8} {r['buf_numel']:>12} {r['K']:>5} {r['best_a']:>10.3f} "
                  f"{r['best_a_label']:>14} {r['avg_b']:>10.3f} {r['speedup']:>8.2f}x")

    parallel_state.destroy_model_parallel()


if __name__ == '__main__':
    main()
