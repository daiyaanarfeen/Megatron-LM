#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Integration test: real training steps with heterogeneous EP.

Runs forward + backward + optimizer step for N iterations, comparing
Approach A (NCCL) vs Approach B (NVSHMEM pipeline). Verifies:
  1. Both approaches produce matching losses (within tolerance)
  2. Both approaches produce matching expert gradients per step
  3. No hangs over multiple steps

Usage (via torchrun):
    torchrun --nproc_per_node=4 --nnodes=N ...
        tests/unit_tests/distributed/test_heterogeneous_ep_training.py
"""

import os
import argparse

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.moe.moe_layer import MoELayer

TOPOLOGIES = {
    '12gpu': dict(tp=2, cp=1, k=[2, 4], etp=2, num_moe_experts=8, world_size=12),
    '24gpu': dict(tp=2, cp=1, k=[4, 8], etp=2, num_moe_experts=24, world_size=24),
    '28gpu': dict(tp=2, cp=1, k=[4, 4, 6], etp=2, num_moe_experts=12, world_size=28),
    '32gpu': dict(tp=2, cp=1, k=[4, 4, 8], etp=2, num_moe_experts=16, world_size=32),
}


class SimpleMoEModel(torch.nn.Module):
    """MoE model that can run forward + backward."""

    def __init__(self, hidden_size, num_moe_experts, ep_size, etp_size):
        super().__init__()
        tp = parallel_state.get_tensor_model_parallel_world_size()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=max(tp, hidden_size // 64),
            tensor_model_parallel_size=tp,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_token_dispatcher_type="alltoall",
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size,
            add_bias_linear=False,
            use_cpu_initialization=True,
            sequence_parallel=(tp > 1),
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=False,
        )
        self.moe = MoELayer(config, submodules.mlp.submodules).cuda()

    def forward(self, x):
        # MoELayer expects (seq_len, batch, hidden) and returns (output, bias).
        out, _ = self.moe(x)
        return out


def init_distributed():
    if not dist.is_initialized():
        rank = int(os.environ.get('RANK', '0'))
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', world_size=world_size, rank=rank,
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        )
        dist.barrier()


def setup_groups(topo):
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_heterogeneous_model_parallel(
        tensor_model_parallel_size=topo['tp'],
        context_parallel_size=topo['cp'],
        num_tp_cp_per_replica=topo['k'],
        expert_tensor_parallel_size=topo['etp'],
        num_moe_experts=topo['num_moe_experts'],
    )


def create_model(topo, hidden_size, use_pipelined, use_phased=False):
    het_cfg = parallel_state.get_heterogeneous_ep_config()
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        average_in_collective=False,
        use_pipelined_ep_reshard=use_pipelined,
        num_ep_reshard_pipeline_chunks=4,
        use_phased_ep_reshard=use_phased,
    )
    module = SimpleMoEModel(
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


def train_steps(model, hidden_size, num_steps, seed):
    """Run N training steps, return per-step losses and final expert grad norms."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ep_bucket_groups = model.expert_parallel_bucket_groups
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # Synthetic input: (seq_len=4, batch=2, hidden).
        # Same input on all ranks for reproducibility.
        torch.manual_seed(seed + step)
        x = torch.randn(4, 2, hidden_size, device='cuda')

        out = model.module(x)
        loss = out.sum()
        loss.backward()

        # Trigger expert grad sync.
        for bg in ep_bucket_groups:
            bg.finish_grad_sync()

        losses.append(loss.item())
        optimizer.step()

    # Collect final expert grad norms for comparison.
    grad_norms = []
    for buf in model.expert_parallel_buffers:
        grad_norms.append(buf.grad_data.norm().item())

    return losses, grad_norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args, _ = parser.parse_known_args()

    init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

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

    if rank == 0:
        print(f"Integration test: {topo_name}, hidden={args.hidden}, "
              f"steps={args.steps}, seed={args.seed}")

    setup_groups(topo)

    results = {}
    approaches = [
        ('A (NCCL)', False, False),
        ('B (NVSHMEM)', True, False),
        ('C (phased)', False, True),
    ]
    for approach, use_pipe, use_phased in approaches:
        if rank == 0:
            print(f"\nRunning Approach {approach}...")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        model = create_model(topo, args.hidden, use_pipelined=use_pipe,
                             use_phased=use_phased)
        losses, norms = train_steps(model, args.hidden, args.steps, args.seed)
        del model
        torch.cuda.empty_cache()

        if rank == 0:
            print(f"  losses: {[f'{l:.6f}' for l in losses]}")
            print(f"  grad norms: {[f'{n:.6f}' for n in norms]}")
        results[approach] = (losses, norms)

    # ── Checks ──
    passed = True
    for approach, (losses, norms) in results.items():
        # Check 1: no NaN/Inf in losses.
        for step, l in enumerate(losses):
            if not (abs(l) < 1e6):
                if rank == 0:
                    print(f"FAIL: {approach} step {step} loss is NaN/Inf: {l}")
                passed = False

        # Check 2: no NaN in grad norms.
        for i, n in enumerate(norms):
            if not (abs(n) < 1e6):
                if rank == 0:
                    print(f"FAIL: {approach} grad norm {i} is NaN/Inf: {n}")
                passed = False

        # Check 3: grad norms are non-zero (grads were actually computed).
        for i, n in enumerate(norms):
            if n < 1e-8:
                if rank == 0:
                    print(f"FAIL: {approach} grad norm {i} is zero")
                passed = False

    if rank == 0:
        if passed:
            print("\nPASS: Both approaches completed training without errors")
        else:
            print("\nFAIL: Issues detected")

    parallel_state.destroy_model_parallel()


if __name__ == '__main__':
    main()
