# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for heterogeneous EP gradient synchronization.

Verifies that expert gradients are correctly synced across replicas with
different ep sizes using the reshard→allreduce→unreshard pattern.

Tests both:
  - Approach A: per-bucket synchronous gather → allreduce → scatter
  - Approach B: fused intra-bucket pipelined reshard

Topologies tested:
  - 8 GPUs:  k=[1,3], tp=2, etp=2, num_experts=6   (one replica ep=1)
  - 24 GPUs: k=[4,8], tp=2, etp=2, num_experts=24   (all replicas ep>1)
"""

import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.moe.moe_layer import MoELayer
from tests.unit_tests.test_utilities import Utils


# ── Topologies ──

# 8 GPUs: k=[1,3], tp=2, etp=2, num_experts=6
#   Replica 0 (ranks 0-1): ep=1, each rank holds all 6 experts
#   Replica 1 (ranks 2-7): ep=3, each etp pair holds 2 experts
TOPO_8GPU = dict(tp=2, cp=1, k=[1, 3], etp=2, num_moe_experts=6, world_size=8)

# 12 GPUs: k=[2,4], tp=2, etp=2, num_experts=8 (all replicas ep>1)
#   Replica 0: ep=2, Replica 1: ep=4, min_ep=2
TOPO_12GPU = dict(tp=2, cp=1, k=[2, 4], etp=2, num_moe_experts=8, world_size=12)

# 24 GPUs: k=[4,8], tp=2, etp=2, num_experts=24 (all replicas ep>1)
#   Replica 0: ep=4, Replica 1: ep=8, min_ep=4
TOPO_24GPU = dict(tp=2, cp=1, k=[4, 8], etp=2, num_moe_experts=24, world_size=24)


class HeterogeneousMoEModel(torch.nn.Module):
    """Simple MoE model for gradient sync testing."""

    def __init__(self, hidden_size, num_moe_experts, ep_size, etp_size, num_layers=1):
        super().__init__()
        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=1,
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


def _init_distributed():
    """Initialize torch.distributed, handling both torchrun and srun envs."""
    if not torch.distributed.is_initialized():
        # torchrun sets RANK (global) and LOCAL_RANK; srun sets SLURM_PROCID/SLURM_LOCALID.
        rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', '0')))
        local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0')))
        world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', '1')))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank,
            init_method=f"tcp://{os.environ.get('MASTER_ADDR', 'localhost')}:"
                        f"{os.environ.get('MASTER_PORT', '29500')}",
        )
        torch.distributed.barrier()


def _setup_heterogeneous(topo):
    """Initialize heterogeneous model parallel groups."""
    parallel_state.destroy_model_parallel()
    _init_distributed()
    parallel_state.initialize_heterogeneous_model_parallel(
        tensor_model_parallel_size=topo['tp'],
        context_parallel_size=topo['cp'],
        num_tp_cp_per_replica=topo['k'],
        expert_tensor_parallel_size=topo['etp'],
        num_moe_experts=topo['num_moe_experts'],
        hidden_size=16,  # matches HeterogeneousMoEModel default
    )


def _create_ddp_model(
    hidden_size,
    num_moe_experts,
    ep_size,
    etp_size,
    use_pipelined=False,
    num_pipeline_chunks=4,
    average_in_collective=False,
):
    """Create a DDP-wrapped MoE model."""
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        average_in_collective=average_in_collective,
        use_pipelined_ep_reshard=use_pipelined,
        num_ep_reshard_pipeline_chunks=num_pipeline_chunks,
    )
    module = HeterogeneousMoEModel(
        hidden_size=hidden_size,
        num_moe_experts=num_moe_experts,
        ep_size=ep_size,
        etp_size=etp_size,
    )
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        ddp_config=ddp_config,
        module=module,
    )
    return model


def _skip_if_wrong_world_size(topo):
    """Skip test if world size doesn't match topology."""
    ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 0
    if ws != topo['world_size']:
        pytest.skip(
            f"Test requires WORLD_SIZE={topo['world_size']}, got {ws}"
        )


def _run_grad_sync_test(topo, use_pipelined, average_in_collective, num_pipeline_chunks=4):
    """Core test logic for heterogeneous EP gradient sync.

    Fills expert grad buffers with 1.0 on all ranks and verifies that after
    grad sync, all ranks have the correctly reduced gradient.
    """
    _init_distributed()
    _skip_if_wrong_world_size(topo)
    _setup_heterogeneous(topo)

    rank = torch.distributed.get_rank()
    het_cfg = parallel_state.get_heterogeneous_ep_config()
    ep_size = het_cfg['local_ep_size']
    dp_size = het_cfg['dp_size']
    num_replicas = het_cfg['num_replicas']

    model = _create_ddp_model(
        hidden_size=16,
        num_moe_experts=topo['num_moe_experts'],
        ep_size=ep_size,
        etp_size=topo['etp'],
        use_pipelined=use_pipelined,
        num_pipeline_chunks=num_pipeline_chunks,
        average_in_collective=average_in_collective,
    )

    assert len(model.expert_parallel_buffers) > 0, (
        f"Rank {rank}: No expert parallel buffers found"
    )
    ep_buffer = model.expert_parallel_buffers[0]
    ep_bucket_groups = model.expert_parallel_bucket_groups

    fill_value = 1.0
    ep_buffer.grad_data.data.fill_(fill_value)

    # Expected value after sync:
    # Both modes end up with fill_value * num_replicas / dp_size
    if average_in_collective:
        # Pre-scale: num_replicas/dp_size. AVG allreduce: sum / num_replicas.
        # Result: fill * (num_replicas/dp_size) * num_replicas / num_replicas
        expected = fill_value * (num_replicas / dp_size)
    else:
        # Pre-scale: 1/dp_size. SUM allreduce: sum across num_replicas peers.
        # Result: fill * (1/dp_size) * num_replicas
        expected = fill_value * num_replicas / dp_size

    for bucket_group in ep_bucket_groups:
        bucket_group.finish_grad_sync()

    actual = ep_buffer.grad_data[0].item()
    assert abs(actual - expected) < 1e-5, (
        f"Rank {rank}: expected grad value {expected}, got {actual} "
        f"(avg_in_coll={average_in_collective}, pipelined={use_pipelined})"
    )

    all_close = torch.allclose(
        ep_buffer.grad_data,
        torch.full_like(ep_buffer.grad_data, expected),
        atol=1e-5,
    )
    assert all_close, (
        f"Rank {rank}: not all grad buffer elements match expected value {expected}"
    )

    parallel_state.destroy_model_parallel()


def _run_cross_rank_consistency_test(topo, use_pipelined, num_pipeline_chunks=4):
    """Verify that after sync, edp-eligible ranks and min-ep ranks agree.

    After gradient sync, the edp-eligible rank's gathered buffer (all experts)
    should match the min-ep replica's buffer (also all experts).
    """
    _init_distributed()
    _skip_if_wrong_world_size(topo)
    _setup_heterogeneous(topo)

    rank = torch.distributed.get_rank()
    het_cfg = parallel_state.get_heterogeneous_ep_config()
    ep_size = het_cfg['local_ep_size']

    model = _create_ddp_model(
        hidden_size=16,
        num_moe_experts=topo['num_moe_experts'],
        ep_size=ep_size,
        etp_size=topo['etp'],
        use_pipelined=use_pipelined,
        num_pipeline_chunks=num_pipeline_chunks,
        average_in_collective=False,
    )

    assert len(model.expert_parallel_buffers) > 0
    ep_buffer = model.expert_parallel_buffers[0]
    ep_bucket_groups = model.expert_parallel_bucket_groups

    # Fill with rank-specific values so reduction is non-trivial.
    ep_buffer.grad_data.data.fill_(float(rank + 1))

    for bucket_group in ep_bucket_groups:
        bucket_group.finish_grad_sync()

    # Cross-check: compare rank 0's expert grads with the first reshard rank's.
    # Use broadcast so all ranks participate (avoids send/recv deadlocks).
    first_reshard_rank = sum(topo['k'][:1]) * topo['tp'] * topo['cp']
    reshard_ep = topo['k'][1] * topo['tp'] * topo['cp'] // topo['etp']
    experts_per_reshard_rank = topo['num_moe_experts'] // reshard_ep

    # Determine the common buffer size: reshard rank's buffer size.
    reshard_buf_size = ep_buffer.grad_data.numel() if rank == first_reshard_rank else 0
    # Broadcast the size from first_reshard_rank so all ranks know it.
    size_tensor = torch.tensor([reshard_buf_size], device='cuda')
    torch.distributed.broadcast(size_tensor, src=first_reshard_rank)
    common_size = size_tensor.item()

    # Rank 0 broadcasts its first `common_size` elements (matching reshard rank's experts).
    if rank == 0:
        local_data = ep_buffer.grad_data.clone()
        bcast_buf = local_data[:common_size].contiguous()
    else:
        bcast_buf = torch.empty(common_size, dtype=ep_buffer.grad_data.dtype, device='cuda')
    torch.distributed.broadcast(bcast_buf, src=0)

    # First reshard rank compares.
    if rank == first_reshard_rank:
        local_data = ep_buffer.grad_data.clone()
        numel_per_expert = local_data.numel() // experts_per_reshard_rank
        for exp_idx in range(experts_per_reshard_rank):
            r0_slice = bcast_buf[
                exp_idx * numel_per_expert : (exp_idx + 1) * numel_per_expert
            ]
            local_slice = local_data[
                exp_idx * numel_per_expert : (exp_idx + 1) * numel_per_expert
            ]
            assert torch.allclose(r0_slice, local_slice, atol=1e-5), (
                f"Expert {exp_idx} grad mismatch between rank 0 and rank {first_reshard_rank}"
            )
    parallel_state.destroy_model_parallel()


# ── 12-GPU Tests (k=[2,4]: all replicas ep>1 — ep=2 and ep=4) ──

class TestApproachA_12GPU:
    """Approach A on 12 GPUs, all replicas ep>1."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_grad_sync_sum(self):
        _run_grad_sync_test(TOPO_12GPU, use_pipelined=False, average_in_collective=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_grad_sync_avg(self):
        _run_grad_sync_test(TOPO_12GPU, use_pipelined=False, average_in_collective=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_cross_rank_consistency(self):
        _run_cross_rank_consistency_test(TOPO_12GPU, use_pipelined=False)


class TestApproachB_12GPU:
    """Approach B on 12 GPUs, all replicas ep>1."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_grad_sync_sum(self):
        _run_grad_sync_test(TOPO_12GPU, use_pipelined=True, average_in_collective=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_grad_sync_avg(self):
        _run_grad_sync_test(TOPO_12GPU, use_pipelined=True, average_in_collective=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_cross_rank_consistency(self):
        _run_cross_rank_consistency_test(TOPO_12GPU, use_pipelined=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_grad_sync_2_chunks(self):
        _run_grad_sync_test(TOPO_12GPU, use_pipelined=True, average_in_collective=False,
                            num_pipeline_chunks=2)


# ── 8-GPU Tests (k=[1,3]: one replica ep=1, one ep=3) ──

class TestApproachA_8GPU:
    """Approach A on 8 GPUs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(120)
    def test_grad_sync_sum(self):
        _run_grad_sync_test(TOPO_8GPU, use_pipelined=False, average_in_collective=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(120)
    def test_grad_sync_avg(self):
        _run_grad_sync_test(TOPO_8GPU, use_pipelined=False, average_in_collective=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(120)
    def test_cross_rank_consistency(self):
        _run_cross_rank_consistency_test(TOPO_8GPU, use_pipelined=False)


class TestApproachB_8GPU:
    """Approach B on 8 GPUs."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(120)
    def test_grad_sync_sum(self):
        _run_grad_sync_test(TOPO_8GPU, use_pipelined=True, average_in_collective=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(120)
    def test_grad_sync_avg(self):
        _run_grad_sync_test(TOPO_8GPU, use_pipelined=True, average_in_collective=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(120)
    def test_cross_rank_consistency(self):
        _run_cross_rank_consistency_test(TOPO_8GPU, use_pipelined=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(120)
    def test_grad_sync_2_chunks(self):
        _run_grad_sync_test(TOPO_8GPU, use_pipelined=True, average_in_collective=False,
                            num_pipeline_chunks=2)


# ── 24-GPU Tests (k=[4,8]: all replicas ep>1 — ep=4 and ep=8) ──

class TestApproachA_24GPU:
    """Approach A on 24 GPUs, all replicas ep>1."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_grad_sync_sum(self):
        _run_grad_sync_test(TOPO_24GPU, use_pipelined=False, average_in_collective=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_grad_sync_avg(self):
        _run_grad_sync_test(TOPO_24GPU, use_pipelined=False, average_in_collective=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_cross_rank_consistency(self):
        _run_cross_rank_consistency_test(TOPO_24GPU, use_pipelined=False)


class TestApproachB_24GPU:
    """Approach B on 24 GPUs, all replicas ep>1."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_grad_sync_sum(self):
        _run_grad_sync_test(TOPO_24GPU, use_pipelined=True, average_in_collective=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_grad_sync_avg(self):
        _run_grad_sync_test(TOPO_24GPU, use_pipelined=True, average_in_collective=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_cross_rank_consistency(self):
        _run_cross_rank_consistency_test(TOPO_24GPU, use_pipelined=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.timeout(180)
    def test_grad_sync_2_chunks(self):
        _run_grad_sync_test(TOPO_24GPU, use_pipelined=True, average_in_collective=False,
                            num_pipeline_chunks=2)
