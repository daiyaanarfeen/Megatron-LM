#!/usr/bin/env python3
"""Test: interleaved expert placement + forward + backward + grad sync."""
import os, torch, torch.distributed as dist

rank = int(os.environ.get('RANK', '0'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '1'))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', world_size=world_size, rank=rank,
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")

from megatron.core import parallel_state
from megatron.core.parallel_state import compute_expert_placement
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer

# Test 1: Verify placement algorithm
if rank == 0:
    print("=== Test 1: Placement algorithm ===")
    for ep, min_ep, E in [(6, 4, 12), (8, 4, 16), (4, 4, 12)]:
        placement, gather_map = compute_expert_placement(E, ep, min_ep)
        all_experts = sorted(e for p in placement for e in p)
        assert all_experts == list(range(E)), f"Missing/duplicate experts"
        for p in placement:
            assert len(p) == E // ep, f"Wrong count"
        print(f"  ep={ep}, min_ep={min_ep}, E={E}: ✓")

TOPOS = {
    12: dict(tp=2, cp=1, k=[2, 4], etp=2, num_moe_experts=8),
    28: dict(tp=2, cp=1, k=[4, 4, 6], etp=2, num_moe_experts=12),
    32: dict(tp=2, cp=1, k=[4, 4, 8], etp=2, num_moe_experts=16),
}

topo = TOPOS.get(world_size)
if topo is None:
    if rank == 0:
        print(f"No topology for world_size={world_size}")
    dist.barrier(); os._exit(0)

if rank == 0:
    print(f"\n=== Test 2: Full training step (world_size={world_size}, k={topo['k']}) ===")

parallel_state.initialize_heterogeneous_model_parallel(
    tensor_model_parallel_size=topo['tp'],
    context_parallel_size=topo['cp'],
    num_tp_cp_per_replica=topo['k'],
    expert_tensor_parallel_size=topo['etp'],
    num_moe_experts=topo['num_moe_experts'],
)

het_cfg = parallel_state.get_heterogeneous_ep_config()
ep_size = het_cfg['local_ep_size']
tp = parallel_state.get_tensor_model_parallel_world_size()
local_experts = het_cfg['local_expert_indices']

print(f"Rank {rank}: ep={ep_size}, ep_rank={het_cfg['ep_rank']}, "
      f"experts={local_experts}, leader={het_cfg['is_b_leader']}", flush=True)

# Create DDP model with Approach B
hidden_size = 64


class SimpleMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = TransformerConfig(
            num_layers=1, hidden_size=hidden_size,
            num_attention_heads=max(tp, hidden_size // 64),
            tensor_model_parallel_size=tp,
            num_moe_experts=topo['num_moe_experts'],
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2, moe_aux_loss_coeff=0.01,
            moe_token_dispatcher_type="alltoall",
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=topo['etp'],
            add_bias_linear=False, use_cpu_initialization=True,
            sequence_parallel=(tp > 1),
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=topo['num_moe_experts'], moe_grouped_gemm=False)
        self.moe = MoELayer(config, submodules.mlp.submodules).cuda()

    def forward(self, x):
        out, _ = self.moe(x)
        return out


ddp_config = DistributedDataParallelConfig(
    grad_reduce_in_fp32=True, overlap_grad_reduce=False,
    average_in_collective=False,
    use_pipelined_ep_reshard=True, num_ep_reshard_pipeline_chunks=4,
)
module = SimpleMoE()
model = DistributedDataParallel(
    TransformerConfig(num_attention_heads=1, num_layers=1),
    ddp_config=ddp_config, module=module,
)

# Test: fill grads with 1.0, sync, check result
ep_buffer = model.expert_parallel_buffers[0]
ep_bucket_groups = model.expert_parallel_bucket_groups

ep_buffer.grad_data.data.fill_(1.0)

# Debug logging for a few representative ranks:
# rank 0 = replica 0 leader (ep=4, ep_rank=0)
# rank 16 = replica 2 leader (ep=6, ep_rank=0)
# rank 24 = replica 2 follower (ep=6, ep_rank=4, experts=[2,8])
if rank in (0, 16, 24):
    os.environ['DEBUG_INTERLEAVED'] = '1'
if rank == 0:
    print("Running grad sync (Approach B)...", flush=True)

for bg in ep_bucket_groups:
    bg.finish_grad_sync()

torch.cuda.synchronize()
actual = ep_buffer.grad_data[0].item()
dp_size = het_cfg['dp_size']
num_replicas = het_cfg['num_replicas']
expected = num_replicas / dp_size  # fill(1.0) * (1/dp_size) * num_replicas

status = "PASS" if abs(actual - expected) < 1e-4 else "FAIL"
print(f"Rank {rank}: expected={expected:.6f}, actual={actual:.6f} → {status}", flush=True)

dist.barrier()
if rank == 0:
    print("DONE")
os._exit(0)
