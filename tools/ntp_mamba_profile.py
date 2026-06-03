#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Benchmark Megatron Mamba layers with uniform TP or opt-in NTP.

This intentionally lives under tools/ so NTP can be exercised without changing
Megatron library files. It uses real MCore MambaModel forward/backward/DDP
training steps with random token batches, not synthetic kernels.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mamba import MambaModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["uniform", "ntp"], required=True)
    parser.add_argument("--base-tp", type=int, default=4)
    parser.add_argument("--reduced-tp", type=int, default=2)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--mamba-head-dim", type=int, default=64)
    parser.add_argument("--mamba-num-heads", type=int, default=16)
    parser.add_argument("--mamba-num-groups", type=int, default=4)
    parser.add_argument("--mamba-state-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--reduced-micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--bucket-size", type=int, default=200_000_000)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--trace-ranks", default="0")
    parser.add_argument("--profile-wait", type=int, default=1)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-active", type=int, default=2)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--sync-step-timing", action="store_true")
    return parser.parse_args()


def _new_group(group_ranks, rank):
    group = dist.new_group(ranks=group_ranks)
    return group if rank in group_ranks else None


def init_uniform(base_tp):
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=base_tp,
        pipeline_model_parallel_size=1,
        create_gloo_process_groups=False,
    )


def init_packed_ntp(base_tp, reduced_tp):
    """Build packed TP{reduced}+TP{base} process groups with no spare processes."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected = base_tp + reduced_tp
    if world_size != expected:
        raise RuntimeError(f"NTP packed layout requires WORLD_SIZE={expected}, got {world_size}")

    reduced_ranks = list(range(reduced_tp))
    healthy_ranks = list(range(reduced_tp, reduced_tp + base_tp))
    healthy_core = healthy_ranks[:reduced_tp]
    healthy_extra = healthy_ranks[reduced_tp:]

    tp_domains = [reduced_ranks, healthy_ranks]
    dp_domains = [[reduced_ranks[i], healthy_core[i]] for i in range(reduced_tp)]
    dp_domains.extend([[rank_] for rank_ in healthy_extra])

    tp_groups = {}
    for ranks in tp_domains:
        group = _new_group(ranks, rank)
        for group_rank in ranks:
            tp_groups[group_rank] = (group, ranks)

    dp_groups = {}
    for ranks in dp_domains:
        group = _new_group(ranks, rank)
        for group_rank in ranks:
            dp_groups[group_rank] = (group, ranks)

    singleton_groups = {}
    for group_rank in range(world_size):
        group = _new_group([group_rank], rank)
        singleton_groups[group_rank] = (group, [group_rank])

    tp_group, tp_ranks = tp_groups[rank]
    dp_group, dp_ranks = dp_groups[rank]
    singleton_group, singleton_ranks = singleton_groups[rank]
    if rank in reduced_ranks:
        tp_rank = rank
        tp_size = reduced_tp
        dp_rank = 0
    else:
        tp_rank = rank - reduced_tp
        tp_size = base_tp
        dp_rank = 1

    parallel_state._TENSOR_MODEL_PARALLEL_GROUP = tp_group
    parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = tp_ranks
    parallel_state._MODEL_PARALLEL_GROUP = tp_group
    parallel_state._MODEL_PARALLEL_GLOBAL_RANKS = tp_ranks
    parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = singleton_group
    parallel_state._PIPELINE_GLOBAL_RANKS = singleton_ranks
    parallel_state._CONTEXT_PARALLEL_GROUP = singleton_group
    parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS = singleton_ranks
    parallel_state._HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = [singleton_group]
    parallel_state._TENSOR_AND_CONTEXT_PARALLEL_GROUP = tp_group
    parallel_state._DATA_PARALLEL_GROUP = dp_group
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP = dp_group
    parallel_state._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = dp_group
    parallel_state._DATA_PARALLEL_GLOBAL_RANKS = dp_ranks
    parallel_state._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = dp_ranks
    parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = dist.group.WORLD
    parallel_state._EMBEDDING_GROUP = singleton_group
    parallel_state._EMBEDDING_GLOBAL_RANKS = singleton_ranks
    parallel_state._POSITION_EMBEDDING_GROUP = singleton_group
    parallel_state._POSITION_EMBEDDING_GLOBAL_RANKS = singleton_ranks
    parallel_state._EXPERT_MODEL_PARALLEL_GROUP = singleton_group
    parallel_state._EXPERT_MODEL_PARALLEL_RANKS = singleton_ranks
    parallel_state._EXPERT_TENSOR_PARALLEL_GROUP = tp_group
    parallel_state._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = tp_group
    parallel_state._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = tp_group
    parallel_state._EXPERT_DATA_PARALLEL_GROUP = dp_group
    parallel_state._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = dp_group
    parallel_state._INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = dp_group

    parallel_state.set_tensor_model_parallel_world_size(tp_size)
    parallel_state.set_tensor_model_parallel_rank(tp_rank)
    parallel_state.set_pipeline_model_parallel_world_size(1)
    parallel_state.set_pipeline_model_parallel_rank(0)
    parallel_state.set_data_parallel_rank(dp_rank)
    parallel_state._set_global_memory_buffer()


def _ntp_map_one_param(param, ntp_config, num_shards):
    from megatron.core.distributed.nonuniform_tp import _ntp_get_non_active_ranks

    dp_rank = parallel_state.get_data_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if _ntp_get_non_active_ranks(ntp_config, dp_rank, cp_rank, pp_rank) is not None:
        return False
    if not getattr(param, "tensor_model_parallel", False) or not hasattr(param, "partition_dim"):
        return False

    reduced_tp = ntp_config.tp_base - ntp_config.tp_spares
    shard_ids = torch.arange(num_shards)
    sync_partitions = list(shard_ids.chunk(reduced_tp))
    comp_partitions = sync_partitions + [
        torch.empty(int(len(shard_ids) / ntp_config.tp_base), dtype=torch.int)
        for _ in range(ntp_config.tp_spares)
    ]
    comp_2_sync = [[] for _ in range(ntp_config.tp_base)]
    sync_idx = 0
    for spare_idx in range(reduced_tp, ntp_config.tp_base):
        for shard_idx in range(len(comp_partitions[spare_idx])):
            comp_partitions[spare_idx][shard_idx] = comp_partitions[sync_idx][-1]
            comp_partitions[sync_idx] = comp_partitions[sync_idx][:-1]
            comp_2_sync[spare_idx].append(sync_idx)
            sync_idx = (sync_idx + 1) % reduced_tp

    param_splits = [
        torch.bincount(torch.tensor(c2s, dtype=torch.int), minlength=ntp_config.tp_base)
        for c2s in comp_2_sync
    ]
    partition_dim = param.partition_dim
    shard_size = int(param.shape[partition_dim] * ntp_config.tp_base / len(shard_ids))
    param.send_splits = [(split * shard_size).tolist() for split in param_splits]
    param.recv_splits = [
        [param.send_splits[src][dst] for src in range(len(param.send_splits))]
        for dst in range(ntp_config.tp_base)
    ]
    return True


def apply_mamba_ntp_mappings(model, ntp_config):
    from megatron.core.distributed.nonuniform_tp import ntp_map

    mapped = 0
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    for module in model.modules():
        if module.__class__.__name__ == "MambaMixer":
            for child_name in ("in_proj", "out_proj"):
                child = getattr(module, child_name, None)
                if child is not None:
                    for param in child.parameters(recurse=True):
                        if hasattr(param, "partition_dim"):
                            full = int(param.shape[param.partition_dim]) * tp_size
                            mapped += int(_ntp_map_one_param(param, ntp_config, full))
            for name in ("dt_bias", "A_log", "D"):
                param = getattr(module, name, None)
                if param is not None and hasattr(param, "partition_dim"):
                    full = int(param.shape[param.partition_dim]) * tp_size
                    mapped += int(_ntp_map_one_param(param, ntp_config, full))
            if hasattr(module, "conv1d"):
                for param in module.conv1d.parameters(recurse=True):
                    if hasattr(param, "partition_dim"):
                        full = int(param.shape[param.partition_dim]) * tp_size
                        mapped += int(_ntp_map_one_param(param, ntp_config, full))
            norm = getattr(module, "norm", None)
            if norm is not None:
                for param in norm.parameters(recurse=True):
                    if hasattr(param, "partition_dim"):
                        full = int(param.shape[param.partition_dim]) * tp_size
                        mapped += int(_ntp_map_one_param(param, ntp_config, full))
    if hasattr(model, "embedding") and hasattr(model.embedding, "word_embeddings"):
        ntp_map(model.embedding.word_embeddings, ntp_config, model.vocab_size)
    if hasattr(model, "output_layer"):
        ntp_map(model.output_layer, ntp_config, model.vocab_size)
    return mapped


def build_model(args, ntp_config):
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    config = TransformerConfig(
        num_layers=args.layers,
        hidden_size=args.hidden_size,
        num_attention_heads=max(1, args.hidden_size // args.mamba_head_dim),
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        sequence_parallel=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        perform_initialization=True,
        normalization="RMSNorm",
        mamba_state_dim=args.mamba_state_dim,
        mamba_head_dim=args.mamba_head_dim,
        mamba_num_heads=args.mamba_num_heads,
        mamba_num_groups=args.mamba_num_groups,
        use_mamba_mem_eff_path=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    model = MambaModel(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_len,
        hybrid_layer_pattern="M" * args.layers,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="none",
        pg_collection=pg_collection,
    ).cuda()
    model.bfloat16()
    for param in model.parameters():
        tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
    mapped = 0
    if ntp_config is not None:
        mapped = apply_mamba_ntp_mappings(model, ntp_config)
    return model, config, pg_collection, mapped


def local_micro_batch_size(args):
    if args.mode == "ntp" and parallel_state.get_tensor_model_parallel_world_size() != args.base_tp:
        return args.reduced_micro_batch_size
    return args.micro_batch_size


def make_batch(args, device):
    dp_rank = parallel_state.get_data_parallel_rank()
    gen = torch.Generator(device=device)
    gen.manual_seed(12345 + dp_rank)
    mb = local_micro_batch_size(args)
    tokens = torch.randint(args.vocab_size, (mb, args.seq_len), generator=gen, device=device)
    labels = torch.randint(args.vocab_size, (mb, args.seq_len), generator=gen, device=device)
    position_ids = torch.arange(args.seq_len, device=device).unsqueeze(0).expand(mb, -1)
    return tokens, labels, position_ids


def train_step(ddp_model, optimizer, args):
    optimizer.zero_grad(set_to_none=True)
    ddp_model.zero_grad_buffer()
    total_loss = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    for _ in range(args.grad_accum_steps):
        tokens, labels, position_ids = make_batch(args, torch.cuda.current_device())
        losses = ddp_model(tokens, position_ids, None, labels=labels)
        loss = losses.float().mean() / args.grad_accum_steps
        loss.backward()
        total_loss += loss.detach()
    ddp_model.finish_grad_sync()
    optimizer.step()
    return float(total_loss.item())


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    if args.mode == "uniform":
        init_uniform(args.base_tp)
        ntp_config = None
        from megatron.core.distributed import DistributedDataParallel
        ddp_cls = DistributedDataParallel
    else:
        from megatron.core.distributed.nonuniform_tp import (
            NonuniformTPConfig,
            NonuniformTPDistributedDataParallel,
            NonuniformTPParamAndGradBuffer,
        )
        init_packed_ntp(args.base_tp, args.reduced_tp)
        ntp_config = NonuniformTPConfig(
            tp_base=args.base_tp,
            tp_spares=args.base_tp - args.reduced_tp,
            num_reduced_tp_dp_ranks=1,
            non_active_ranks_per_dp={(0, 0, 0): list(range(args.reduced_tp, args.base_tp))},
        )
        original_ntp_buffer_init = NonuniformTPParamAndGradBuffer.__init__

        def compat_ntp_buffer_init(self, *buffer_args, **buffer_kwargs):
            buffer_kwargs.pop("param_layout", None)
            return original_ntp_buffer_init(self, *buffer_args, **buffer_kwargs)

        NonuniformTPParamAndGradBuffer.__init__ = compat_ntp_buffer_init
        ddp_cls = lambda **kwargs: NonuniformTPDistributedDataParallel(
            **kwargs, ntp_config=ntp_config
        )

    model_parallel_cuda_manual_seed(1234)
    model, config, pg_collection, mapped = build_model(args, ntp_config)
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=True,
        bucket_size=args.bucket_size,
        average_in_collective=True,
    )
    ddp_model = ddp_cls(
        config=config,
        ddp_config=ddp_config,
        module=model,
        disable_bucketing=False,
        pg_collection=pg_collection,
    )
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    trace_ranks = {int(x) for x in args.trace_ranks.split(",") if x}
    profiler = None
    if rank in trace_ranks:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=args.profile_wait, warmup=args.profile_warmup, active=args.profile_active
            ),
            record_shapes=args.record_shapes,
        )
        profiler.start()

    losses = []
    iter_ms = []
    for step in range(args.warmup_steps + args.steps):
        torch.cuda.synchronize()
        if args.sync_step_timing:
            dist.barrier()
        start = time.perf_counter()
        loss = train_step(ddp_model, optimizer, args)
        torch.cuda.synchronize()
        if args.sync_step_timing:
            dist.barrier()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if profiler is not None:
            profiler.step()
        if step >= args.warmup_steps:
            losses.append(loss)
            iter_ms.append(elapsed_ms)
            if rank == 0:
                print(f"iteration {step - args.warmup_steps + 1}: loss={loss:.6f} time_ms={elapsed_ms:.3f}", flush=True)

    if profiler is not None:
        profiler.stop()
        profiler.export_chrome_trace(str(args.trace_dir / f"rank-{rank}.json"))

    summary = {
        "mode": args.mode,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "tp_size": parallel_state.get_tensor_model_parallel_world_size(),
        "tp_rank": parallel_state.get_tensor_model_parallel_rank(),
        "dp_rank": parallel_state.get_data_parallel_rank(),
        "local_micro_batch_size": local_micro_batch_size(args),
        "local_samples_per_step": local_micro_batch_size(args) * args.grad_accum_steps,
        "mapped_mamba_params": mapped,
        "loss_last": losses[-1] if losses else None,
        "iter_ms": iter_ms,
        "iter_ms_avg": statistics.mean(iter_ms) if iter_ms else None,
        "iter_ms_median": statistics.median(iter_ms) if iter_ms else None,
    }
    (args.trace_dir / f"summary-rank-{rank}.json").write_text(json.dumps(summary, indent=2))
    if rank == 0:
        print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
