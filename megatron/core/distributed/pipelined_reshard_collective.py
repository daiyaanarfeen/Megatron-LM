# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Fused intra-bucket pipelined reshard collective for heterogeneous EP.

Implements Approach B from the heterogeneous EP gradient sync design:
a single bucket's gradient buffer is internally split into K chunks,
pipelined through three stages:

    Chunk 0:  [gather]───→───[allreduce]────→────[scatter]
    Chunk 1:                 [gather]───→───[allreduce]────→────[scatter]
    Chunk 2:                                [gather]───→───[allreduce]────→────[scatter]

Gather/scatter are intra-replica (NVLink, fast). Allreduce is cross-replica
(potentially cross-node, slower). The pipeline hides gather/scatter latency
behind the allreduce.

Three CUDA streams coordinate the pipeline:
  - gather_stream: runs gather collectives
  - allreduce_stream: runs NCCL allreduce
  - scatter_stream: runs scatter collectives

CUDA events synchronize dependencies between stages within and across chunks.
"""

import torch
import torch.distributed


class PipelinedReshardCollective:
    """Chunked pipeline: gather → allreduce → scatter within a single buffer.

    For large-scale settings with few buckets where cross-bucket pipelining
    (Approach A) provides insufficient overlap.

    The chunking is based on the TOTAL allreduce buffer size (L_total), which
    is the same on both sides of the edp allreduce:
      - Min-ep replica: L_total = grad_data.numel() (holds all experts)
      - Reshard replica: L_total = gather_buffer.numel() = local_ep_size * grad_data.numel()

    This ensures both sides allreduce matching chunk sizes.

    Args:
        ep_group: Expert model parallel process group (intra-replica).
        edp_group: Expert data parallel process group (cross-replica).
        local_ep_size: Number of EP ranks in this replica.
        is_edp_eligible: Whether this rank participates in the allreduce.
        num_chunks: Number of pipeline chunks (K). Sweet spot: 4-8.
        reduce_op: Reduction operation for allreduce (SUM or AVG).
    """

    def __init__(
        self,
        ep_group: torch.distributed.ProcessGroup,
        edp_group: torch.distributed.ProcessGroup,
        local_ep_size: int,
        is_edp_eligible: bool,
        num_chunks: int = 4,
        reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
    ):
        self.ep_group = ep_group
        self.edp_group = edp_group
        self.local_ep_size = local_ep_size
        self.is_edp_eligible = is_edp_eligible
        self.num_chunks = num_chunks
        self.reduce_op = reduce_op

        device = torch.cuda.current_device()
        self.gather_stream = torch.cuda.Stream(device=device)
        self.allreduce_stream = torch.cuda.Stream(device=device)
        self.scatter_stream = torch.cuda.Stream(device=device)

    def execute(self, grad_data: torch.Tensor, gather_buffer: torch.Tensor):
        """Run the pipelined reshard collective on a reshard replica.

        Chunks are based on L_total = local_ep_size * L_local, ensuring the
        allreduce chunk sizes match the min-ep replica's execute_min_ep().

        Args:
            grad_data: This rank's local gradient buffer (size L_local).
            gather_buffer: Pre-allocated buffer on edp-eligible ranks
                (size L_total = local_ep_size * L_local). None on extra ranks.
        """
        L_local = grad_data.numel()
        L_total = L_local * self.local_ep_size
        K = self.num_chunks
        # Chunk the total allreduce buffer; each chunk must be divisible by
        # local_ep_size so that the per-rank portion is an integer.
        total_chunk_size = ((L_total + K - 1) // K)
        # Round up to multiple of local_ep_size for clean division.
        total_chunk_size = (
            (total_chunk_size + self.local_ep_size - 1)
            // self.local_ep_size
            * self.local_ep_size
        )
        local_chunk_size = total_chunk_size // self.local_ep_size

        default_stream = torch.cuda.current_stream()

        # Pre-create events for inter-stream synchronization.
        gather_done = [torch.cuda.Event() for _ in range(K)]
        allreduce_done = [torch.cuda.Event() for _ in range(K)]
        scatter_done = [torch.cuda.Event() for _ in range(K)]

        num_actual_chunks = 0
        for i in range(K):
            local_start = i * local_chunk_size
            local_end = min(local_start + local_chunk_size, L_local)
            if local_end <= local_start:
                break
            num_actual_chunks = i + 1
            actual_local = local_end - local_start
            actual_total = actual_local * self.local_ep_size

            local_chunk = grad_data[local_start:local_end]

            # ── Stage 1: Gather ──
            self.gather_stream.wait_stream(default_stream)
            if i > 0:
                self.gather_stream.wait_event(scatter_done[i - 1])

            with torch.cuda.stream(self.gather_stream):
                if self.is_edp_eligible:
                    total_start = i * total_chunk_size
                    total_end = total_start + actual_total
                    gather_chunk_buf = gather_buffer[total_start:total_end]

                    gather_list = [
                        gather_chunk_buf[j * actual_local : (j + 1) * actual_local]
                        for j in range(self.local_ep_size)
                    ]
                else:
                    gather_chunk_buf = None
                    gather_list = [
                        torch.empty(
                            actual_local, dtype=grad_data.dtype, device=grad_data.device
                        )
                        for _ in range(self.local_ep_size)
                    ]
                torch.distributed.all_gather(
                    gather_list, local_chunk, group=self.ep_group
                )
            gather_done[i].record(self.gather_stream)

            # ── Stage 2: Allreduce ──
            self.allreduce_stream.wait_event(gather_done[i])
            if i > 0:
                self.allreduce_stream.wait_event(allreduce_done[i - 1])

            with torch.cuda.stream(self.allreduce_stream):
                if self.is_edp_eligible:
                    torch.distributed.all_reduce(
                        gather_chunk_buf, op=self.reduce_op, group=self.edp_group
                    )
            allreduce_done[i].record(self.allreduce_stream)

            # ── Stage 3: Scatter ──
            self.scatter_stream.wait_event(allreduce_done[i])

            with torch.cuda.stream(self.scatter_stream):
                if self.is_edp_eligible:
                    scatter_list = [
                        gather_chunk_buf[j * actual_local : (j + 1) * actual_local]
                        for j in range(self.local_ep_size)
                    ]
                else:
                    scatter_list = None
                torch.distributed.scatter(
                    local_chunk, scatter_list, src=0, group=self.ep_group
                )
            scatter_done[i].record(self.scatter_stream)

        # Wait for all scatter operations to complete before returning.
        if num_actual_chunks > 0:
            default_stream.wait_event(scatter_done[num_actual_chunks - 1])

    def execute_min_ep(self, grad_data: torch.Tensor):
        """Chunked allreduce for min-ep replica (no gather/scatter needed).

        Chunks the total allreduce buffer (grad_data, size L_total = all experts)
        with the SAME chunk boundaries as execute() on the reshard side.
        Since L_total is the same on both sides, and total_chunk_size is computed
        identically, the allreduce chunk sizes match.

        Args:
            grad_data: This rank's local gradient buffer (size L_total = all experts).
        """
        L_total = grad_data.numel()
        K = self.num_chunks
        total_chunk_size = ((L_total + K - 1) // K)
        # Match rounding from execute(): round up to multiple of local_ep_size.
        total_chunk_size = (
            (total_chunk_size + self.local_ep_size - 1)
            // self.local_ep_size
            * self.local_ep_size
        )

        default_stream = torch.cuda.current_stream()
        allreduce_done = [torch.cuda.Event() for _ in range(K)]

        num_actual_chunks = 0
        for i in range(K):
            start = i * total_chunk_size
            end = min(start + total_chunk_size, L_total)
            if end <= start:
                break
            num_actual_chunks = i + 1

            local_chunk = grad_data[start:end]

            self.allreduce_stream.wait_stream(default_stream)
            if i > 0:
                self.allreduce_stream.wait_event(allreduce_done[i - 1])

            with torch.cuda.stream(self.allreduce_stream):
                torch.distributed.all_reduce(
                    local_chunk, op=self.reduce_op, group=self.edp_group
                )
            allreduce_done[i].record(self.allreduce_stream)

        if num_actual_chunks > 0:
            default_stream.wait_event(allreduce_done[num_actual_chunks - 1])
