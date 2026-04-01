# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Fused intra-bucket pipelined reshard collective for heterogeneous EP.

Implements Approach B from the heterogeneous EP gradient sync design:
a single bucket's gradient buffer is internally split into K chunks,
pipelined through three stages:

    Chunk 0:  [gather]───→───[allreduce]────→────[distribute]
    Chunk 1:                 [gather]───→───[allreduce]────→────[distribute]
    Chunk 2:                                [gather]───→───[allreduce]────→────[distribute]

Gather is intra-replica all_gather. Allreduce is cross-replica on edp group.
Distribute is intra-replica all_reduce (only ep_rank=0 non-zero, others zeroed).

Three CUDA streams coordinate the pipeline:
  - gather_stream: runs all_gather collectives
  - allreduce_stream: runs edp allreduce + extra-rank no-op allreduce
  - distribute_stream: runs ep all_reduce for distribution

CUDA events synchronize dependencies between stages within and across chunks.
"""

import torch
import torch.distributed


class PipelinedReshardCollective:
    """Chunked pipeline: gather → allreduce → distribute within a single buffer.

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
        ep_rank: This rank's position within the ep group.
        is_edp_eligible: Whether this rank participates in the cross-replica allreduce.
        num_chunks: Number of pipeline chunks (K). Sweet spot: 4-8.
        reduce_op: Reduction operation for the edp allreduce (SUM or AVG).
    """

    def __init__(
        self,
        ep_group: torch.distributed.ProcessGroup,
        edp_group: torch.distributed.ProcessGroup,
        local_ep_size: int,
        ep_rank: int,
        is_edp_eligible: bool,
        num_chunks: int = 4,
        reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
    ):
        self.ep_group = ep_group
        self.edp_group = edp_group
        self.local_ep_size = local_ep_size
        self.ep_rank = ep_rank
        self.is_edp_eligible = is_edp_eligible
        self.num_chunks = num_chunks
        self.reduce_op = reduce_op

        device = torch.cuda.current_device()
        self.gather_stream = torch.cuda.Stream(device=device)
        self.allreduce_stream = torch.cuda.Stream(device=device)
        self.distribute_stream = torch.cuda.Stream(device=device)

    def _compute_chunk_sizes(self, L_total):
        """Compute chunk sizes aligned to local_ep_size."""
        K = self.num_chunks
        total_chunk_size = (L_total + K - 1) // K
        # Round up to multiple of local_ep_size for clean division.
        total_chunk_size = (
            (total_chunk_size + self.local_ep_size - 1)
            // self.local_ep_size
            * self.local_ep_size
        )
        local_chunk_size = total_chunk_size // self.local_ep_size
        return total_chunk_size, local_chunk_size

    def execute(self, grad_data: torch.Tensor, gather_buffer: torch.Tensor):
        """Run the pipelined reshard collective for ranks with ep > 1.

        All ranks in the ep group call this. edp-eligible ranks must provide
        gather_buffer; extra ranks pass None.

        Args:
            grad_data: This rank's local gradient buffer (size L_local).
            gather_buffer: Pre-allocated buffer on edp-eligible ranks
                (size L_total = local_ep_size * L_local). None on extra ranks.
        """
        L_local = grad_data.numel()
        L_total = L_local * self.local_ep_size
        K = self.num_chunks
        total_chunk_size, local_chunk_size = self._compute_chunk_sizes(L_total)

        default_stream = torch.cuda.current_stream()

        gather_done = [torch.cuda.Event() for _ in range(K)]
        allreduce_done = [torch.cuda.Event() for _ in range(K)]
        distribute_done = [torch.cuda.Event() for _ in range(K)]

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
            total_start = i * total_chunk_size
            total_end = total_start + actual_total

            # ── Stage 1: Gather ──
            self.gather_stream.wait_stream(default_stream)
            if i > 0:
                self.gather_stream.wait_event(distribute_done[i - 1])

            with torch.cuda.stream(self.gather_stream):
                if self.is_edp_eligible:
                    gather_chunk_buf = gather_buffer[total_start:total_end]
                    gather_list = [
                        gather_chunk_buf[j * actual_local : (j + 1) * actual_local]
                        for j in range(self.local_ep_size)
                    ]
                else:
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

            # ── Stage 2: Allreduce on edp group ──
            # All ranks must call allreduce (extra ranks do no-op on single-rank
            # edp group) to avoid racing ahead to stage 3.
            self.allreduce_stream.wait_event(gather_done[i])
            if i > 0:
                self.allreduce_stream.wait_event(allreduce_done[i - 1])

            with torch.cuda.stream(self.allreduce_stream):
                if self.is_edp_eligible:
                    gather_chunk_buf = gather_buffer[total_start:total_end]
                    torch.distributed.all_reduce(
                        gather_chunk_buf, op=self.reduce_op, group=self.edp_group
                    )
                else:
                    torch.distributed.all_reduce(
                        local_chunk, op=self.reduce_op, group=self.edp_group
                    )
            allreduce_done[i].record(self.allreduce_stream)

            # ── Stage 3: Distribute via all_reduce on ep group ──
            # Only ep_rank=0 keeps gather_chunk_buf; others use zeros.
            self.distribute_stream.wait_event(allreduce_done[i])

            with torch.cuda.stream(self.distribute_stream):
                if self.ep_rank == 0 and self.is_edp_eligible:
                    dist_buf = gather_buffer[total_start:total_end]
                else:
                    dist_buf = torch.zeros(
                        actual_total, dtype=grad_data.dtype, device=grad_data.device
                    )
                torch.distributed.all_reduce(
                    dist_buf, op=torch.distributed.ReduceOp.SUM, group=self.ep_group
                )
                local_chunk.copy_(
                    dist_buf[self.ep_rank * actual_local : (self.ep_rank + 1) * actual_local]
                )
            distribute_done[i].record(self.distribute_stream)

        if num_actual_chunks > 0:
            default_stream.wait_event(distribute_done[num_actual_chunks - 1])

    def execute_min_ep(self, grad_data: torch.Tensor, gather_buffer: torch.Tensor):
        """Chunked gather → allreduce → distribute for ranks with ep > 1.

        Same 3-stage pipeline as execute(), used by all replicas with ep > 1
        (including the min-ep replica when min_ep > 1).

        Args:
            grad_data: This rank's local gradient buffer (size L_local).
            gather_buffer: Pre-allocated gather buffer (size L_total).
        """
        # When local_ep_size > 1, min-ep replicas also need gather/distribute.
        # Delegate to the same execute() method.
        self.execute(grad_data, gather_buffer)

    def execute_ep1(self, grad_data: torch.Tensor):
        """Chunked allreduce for ep=1 replicas (no gather/distribute needed).

        Args:
            grad_data: This rank's local gradient buffer (size L_total = all experts).
        """
        L_total = grad_data.numel()
        K = self.num_chunks
        total_chunk_size, _ = self._compute_chunk_sizes(L_total)

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
