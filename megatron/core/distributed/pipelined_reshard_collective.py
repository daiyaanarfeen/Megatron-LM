# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Fused intra-bucket pipelined reshard collective for heterogeneous EP.

Implements Approach B from the heterogeneous EP gradient sync design:
a single bucket's gradient buffer is internally split into K chunks,
pipelined through three stages:

    Chunk 0:  [gather]───→───[allreduce]────→────[scatter]
    Chunk 1:                 [gather]───→───[allreduce]────→────[scatter]
    Chunk 2:                                [gather]───→───[allreduce]────→────[scatter]

Stages 1 (gather) and 3 (scatter) use NVSHMEM P2P put operations over
NVLink — single kernel launch per put, no NCCL overhead. Stage 2 uses
NCCL allreduce for the cross-replica reduction.

NVSHMEM provides symmetric memory mapped across all PEs in the ep group,
enabling direct GPU-to-GPU writes over NVLink (including multi-node NVLink
in NVL72 domains).

Two CUDA streams coordinate the pipeline:
  - default stream: runs NVSHMEM put operations for gather/scatter
  - allreduce_stream: runs NCCL allreduce

CUDA events + nvshmem barriers synchronize between stages.
"""

import logging
from typing import Optional

import torch
import torch.distributed

try:
    import nvshmem.core

    HAVE_NVSHMEM = True
except ImportError:
    HAVE_NVSHMEM = False

logger = logging.getLogger(__name__)

class PipelinedReshardCollective:
    """Chunked pipeline: gather → allreduce → scatter within a single buffer.

    Uses NVSHMEM for gather/scatter (NVLink P2P put) and NCCL only for the
    cross-replica allreduce.

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
        self.allreduce_stream = torch.cuda.Stream(device=device)

        # NVSHMEM state — read from het EP config (initialized in parallel_state).
        self._use_nvshmem = False
        self._nvshmem_stream = None
        self._ep_peer_pes = None
        self._gather_symm_buf = None
        self._local_symm_buf = None

    def set_nvshmem_state(self, het_ep_config: dict):
        """Adopt pre-initialized NVSHMEM state from the het EP config.

        Called by set_heterogeneous_ep_config in param_and_grad_buffer.py
        after NVSHMEM is initialized in parallel_state.py.
        """
        if het_ep_config.get('nvshmem_initialized', False) and self.local_ep_size > 1:
            self._use_nvshmem = True
            self._nvshmem_stream = het_ep_config['nvshmem_stream']
            self._ep_peer_pes = het_ep_config['ep_peer_pes']
            logger.info(
                f"PipelinedReshardCollective: using NVSHMEM "
                f"(ep_rank={self.ep_rank}, ep_size={self.local_ep_size}, "
                f"ep_peer_pes={self._ep_peer_pes})"
            )

    def _ensure_symm_buffers(self, gather_numel: int, local_numel: int, dtype: torch.dtype):
        """Allocate or resize NVSHMEM symmetric buffers."""
        elem_size = torch.tensor([], dtype=dtype).element_size()

        # Gather buffer: ep_rank=0 needs space for all ep peers' data.
        gather_bytes = gather_numel * elem_size
        if self._gather_symm_buf is None or self._gather_symm_buf.numel() < gather_bytes:
            if self._gather_symm_buf is not None:
                nvshmem.core.interop.torch.free_tensor(self._gather_symm_buf)
            self._gather_symm_buf = nvshmem.core.interop.torch.bytetensor(
                (gather_bytes,), dtype=torch.uint8
            )
            self._gather_symm_buf.zero_()

        # Local buffer: each rank needs space for its own data (for scatter receive).
        local_bytes = local_numel * elem_size
        if self._local_symm_buf is None or self._local_symm_buf.numel() < local_bytes:
            if self._local_symm_buf is not None:
                nvshmem.core.interop.torch.free_tensor(self._local_symm_buf)
            self._local_symm_buf = nvshmem.core.interop.torch.bytetensor(
                (local_bytes,), dtype=torch.uint8
            )
            self._local_symm_buf.zero_()

        torch.cuda.synchronize()
        nvshmem.core.barrier_all(stream=self._nvshmem_stream)

    def _get_gather_view(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """View of gather symmetric buffer as the given dtype."""
        elem_size = torch.tensor([], dtype=dtype).element_size()
        return self._gather_symm_buf[: numel * elem_size].view(dtype)[:numel]

    def _get_local_view(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """View of local symmetric buffer as the given dtype."""
        elem_size = torch.tensor([], dtype=dtype).element_size()
        return self._local_symm_buf[: numel * elem_size].view(dtype)[:numel]

    def execute(self, grad_data: torch.Tensor, gather_buffer: Optional[torch.Tensor]):
        """Run the pipelined reshard collective for ranks with ep > 1.

        Args:
            grad_data: This rank's local gradient buffer (size L_local).
            gather_buffer: Pre-allocated buffer on edp-eligible ranks
                (size L_total = local_ep_size * L_local). None on extra ranks.
        """
        if not self._use_nvshmem:
            self._execute_nccl_fallback(grad_data, gather_buffer)
            return

        L_local = grad_data.numel()
        L_total = L_local * self.local_ep_size
        K = self.num_chunks
        total_chunk = (L_total + K - 1) // K
        total_chunk = (
            (total_chunk + self.local_ep_size - 1)
            // self.local_ep_size
            * self.local_ep_size
        )
        local_chunk = total_chunk // self.local_ep_size

        # Ensure symmetric buffers are allocated.
        self._ensure_symm_buffers(L_total, L_local, grad_data.dtype)

        default_stream = torch.cuda.current_stream()
        allreduce_done = [torch.cuda.Event() for _ in range(K)]
        gather_done = [torch.cuda.Event() for _ in range(K)]

        # Get PyTorch stream wrapper for NVSHMEM stream.
        _, nvshmem_stream_ptr = self._nvshmem_stream.__cuda_stream__()
        torch_nvshmem_stream = torch.cuda.ExternalStream(nvshmem_stream_ptr)

        num_actual = 0
        for i in range(K):
            l_start = i * local_chunk
            l_end = min(l_start + local_chunk, L_local)
            if l_end <= l_start:
                break
            num_actual = i + 1
            actual_local = l_end - l_start
            actual_total = actual_local * self.local_ep_size

            local_slice = grad_data[l_start:l_end]
            t_start = i * total_chunk
            t_end = t_start + actual_total

            # ── Stage 1: Gather via NVSHMEM put ──
            # Each ep rank puts its local_slice into ep_rank=0's gather buffer
            # at the offset corresponding to its ep_rank.
            if i > 0:
                default_stream.wait_event(allreduce_done[i - 1])

            gather_view = self._get_gather_view(actual_total, grad_data.dtype)
            dest_offset = self.ep_rank * actual_local
            dest_slice = gather_view[dest_offset : dest_offset + actual_local]

            # Copy local grad data into symmetric send buffer, then put.
            local_symm = self._get_local_view(actual_local, grad_data.dtype)
            local_symm.copy_(local_slice)

            # Put to ep_rank=0's gather buffer using global PE.
            ep_rank0_pe = self._ep_peer_pes[0]
            nvshmem.core.put(
                dest_slice,      # destination on ep_rank=0 (symmetric)
                local_symm,      # source (symmetric)
                ep_rank0_pe,     # dest PE = global rank of ep_rank 0
                stream=self._nvshmem_stream,
            )
            nvshmem.core.quiet(stream=self._nvshmem_stream)
            # Global barrier: all world ranks sync. Safe because all ranks
            # execute the same 3-stage sequence (NVSHMEM is world-scoped).
            nvshmem.core.barrier_all(stream=self._nvshmem_stream)
            torch_nvshmem_stream.synchronize()
            gather_done[i].record(default_stream)

            # ── Stage 2: Allreduce on edp group (NCCL, edp-eligible only) ──
            self.allreduce_stream.wait_event(gather_done[i])
            with torch.cuda.stream(self.allreduce_stream):
                if self.is_edp_eligible and self.ep_rank == 0:
                    allreduce_buf = gather_buffer[t_start:t_end]
                    allreduce_buf.copy_(gather_view)
                    torch.distributed.all_reduce(
                        allreduce_buf, op=self.reduce_op, group=self.edp_group
                    )
                    gather_view.copy_(allreduce_buf)
                elif self.is_edp_eligible:
                    allreduce_buf = gather_buffer[t_start:t_end]
                    torch.distributed.all_reduce(
                        allreduce_buf, op=self.reduce_op, group=self.edp_group
                    )
                else:
                    # Extra ranks: no-op allreduce on single-rank edp group.
                    torch.distributed.all_reduce(
                        local_slice, op=self.reduce_op, group=self.edp_group
                    )
            allreduce_done[i].record(self.allreduce_stream)

            # ── Stage 3: Scatter via NVSHMEM put ──
            default_stream.wait_event(allreduce_done[i])

            if self.ep_rank == 0:
                for peer in range(self.local_ep_size):
                    peer_pe = self._ep_peer_pes[peer]
                    src_offset = peer * actual_local
                    src_slice = gather_view[src_offset : src_offset + actual_local]
                    local_view = self._get_local_view(actual_local, grad_data.dtype)
                    nvshmem.core.put(
                        local_view,   # dest on remote PE (symmetric)
                        src_slice,    # source (symmetric)
                        peer_pe,      # dest PE = global rank
                        stream=self._nvshmem_stream,
                    )

            nvshmem.core.quiet(stream=self._nvshmem_stream)
            nvshmem.core.barrier_all(stream=self._nvshmem_stream)
            torch_nvshmem_stream.synchronize()

            local_view = self._get_local_view(actual_local, grad_data.dtype)
            local_slice.copy_(local_view)

        if num_actual > 0:
            torch.cuda.synchronize()

    def execute_ep1(self, grad_data: torch.Tensor):
        """Chunked allreduce for ep=1 replicas (no gather/scatter needed)."""
        L_total = grad_data.numel()
        K = self.num_chunks
        total_chunk = (L_total + K - 1) // K
        total_chunk = (
            (total_chunk + self.local_ep_size - 1)
            // self.local_ep_size
            * self.local_ep_size
        )

        default_stream = torch.cuda.current_stream()
        allreduce_done = [torch.cuda.Event() for _ in range(K)]

        num_actual = 0
        for i in range(K):
            start = i * total_chunk
            end = min(start + total_chunk, L_total)
            if end <= start:
                break
            num_actual = i + 1

            chunk = grad_data[start:end]

            self.allreduce_stream.wait_stream(default_stream)
            if i > 0:
                self.allreduce_stream.wait_event(allreduce_done[i - 1])

            with torch.cuda.stream(self.allreduce_stream):
                torch.distributed.all_reduce(
                    chunk, op=self.reduce_op, group=self.edp_group
                )
            allreduce_done[i].record(self.allreduce_stream)

        if num_actual > 0:
            default_stream.wait_event(allreduce_done[num_actual - 1])

    def _execute_nccl_fallback(self, grad_data: torch.Tensor, gather_buffer: Optional[torch.Tensor]):
        """NCCL-only fallback when NVSHMEM is not available."""
        L_local = grad_data.numel()
        L_total = L_local * self.local_ep_size
        K = self.num_chunks
        total_chunk = (L_total + K - 1) // K
        total_chunk = (
            (total_chunk + self.local_ep_size - 1)
            // self.local_ep_size
            * self.local_ep_size
        )
        local_chunk = total_chunk // self.local_ep_size

        default_stream = torch.cuda.current_stream()
        gather_done = [torch.cuda.Event() for _ in range(K)]
        allreduce_done = [torch.cuda.Event() for _ in range(K)]
        distribute_done = [torch.cuda.Event() for _ in range(K)]
        device = torch.cuda.current_device()
        distribute_stream = torch.cuda.Stream(device=device)

        num_actual = 0
        for i in range(K):
            l_start = i * local_chunk
            l_end = min(l_start + local_chunk, L_local)
            if l_end <= l_start:
                break
            num_actual = i + 1
            actual_local = l_end - l_start
            actual_total = actual_local * self.local_ep_size

            local_slice = grad_data[l_start:l_end]
            t_start = i * total_chunk
            t_end = t_start + actual_total

            if i > 0:
                default_stream.wait_event(distribute_done[i - 1])

            if self.is_edp_eligible:
                chunk_buf = gather_buffer[t_start:t_end]
                gather_list = list(chunk_buf.chunk(self.local_ep_size))
            else:
                gather_list = [
                    torch.empty(actual_local, dtype=grad_data.dtype, device=grad_data.device)
                    for _ in range(self.local_ep_size)
                ]
            torch.distributed.all_gather(gather_list, local_slice, group=self.ep_group)
            gather_done[i].record(default_stream)

            self.allreduce_stream.wait_event(gather_done[i])
            if i > 0:
                self.allreduce_stream.wait_event(allreduce_done[i - 1])
            with torch.cuda.stream(self.allreduce_stream):
                if self.is_edp_eligible:
                    torch.distributed.all_reduce(
                        chunk_buf, op=self.reduce_op, group=self.edp_group
                    )
                else:
                    torch.distributed.all_reduce(
                        local_slice, op=self.reduce_op, group=self.edp_group
                    )
            allreduce_done[i].record(self.allreduce_stream)

            distribute_stream.wait_event(allreduce_done[i])
            with torch.cuda.stream(distribute_stream):
                if self.ep_rank == 0 and self.is_edp_eligible:
                    dist_buf = gather_buffer[t_start:t_end]
                else:
                    dist_buf = torch.zeros(
                        actual_total, dtype=grad_data.dtype, device=grad_data.device
                    )
                torch.distributed.all_reduce(
                    dist_buf, op=torch.distributed.ReduceOp.SUM, group=self.ep_group
                )
                local_slice.copy_(
                    dist_buf[self.ep_rank * actual_local : (self.ep_rank + 1) * actual_local]
                )
            distribute_done[i].record(distribute_stream)

        if num_actual > 0:
            default_stream.wait_event(distribute_done[num_actual - 1])
