# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Fused intra-bucket pipelined reshard collective for heterogeneous EP.

Approach B: uses NVSHMEM P2P put for gather/scatter stages (NVLink),
NCCL allreduce for cross-replica reduction. NVSHMEM is world-scoped
(PE = global rank), initialized in parallel_state.py.

Stages 1 and 3 use nvshmem.core.rma.put() with sliced symmetric tensors
for offset-based transfers. Sync between stages uses NCCL barrier on the
ep group (no NVSHMEM collectives, avoiding cross-library deadlocks).
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
    """NVSHMEM-based gather → allreduce → scatter for heterogeneous EP.

    Args:
        ep_group: Expert model parallel process group (intra-replica).
        edp_group: Expert data parallel process group (cross-replica).
        local_ep_size: Number of EP ranks in this replica.
        ep_rank: This rank's position within the ep group.
        is_edp_eligible: Whether this rank participates in cross-replica allreduce.
        num_chunks: Number of pipeline chunks (K).
        reduce_op: Reduction operation for the edp allreduce.
    """

    def __init__(self, ep_group, edp_group, local_ep_size, ep_rank,
                 is_edp_eligible, num_chunks=4, reduce_op=None):
        self.ep_group = ep_group
        self.edp_group = edp_group
        self.local_ep_size = local_ep_size
        self.ep_rank = ep_rank
        self.is_edp_eligible = is_edp_eligible
        self.num_chunks = num_chunks
        self.reduce_op = reduce_op or torch.distributed.ReduceOp.SUM

        self.allreduce_stream = torch.cuda.Stream(device=torch.cuda.current_device())

        # NVSHMEM state — set by set_nvshmem_state().
        self._use_nvshmem = False
        self._nvshmem_stream = None
        self._ep_peer_pes = None
        self._gather_symm_buf = None
        self._local_symm_buf = None

    def set_nvshmem_state(self, het_ep_config: dict):
        """Adopt pre-initialized NVSHMEM state from het EP config."""
        if het_ep_config.get('nvshmem_initialized', False) and self.local_ep_size > 1:
            self._use_nvshmem = True
            self._nvshmem_stream = het_ep_config['nvshmem_stream']
            self._ep_peer_pes = het_ep_config['ep_peer_pes']
            self._gather_symm_buf = het_ep_config['nvshmem_gather_symm_buf']
            self._local_symm_buf = het_ep_config['nvshmem_local_symm_buf']
            logger.info(
                f"PipelinedReshardCollective: NVSHMEM active "
                f"(ep_rank={self.ep_rank}, ep_size={self.local_ep_size}, "
                f"ep_peer_pes={self._ep_peer_pes})"
            )

    def execute(self, grad_data: torch.Tensor, gather_buffer: Optional[torch.Tensor]):
        """Run gather → allreduce → scatter for ranks with ep > 1."""
        if not self._use_nvshmem:
            self._execute_nccl_fallback(grad_data, gather_buffer)
            return

        L_local = grad_data.numel()
        elem_size = grad_data.element_size()
        local_bytes = L_local * elem_size
        total_bytes = local_bytes * self.local_ep_size
        ep_rank0_pe = self._ep_peer_pes[0]

        # Get NVSHMEM stream as PyTorch ExternalStream for event sync.
        _, stream_ptr = self._nvshmem_stream.__cuda_stream__()
        nvshmem_torch_stream = torch.cuda.ExternalStream(stream_ptr)
        default_stream = torch.cuda.current_stream()

        # ── Stage 1: Gather via NVSHMEM put ──
        # Every rank: copy local grads into _local_symm_buf, then put to
        # ep_rank=0's _gather_symm_buf at the correct offset.
        src = self._local_symm_buf[:local_bytes].view(grad_data.dtype)
        src.copy_(grad_data)

        dst_offset = self.ep_rank * local_bytes
        dst = self._gather_symm_buf[dst_offset:dst_offset + local_bytes].view(grad_data.dtype)

        nvshmem.core.put(dst, src, ep_rank0_pe, stream=self._nvshmem_stream)
        nvshmem.core.quiet(stream=self._nvshmem_stream)
        nvshmem_torch_stream.synchronize()

        # Sync ep group: ep_rank=0 waits for all peers' puts to arrive.
        torch.distributed.barrier(group=self.ep_group)

        # ep_rank=0: copy gathered data from symm buf to regular gather_buffer.
        if self.ep_rank == 0:
            gathered = self._gather_symm_buf[:total_bytes].view(grad_data.dtype)
            gather_buffer[:L_local * self.local_ep_size].copy_(gathered)

        # ── Stage 2: Allreduce on edp group (NCCL only) ──
        allreduce_event = torch.cuda.Event()
        with torch.cuda.stream(self.allreduce_stream):
            if self.is_edp_eligible:
                ar_buf = gather_buffer[:L_local * self.local_ep_size]
                torch.distributed.all_reduce(ar_buf, op=self.reduce_op, group=self.edp_group)
            else:
                torch.distributed.all_reduce(
                    grad_data, op=self.reduce_op, group=self.edp_group
                )
        allreduce_event.record(self.allreduce_stream)
        default_stream.wait_event(allreduce_event)

        # ── Stage 3: Scatter via NVSHMEM put ──
        # ep_rank=0: copy allreduced gather_buffer back to symm buf, then
        # put each peer's slice to their _local_symm_buf.
        if self.ep_rank == 0:
            self._gather_symm_buf[:total_bytes].view(grad_data.dtype).copy_(
                gather_buffer[:L_local * self.local_ep_size]
            )
            for peer in range(self.local_ep_size):
                peer_pe = self._ep_peer_pes[peer]
                src_off = peer * local_bytes
                src_view = self._gather_symm_buf[src_off:src_off + local_bytes]
                dst_view = self._local_symm_buf[:local_bytes]
                nvshmem.core.put(dst_view, src_view, peer_pe, stream=self._nvshmem_stream)
            nvshmem.core.quiet(stream=self._nvshmem_stream)
            nvshmem_torch_stream.synchronize()

        # Sync ep group: all peers wait for scatter to complete.
        torch.distributed.barrier(group=self.ep_group)

        # Copy received data from symm buf to grad_data.
        grad_data.copy_(self._local_symm_buf[:local_bytes].view(grad_data.dtype))

    def execute_ep1(self, grad_data: torch.Tensor):
        """Simple allreduce for ep=1 replicas."""
        torch.distributed.all_reduce(grad_data, op=self.reduce_op, group=self.edp_group)

    def _execute_nccl_fallback(self, grad_data, gather_buffer):
        """NCCL-only fallback (same as Approach A)."""
        L_local = grad_data.numel()
        chunk_size = L_local

        if self.local_ep_size == 1:
            torch.distributed.all_reduce(grad_data, op=self.reduce_op, group=self.edp_group)
            return

        # Stage 1: all_gather on ep group.
        if self.is_edp_eligible:
            gather_list = list(gather_buffer[:L_local * self.local_ep_size].chunk(self.local_ep_size))
        else:
            gather_list = [
                torch.empty(chunk_size, dtype=grad_data.dtype, device=grad_data.device)
                for _ in range(self.local_ep_size)
            ]
        torch.distributed.all_gather(gather_list, grad_data, group=self.ep_group)

        # Stage 2: allreduce on edp group.
        if self.is_edp_eligible:
            ar_buf = gather_buffer[:L_local * self.local_ep_size]
            torch.distributed.all_reduce(ar_buf, op=self.reduce_op, group=self.edp_group)
        else:
            torch.distributed.all_reduce(grad_data, op=self.reduce_op, group=self.edp_group)

        # Stage 3: distribute via all_reduce on ep group.
        ep_r = self.ep_rank
        if ep_r == 0 and self.is_edp_eligible:
            dist_buf = gather_buffer[:L_local * self.local_ep_size]
        else:
            dist_buf = torch.zeros(
                L_local * self.local_ep_size, dtype=grad_data.dtype, device=grad_data.device
            )
        torch.distributed.all_reduce(dist_buf, op=torch.distributed.ReduceOp.SUM, group=self.ep_group)
        grad_data.copy_(dist_buf[ep_r * chunk_size : (ep_r + 1) * chunk_size])
