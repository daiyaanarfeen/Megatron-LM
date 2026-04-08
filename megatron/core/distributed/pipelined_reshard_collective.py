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
        self._gather_slots = None
        self._local_symm_buf = None

    def set_nvshmem_state(self, het_ep_config: dict):
        """Adopt pre-initialized NVSHMEM state from het EP config."""
        if self.local_ep_size <= 1:
            return  # ep=1 uses execute_ep1, no NVSHMEM needed.
        if not het_ep_config.get('nvshmem_initialized', False):
            raise RuntimeError(
                "PipelinedReshardCollective requires NVSHMEM but initialization failed. "
                "Check that nvshmem4py-cu12 and nvidia-nvshmem-cu12 are installed, "
                "libnvshmem_host.so.3 is on LD_LIBRARY_PATH, and nodes are in the "
                "same NVL domain."
            )
        self._use_nvshmem = True
        self._nvshmem_stream = het_ep_config['nvshmem_stream']
        self._ep_peer_pes = het_ep_config['ep_peer_pes']
        self._gather_slots = het_ep_config['nvshmem_gather_slots']
        self._local_symm_buf = het_ep_config['nvshmem_local_symm_buf']
        logger.info(
            f"PipelinedReshardCollective: NVSHMEM active "
            f"(ep_rank={self.ep_rank}, ep_size={self.local_ep_size}, "
            f"ep_peer_pes={self._ep_peer_pes})"
        )

    def execute(self, grad_data: torch.Tensor, gather_buffer: Optional[torch.Tensor]):
        """Run gather → allreduce → scatter for ranks with ep > 1."""
        if not self._use_nvshmem:
            raise RuntimeError(
                "PipelinedReshardCollective requires NVSHMEM but it is not initialized. "
                "Ensure nvshmem4py-cu12 and nvidia-nvshmem-cu12 are installed, "
                "libnvshmem_host.so.3 is on LD_LIBRARY_PATH, and all nodes are "
                "in the same NVL domain (--segment=N in sbatch)."
            )

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
        # Each rank puts its grads into ep_rank=0's gather_slot[ep_rank].
        # Each slot is a separate NVSHMEM allocation — no slicing needed.
        # Copy local grads into _local_symm_buf (NVSHMEM src).
        self._local_symm_buf[:local_bytes].view(grad_data.dtype).copy_(grad_data)
        torch.cuda.synchronize()

        # Put _local_symm_buf → ep_rank=0's gather_slot[my_ep_rank].
        my_slot = self._gather_slots[self.ep_rank]
        nvshmem.core.put(my_slot, self._local_symm_buf, ep_rank0_pe,
                         stream=self._nvshmem_stream)
        nvshmem.core.quiet(stream=self._nvshmem_stream)
        nvshmem_torch_stream.synchronize()

        # Sync: all puts complete before ep_rank=0 reads.
        torch.distributed.barrier(group=self.ep_group)

        # ep_rank=0: copy from gather_slots to regular gather_buffer.
        if self.ep_rank == 0:
            for peer in range(self.local_ep_size):
                slot = self._gather_slots[peer]
                gather_buffer[peer * L_local : (peer + 1) * L_local].copy_(
                    slot[:local_bytes].view(grad_data.dtype)
                )

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
        # ep_rank=0: copy each peer's synced grads into gather_slots, then
        # put each slot to the peer's _local_symm_buf.
        if self.ep_rank == 0:
            for peer in range(self.local_ep_size):
                slot = self._gather_slots[peer]
                slot[:local_bytes].view(grad_data.dtype).copy_(
                    gather_buffer[peer * L_local : (peer + 1) * L_local]
                )
            torch.cuda.synchronize()
            for peer in range(self.local_ep_size):
                peer_pe = self._ep_peer_pes[peer]
                nvshmem.core.put(self._local_symm_buf, self._gather_slots[peer],
                                 peer_pe, stream=self._nvshmem_stream)
            nvshmem.core.quiet(stream=self._nvshmem_stream)
            nvshmem_torch_stream.synchronize()

        # Sync: all peers wait for scatter to complete.
        torch.distributed.barrier(group=self.ep_group)

        # Copy received data from symm buf to grad_data.
        grad_data.copy_(self._local_symm_buf[:local_bytes].view(grad_data.dtype))

    def execute_ep1(self, grad_data: torch.Tensor):
        """Simple allreduce for ep=1 replicas."""
        torch.distributed.all_reduce(grad_data, op=self.reduce_op, group=self.edp_group)

