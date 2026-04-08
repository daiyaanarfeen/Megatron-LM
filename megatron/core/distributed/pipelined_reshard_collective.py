# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""NVSHMEM-based pipelined reshard collective for heterogeneous EP.

Uses NVSHMEM P2P put for gather/scatter (NVLink), NCCL allreduce for
cross-replica reduction. All ranks put concurrently to ep_rank=0's
symmetric gather slots. No NVSHMEM collectives — only put + quiet.
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
    """NVSHMEM gather → NCCL allreduce → NVSHMEM scatter."""

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

        self._use_nvshmem = False
        self._nvshmem_stream = None
        self._ep_peer_pes = None
        self._gather_slots = None
        self._local_symm_buf = None

    def set_nvshmem_state(self, het_ep_config: dict):
        if self.local_ep_size <= 1:
            return
        if not het_ep_config.get('nvshmem_initialized', False):
            raise RuntimeError(
                "PipelinedReshardCollective requires NVSHMEM but initialization failed."
            )
        self._use_nvshmem = True
        self._nvshmem_stream = het_ep_config['nvshmem_stream']
        self._ep_peer_pes = het_ep_config['ep_peer_pes']
        self._gather_slots = het_ep_config['nvshmem_gather_slots']
        self._local_symm_buf = het_ep_config['nvshmem_local_symm_buf']

    def execute(self, grad_data: torch.Tensor, gather_buffer: Optional[torch.Tensor]):
        if not self._use_nvshmem:
            raise RuntimeError("NVSHMEM not initialized for PipelinedReshardCollective")

        L = grad_data.numel()
        elem = grad_data.element_size()
        nbytes = L * elem
        ep0_pe = self._ep_peer_pes[0]

        # PyTorch stream wrapper for NVSHMEM stream (for event sync).
        _, sp = self._nvshmem_stream.__cuda_stream__()
        nv_torch = torch.cuda.ExternalStream(sp)

        # Verify slot buffers are large enough.
        if nbytes > self._local_symm_buf.numel():
            raise RuntimeError(
                f"NVSHMEM slot buffer too small: need {nbytes} bytes, "
                f"have {self._local_symm_buf.numel()}. Increase slot_size in parallel_state.py"
            )

        # ── Stage 1: Gather — all ranks put to ep_rank=0 concurrently ──
        self._local_symm_buf[:nbytes].view(grad_data.dtype).copy_(grad_data)
        my_slot = self._gather_slots[self.ep_rank]
        nvshmem.core.put(my_slot[:nbytes], self._local_symm_buf[:nbytes],
                         ep0_pe, stream=self._nvshmem_stream)
        nvshmem.core.quiet(stream=self._nvshmem_stream)
        # Event-based sync: nvshmem stream → default stream.
        gather_event = torch.cuda.Event()
        gather_event.record(nv_torch)
        torch.cuda.current_stream().wait_event(gather_event)
        # Single barrier: all puts delivered, PE 0 can read slots.
        torch.distributed.barrier(group=self.ep_group)

        # PE 0: assemble from slots into gather_buffer.
        if self.ep_rank == 0:
            for peer in range(self.local_ep_size):
                gather_buffer[peer * L:(peer + 1) * L].copy_(
                    self._gather_slots[peer][:nbytes].view(grad_data.dtype)
                )

        # ── Stage 2: Allreduce (NCCL, separate stream) ──
        ar_event = torch.cuda.Event()
        with torch.cuda.stream(self.allreduce_stream):
            if self.is_edp_eligible:
                torch.distributed.all_reduce(
                    gather_buffer[:L * self.local_ep_size],
                    op=self.reduce_op, group=self.edp_group)
            else:
                torch.distributed.all_reduce(
                    grad_data, op=self.reduce_op, group=self.edp_group)
        ar_event.record(self.allreduce_stream)
        torch.cuda.current_stream().wait_event(ar_event)

        # ── Stage 3: Scatter — PE 0 puts to all peers concurrently ──
        if self.ep_rank == 0:
            # Pack slots from gather_buffer.
            for peer in range(self.local_ep_size):
                self._gather_slots[peer][:nbytes].view(grad_data.dtype).copy_(
                    gather_buffer[peer * L:(peer + 1) * L]
                )
            torch.cuda.synchronize()  # ensure copies visible before put
            # Batch puts to all peers.
            for peer in range(self.local_ep_size):
                peer_pe = self._ep_peer_pes[peer]
                nvshmem.core.put(
                    self._local_symm_buf[:nbytes], self._gather_slots[peer][:nbytes],
                    peer_pe, stream=self._nvshmem_stream)
            nvshmem.core.quiet(stream=self._nvshmem_stream)
            scatter_event = torch.cuda.Event()
            scatter_event.record(nv_torch)
            torch.cuda.current_stream().wait_event(scatter_event)

        # Single barrier: scatter complete, all peers can read.
        torch.distributed.barrier(group=self.ep_group)

        # Copy from symmetric buffer to grad_data.
        grad_data.copy_(self._local_symm_buf[:nbytes].view(grad_data.dtype))

    def execute_ep1(self, grad_data: torch.Tensor):
        torch.distributed.all_reduce(grad_data, op=self.reduce_op, group=self.edp_group)
