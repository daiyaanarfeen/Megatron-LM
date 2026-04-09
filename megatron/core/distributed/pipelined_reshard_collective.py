# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""NVSHMEM-based pipelined reshard collective for heterogeneous EP.

Uses NVSHMEM P2P put for gather/scatter (NVLink), NCCL allreduce for
cross-replica reduction. NVSHMEM team barriers for ep-group sync.
No host-side synchronize() in the critical path — all sync via CUDA events.
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
        self._ep_team = None

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
        self._ep_team = het_ep_config['nvshmem_ep_team']

    def execute(self, grad_data: torch.Tensor, gather_buffer: Optional[torch.Tensor]):
        if not self._use_nvshmem:
            raise RuntimeError("NVSHMEM not initialized for PipelinedReshardCollective")

        L = grad_data.numel()
        elem = grad_data.element_size()
        nbytes = L * elem
        ep0_pe = self._ep_peer_pes[0]

        if nbytes > self._local_symm_buf.numel():
            raise RuntimeError(
                f"NVSHMEM slot too small: need {nbytes} bytes, have {self._local_symm_buf.numel()}"
            )

        _, sp = self._nvshmem_stream.__cuda_stream__()
        nv_torch = torch.cuda.ExternalStream(sp)
        default = torch.cuda.current_stream()

        # ── Stage 1: Gather ──
        # Copy grad_data → local_symm_buf (on default stream).
        self._local_symm_buf[:nbytes].view(grad_data.dtype).copy_(grad_data)
        # Make nvshmem stream wait for the copy.
        copy_ev = torch.cuda.Event()
        copy_ev.record(default)
        nv_torch.wait_event(copy_ev)
        # All ranks put concurrently to ep_rank=0's gather slots.
        nvshmem.core.put(
            self._gather_slots[self.ep_rank][:nbytes],
            self._local_symm_buf[:nbytes],
            ep0_pe, stream=self._nvshmem_stream)
        nvshmem.core.quiet(stream=self._nvshmem_stream)
        # Team barrier: all ep peers' puts complete and visible.
        nvshmem.core.collective.barrier(team=self._ep_team, stream=self._nvshmem_stream)
        # Default stream waits for barrier completion (no host sync).
        barrier1_ev = torch.cuda.Event()
        barrier1_ev.record(nv_torch)
        default.wait_event(barrier1_ev)

        # PE 0: copy from gather slots to regular gather_buffer (on default stream).
        if self.ep_rank == 0:
            for peer in range(self.local_ep_size):
                gather_buffer[peer * L:(peer + 1) * L].copy_(
                    self._gather_slots[peer][:nbytes].view(grad_data.dtype))

        # ── Stage 2: Allreduce (NCCL) ──
        ar_ev = torch.cuda.Event()
        with torch.cuda.stream(self.allreduce_stream):
            if self.is_edp_eligible:
                torch.distributed.all_reduce(
                    gather_buffer[:L * self.local_ep_size],
                    op=self.reduce_op, group=self.edp_group)
            else:
                torch.distributed.all_reduce(
                    grad_data, op=self.reduce_op, group=self.edp_group)
        ar_ev.record(self.allreduce_stream)
        default.wait_event(ar_ev)

        # ── Stage 3: Scatter ──
        if self.ep_rank == 0:
            # Copy from gather_buffer to gather slots (on default stream).
            for peer in range(self.local_ep_size):
                self._gather_slots[peer][:nbytes].view(grad_data.dtype).copy_(
                    gather_buffer[peer * L:(peer + 1) * L])
            # Make nvshmem stream wait for the copies.
            pack_ev = torch.cuda.Event()
            pack_ev.record(default)
            nv_torch.wait_event(pack_ev)
            # Batch puts to all peers.
            for peer in range(self.local_ep_size):
                nvshmem.core.put(
                    self._local_symm_buf[:nbytes],
                    self._gather_slots[peer][:nbytes],
                    self._ep_peer_pes[peer], stream=self._nvshmem_stream)
            nvshmem.core.quiet(stream=self._nvshmem_stream)

        # Team barrier: scatter complete.
        nvshmem.core.collective.barrier(team=self._ep_team, stream=self._nvshmem_stream)
        barrier2_ev = torch.cuda.Event()
        barrier2_ev.record(nv_torch)
        default.wait_event(barrier2_ev)

        # Copy from symmetric buffer to grad_data (on default stream).
        grad_data.copy_(self._local_symm_buf[:nbytes].view(grad_data.dtype))

    def execute_ep1(self, grad_data: torch.Tensor):
        torch.distributed.all_reduce(grad_data, op=self.reduce_op, group=self.edp_group)
