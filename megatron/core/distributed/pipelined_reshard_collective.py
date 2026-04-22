# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""NVSHMEM pipelined reshard collective for heterogeneous EP.

Multi-leader design: num_leaders = min_ep_size. Each leader gathers from a
sub-group of ratio = local_ep / min_ep ep peers, allreduces with the
cross-replica leader via NVSHMEM ring, then scatters back.

Uses NVSHMEM for ALL communication (no NCCL in the pipeline loop):
  - Gather/scatter: put + signal_op/signal_wait between ep peers
  - Cross-replica allreduce: ring reduce-scatter + allgather via NVSHMEM put

Per-chunk math:
  slot_elems = chunk_size / element_size
  max_ratio = max_ep_size / min_ep_size
  effective_ar_chunk = (slot_elems // max_ratio) * max_ratio
  per_member_elems = effective_ar_chunk // ratio
  K = ceil(per_rank_numel / per_member_elems)  (same for all replicas)
"""

import logging
from typing import List, Optional

import torch
import torch.distributed

try:
    import nvshmem.core
    HAVE_NVSHMEM = True
except ImportError:
    HAVE_NVSHMEM = False

logger = logging.getLogger(__name__)


class PipelinedReshardCollective:

    def __init__(self, ep_group, edp_group, local_ep_size, ep_rank,
                 is_edp_eligible, num_chunks=4, reduce_op=None):
        self.ep_group = ep_group
        self.edp_group = edp_group  # original edp group (used by Approach A)
        self.local_ep_size = local_ep_size
        self.ep_rank = ep_rank
        self.is_edp_eligible = is_edp_eligible
        self.num_chunks = num_chunks
        self.reduce_op = reduce_op or torch.distributed.ReduceOp.SUM

        # NVSHMEM state (set via set_nvshmem_state).
        self._use_nvshmem = False
        self._nvshmem_stream = None
        self._ep_peer_pes = None
        self._gather_slots = None
        self._local_slots = [None, None]
        self._chunk_size = None
        self._b_edp_group = None

        # Multi-leader state.
        self._ratio = 1
        self._max_ratio = 1
        self._is_leader = False
        self._my_leader_idx = 0
        self._my_sub_rank = 0
        self._my_leader_ep_rank = 0
        self._min_ep_size = 1
        self._max_ep_size = 1
        self._gather_signals = None
        self._scatter_signal = None
        self._gather_states = [None, None]  # epoch handshake for double-buffered gather
        self._scatter_states = [None, None]  # epoch handshake for double-buffered scatter

        # Ring allreduce state (double-buffered exchange).
        self._use_nvshmem_ring = False
        self._exchange_bufs = [None, None]   # ping-pong buffers
        self._exchange_signals = [None, None]
        self._exchange_acks = [None, None]  # ack signals for buffer reuse
        self._ring_peers: List[int] = []  # PEs in ring order
        self._ring_size = 1
        self._my_ring_idx = 0
        self._ring_next_pe = 0
        self._ring_prev_pe = 0
        self._num_replicas = 1  # for signal_base advancement on non-leaders

        # Interleaved expert placement routing.
        self._expert_placement = None   # [ep_rank] -> [expert_ids]
        self._expert_gather_map = None  # {follower_rank: [(local_idx, leader, slot)]}
        self._needs_reshard = False
        self._local_expert_indices = None
        self._experts_per_leader = 0    # E / min_ep

        # Lazy-init events.
        self._events_assembly = []
        self._signal_base = 0

    def _setup_ring_state(self, het_ep_config: dict):
        """Set up ring allreduce state from config. Called for both ep>1 and ep=1."""
        self._num_replicas = het_ep_config.get('num_replicas', 1)
        bufs = het_ep_config.get('nvshmem_exchange_bufs')
        sigs = het_ep_config.get('nvshmem_exchange_signals')
        if bufs:
            self._exchange_bufs = bufs
        if sigs:
            self._exchange_signals = sigs
        acks = het_ep_config.get('nvshmem_exchange_acks')
        if acks:
            self._exchange_acks = acks
        peers = het_ep_config.get('b_edp_peer_pes')
        if peers and len(peers) > 1 and self._exchange_bufs[0] is not None:
            self._use_nvshmem_ring = True
            self._ring_peers = peers
            self._ring_size = len(peers)
            my_pe = torch.distributed.get_rank()
            self._my_ring_idx = peers.index(my_pe)
            self._ring_next_pe = peers[(self._my_ring_idx + 1) % self._ring_size]
            self._ring_prev_pe = peers[(self._my_ring_idx - 1) % self._ring_size]

    def set_nvshmem_state(self, het_ep_config: dict):
        self._chunk_size = het_ep_config.get('nvshmem_chunk_size')
        self._min_ep_size = het_ep_config.get('min_ep_size', 1)
        self._max_ep_size = het_ep_config.get('max_ep_size', 1)
        self._max_ratio = (self._max_ep_size // self._min_ep_size
                           if self._min_ep_size > 0 else 1)
        self._b_edp_group = het_ep_config.get('nvshmem_edp_group')

        if self.local_ep_size <= 1:
            # ep=1: no NVSHMEM gather/scatter, but still need ring for
            # cross-replica allreduce and local_slot for ring staging.
            if het_ep_config.get('nvshmem_initialized', False):
                self._nvshmem_stream = het_ep_config['nvshmem_stream']
                ls = het_ep_config.get('nvshmem_local_slots')
                if ls:
                    self._local_slots = ls
                self._setup_ring_state(het_ep_config)
                self._het_ep_config = het_ep_config
                self._signal_base = het_ep_config.get('_signal_base', 1000000)
            self._use_nvshmem = False
            return

        if not het_ep_config.get('nvshmem_initialized', False):
            raise RuntimeError("NVSHMEM not initialized")
        self._use_nvshmem = True
        self._nvshmem_stream = het_ep_config['nvshmem_stream']
        self._ep_peer_pes = het_ep_config['ep_peer_pes']
        self._gather_slots = het_ep_config['nvshmem_gather_slots']  # [parity][sub_rank]
        ls = het_ep_config.get('nvshmem_local_slots')
        if ls:
            self._local_slots = ls
        self._gather_signals = het_ep_config['nvshmem_gather_signals']
        self._scatter_signal = het_ep_config['nvshmem_scatter_signal']
        gs = het_ep_config.get('nvshmem_gather_states')
        if gs:
            self._gather_states = gs
        ss = het_ep_config.get('nvshmem_scatter_states')
        if ss:
            self._scatter_states = ss

        # Multi-leader fields.
        self._is_leader = het_ep_config.get('is_b_leader', self.ep_rank == 0)
        self._needs_reshard = het_ep_config.get('needs_reshard', False)
        self._expert_placement = het_ep_config.get('expert_placement')
        self._expert_gather_map = het_ep_config.get('expert_gather_map')
        self._local_expert_indices = het_ep_config.get('local_expert_indices')
        min_ep = het_ep_config.get('min_ep_size', 1)
        num_experts = sum(len(p) for p in self._expert_placement) if self._expert_placement else 0
        self._experts_per_leader = num_experts // min_ep if min_ep > 0 else 0

        # Ring allreduce.
        self._setup_ring_state(het_ep_config)

        # Persistent signal counter (survives across model re-creations).
        # Start high to avoid matching any stale values left in signal buffers
        # from NVSHMEM init or previous sessions.
        self._het_ep_config = het_ep_config
        self._signal_base = het_ep_config.get('_signal_base', 1000000)

    def _ensure_events(self, K):
        """Lazily allocate CUDA events for inter-stream sync."""
        while len(self._events_assembly) < K:
            self._events_assembly.append(torch.cuda.Event())

    def _compute_chunk_params(self, numel, elem_size):
        """Compute per-chunk parameters consistent across all replicas.

        Returns (per_member_elems, effective_ar_chunk, K).
        """
        slot_elems = self._chunk_size // elem_size
        max_ratio = self._max_ratio
        # Round slot_elems down to nearest multiple of max_ratio so that
        # per_member * ratio = effective_ar_chunk is the same for all replicas.
        effective_ar_chunk = (slot_elems // max_ratio) * max_ratio
        assert effective_ar_chunk > 0, (
            f"slot_elems={slot_elems} too small for max_ratio={max_ratio}")
        per_member_elems = effective_ar_chunk // self._ratio
        assert per_member_elems > 0, (
            f"effective_ar_chunk={effective_ar_chunk} too small for ratio={self._ratio}")
        K = (numel + per_member_elems - 1) // per_member_elems
        return per_member_elems, effective_ar_chunk, K

    def _ring_allreduce(self, data: torch.Tensor, n_elems: int,
                        send_staging: torch.Tensor, nv_torch,
                        ring_signal_base: Optional[int] = None):
        """Ring allreduce on data[0:n_elems] across ring peers via NVSHMEM.

        Implements reduce-scatter (N-1 steps) then allgather (N-1 steps).
        Each step: put one sub-chunk to next ring neighbor, wait for one from
        prev neighbor, accumulate (reduce-scatter) or copy (allgather).

        Args:
            data: Buffer to allreduce in-place. Must have >= n_elems elements.
            n_elems: Number of elements to allreduce.
            send_staging: Symmetric NVSHMEM buffer for staging puts. Must be
                >= (n_elems // ring_size) * elem_size bytes.
            nv_torch: The NVSHMEM stream as a torch.cuda.ExternalStream.
            ring_signal_base: If provided, use this as starting signal counter
                for the ring (overrides self._signal_base). All ring members
                must pass the same value.
        """
        N = self._ring_size
        if N <= 1:
            return  # single-rank group, nothing to do

        # Use provided ring_signal_base if given, otherwise self._signal_base.
        if ring_signal_base is not None:
            saved_base = self._signal_base
            self._signal_base = ring_signal_base

        dtype = data.dtype
        elem = data.element_size()
        my_idx = self._my_ring_idx
        next_pe = self._ring_next_pe

        # Split data into N sub-chunks. Last sub-chunk may be smaller.
        base_sub_n = n_elems // N
        remainder = n_elems % N
        # sub_chunk i has base_sub_n + (1 if i < remainder else 0) elements.
        # Compute offsets for each sub-chunk.
        sub_offsets = []
        sub_sizes = []
        off = 0
        for i in range(N):
            sz = base_sub_n + (1 if i < remainder else 0)
            sub_offsets.append(off)
            sub_sizes.append(sz)
            off += sz

        # Ping-pong buffer selection with ack-based reuse protection.
        # Step s uses xbuf[s%2]. After reading from xbuf[parity], signal
        # prev_pe that buffer is free (ack). Before putting to next_pe's
        # xbuf[parity], wait for next_pe's ack. This prevents the N>2
        # race where step s+2 overwrites xbuf[parity] before step s reads it.
        step = 0
        prev_pe = self._ring_prev_pe
        # Ack counter: increments each time a parity's buffer is freed.
        ack_count = [0, 0]  # per-parity ack counter

        def _ring_step(send_idx, recv_idx, accumulate):
            nonlocal step
            send_n = sub_sizes[send_idx]
            recv_n = sub_sizes[recv_idx]
            send_off = sub_offsets[send_idx]
            recv_off = sub_offsets[recv_idx]
            send_bytes = send_n * elem
            recv_bytes = recv_n * elem

            parity = step % 2
            xbuf = self._exchange_bufs[parity]
            xsig = self._exchange_signals[parity]
            sig_val = self._signal_base
            self._signal_base += 1
            step += 1

            # Wait for next_pe to ack that its xbuf[parity] is free.
            # First use of each parity (ack_count=0): ack initialized to 0, passes.
            if self._exchange_acks[0] is not None:
                nvshmem.core.signal_wait(
                    self._exchange_acks[parity],
                    ack_count[parity],
                    nvshmem.core.ComparisonType.CMP_GE,
                    stream=self._nvshmem_stream)

            # Copy + put + signal to next_pe.
            with torch.cuda.stream(nv_torch):
                send_staging[:send_bytes].view(dtype)[:send_n].copy_(
                    data[send_off:send_off + send_n])
            nvshmem.core.put(
                xbuf[:send_bytes], send_staging[:send_bytes],
                next_pe, stream=self._nvshmem_stream)
            nvshmem.core.signal_op(
                xsig, sig_val,
                nvshmem.core.SignalOp.SIGNAL_SET,
                next_pe, stream=self._nvshmem_stream)
            nvshmem.core.quiet(stream=self._nvshmem_stream)

            # Wait for data from prev_pe.
            nvshmem.core.signal_wait(
                xsig, sig_val,
                nvshmem.core.ComparisonType.CMP_GE,
                stream=self._nvshmem_stream)

            # Read from own xbuf[parity].
            if accumulate:
                with torch.cuda.stream(nv_torch):
                    data[recv_off:recv_off + recv_n].add_(
                        xbuf[:recv_bytes].view(dtype)[:recv_n])
            else:
                with torch.cuda.stream(nv_torch):
                    data[recv_off:recv_off + recv_n].copy_(
                        xbuf[:recv_bytes].view(dtype)[:recv_n])

            # Ack prev_pe: "I've read xbuf[parity], safe to overwrite."
            ack_count[parity] += 1
            if self._exchange_acks[0] is not None:
                nvshmem.core.signal_op(
                    self._exchange_acks[parity],
                    ack_count[parity],
                    nvshmem.core.SignalOp.SIGNAL_SET,
                    prev_pe, stream=self._nvshmem_stream)

        # ── Phase 1: Reduce-scatter (N-1 steps) ──
        for s in range(N - 1):
            send_idx = (my_idx - s) % N
            recv_idx = (my_idx - s - 1) % N
            _ring_step(send_idx, recv_idx, accumulate=True)

        # ── Phase 2: Allgather (N-1 steps) ──
        for s in range(N - 1):
            send_idx = (my_idx - s + 1) % N
            recv_idx = (my_idx - s) % N
            _ring_step(send_idx, recv_idx, accumulate=False)

        # Restore signal_base if we overrode it.
        if ring_signal_base is not None:
            self._signal_base = saved_base

    def execute(self, grad_data: torch.Tensor, gather_buffer: Optional[torch.Tensor]):
        """Dispatch to the appropriate execution path.

        For interleaved expert placement: per-expert routing.
        For sub-group placement (legacy): chunk-based pipelining.
        """
        if not self._use_nvshmem:
            raise RuntimeError("NVSHMEM not initialized for execute()")

        if self._expert_gather_map is not None:
            return self._execute_interleaved(grad_data, gather_buffer)
        return self._execute_chunked(grad_data, gather_buffer)

    def _execute_interleaved(self, grad_data: torch.Tensor,
                              gather_buffer: Optional[torch.Tensor]):
        """Per-expert gather/scatter for interleaved expert placement.

        Signal protocol: epoch-based. Each execute() call increments epoch.
        Gather signal for follower f, route k: gather_signals[f] = epoch * MAX_ROUTES + k.
        Scatter signal for follower f, route k: scatter_signal on f = epoch * MAX_ROUTES + k.
        Both sides compute the same values from the static routing map.
        Ring: uses its own signal_base (managed by _ring_allreduce).
        """
        _, sp = self._nvshmem_stream.__cuda_stream__()
        nv_torch = torch.cuda.ExternalStream(sp)

        dtype = grad_data.dtype
        elem = grad_data.element_size()
        min_ep = self._min_ep_size
        experts_per_leader = self._experts_per_leader
        local_experts = self._local_expert_indices
        experts_per_rank = len(local_experts)
        params_per_expert = grad_data.numel() // experts_per_rank
        P = params_per_expert
        Pbytes = P * elem

        default_stream = torch.cuda.current_stream()
        ev_ready = torch.cuda.Event()
        ev_done = torch.cuda.Event()

        epoch = self._signal_base
        self._signal_base += 1
        MAX_ROUTES = max(len(r) for r in self._expert_gather_map.values()) if self._expert_gather_map else 1

        # Ring signal_base: must be identical across all ring members.
        # Compute deterministically from epoch (not from gather signal increments).
        # Reserve a separate range for ring signals.
        ring_signal_base = epoch * 10000  # large offset to avoid collisions

        import os
        _debug = os.environ.get('DEBUG_INTERLEAVED', '')
        _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        def _log(msg):
            if _debug:
                print(f"[R{_rank} ep_rank={self.ep_rank}] {msg}", flush=True)

        _log(f"execute_interleaved: epoch={epoch}, needs_reshard={self._needs_reshard}, "
             f"is_leader={self._is_leader}, local_experts={local_experts}, "
             f"P={P}, experts_per_leader={experts_per_leader}")
        _log(f"  grad_data[:4]={grad_data[:4].tolist()}, numel={grad_data.numel()}")

        if not self._needs_reshard:
            # ── Min-ep replica: ring allreduce directly on grad_data ──
            _log(f"  min-ep path: ring_allreduce on {grad_data.numel()} elems")
            self._ring_allreduce(grad_data, grad_data.numel(),
                                self._local_slots[0], nv_torch,
                                ring_signal_base=ring_signal_base)
            _log(f"  min-ep after ring: grad_data[:4]={grad_data[:4].tolist()}")
        elif self._is_leader:
            # ── Reshard leader: gather → ring → scatter ──
            leader_idx = self.ep_rank
            ar_buf_size = experts_per_leader * P
            ar_buf = gather_buffer[:ar_buf_size]

            # Step 1: Copy own experts into ar_buf.
            ev_ready.record(default_stream)
            nv_torch.wait_event(ev_ready)
            with torch.cuda.stream(nv_torch):
                for local_idx, expert_id in enumerate(local_experts):
                    slot = expert_id - leader_idx * experts_per_leader
                    _log(f"  leader copy own: local_idx={local_idx}, expert={expert_id}, "
                         f"slot={slot}, val={grad_data[local_idx*P].item():.4f}")
                    ar_buf[slot * P:(slot + 1) * P].copy_(
                        grad_data[local_idx * P:(local_idx + 1) * P])

            # Step 2: Receive follower experts.
            # Build receive list: which (follower, route_idx) sends to this leader.
            receive_list = []
            for f_rank, routes in self._expert_gather_map.items():
                for route_idx, (local_idx, dest_leader, dest_slot) in enumerate(routes):
                    if dest_leader == leader_idx:
                        # Signal value: epoch * MAX_ROUTES + route_idx.
                        # gather_signals[f_rank] is per-follower.
                        sig_val = epoch * MAX_ROUTES + route_idx
                        receive_list.append((f_rank, local_idx, dest_slot, sig_val))

            _log(f"  leader receive_list: {[(f,li,ds,sv) for f,li,ds,sv in receive_list]}")
            for f_rank, f_local_idx, dest_slot, sig_val in receive_list:
                _log(f"  leader wait: gather_signals[{f_rank}] >= {sig_val}")
                nvshmem.core.signal_wait(
                    self._gather_signals[f_rank], sig_val,
                    nvshmem.core.ComparisonType.CMP_GE,
                    stream=self._nvshmem_stream,
                )
                nv_torch.synchronize()
                _log(f"  leader received from f_rank={f_rank}: slot[{f_local_idx}][:1]="
                     f"{self._gather_slots[0][f_local_idx][:elem].view(dtype)[0].item():.4f}, "
                     f"copying to ar_buf slot {dest_slot}")
                with torch.cuda.stream(nv_torch):
                    ar_buf[dest_slot * P:(dest_slot + 1) * P].copy_(
                        self._gather_slots[0][f_local_idx][:Pbytes].view(dtype)[:P])

            nv_torch.synchronize()
            _log(f"  leader ar_buf before ring: [{', '.join(f'{ar_buf[i*P].item():.4f}' for i in range(experts_per_leader))}]")

            # Step 3: Ring allreduce (ring_signal_base ensures all ring
            # members use the same signal values regardless of gather phase).
            self._ring_allreduce(ar_buf, ar_buf_size,
                                self._gather_slots[0][0], nv_torch,
                                ring_signal_base=ring_signal_base)

            nv_torch.synchronize()
            _log(f"  leader ar_buf after ring: [{', '.join(f'{ar_buf[i*P].item():.4f}' for i in range(experts_per_leader))}]")

            # Step 4: Copy own experts back to grad_data.
            with torch.cuda.stream(nv_torch):
                for local_idx, expert_id in enumerate(local_experts):
                    slot = expert_id - leader_idx * experts_per_leader
                    grad_data[local_idx * P:(local_idx + 1) * P].copy_(
                        ar_buf[slot * P:(slot + 1) * P])

            # Step 5: Scatter follower experts back.
            for f_rank, f_local_idx, dest_slot, _ in receive_list:
                scatter_sig = epoch * MAX_ROUTES + f_local_idx
                _log(f"  leader scatter: f_rank={f_rank}, f_local_idx={f_local_idx}, "
                     f"dest_slot={dest_slot}, scatter_sig={scatter_sig}, "
                     f"val={ar_buf[dest_slot*P].item():.4f}")
                with torch.cuda.stream(nv_torch):
                    self._gather_slots[0][f_local_idx][:Pbytes].view(dtype)[:P].copy_(
                        ar_buf[dest_slot * P:(dest_slot + 1) * P])
                f_pe = self._ep_peer_pes[f_rank]
                nvshmem.core.put(
                    self._local_slots[0][:Pbytes],
                    self._gather_slots[0][f_local_idx][:Pbytes],
                    f_pe, stream=self._nvshmem_stream,
                )
                nvshmem.core.signal_op(
                    self._scatter_signal, scatter_sig,
                    nvshmem.core.SignalOp.SIGNAL_SET,
                    f_pe, stream=self._nvshmem_stream,
                )
            nvshmem.core.quiet(stream=self._nvshmem_stream)
            ev_done.record(nv_torch)
            default_stream.wait_event(ev_done)

        else:
            # ── Reshard follower: send experts → leaders, wait for scatter ──
            my_routes = self._expert_gather_map.get(self.ep_rank, [])

            ev_ready.record(default_stream)
            nv_torch.wait_event(ev_ready)

            _log(f"  follower routes: {my_routes}")
            # Gather: put each expert to its leader.
            for route_idx, (local_idx, dest_leader, dest_slot) in enumerate(my_routes):
                sig_val = epoch * MAX_ROUTES + route_idx
                leader_pe = self._ep_peer_pes[dest_leader]
                _log(f"  follower put: route_idx={route_idx}, local_idx={local_idx}, "
                     f"expert={local_experts[local_idx]}, dest_leader={dest_leader}, "
                     f"dest_slot={dest_slot}, sig_val={sig_val}, leader_pe={leader_pe}, "
                     f"val={grad_data[local_idx*P].item():.4f}")
                with torch.cuda.stream(nv_torch):
                    self._local_slots[0][:Pbytes].view(dtype)[:P].copy_(
                        grad_data[local_idx * P:(local_idx + 1) * P])
                nvshmem.core.put(
                    self._gather_slots[0][local_idx][:Pbytes],
                    self._local_slots[0][:Pbytes],
                    leader_pe, stream=self._nvshmem_stream,
                )
                nvshmem.core.signal_op(
                    self._gather_signals[self.ep_rank], sig_val,
                    nvshmem.core.SignalOp.SIGNAL_SET,
                    leader_pe, stream=self._nvshmem_stream,
                )
            nvshmem.core.quiet(stream=self._nvshmem_stream)

            # Scatter: wait for each expert to come back.
            for route_idx, (local_idx, dest_leader, dest_slot) in enumerate(my_routes):
                scatter_sig = epoch * MAX_ROUTES + local_idx
                _log(f"  follower scatter wait: route_idx={route_idx}, local_idx={local_idx}, "
                     f"scatter_sig={scatter_sig}")
                nvshmem.core.signal_wait(
                    self._scatter_signal, scatter_sig,
                    nvshmem.core.ComparisonType.CMP_GE,
                    stream=self._nvshmem_stream,
                )
                ev_done.record(nv_torch)
                default_stream.wait_event(ev_done)
                grad_data[local_idx * P:(local_idx + 1) * P].copy_(
                    self._local_slots[0][:Pbytes].view(dtype)[:P])
                _log(f"  follower scatter received: local_idx={local_idx}, "
                     f"val={grad_data[local_idx*P].item():.4f}")

        # Persist epoch counter.
        if self._het_ep_config is not None:
            self._het_ep_config['_signal_base'] = self._signal_base

    def _execute_chunked(self, grad_data: torch.Tensor,
                          gather_buffer: Optional[torch.Tensor]):
        """Chunk-based pipelined execution for sub-group placement (legacy).

        Used when ep % min_ep == 0 (integer ratio sub-groups).
        """

        L = grad_data.numel()
        elem = grad_data.element_size()
        dtype = grad_data.dtype
        ratio = self._ratio

        per_member_elems, effective_ar_chunk, K = self._compute_chunk_params(L, elem)

        self._ensure_events(K)

        # NVSHMEM stream as torch stream.
        _, sp = self._nvshmem_stream.__cuda_stream__()
        nv_torch = torch.cuda.ExternalStream(sp)

        # Leader's double-buffered allreduce buffers (two regions of gather_buffer).
        if self._is_leader and gather_buffer is not None:
            ar_bufs = [
                gather_buffer[:effective_ar_chunk],
                gather_buffer[effective_ar_chunk:2 * effective_ar_chunk],
            ]
        else:
            ar_bufs = [None, None]

        # Leader PE for this rank's sub-group.
        leader_pe = self._ep_peer_pes[self._my_leader_ep_rank]

        default_stream = torch.cuda.current_stream()
        ev_grad_ready = torch.cuda.Event()
        ev_scatter_done = torch.cuda.Event()

        if self._is_leader:
            # ── Leader: sequential per-chunk processing ──
            # Non-leaders' puts arrive ahead of time; signal_waits are pre-satisfied.
            for i in range(K):
                start = i * per_member_elems
                end = min(start + per_member_elems, L)
                n = end - start
                nbytes = n * elem
                parity = i % 2
                epoch = i // 2
                gather_sig = self._signal_base
                self._signal_base += 1

                # Stage 1: Wait for all peers' gather data.
                for j in range(ratio):
                    if j == self._my_sub_rank:
                        continue
                    nvshmem.core.signal_wait(
                        self._gather_signals[j],
                        gather_sig,
                        nvshmem.core.ComparisonType.CMP_GE,
                        stream=self._nvshmem_stream,
                    )

                # Leader also puts its own data (as a sub-group member).
                # Wait for own grad_data readiness, copy to gather staging, put.
                ev_grad_ready.record(default_stream)
                nv_torch.wait_event(ev_grad_ready)
                gather_staging = self._local_slots[0]
                grad_chunk = grad_data[start:end]
                with torch.cuda.stream(nv_torch):
                    gather_staging[:nbytes].view(dtype)[:n].copy_(grad_chunk)
                my_slot = self._gather_slots[parity][self._my_sub_rank]
                nvshmem.core.put(
                    my_slot[:nbytes], gather_staging[:nbytes],
                    self._ep_peer_pes[self._my_leader_ep_rank],
                    stream=self._nvshmem_stream)
                nvshmem.core.quiet(stream=self._nvshmem_stream)

                # Assemble gather_slots[parity] → ar_buf.
                ar_buf = ar_bufs[parity]
                ar_n = ratio * n
                with torch.cuda.stream(nv_torch):
                    for j in range(ratio):
                        ar_buf[j * n:(j + 1) * n].copy_(
                            self._gather_slots[parity][j][:nbytes].view(dtype)[:n])

                # Stage 2: Ring allreduce.
                self._ring_allreduce(
                    ar_buf, ar_n, self._gather_slots[parity][0], nv_torch)

                # Stage 3: Scatter.
                # Wait for peers' scatter ack (local_slots[parity] freed).
                if self._scatter_states[0] is not None:
                    nvshmem.core.signal_wait(
                        self._scatter_states[parity],
                        epoch * ratio,
                        nvshmem.core.ComparisonType.CMP_GE,
                        stream=self._nvshmem_stream,
                    )

                scatter_sig = self._signal_base
                self._signal_base += 1

                # Pack ar_buf → gather_slots[parity] for scatter puts.
                with torch.cuda.stream(nv_torch):
                    for j in range(ratio):
                        self._gather_slots[parity][j][:nbytes].view(dtype)[:n].copy_(
                            ar_buf[j * n:(j + 1) * n])

                scatter_local = self._local_slots[parity]
                for j in range(ratio):
                    peer_ep_rank = self._my_leader_ep_rank + j
                    peer_pe = self._ep_peer_pes[peer_ep_rank]
                    nvshmem.core.put(
                        scatter_local[:nbytes],
                        self._gather_slots[parity][j][:nbytes],
                        peer_pe, stream=self._nvshmem_stream,
                    )
                    nvshmem.core.signal_op(
                        self._scatter_signal,
                        scatter_sig,
                        nvshmem.core.SignalOp.SIGNAL_SET,
                        peer_pe,
                        stream=self._nvshmem_stream,
                    )
                # Quiet ensures scatter_put completes before we set EMPTY.
                nvshmem.core.quiet(stream=self._nvshmem_stream)

                # NOW safe to mark gather_slots[parity] as EMPTY.
                if self._gather_states[0] is not None:
                    for j in range(ratio):
                        peer_ep_rank = self._my_leader_ep_rank + j
                        peer_pe = self._ep_peer_pes[peer_ep_rank]
                        nvshmem.core.signal_op(
                            self._gather_states[parity],
                            (epoch + 1) * 2,
                            nvshmem.core.SignalOp.SIGNAL_SET,
                            peer_pe,
                            stream=self._nvshmem_stream,
                        )

                # Leader also receives scatter to itself — copy result.
                nvshmem.core.signal_wait(
                    self._scatter_signal,
                    scatter_sig,
                    nvshmem.core.ComparisonType.CMP_GE,
                    stream=self._nvshmem_stream,
                )
                ev_scatter_done.record(nv_torch)
                default_stream.wait_event(ev_scatter_done)
                grad_chunk.copy_(self._local_slots[parity][:nbytes].view(dtype)[:n])

                # Leader acks its own scatter (for scatter_states consistency).
                if self._scatter_states[0] is not None:
                    nvshmem.core.signal_op(
                        self._scatter_states[parity],
                        1,
                        nvshmem.core.SignalOp.SIGNAL_ADD,
                        self._ep_peer_pes[self._my_leader_ep_rank],
                        stream=self._nvshmem_stream,
                    )

        else:
            # ── Non-leader: interleaved gather puts + scatter waits ──
            # Depth-2 pipeline: put gather(i), and if i>=2, process scatter(i-2).
            # This avoids deadlock: the leader needs scatter_ack before reusing
            # local_slots[parity], so the non-leader must ack promptly.
            leader_pe = self._ep_peer_pes[self._my_leader_ep_rank]
            leader_ring_signals = 2 * (self._num_replicas - 1)

            # Track scatter_sig values for deferred scatter processing.
            scatter_sigs = []

            for i in range(K):
                start_i = i * per_member_elems
                end_i = min(start_i + per_member_elems, L)
                n_i = end_i - start_i
                nbytes_i = n_i * elem
                parity_i = i % 2
                epoch_i = i // 2

                # ── Gather put for chunk i ──
                gather_sig = self._signal_base
                self._signal_base += 1

                if self._gather_states[0] is not None:
                    nvshmem.core.signal_wait(
                        self._gather_states[parity_i],
                        epoch_i * 2,
                        nvshmem.core.ComparisonType.CMP_GE,
                        stream=self._nvshmem_stream,
                    )

                ev_grad_ready.record(default_stream)
                nv_torch.wait_event(ev_grad_ready)

                gather_staging = self._local_slots[0]
                with torch.cuda.stream(nv_torch):
                    gather_staging[:nbytes_i].view(dtype)[:n_i].copy_(
                        grad_data[start_i:end_i])

                my_slot = self._gather_slots[parity_i][self._my_sub_rank]
                nvshmem.core.put(
                    my_slot[:nbytes_i], gather_staging[:nbytes_i],
                    leader_pe, stream=self._nvshmem_stream)
                nvshmem.core.signal_op(
                    self._gather_signals[self._my_sub_rank],
                    gather_sig,
                    nvshmem.core.SignalOp.SIGNAL_SET,
                    leader_pe,
                    stream=self._nvshmem_stream,
                )

                # Compute this chunk's scatter_sig (leader will use it later).
                self._signal_base += leader_ring_signals  # skip ring signals
                scatter_sig = self._signal_base
                self._signal_base += 1
                scatter_sigs.append(scatter_sig)

                # ── Process scatter for chunk i-2 (if exists) ──
                # This frees local_slots[parity] for the leader to reuse.
                if i >= 2:
                    si = i - 2
                    start_s = si * per_member_elems
                    end_s = min(start_s + per_member_elems, L)
                    n_s = end_s - start_s
                    nbytes_s = n_s * elem
                    parity_s = si % 2

                    nvshmem.core.signal_wait(
                        self._scatter_signal,
                        scatter_sigs[si],
                        nvshmem.core.ComparisonType.CMP_GE,
                        stream=self._nvshmem_stream,
                    )
                    ev_scatter_done.record(nv_torch)
                    default_stream.wait_event(ev_scatter_done)
                    grad_data[start_s:end_s].copy_(
                        self._local_slots[parity_s][:nbytes_s].view(dtype)[:n_s])

                    if self._scatter_states[0] is not None:
                        nvshmem.core.signal_op(
                            self._scatter_states[parity_s],
                            1,
                            nvshmem.core.SignalOp.SIGNAL_ADD,
                            leader_pe,
                            stream=self._nvshmem_stream,
                        )

            # Quiet after all gather puts.
            nvshmem.core.quiet(stream=self._nvshmem_stream)

            # Drain: process scatter for last 2 chunks (or fewer if K<2).
            for si in range(max(0, K - 2), K):
                start_s = si * per_member_elems
                end_s = min(start_s + per_member_elems, L)
                n_s = end_s - start_s
                nbytes_s = n_s * elem
                parity_s = si % 2

                nvshmem.core.signal_wait(
                    self._scatter_signal,
                    scatter_sigs[si],
                    nvshmem.core.ComparisonType.CMP_GE,
                    stream=self._nvshmem_stream,
                )
                ev_scatter_done.record(nv_torch)
                default_stream.wait_event(ev_scatter_done)
                grad_data[start_s:end_s].copy_(
                    self._local_slots[parity_s][:nbytes_s].view(dtype)[:n_s])

                if self._scatter_states[0] is not None:
                    nvshmem.core.signal_op(
                        self._scatter_states[parity_s],
                        1,
                        nvshmem.core.SignalOp.SIGNAL_ADD,
                        leader_pe,
                        stream=self._nvshmem_stream,
                    )

        # Persist signal_base for next execute() call (across model re-creations).
        if self._het_ep_config is not None:
            self._het_ep_config['_signal_base'] = self._signal_base

    def execute_ep1(self, grad_data: torch.Tensor):
        """Chunked ring allreduce for ep=1 ranks.

        ep=1 ranks don't do NVSHMEM gather/scatter (no ep peers). They
        participate in the cross-replica ring allreduce directly on their
        grad_data, chunked to match the reshard side's K and chunk sizes.

        Uses local_slots[0] as symmetric staging buffer for ring puts.
        """
        if self._use_nvshmem_ring and self._chunk_size is not None:
            _, effective_ar_chunk, K = self._compute_chunk_params(
                grad_data.numel(), grad_data.element_size())

            _, sp = self._nvshmem_stream.__cuda_stream__()
            nv_torch = torch.cuda.ExternalStream(sp)

            ring_staging = self._local_slots[0]
            for i in range(K):
                start = i * effective_ar_chunk
                end = min(start + effective_ar_chunk, grad_data.numel())
                chunk = grad_data[start:end]
                self._ring_allreduce(
                    chunk, end - start, ring_staging, nv_torch)

            # Persist signal_base.
            if self._het_ep_config is not None:
                self._het_ep_config['_signal_base'] = self._signal_base
        elif self._b_edp_group is not None and self._chunk_size is not None:
            # Fallback: chunked NCCL allreduce (no NVSHMEM available).
            _, effective_ar_chunk, K = self._compute_chunk_params(
                grad_data.numel(), grad_data.element_size())
            for i in range(K):
                start = i * effective_ar_chunk
                end = min(start + effective_ar_chunk, grad_data.numel())
                chunk = grad_data[start:end]
                torch.distributed.all_reduce(
                    chunk, op=self.reduce_op, group=self._b_edp_group)
        else:
            # Fallback: simple allreduce on original edp group.
            torch.distributed.all_reduce(
                grad_data, op=self.reduce_op, group=self.edp_group)
