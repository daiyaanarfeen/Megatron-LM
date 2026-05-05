# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Opt-in DDP wrapper and bucket groups for heterogeneous expert parallelism."""

from dataclasses import dataclass
from enum import Enum
import logging
import os
import re
from typing import Dict, List, Optional, Union

import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig

from .distributed_data_parallel import DistributedDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBucket, _ParamAndGradBucketGroup
from .pipelined_reshard_collective import PipelinedReshardCollective


logger = logging.getLogger(__name__)


class HeterogeneousEPApproach(str, Enum):
    """Gradient synchronization approach for heterogeneous EP expert params."""

    NCCL = "nccl"
    NVSHMEM = "nvshmem"
    PHASED = "phased"


def _debug_heterogeneous_ep(message: str) -> None:
    """Print opt-in hetero EP debug messages when explicitly requested."""
    if os.environ.get("MEGATRON_HET_EP_DEBUG") != "1":
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = -1
    print(f"[het-ep-debug r{rank}] {message}", flush=True)


class _CudaEventHandle:
    """Small wait handle compatible with Megatron bucket-group finish logic."""

    def __init__(self, events: List[torch.cuda.Event]):
        self.events = events

    def wait(self):
        current_stream = torch.cuda.current_stream()
        for event in self.events:
            current_stream.wait_event(event)
        self.events = []


@dataclass
class HeterogeneousEPConfig:
    """User-facing configuration for heterogeneous EP gradient sync."""

    approach: Union[HeterogeneousEPApproach, str] = HeterogeneousEPApproach.NCCL
    num_pipeline_chunks: Optional[int] = None

    def __post_init__(self):
        self.approach = HeterogeneousEPApproach(self.approach)
        if self.num_pipeline_chunks is not None and self.num_pipeline_chunks < 1:
            raise ValueError("num_pipeline_chunks must be >= 1")


def heterogeneous_ep_config_from_ddp_config(
    ddp_config: DistributedDataParallelConfig,
) -> HeterogeneousEPConfig:
    """Build a heterogeneous EP config from legacy DDP flags."""
    if ddp_config.use_phased_ep_reshard:
        approach = HeterogeneousEPApproach.PHASED
    elif ddp_config.use_pipelined_ep_reshard:
        approach = HeterogeneousEPApproach.NVSHMEM
    else:
        approach = HeterogeneousEPApproach.NCCL

    return HeterogeneousEPConfig(
        approach=approach,
        num_pipeline_chunks=ddp_config.num_ep_reshard_pipeline_chunks,
    )


class _HeterogeneousEPPipelinedReshardCollective(PipelinedReshardCollective):
    """Opt-in wrapper that preserves DDP AVG semantics for the NVSHMEM ring."""

    def _record_nvshmem_event(self) -> Optional[torch.cuda.Event]:
        """Record completion of work enqueued on the NVSHMEM stream."""
        nvshmem_stream = getattr(self, "_nvshmem_stream", None)
        if nvshmem_stream is None:
            return None
        _, stream_ptr = nvshmem_stream.__cuda_stream__()
        nv_torch = torch.cuda.ExternalStream(stream_ptr)
        event = torch.cuda.Event()
        event.record(nv_torch)
        return event

    def execute(self, grad_data: torch.Tensor, gather_buffer: Optional[torch.Tensor]):
        if self._expert_gather_map is not None:
            self._execute_interleaved(grad_data, gather_buffer)
        else:
            super().execute(grad_data, gather_buffer)
        self._last_event = self._record_nvshmem_event()

    def execute_ep1(self, grad_data: torch.Tensor):
        super().execute_ep1(grad_data)
        self._last_event = self._record_nvshmem_event()

    def _ring_allreduce(
        self,
        data: torch.Tensor,
        n_elems: int,
        send_staging: torch.Tensor,
        nv_torch,
        ring_signal_base: Optional[int] = None,
    ):
        import nvshmem.core

        n_ranks = self._ring_size
        if n_ranks <= 1:
            return

        if ring_signal_base is not None:
            saved_base = self._signal_base
            self._signal_base = ring_signal_base

        dtype = data.dtype
        elem = data.element_size()
        my_idx = self._my_ring_idx
        next_pe = self._ring_next_pe
        prev_pe = self._ring_prev_pe

        sub_offsets = []
        sub_sizes = []
        offset = 0
        base_sub_n = n_elems // n_ranks
        remainder = n_elems % n_ranks
        for idx in range(n_ranks):
            size = base_sub_n + (1 if idx < remainder else 0)
            sub_offsets.append(offset)
            sub_sizes.append(size)
            offset += size

        step = 0
        staging = [self._local_slots[0], self._local_slots[1]]
        if self._het_ep_config is not None:
            ack_targets = self._het_ep_config.setdefault('_exchange_ack_targets', [0, 0])
        else:
            if not hasattr(self, "_exchange_ack_targets"):
                self._exchange_ack_targets = [0, 0]
            ack_targets = self._exchange_ack_targets

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
            sbuf = staging[parity]
            step += 1

            if self._exchange_acks[0] is not None:
                nvshmem.core.signal_wait(
                    self._exchange_acks[parity],
                    ack_targets[parity],
                    nvshmem.core.ComparisonType.CMP_GE,
                    stream=self._nvshmem_stream,
                )

            with torch.cuda.stream(nv_torch):
                sbuf[:send_bytes].view(dtype)[:send_n].copy_(data[send_off : send_off + send_n])
            nvshmem.core.put(xbuf[:send_bytes], sbuf[:send_bytes], next_pe, stream=self._nvshmem_stream)
            nvshmem.core.signal_op(
                xsig,
                sig_val,
                nvshmem.core.SignalOp.SIGNAL_SET,
                next_pe,
                stream=self._nvshmem_stream,
            )

            nvshmem.core.signal_wait(
                xsig,
                sig_val,
                nvshmem.core.ComparisonType.CMP_GE,
                stream=self._nvshmem_stream,
            )

            if accumulate:
                with torch.cuda.stream(nv_torch):
                    data[recv_off : recv_off + recv_n].add_(xbuf[:recv_bytes].view(dtype)[:recv_n])
            else:
                with torch.cuda.stream(nv_torch):
                    data[recv_off : recv_off + recv_n].copy_(xbuf[:recv_bytes].view(dtype)[:recv_n])

            ack_targets[parity] = sig_val
            if self._exchange_acks[0] is not None:
                nvshmem.core.signal_op(
                    self._exchange_acks[parity],
                    sig_val,
                    nvshmem.core.SignalOp.SIGNAL_SET,
                    prev_pe,
                    stream=self._nvshmem_stream,
                )

        for step_idx in range(n_ranks - 1):
            _ring_step((my_idx - step_idx) % n_ranks, (my_idx - step_idx - 1) % n_ranks, True)

        for step_idx in range(n_ranks - 1):
            _ring_step(
                (my_idx - step_idx + 1) % n_ranks,
                (my_idx - step_idx) % n_ranks,
                False,
            )

        nvshmem.core.quiet(stream=self._nvshmem_stream)

        if ring_signal_base is not None:
            self._signal_base = saved_base

        if self.reduce_op == torch.distributed.ReduceOp.AVG and self._ring_size > 1:
            with torch.cuda.stream(nv_torch):
                data[:n_elems].div_(self._ring_size)

    def _pack_expert(self, source: torch.Tensor, dest: torch.Tensor, slices):
        offset = 0
        for start, end in slices:
            next_offset = offset + (end - start)
            dest[offset:next_offset].copy_(source[start:end])
            offset = next_offset

    def _unpack_expert(self, source: torch.Tensor, dest: torch.Tensor, slices):
        offset = 0
        for start, end in slices:
            next_offset = offset + (end - start)
            dest[start:end].copy_(source[offset:next_offset])
            offset = next_offset

    def _execute_interleaved(self, grad_data: torch.Tensor, gather_buffer: Optional[torch.Tensor]):
        """Interleaved routing using explicit expert slice metadata."""
        import nvshmem.core

        _, stream_ptr = self._nvshmem_stream.__cuda_stream__()
        nv_torch = torch.cuda.ExternalStream(stream_ptr)

        expert_slices = getattr(self, "_expert_grad_slices", None)
        if expert_slices is None:
            raise RuntimeError("Missing expert slice metadata for heterogeneous EP NVSHMEM sync")

        dtype = grad_data.dtype
        elem = grad_data.element_size()
        experts_per_leader = self._experts_per_leader
        local_experts = self._local_expert_indices
        p_per_expert = [sum(end - start for start, end in slices) for slices in expert_slices]
        assert len(set(p_per_expert)) == 1, f"Expert bucket slices are not balanced: {p_per_expert}"
        per_expert = p_per_expert[0]
        per_expert_bytes = per_expert * elem

        default_stream = torch.cuda.current_stream()
        ev_ready = torch.cuda.Event()
        ev_ready.record(default_stream)
        nv_torch.wait_event(ev_ready)

        epoch = self._signal_base
        self._signal_base += 1
        max_routes = (
            max(len(routes) for routes in self._expert_gather_map.values())
            if self._expert_gather_map
            else 1
        )

        ring_steps = 2 * (self._ring_size - 1) if self._ring_size > 1 else 0
        ring_base = epoch * 100000

        if self._is_leader or not self._needs_reshard:
            leader_idx = self.ep_rank
            if gather_buffer is None:
                raise RuntimeError("Leader ranks require a gather buffer for packed expert sync")
            data_buf = gather_buffer[: experts_per_leader * per_expert]

            with torch.cuda.stream(nv_torch):
                for local_idx, expert_id in enumerate(local_experts):
                    slot = expert_id - leader_idx * experts_per_leader
                    self._pack_expert(
                        grad_data,
                        data_buf[slot * per_expert : (slot + 1) * per_expert],
                        expert_slices[local_idx],
                    )

            slot_receive = {}
            if self._needs_reshard:
                for f_rank, routes in self._expert_gather_map.items():
                    for route_idx, (f_local_idx, dest_leader, dest_slot) in enumerate(routes):
                        if dest_leader == leader_idx:
                            sig_val = epoch * max_routes + route_idx
                            slot_receive[dest_slot] = (f_rank, f_local_idx, sig_val)

            for slot in range(experts_per_leader):
                slot_offset = slot * per_expert
                expert_ring_base = ring_base + slot * (ring_steps + 1)

                if slot in slot_receive:
                    f_rank, f_local_idx, sig_val = slot_receive[slot]
                    nvshmem.core.signal_wait(
                        self._gather_signals[f_rank],
                        sig_val,
                        nvshmem.core.ComparisonType.CMP_GE,
                        stream=self._nvshmem_stream,
                    )
                    with torch.cuda.stream(nv_torch):
                        data_buf[slot_offset : slot_offset + per_expert].copy_(
                            self._gather_slots[0][f_rank][:per_expert_bytes].view(dtype)[
                                :per_expert
                            ]
                        )

                self._ring_allreduce(
                    data_buf[slot_offset : slot_offset + per_expert],
                    per_expert,
                    self._local_slots[0],
                    nv_torch,
                    ring_signal_base=expert_ring_base,
                )

                if slot in slot_receive:
                    f_rank, f_local_idx, _ = slot_receive[slot]
                    scatter_sig = epoch * max_routes + f_local_idx
                    with torch.cuda.stream(nv_torch):
                        self._gather_slots[0][f_rank][:per_expert_bytes].view(dtype)[
                            :per_expert
                        ].copy_(data_buf[slot_offset : slot_offset + per_expert])
                    f_pe = self._ep_peer_pes[f_rank]
                    nvshmem.core.put(
                        self._gather_slots[1][f_local_idx][:per_expert_bytes],
                        self._gather_slots[0][f_rank][:per_expert_bytes],
                        f_pe,
                        stream=self._nvshmem_stream,
                    )
                    nvshmem.core.signal_op(
                        self._gather_signals[f_local_idx],
                        scatter_sig,
                        nvshmem.core.SignalOp.SIGNAL_SET,
                        f_pe,
                        stream=self._nvshmem_stream,
                    )

            nvshmem.core.quiet(stream=self._nvshmem_stream)

            with torch.cuda.stream(nv_torch):
                for local_idx, expert_id in enumerate(local_experts):
                    slot = expert_id - leader_idx * experts_per_leader
                    self._unpack_expert(
                        data_buf[slot * per_expert : (slot + 1) * per_expert],
                        grad_data,
                        expert_slices[local_idx],
                    )
        else:
            my_routes = self._expert_gather_map.get(self.ep_rank, [])

            for route_idx, (local_idx, dest_leader, _dest_slot) in enumerate(my_routes):
                sig_val = epoch * max_routes + route_idx
                leader_pe = self._ep_peer_pes[dest_leader]
                staging = self._gather_slots[1][local_idx]
                with torch.cuda.stream(nv_torch):
                    self._pack_expert(
                        grad_data,
                        staging[:per_expert_bytes].view(dtype)[:per_expert],
                        expert_slices[local_idx],
                    )
                nvshmem.core.put(
                    self._gather_slots[0][self.ep_rank][:per_expert_bytes],
                    staging[:per_expert_bytes],
                    leader_pe,
                    stream=self._nvshmem_stream,
                )
                nvshmem.core.signal_op(
                    self._gather_signals[self.ep_rank],
                    sig_val,
                    nvshmem.core.SignalOp.SIGNAL_SET,
                    leader_pe,
                    stream=self._nvshmem_stream,
                )
            nvshmem.core.quiet(stream=self._nvshmem_stream)

            for local_idx, _dest_leader, _dest_slot in my_routes:
                scatter_sig = epoch * max_routes + local_idx
                nvshmem.core.signal_wait(
                    self._gather_signals[local_idx],
                    scatter_sig,
                    nvshmem.core.ComparisonType.CMP_GE,
                    stream=self._nvshmem_stream,
                )
                with torch.cuda.stream(nv_torch):
                    self._unpack_expert(
                        self._gather_slots[1][local_idx][:per_expert_bytes].view(dtype)[
                            :per_expert
                        ],
                        grad_data,
                        expert_slices[local_idx],
                    )

        if self._het_ep_config is not None:
            self._het_ep_config['_signal_base'] = self._signal_base


class HeterogeneousEPParamAndGradBucketGroup(_ParamAndGradBucketGroup):
    """EP-aware bucket group that owns heterogeneous EP grad synchronization."""

    def configure_heterogeneous_ep(
        self,
        runtime_config: dict,
        heterogeneous_ep_config: HeterogeneousEPConfig,
        param_to_name: Optional[Dict[torch.nn.Parameter, str]] = None,
    ) -> None:
        """Configure this bucket group for heterogeneous EP gradient sync."""
        self._het_ep_config = runtime_config
        self._heterogeneous_ep_config = heterogeneous_ep_config
        self._param_to_name = param_to_name or {}
        self._expert_grad_slices = None
        self._gather_buffers = None
        self._pipelined_collectives = None
        self._use_phased_ep = (
            heterogeneous_ep_config.approach == HeterogeneousEPApproach.PHASED
        )
        self._phased_gather_send = None
        self._phased_gather_recv = None
        self._phased_gather_done = None
        self._heterogeneous_ep_grad_sync_started = False
        runtime_config.setdefault('_next_signal_base', 1_000_000)
        runtime_config.setdefault('_exchange_ack_targets', [0, 0])

        use_pipelined = (
            heterogeneous_ep_config.approach == HeterogeneousEPApproach.NVSHMEM
        )
        use_phased = self._use_phased_ep
        num_pipeline_chunks = (
            heterogeneous_ep_config.num_pipeline_chunks
            if heterogeneous_ep_config.num_pipeline_chunks is not None
            else self.ddp_config.num_ep_reshard_pipeline_chunks
        )

        # Approach A: is_edp_eligible gathers from all ep peers.
        # Approach B: is_b_leader gathers from its subgroup.
        # Approach C: is_b_leader receives via all_to_all.
        should_alloc_gather = (
            runtime_config.get('is_b_leader', runtime_config['is_edp_eligible'])
            if (use_pipelined or use_phased)
            else runtime_config['is_edp_eligible']
        )
        if runtime_config['local_ep_size'] > 1 and should_alloc_gather:
            self._gather_buffers = []
            for bucket in self.buckets:
                if use_pipelined:
                    gather_numel = self._pipelined_gather_buffer_numel(
                        bucket.grad_data.numel(),
                        bucket.grad_data.element_size(),
                        runtime_config,
                    )
                else:
                    gather_numel = bucket.grad_data.numel() * runtime_config['local_ep_size']

                if gather_numel > 0:
                    gather_buf = torch.zeros(
                        gather_numel,
                        dtype=bucket.grad_data.dtype,
                        device=bucket.grad_data.device,
                    )
                else:
                    gather_buf = None
                self._gather_buffers.append(gather_buf)

        if use_phased and runtime_config['local_ep_size'] > 1:
            self._setup_phased_splits(runtime_config)

        if use_pipelined and runtime_config.get('expert_gather_map') is not None:
            self._expert_grad_slices = self._build_expert_grad_slices(runtime_config)

        if use_pipelined:
            reduce_op = torch.distributed.ReduceOp.SUM
            if self.ddp_config.average_in_collective:
                reduce_op = torch.distributed.ReduceOp.AVG

            self._pipelined_collectives = _HeterogeneousEPPipelinedReshardCollective(
                ep_group=runtime_config['ep_group'],
                edp_group=runtime_config['edp_group'],
                local_ep_size=runtime_config['local_ep_size'],
                ep_rank=runtime_config['ep_rank'],
                is_edp_eligible=runtime_config['is_edp_eligible'],
                num_chunks=num_pipeline_chunks,
                reduce_op=reduce_op,
            )
            self._pipelined_collectives.set_nvshmem_state(runtime_config)

    def _build_expert_grad_slices(self, runtime_config: dict):
        """Build bucket-local grad slices grouped by local expert index."""
        local_expert_count = len(runtime_config['local_expert_indices'])
        expert_slices_by_bucket = []
        expert_name_re = re.compile(r"\.local_experts\.(\d+)\.")

        for bucket in self.buckets:
            slices = [[] for _ in range(local_expert_count)]
            for param in bucket.params_list:
                name = self._param_to_name.get(param, "")
                match = expert_name_re.search(name)
                if match is None:
                    continue
                local_idx = int(match.group(1))
                if local_idx >= local_expert_count:
                    raise RuntimeError(
                        f"Expert local index {local_idx} is out of range for {name}"
                    )
                start, end = bucket.param_to_index[param]
                slices[local_idx].append((start, end))

            sizes = []
            for local_idx, local_slices in enumerate(slices):
                if not local_slices:
                    raise RuntimeError(
                        f"Bucket {bucket.bucket_id} has no slices for local expert {local_idx}"
                    )
                local_slices.sort(key=lambda item: item[0])
                sizes.append(sum(end - start for start, end in local_slices))

            if len(set(sizes)) != 1:
                raise RuntimeError(
                    f"Bucket {bucket.bucket_id} is not expert-balanced; per-expert sizes={sizes}"
                )
            expert_slices_by_bucket.append(slices)

        return expert_slices_by_bucket

    def _sync_pipelined_signal_base(self) -> None:
        """Refresh the NVSHMEM signal epoch shared by split expert buckets."""
        if self._pipelined_collectives is None or self._het_ep_config is None:
            return
        self._pipelined_collectives._signal_base = self._het_ep_config.get(
            '_signal_base',
            self._pipelined_collectives._signal_base,
        )

    def _pipelined_gather_buffer_numel(
        self,
        bucket_numel: int,
        element_size: int,
        runtime_config: dict,
    ) -> int:
        """Return the leader staging size needed by the NVSHMEM path."""
        if runtime_config.get('expert_gather_map') is not None:
            local_expert_count = len(runtime_config['local_expert_indices'])
            num_experts = sum(len(p) for p in runtime_config['expert_placement'])
            experts_per_leader = num_experts // runtime_config['min_ep_size']
            return experts_per_leader * (bucket_numel // local_expert_count)

        chunk_size = runtime_config['nvshmem_chunk_size']
        max_ratio = runtime_config['max_ep_size'] // runtime_config['min_ep_size']
        slot_elems = chunk_size // element_size
        effective_ar_chunk = (slot_elems // max_ratio) * max_ratio
        return 2 * effective_ar_chunk

    def set_heterogeneous_ep_config(
        self,
        config: dict,
        use_pipelined: bool = False,
        num_pipeline_chunks: int = 4,
        use_phased: bool = False,
    ) -> None:
        """Compatibility shim for the old bucket-group configuration method."""
        if use_phased:
            approach = HeterogeneousEPApproach.PHASED
        elif use_pipelined:
            approach = HeterogeneousEPApproach.NVSHMEM
        else:
            approach = HeterogeneousEPApproach.NCCL
        self.configure_heterogeneous_ep(
            config,
            HeterogeneousEPConfig(
                approach=approach,
                num_pipeline_chunks=num_pipeline_chunks,
            ),
        )

    def _setup_phased_splits(self, config):
        """Precompute all_to_all split sizes for Approach C gather/scatter."""
        placement = config['expert_placement']
        ep_size = config['local_ep_size']
        min_ep = config['min_ep_size']
        ep_rank = config['ep_rank']
        num_experts = sum(len(p) for p in placement)
        experts_per_leader = num_experts // min_ep

        self._phased_gather_send = []
        self._phased_gather_recv = []
        self._phased_gather_done = []

        for bucket in self.buckets:
            local_expert_count = len(config['local_expert_indices'])
            params_per_expert = bucket.grad_data.numel() // local_expert_count
            send_splits = [0] * ep_size
            recv_splits = [0] * ep_size

            for rank in range(ep_size):
                for expert_id in placement[rank]:
                    leader = expert_id // experts_per_leader
                    if leader < min_ep:
                        if rank == ep_rank:
                            send_splits[leader] += params_per_expert
                        if leader == ep_rank:
                            recv_splits[rank] += params_per_expert

            self._phased_gather_send.append(send_splits)
            self._phased_gather_recv.append(recv_splits)
            self._phased_gather_done.append(False)

    def _start_phased_gather(self, bucket_idx):
        """Approach C: gather phase on the EP group for one bucket."""
        cfg = self._het_ep_config
        ep_group = cfg['ep_group']
        is_leader = cfg.get('is_b_leader', False)
        bucket = self.buckets[bucket_idx]

        send_splits = self._phased_gather_send[bucket_idx]
        recv_splits = self._phased_gather_recv[bucket_idx]
        recv_total = sum(recv_splits)

        input_tensor = bucket.grad_data
        if is_leader:
            output_tensor = self._gather_buffers[bucket_idx][:recv_total]
        else:
            output_tensor = torch.empty(
                recv_total,
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )

        torch.distributed.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=ep_group,
        )
        self._phased_gather_done[bucket_idx] = True

    def _start_phased_allreduce(self, bucket_idx):
        """Approach C: allreduce phase on the EDP group for one bucket."""
        cfg = self._het_ep_config
        edp_group = cfg['edp_group']
        is_leader = cfg.get('is_b_leader', False)

        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        if is_leader:
            recv_total = sum(self._phased_gather_recv[bucket_idx])
            torch.distributed.all_reduce(
                self._gather_buffers[bucket_idx][:recv_total],
                op=reduce_op,
                group=edp_group,
            )
        else:
            torch.distributed.all_reduce(
                self.buckets[bucket_idx].grad_data,
                op=reduce_op,
                group=edp_group,
            )

    def _finish_phased_scatter_all(self):
        """Approach C: scatter all buckets with the reverse EP all_to_all."""
        cfg = self._het_ep_config
        ep_group = cfg['ep_group']
        is_leader = cfg.get('is_b_leader', False)

        for bucket_idx, bucket in enumerate(self.buckets):
            send_splits = self._phased_gather_recv[bucket_idx]
            recv_splits = self._phased_gather_send[bucket_idx]
            send_total = sum(send_splits)

            if is_leader:
                input_tensor = self._gather_buffers[bucket_idx][:send_total]
            else:
                input_tensor = torch.empty(
                    send_total,
                    dtype=bucket.grad_data.dtype,
                    device=bucket.grad_data.device,
                )

            torch.distributed.all_to_all_single(
                bucket.grad_data,
                input_tensor,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=ep_group,
            )

    def _start_heterogeneous_ep_grad_sync(self):
        """Gradient sync for heterogeneous EP: gather, allreduce, scatter."""
        cfg = self._het_ep_config
        ep_group = cfg['ep_group']
        edp_group = cfg['edp_group']
        local_ep_size = cfg['local_ep_size']
        is_edp_eligible = cfg['is_edp_eligible']

        # Approach C: phased gather + allreduce now, scatter in finish_grad_sync.
        if self._use_phased_ep and local_ep_size > 1:
            for idx in range(len(self.buckets)):
                self._start_phased_gather(idx)
                self._start_phased_allreduce(idx)
            return []

        # Approach B: fused intra-bucket NVSHMEM pipeline.
        if self._pipelined_collectives is not None:
            is_b_leader = cfg.get('is_b_leader', is_edp_eligible)
            events = []
            for idx, bucket in enumerate(self.buckets):
                self._pipelined_collectives._signal_base = self._reserve_nvshmem_signal_base()
                if self._expert_grad_slices is not None:
                    self._pipelined_collectives._expert_grad_slices = self._expert_grad_slices[idx]
                local_expert_count = max(1, len(cfg.get('local_expert_indices', [])))
                per_expert_numel = bucket.grad_data.numel() // local_expert_count
                _debug_heterogeneous_ep(
                    "nvshmem bucket start "
                    f"group={id(self)} bucket={idx} numel={bucket.grad_data.numel()} "
                    f"per_expert={per_expert_numel} dtype={bucket.grad_data.dtype} "
                    f"per_expert_bytes={per_expert_numel * bucket.grad_data.element_size()} "
                    f"ep_rank={cfg['ep_rank']} "
                    f"local_ep={local_ep_size} leader={is_b_leader} "
                    f"needs_reshard={cfg.get('needs_reshard', False)} "
                    f"signal_base={self._pipelined_collectives._signal_base}"
                )
                if local_ep_size == 1:
                    self._pipelined_collectives.execute_ep1(bucket.grad_data)
                else:
                    gather_buf = (
                        self._gather_buffers[idx]
                        if is_b_leader and self._gather_buffers is not None
                        else None
                    )
                    self._pipelined_collectives.execute(bucket.grad_data, gather_buf)
                event = getattr(self._pipelined_collectives, "_last_event", None)
                if event is not None:
                    events.append(event)
                _debug_heterogeneous_ep(
                    "nvshmem bucket done "
                    f"group={id(self)} bucket={idx} "
                    f"signal_base={self._pipelined_collectives._signal_base}"
                )
            return events

        # Approach A: NCCL gather, EDP allreduce, EP scatter.
        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        for idx, bucket in enumerate(self.buckets):
            grad_data = bucket.grad_data
            chunk_size = grad_data.numel()

            if local_ep_size == 1:
                torch.distributed.all_reduce(grad_data, op=reduce_op, group=edp_group)
            else:
                if is_edp_eligible:
                    gather_buf = self._gather_buffers[idx]
                    gather_list = [
                        gather_buf[i * chunk_size : (i + 1) * chunk_size]
                        for i in range(local_ep_size)
                    ]
                else:
                    gather_list = [
                        torch.empty(chunk_size, dtype=grad_data.dtype, device=grad_data.device)
                        for _ in range(local_ep_size)
                    ]
                torch.distributed.all_gather(gather_list, grad_data, group=ep_group)

                if is_edp_eligible:
                    torch.distributed.all_reduce(gather_buf, op=reduce_op, group=edp_group)
                else:
                    torch.distributed.all_reduce(grad_data, op=reduce_op, group=edp_group)

                ep_rank = cfg['ep_rank']
                if ep_rank != 0:
                    if is_edp_eligible:
                        gather_buf.zero_()
                    else:
                        gather_buf = torch.zeros(
                            chunk_size * local_ep_size,
                            dtype=grad_data.dtype,
                            device=grad_data.device,
                        )
                torch.distributed.all_reduce(
                    gather_buf,
                    op=torch.distributed.ReduceOp.SUM,
                    group=ep_group,
                )
                grad_data.copy_(gather_buf[ep_rank * chunk_size : (ep_rank + 1) * chunk_size])

        return []

    def _reserve_nvshmem_signal_base(self) -> int:
        """Reserve a monotonically increasing signal epoch for shared NVSHMEM buffers."""
        base = self._het_ep_config.get('_next_signal_base', 1_000_000)
        self._het_ep_config['_next_signal_base'] = base + 1_000_000
        return base

    def _should_finish_phased_scatter(self):
        return self._use_phased_ep and self._het_ep_config['local_ep_size'] > 1

    def start_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Start heterogeneous EP gradient synchronization for this bucket group."""
        if self._heterogeneous_ep_grad_sync_started:
            return

        if self.is_first_batch and self.grad_reduce_handle is not None:
            return

        assert (
            self.grad_reduce_handle is None
        ), "Should not have multiple communication calls outstanding at once"

        if self.ddp_config.check_for_nan_in_grad or self.ddp_config.check_for_large_grads:
            self.check_grads(
                check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
                check_for_large=self.ddp_config.check_for_large_grads,
            )

        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor

        _debug_heterogeneous_ep(
            f"expert start_grad_sync group={id(self)} buckets={len(self.buckets)}"
        )
        events = self._start_heterogeneous_ep_grad_sync()
        self._heterogeneous_ep_grad_sync_started = True
        self.grad_reduce_handle = _CudaEventHandle(events) if events else None
        _debug_heterogeneous_ep(f"expert end_grad_sync group={id(self)}")

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Finish heterogeneous EP gradient synchronization for this bucket group."""
        self.param_gather_dispatched = False

        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            if self.grad_reduce_handle is not None:
                self.grad_reduce_handle.wait()
                self.grad_reduce_handle = None
            if self._should_finish_phased_scatter():
                self._finish_phased_scatter_all()
            return

        if not self.is_first_batch:
            assert self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts, (
                f"Communication call has not been issued for this bucket "
                f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
                "params have grad available)"
            )

        # NVSHMEM buffers/signals are shared across split expert buckets. Launch
        # in deterministic bucket-group order while keeping completion asynchronous.
        if self.is_first_batch:
            self.start_grad_sync(force_all_reduce=force_all_reduce)

        if self.ddp_config.num_distributed_optimizer_instances > 1:
            torch.cuda.default_stream().wait_stream(self.communication_stream)
            return

        assert self._heterogeneous_ep_grad_sync_started, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.per_param_grad_ready_counts)}/{len(self.params)} "
            "params have grad available)"
        )
        if self.grad_reduce_handle is not None:
            self.grad_reduce_handle.wait()
            self.grad_reduce_handle = None

        if self._should_finish_phased_scatter():
            self._finish_phased_scatter_all()

    def register_grad_ready(
        self, param: torch.nn.Parameter, force_all_reduce: Optional[bool] = False
    ):
        """Track ready grads without launching rank-order-sensitive EP collectives."""
        assert (
            self.ddp_config.overlap_grad_reduce
        ), "register_grad_ready() should only be called when overlap_grad_reduce is True"
        if self.is_last_microbatch:
            assert param in self.param_to_bucket, "Param is not in the bucket group"
            if param not in self.per_param_grad_ready_counts:
                self.per_param_grad_ready_counts[param] = 0
            self.per_param_grad_ready_counts[param] += 1
            if not self.is_first_batch:
                if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
                    assert len(self.per_param_grad_ready_counts) == len(self.params)
                    self._heterogeneous_ep_ready = True
                    self._try_start_ready_heterogeneous_ep_bucket_groups(
                        force_all_reduce=force_all_reduce
                    )

    def reset(self):
        """Reset per-iteration metadata and the synchronous hetero sync marker."""
        super().reset()
        self._heterogeneous_ep_grad_sync_started = False
        self._heterogeneous_ep_ready = False
        self.grad_reduce_handle = None
        state = getattr(self, "_heterogeneous_ep_scheduler_state", None)
        if state is not None and getattr(self, "_heterogeneous_ep_group_index", -1) == 0:
            state["next_index"] = 0

    def _try_start_ready_heterogeneous_ep_bucket_groups(
        self, force_all_reduce: Optional[bool] = False
    ):
        state = getattr(self, "_heterogeneous_ep_scheduler_state", None)
        if state is None:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
            return

        groups = state["groups"]
        while state["next_index"] < len(groups):
            group = groups[state["next_index"]]
            if not getattr(group, "_heterogeneous_ep_ready", False):
                break
            group.start_grad_sync(force_all_reduce=force_all_reduce)
            state["next_index"] += 1


def wrap_heterogeneous_ep_bucket_groups(
    bucket_groups: List[_ParamAndGradBucketGroup],
    runtime_config: dict,
    heterogeneous_ep_config: HeterogeneousEPConfig,
    param_to_bucket_group: Optional[Dict[torch.nn.Parameter, _ParamAndGradBucketGroup]] = None,
    param_to_name: Optional[Dict[torch.nn.Parameter, str]] = None,
) -> List[HeterogeneousEPParamAndGradBucketGroup]:
    """Replace generic expert bucket groups with heterogeneous EP-aware groups."""
    wrapped_bucket_groups = []
    old_to_new = {}

    for bucket_group in bucket_groups:
        if isinstance(bucket_group, HeterogeneousEPParamAndGradBucketGroup):
            wrapped_bucket_group = bucket_group
        else:
            wrapped_bucket_group = HeterogeneousEPParamAndGradBucketGroup.__new__(
                HeterogeneousEPParamAndGradBucketGroup
            )
            wrapped_bucket_group.__dict__ = bucket_group.__dict__.copy()

        wrapped_bucket_group.configure_heterogeneous_ep(
            runtime_config,
            heterogeneous_ep_config,
            param_to_name,
        )
        old_to_new[bucket_group] = wrapped_bucket_group
        wrapped_bucket_groups.append(wrapped_bucket_group)

    for wrapped_bucket_group in wrapped_bucket_groups:
        next_bucket_group = wrapped_bucket_group.next_param_gather_bucket_group
        if next_bucket_group in old_to_new:
            wrapped_bucket_group.next_param_gather_bucket_group = old_to_new[next_bucket_group]

    if param_to_bucket_group is not None:
        for wrapped_bucket_group in wrapped_bucket_groups:
            for bucket in wrapped_bucket_group.buckets:
                for param in bucket.params_list:
                    param_to_bucket_group[param] = wrapped_bucket_group

    _configure_heterogeneous_ep_scheduler(wrapped_bucket_groups)
    return wrapped_bucket_groups


def _configure_heterogeneous_ep_scheduler(
    bucket_groups: List[HeterogeneousEPParamAndGradBucketGroup],
) -> None:
    """Attach deterministic async launch state to EP bucket groups."""
    state = {"groups": bucket_groups, "next_index": 0}
    signal_base = 1_000_000
    signal_stride = 1_000_000
    for index, bucket_group in enumerate(bucket_groups):
        bucket_group._heterogeneous_ep_scheduler_state = state
        bucket_group._heterogeneous_ep_group_index = index
        bucket_group._heterogeneous_ep_ready = False
        if bucket_group._pipelined_collectives is not None:
            bucket_group._pipelined_collectives._signal_base = signal_base + index * signal_stride


def build_heterogeneous_ep_layer_bucket_groups(
    buffers,
    ddp_config: DistributedDataParallelConfig,
    runtime_config: dict,
    heterogeneous_ep_config: HeterogeneousEPConfig,
    param_to_bucket_group: Dict[torch.nn.Parameter, _ParamAndGradBucketGroup],
    param_to_name: Dict[torch.nn.Parameter, str],
) -> List[HeterogeneousEPParamAndGradBucketGroup]:
    """Build layer-aligned expert bucket groups from expert grad buffers."""
    layer_re = re.compile(r"\.(?:decoder|encoder)\.layers\.(\d+)\.mlp\.experts\.")
    bucket_groups = []
    missing_names = []

    for buffer in buffers:
        layer_to_params = {}
        for param in buffer.param_index_map:
            name = param_to_name.get(param, "")
            match = layer_re.search(name)
            if match is None:
                missing_names.append(name or "<unnamed expert parameter>")
                continue
            layer_to_params.setdefault(int(match.group(1)), []).append(param)

        if missing_names:
            examples = ", ".join(missing_names[:3])
            raise RuntimeError(
                "Cannot build optimized heterogeneous EP NVSHMEM layer buckets because "
                f"{len(missing_names)} expert params do not match the expected layer naming "
                f"pattern. Examples: {examples}"
            )

        if not layer_to_params:
            raise RuntimeError(
                "Cannot build optimized heterogeneous EP NVSHMEM layer buckets: "
                "no expert-layer parameters were found."
            )

        layer_ranges = []
        for layer, params in layer_to_params.items():
            params = sorted(params, key=lambda param: buffer.param_index_map[param][0])
            starts_ends = [buffer.param_index_map[param][:2] for param in params]
            start = min(start for start, _ in starts_ends)
            end = max(end for _, end in starts_ends)
            total = sum(param_end - param_start for param_start, param_end in starts_ends)
            if total != end - start:
                raise RuntimeError(
                    f"Expert params for layer {layer} are not contiguous in the grad buffer"
                )
            layer_ranges.append((start, end, layer, params))

        for bucket_id, (start, end, _layer, params) in enumerate(sorted(layer_ranges)):
            param_data = None
            if buffer.param_data is not None:
                param_data = buffer.param_data[start:end]
            bucket = _ParamAndGradBucket(
                params=params,
                param_data=param_data,
                grad_data=buffer.grad_data[start:end],
                offset=start,
                numel_unpadded=end - start,
                gradient_scaling_factor=buffer.gradient_scaling_factor,
                bucket_id=bucket_id,
                param_index_map=buffer.param_index_map,
            )
            bucket_group = HeterogeneousEPParamAndGradBucketGroup(
                [bucket],
                ddp_config,
                buffer.data_parallel_group,
                buffer.data_parallel_world_size,
            )
            bucket_group.configure_heterogeneous_ep(
                runtime_config,
                heterogeneous_ep_config,
                param_to_name,
            )
            bucket_groups.append(bucket_group)

    for bucket_group in bucket_groups:
        for bucket in bucket_group.buckets:
            for param in bucket.params_list:
                param_to_bucket_group[param] = bucket_group

    _configure_heterogeneous_ep_scheduler(bucket_groups)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(
                "Built %d layer-aligned heterogeneous EP NVSHMEM expert bucket groups",
                len(bucket_groups),
            )
            print(
                "[heterogeneous_ep] built "
                f"{len(bucket_groups)} layer-aligned NVSHMEM expert bucket groups",
                flush=True,
            )
    return bucket_groups


class HeterogeneousEPDistributedDataParallel(DistributedDataParallel):
    """DDP wrapper that opts expert params into heterogeneous EP grad sync."""

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        heterogeneous_ep_config: Optional[HeterogeneousEPConfig] = None,
        disable_bucketing: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        if not parallel_state.is_heterogeneous_ep():
            raise RuntimeError(
                "HeterogeneousEPDistributedDataParallel requires heterogeneous EP "
                "parallel state. Call initialize_heterogeneous_model_parallel() first."
            )

        if heterogeneous_ep_config is None:
            heterogeneous_ep_config = HeterogeneousEPConfig()
        else:
            heterogeneous_ep_config = HeterogeneousEPConfig(
                approach=heterogeneous_ep_config.approach,
                num_pipeline_chunks=heterogeneous_ep_config.num_pipeline_chunks,
            )
        self.heterogeneous_ep_config = heterogeneous_ep_config

        self._skip_heterogeneous_ep_compat_wrap = True
        original_calculate_per_token_loss = config.calculate_per_token_loss
        original_average_in_collective = ddp_config.average_in_collective
        original_is_heterogeneous_ep = parallel_state.is_heterogeneous_ep
        original_get_heterogeneous_ep_config = parallel_state.get_heterogeneous_ep_config
        try:
            # Base DDP validates homogeneous expert-DP scaling before this opt-in
            # wrapper can replace expert bucket groups. Build generic buffers with
            # neutral scaling, then assign the heterogeneous EP scaling below.
            config.calculate_per_token_loss = True
            ddp_config.average_in_collective = False
            parallel_state.is_heterogeneous_ep = lambda: False
            parallel_state.get_heterogeneous_ep_config = lambda: None
            super().__init__(
                config=config,
                ddp_config=ddp_config,
                module=module,
                disable_bucketing=disable_bucketing,
                pg_collection=pg_collection,
            )
        finally:
            parallel_state.is_heterogeneous_ep = original_is_heterogeneous_ep
            parallel_state.get_heterogeneous_ep_config = original_get_heterogeneous_ep_config
            config.calculate_per_token_loss = original_calculate_per_token_loss
            ddp_config.average_in_collective = original_average_in_collective
            if hasattr(self, "ddp_config"):
                self.ddp_config.average_in_collective = original_average_in_collective

        runtime_config = parallel_state.get_heterogeneous_ep_config()
        self._heterogeneous_ep_config = runtime_config
        self._param_to_name = {param: name for name, param in self.module.named_parameters()}

        def _set_gradient_scaling(bucket_groups, scaling_factor):
            for bucket_group in bucket_groups:
                for bucket in bucket_group.buckets:
                    bucket.gradient_scaling_factor = scaling_factor

        if original_calculate_per_token_loss:
            dense_gradient_scaling_factor = 1.0
            expert_gradient_scaling_factor = 1.0
        elif original_average_in_collective:
            dense_gradient_scaling_factor = 1.0
            # Match standard Megatron's target gradient scale of
            # 1 / dp_cp_group.size(). Heterogeneous EDP collectives average
            # over one rank per replica, so pre-scale by num_replicas/dp_cp.
            expert_gradient_scaling_factor = (
                runtime_config["num_replicas"] / self.dp_cp_group.size()
            )
        else:
            dense_gradient_scaling_factor = 1.0 / self.dp_cp_group.size()
            expert_gradient_scaling_factor = 1.0 / self.dp_cp_group.size()

        _set_gradient_scaling(self.bucket_groups, dense_gradient_scaling_factor)
        _set_gradient_scaling(
            self.expert_parallel_bucket_groups,
            expert_gradient_scaling_factor,
        )
        if self.heterogeneous_ep_config.approach == HeterogeneousEPApproach.NVSHMEM:
            layer_bucket_groups = build_heterogeneous_ep_layer_bucket_groups(
                self.expert_parallel_buffers,
                self.ddp_config,
                runtime_config,
                self.heterogeneous_ep_config,
                self.param_to_bucket_group,
                self._param_to_name,
            )
        else:
            layer_bucket_groups = []

        if layer_bucket_groups:
            self.expert_parallel_bucket_groups = layer_bucket_groups
        elif self.heterogeneous_ep_config.approach == HeterogeneousEPApproach.NVSHMEM:
            raise RuntimeError(
                "Optimized heterogeneous EP NVSHMEM path requires layer-aligned expert buckets, "
                "but none were constructed."
            )
        else:
            self.expert_parallel_bucket_groups = wrap_heterogeneous_ep_bucket_groups(
                self.expert_parallel_bucket_groups,
                runtime_config,
                self.heterogeneous_ep_config,
                self.param_to_bucket_group,
                self._param_to_name,
            )

    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Finish gradient sync with optional opt-in hetero EP debug markers."""
        if os.environ.get("MEGATRON_HET_EP_DEBUG") != "1":
            return super().finish_grad_sync(force_all_reduce=force_all_reduce)

        _debug_heterogeneous_ep(
            f"ddp finish_grad_sync dense={len(self.bucket_groups)} "
            f"expert={len(self.expert_parallel_bucket_groups)}"
        )
        for idx, bucket_group in enumerate(self.bucket_groups):
            _debug_heterogeneous_ep(f"dense bucket_group start {idx}")
            bucket_group.finish_grad_sync(force_all_reduce=force_all_reduce)
            _debug_heterogeneous_ep(f"dense bucket_group done {idx}")
        for idx, bucket_group in enumerate(self.expert_parallel_bucket_groups):
            _debug_heterogeneous_ep(f"expert bucket_group start {idx}")
            bucket_group.finish_grad_sync(force_all_reduce=force_all_reduce)
            _debug_heterogeneous_ep(f"expert bucket_group done {idx}")
        _debug_heterogeneous_ep("ddp finish_grad_sync done")
