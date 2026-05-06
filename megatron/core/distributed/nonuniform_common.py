# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Shared helpers for opt-in nonuniform distributed wrappers.

The helpers in this file intentionally avoid modifying the generic DDP, optimizer,
or param-buffer implementations.  NTP and NEP wrappers import this module to share
DDP subclass construction, bucket-group wrapping, handle tracking, and local buffer
layout utilities.
"""

import inspect
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

from . import distributed_data_parallel as ddp_module


@dataclass
class PerBufferParamLayout:
    """Layout for parameters within one opt-in contiguous DDP buffer."""

    param_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(default_factory=dict)
    side_grad_index_map: Dict[torch.nn.Parameter, Tuple[int, int, int]] = field(
        default_factory=dict
    )
    bucket_indices: List[Tuple[int, int]] = field(default_factory=list)
    per_bucket_numel_unpadded: List[int] = field(default_factory=list)
    param_indices: List[int] = field(default_factory=list)


@dataclass
class FullParamLayout:
    """Compatibility placeholder for wrappers that pass precomputed layouts."""

    layouts: Dict[object, PerBufferParamLayout] = field(default_factory=dict)


def pad_to_divisor(value: int, divisor: int) -> int:
    """Round up ``value`` to the nearest multiple of ``divisor``."""
    return int(math.ceil(value / divisor) * divisor)


def pad_param_start(param_start_index: int) -> int:
    """Align parameter start index to a 64-element boundary."""
    return pad_to_divisor(param_start_index, 64)


def pad_bucket_end(
    bucket_end_index: int, data_parallel_world_size: int, pad_for_high_nccl_busbw: bool
) -> int:
    """Pad bucket end for DP divisibility and optionally high NCCL bus bandwidth."""
    if pad_for_high_nccl_busbw:
        divisor = math.lcm(data_parallel_world_size, 128, 2**16)
    else:
        divisor = math.lcm(data_parallel_world_size, 128)
    return pad_to_divisor(bucket_end_index, divisor)


class ViewCopyHandle:
    """Wait handle that copies temporary contiguous receive buffers into views."""

    def __init__(self, handle, output_copies):
        self.handle = handle
        self.output_copies = output_copies

    def wait(self):
        self.handle.wait()
        for dst, src in self.output_copies:
            dst.copy_(src)
        self.output_copies = []


class CudaEventHandle:
    """Small wait handle compatible with Megatron bucket-group finish logic."""

    def __init__(self, events: List[torch.cuda.Event]):
        self.events = events

    def wait(self):
        current_stream = torch.cuda.current_stream()
        for event in self.events:
            current_stream.wait_event(event)
        self.events = []


def all_to_all_with_output_views(output_tensors, input_tensors, group, async_op: bool = False):
    """Run all_to_all, preserving non-contiguous output views via temporary buffers."""
    output_list = []
    output_copies = []
    for tensor in output_tensors:
        if tensor.is_contiguous():
            output_list.append(tensor)
        else:
            contiguous = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            output_list.append(contiguous)
            output_copies.append((tensor, contiguous))

    handle = dist.all_to_all(output_list, input_tensors, group=group, async_op=async_op)
    if async_op:
        return ViewCopyHandle(handle, output_copies)

    for dst, src in output_copies:
        dst.copy_(src)
    return None


def wait_handles(handles: Iterable[object]) -> None:
    """Wait every non-None handle in order."""
    for handle in handles:
        if handle is not None:
            handle.wait()


def record_post_sync_handles(bucket_group, state_attr: str, handles: List[object]) -> None:
    """Track post-sync handles and drain them when the last bucket group finishes."""
    state = getattr(bucket_group, state_attr, None)
    if state is None:
        wait_handles(handles)
        return

    state['handles'].extend(handles)
    if bucket_group is state['last_bucket_group']:
        try:
            wait_handles(state['handles'])
        finally:
            state['handles'] = []


def configure_post_sync_handle_tracker(bucket_groups: List[object], state_attr: str) -> None:
    """Attach a shared last-group handle tracker to ordered bucket groups."""
    if not bucket_groups:
        return
    state = {'handles': [], 'last_bucket_group': bucket_groups[-1]}
    for bucket_group in bucket_groups:
        setattr(bucket_group, state_attr, state)


def filter_kwargs_for_callable(fn: Callable, kwargs: Dict[str, object]) -> Dict[str, object]:
    """Return only kwargs accepted by ``fn``."""
    parameters = inspect.signature(fn).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


@contextmanager
def patch_ddp_param_and_grad_buffer(buffer_cls):
    """Temporarily patch DDP's imported _ParamAndGradBuffer binding."""
    original_buffer_class = ddp_module._ParamAndGradBuffer
    ddp_module._ParamAndGradBuffer = buffer_cls
    try:
        yield
    finally:
        ddp_module._ParamAndGradBuffer = original_buffer_class


def clone_bucket_group(bucket_group, wrapper_cls):
    """Clone a DDP bucket group into an opt-in subclass while preserving runtime state."""
    if isinstance(bucket_group, wrapper_cls):
        return bucket_group
    wrapped_bucket_group = wrapper_cls.__new__(wrapper_cls)
    wrapped_bucket_group.__dict__ = bucket_group.__dict__.copy()
    return wrapped_bucket_group


def wrap_bucket_groups_with_subclass(
    bucket_groups: List[object],
    wrapper_cls,
    configure_fn: Callable[[object], None],
    param_to_bucket_group: Optional[Dict[torch.nn.Parameter, object]] = None,
) -> List[object]:
    """Replace generic bucket groups with opt-in subclasses and rebuild next links."""
    wrapped_bucket_groups = []
    old_to_new = {}

    for bucket_group in bucket_groups:
        wrapped_bucket_group = clone_bucket_group(bucket_group, wrapper_cls)
        configure_fn(wrapped_bucket_group)
        old_to_new[bucket_group] = wrapped_bucket_group
        wrapped_bucket_groups.append(wrapped_bucket_group)

    for bucket_group, wrapped_bucket_group in old_to_new.items():
        next_bucket_group = getattr(bucket_group, 'next_param_gather_bucket_group', None)
        if next_bucket_group in old_to_new:
            wrapped_bucket_group.next_param_gather_bucket_group = old_to_new[next_bucket_group]

    if param_to_bucket_group is not None:
        for wrapped_bucket_group in wrapped_bucket_groups:
            for bucket in wrapped_bucket_group.buckets:
                for param in bucket.params_list:
                    param_to_bucket_group[param] = wrapped_bucket_group

    return wrapped_bucket_groups


def configure_ordered_bucket_group_scheduler(
    bucket_groups: List[object],
    state_attr: str,
    index_attr: str,
    ready_attr: str,
) -> None:
    """Attach deterministic launch state to bucket groups that must start in rank order."""
    state = {"groups": bucket_groups, "next_index": 0}
    for index, bucket_group in enumerate(bucket_groups):
        setattr(bucket_group, state_attr, state)
        setattr(bucket_group, index_attr, index)
        setattr(bucket_group, ready_attr, False)


def try_start_ordered_bucket_groups(
    bucket_group,
    state_attr: str,
    ready_attr: str,
    start_fn_name: str,
    *args,
    **kwargs,
) -> None:
    """Start ready bucket groups in deterministic order."""
    state = getattr(bucket_group, state_attr, None)
    if state is None:
        getattr(bucket_group, start_fn_name)(*args, **kwargs)
        return

    groups = state["groups"]
    while state["next_index"] < len(groups):
        group = groups[state["next_index"]]
        if not getattr(group, ready_attr, False):
            break
        getattr(group, start_fn_name)(*args, **kwargs)
        state["next_index"] += 1


def reset_ordered_bucket_group_scheduler(bucket_group, state_attr: str, index_attr: str) -> None:
    """Reset ordered scheduler state at the first bucket group."""
    state = getattr(bucket_group, state_attr, None)
    if state is not None and getattr(bucket_group, index_attr, -1) == 0:
        state["next_index"] = 0


def get_global_rank(group, group_rank: int) -> int:
    """Translate a process-group rank to a global rank."""
    if hasattr(dist, "get_global_rank"):
        return dist.get_global_rank(group, group_rank)
    ranks = getattr(group, "ranks", None)
    if ranks is None:
        raise RuntimeError("Cannot map process-group rank to global rank")
    return ranks[group_rank]
