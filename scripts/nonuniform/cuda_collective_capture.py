"""Collective input-ready-to-completion ranges without waiting on compute.

Start events use the issuing stream. Completion events use per-PG observer
streams ordered after existing Work handles. These ranges include launch/queue
and participant waiting; they are not NCCL kernel-only service durations.
No collective, existing handle, or native compute-stream wait is replaced.
"""

import contextlib
import functools
import threading
import time

import torch
import torch.distributed as dist

_installed = False
_groups = {}
_state = threading.local()


def _group_info(group):
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    if group not in _groups:
        assert dist.get_backend(group) == 'nccl'
        _groups[group] = (
            dict(group_id=str(group.group_name), group_desc=group.group_desc,
                 ranks=dist.get_process_group_ranks(group)),
            torch.cuda.Stream(),
        )
    return _groups[group]


def _payload(kind, args, kwargs):
    if kind in ('all_gather', 'reduce_scatter', 'all_to_all'):
        output, source = args[:2]
    else:
        output = source = args[0]
    result = dict(op=kind, input_bytes=source.numel() * source.element_size(),
                  output_bytes=output.numel() * output.element_size(), dtype=str(source.dtype))
    if kind == 'all_to_all':
        result['input_splits'] = kwargs.get('input_split_sizes')
        result['output_splits'] = kwargs.get('output_split_sizes')
    if kind == 'broadcast':
        result['source_rank'] = kwargs.get('src', args[1] if len(args) > 1 else None)
    return result


def _begin(capture, metadata, operations):
    index = len(capture.records)
    assert index < len(capture.pool), 'Collective event pool exhausted'
    begin, _ = capture.pool[index]
    stream = torch.cuda.current_stream()
    kind = '+'.join(sorted({op['op'] for op in operations}))
    record = dict(name='collective.' + kind, iteration=capture.step_num,
                  thread=threading.get_native_id(), index=index, stream=stream.cuda_stream,
                  collective=dict(**metadata, operations=operations,
                                  interval='input_ready_to_completion'))
    capture.records.append(record)
    begin.record(stream)
    record['cpu_start_ns'] = time.time_ns()
    record['_cpu_start_thread_ns'] = time.thread_time_ns()
    return record


def _finish(capture, record, work, observer):
    record['cpu_end_ns'] = time.time_ns()
    record['thread_cpu_ns'] = time.thread_time_ns() - record.pop('_cpu_start_thread_ns')
    _, finish = capture.pool[record['index']]
    overhead_start = time.time_ns()
    if work is None:
        # Native synchronous collectives already order the issuing stream.
        finish.record(torch.cuda.current_stream())
        record['completion_stream'] = record['stream']
    else:
        # This adds only NCCL -> observer edges, never observer -> compute.
        with torch.cuda.stream(observer):
            work.wait()
            finish.record(observer)
        record['completion_stream'] = observer.cuda_stream
    record['observer_enqueue_cpu_ns'] = time.time_ns() - overhead_start


def install_collective_capture(capture_class):
    """Cover native Megatron coalesced AG/RS and standalone NEP A2A/broadcast."""
    global _installed
    if _installed:
        return
    import megatron.core.distributed.param_and_grad_buffer as buffers

    def wrap_collective(original, kind, group_position):
        @functools.wraps(original)
        def measured(*args, **kwargs):
            group = kwargs.get('group', args[group_position] if len(args) > group_position else None)
            if not dist.is_initialized() or dist.get_backend(group) != 'nccl':
                return original(*args, **kwargs)
            metadata, observer = _group_info(group)
            capture = capture_class.current
            if capture is None or not capture.active:
                return original(*args, **kwargs)
            assert not torch.cuda.is_current_stream_capturing(), 'Collectives inside a graph need kernel profiling'
            operation = _payload(kind, args, kwargs)
            stack = getattr(_state, 'coalescing', [])
            if stack:
                stack[-1]['operations'].append(operation)
                return original(*args, **kwargs)
            record = _begin(capture, metadata, [operation])
            work = original(*args, **kwargs)
            _finish(capture, record, work, observer)
            return work
        return measured

    wrapped = {}
    for attribute, kind, position in (
        ('all_gather_into_tensor', 'all_gather', 2),
        ('reduce_scatter_tensor', 'reduce_scatter', 3),
        ('all_reduce', 'all_reduce', 2),
        ('all_to_all_single', 'all_to_all', 4),
        ('broadcast', 'broadcast', 2),
    ):
        original = getattr(dist, attribute)
        measured = wrap_collective(original, kind, position)
        wrapped[original] = measured
        setattr(dist, attribute, measured)
    # Native Megatron imports these aliases before installing the profiler.
    for attribute in ('dist_all_gather_func', 'dist_reduce_scatter_func'):
        original = getattr(buffers, attribute)
        assert original in wrapped, f'Unsupported collective alias: {attribute}'
        setattr(buffers, attribute, wrapped[original])

    original_manager = buffers._coalescing_manager

    @contextlib.contextmanager
    def measured_manager(group=None, device=None, async_ops=False):
        metadata, observer = _group_info(group)
        capture = capture_class.current
        if capture is None or not capture.active:
            with original_manager(group=group, device=device, async_ops=async_ops) as manager:
                yield manager
            return
        stack = getattr(_state, 'coalescing', None)
        if stack is None:
            stack = _state.coalescing = []
        context = dict(operations=[])
        assert not stack, 'Nested coalescing must be explicitly accounted for'
        record = None
        try:
            with original_manager(group=group, device=device, async_ops=async_ops) as manager:
                stack.append(context)
                yield manager
                assert context['operations'], 'Coalesced collectives were not instrumented'
                # Coalesced NCCL is issued by the native manager on exit.
                record = _begin(capture, metadata, context['operations'])
        finally:
            if stack and stack[-1] is context:
                stack.pop()
        _finish(capture, record, manager if async_ops else None, observer)

    buffers._coalescing_manager = measured_manager
    _installed = True
