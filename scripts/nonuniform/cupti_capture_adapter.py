"""Native GPU activity capture driven by existing training profiler boundaries."""

import ctypes
import functools
import os
import time
from pathlib import Path


class GraphEvents:
    """Sparse asynchronous graph envelopes; elapsed-time reads happen after GPU drain."""

    def __init__(self, torch, lib, capacity=4096):
        self.torch, self.lib, self.capacity = torch, lib, capacity
        self.active = False
        self.records = []
        original = torch.cuda.CUDAGraph.replay

        def replay(graph, *args, **kwargs):
            if not self.active:
                return original(graph, *args, **kwargs)
            assert len(self.records) < len(self.pool), 'Graph-event pool exhausted'
            begin, end = self.pool[len(self.records)]
            stream = torch.cuda.current_stream()
            begin.record(stream)
            result = original(graph, *args, **kwargs)
            end.record(stream)
            self.records.append((begin, end))
            return result

        torch.cuda.CUDAGraph.replay = replay
        lib.gpu_trace_graph_span.argtypes = [ctypes.c_uint64, ctypes.c_uint64]

    def start(self):
        self.pool = [(self.torch.cuda.Event(enable_timing=True),
                      self.torch.cuda.Event(enable_timing=True)) for _ in range(self.capacity)]
        # Materialize CUDA event handles outside the measured window.
        for pair in self.pool:
            for event in pair:
                event.record()
        self.torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            anchor = self.torch.cuda.Event(enable_timing=True)
            anchor.record()
            anchor.synchronize()
            before = time.time_ns()
            anchor.record()
            anchor.synchronize()
            after = time.time_ns()
            samples.append((after-before, anchor, (after+before)//2))
        width, self.anchor, self.epoch = min(samples, key=lambda s:s[0])
        print(f'GPU_GRAPH_EVENT_CLOCK_ERROR_BOUND_NS {width//2}', flush=True)
        assert width < 400000, ('GPU/CPU clock anchor uncertainty', width)
        self.active = True

    def finish(self):
        self.active = False
        # The caller has synchronized all GPU work before reading elapsed times.
        for begin, end in self.records:
            self.lib.gpu_trace_graph_span(
                self.epoch + round(self.anchor.elapsed_time(begin)*1000000),
                self.epoch + round(self.anchor.elapsed_time(end)*1000000),
            )
        print(f'GPU_GRAPH_EVENT_SPANS {len(self.records)}', flush=True)


class CuptiCapture:
    lib = None
    active = False
    disabled = False
    initialized = False

    @classmethod
    def initialize(cls):
        cls.disabled = os.environ.get('CUPTI_COLLECTION_DISABLED') == '1'
        if cls.disabled:
            print('PROFILE_EFFECTIVE mode=control_no_cupti', flush=True)
            return
        if os.environ.get('CUPTI_BACKEND_LIBRARY'):
            ctypes.CDLL(os.environ['CUPTI_BACKEND_LIBRARY'], mode=ctypes.RTLD_GLOBAL)
        cls.lib = ctypes.CDLL(os.environ['CUPTI_TRACE_LIBRARY'])
        # Tiny timestamp/range calls must not hand off the GIL to other threads.
        fast = ctypes.PyDLL(os.environ['CUPTI_TRACE_LIBRARY'])
        cls.lib.gpu_trace_timestamp = fast.gpu_trace_timestamp
        cls.lib.gpu_trace_range = fast.gpu_trace_range
        cls.lib.gpu_trace_timestamp.restype = ctypes.c_uint64
        cls.lib.gpu_trace_range.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64]
        cls.lib.gpu_trace_stop.argtypes = [ctypes.c_char_p]
        if os.environ.get('CUPTI_LATE_INIT') != '1':
            cls.initialize_backend()

    @classmethod
    def initialize_backend(cls):
        if cls.initialized:
            return
        cls.lib.gpu_trace_init()
        before = time.time_ns()
        timestamp = cls.lib.gpu_trace_timestamp()
        after = time.time_ns()
        assert before <= timestamp <= after, (before, timestamp, after)
        cls.lib.gpu_trace_timestamp = time.time_ns
        if os.environ.get('CUPTI_GPU_HES') == '1':
            assert ctypes.CDLL('libcuda.so.1').cuInit(0) == 0
            cls.lib.gpu_trace_enable_hes()
        if hasattr(cls.lib, 'gpu_trace_prepare'):
            cls.lib.gpu_trace_prepare()
        cls.initialized = True

    @classmethod
    def install_stage_labels(cls):
        if cls.disabled:
            return
        from megatron.core.distributed.nonuniform_ep import NonuniformEPNCCLParamAndGradBucketGroup

        def wrap(function, name):
            encoded = ('nep.' + name).encode()

            @functools.wraps(function)
            def measured(*args, **kwargs):
                if not cls.active:
                    return function(*args, **kwargs)
                start = cls.lib.gpu_trace_timestamp()
                try:
                    return function(*args, **kwargs)
                finally:
                    cls.lib.gpu_trace_range(encoded, start, cls.lib.gpu_trace_timestamp())

            return measured

        for name in (
            '_foreach_copy_',
            '_copy_nep_nccl_contexts_to_distopt_grads',
            '_scatter_nep_distopt_owner_params_to_physical_holders',
        ):
            owner = NonuniformEPNCCLParamAndGradBucketGroup
            setattr(owner, name, wrap(getattr(owner, name), name))

    def __init__(self, *, schedule, execution_trace_observer=None, **kwargs):
        import torch

        assert execution_trace_observer is None
        self.execution_trace_observer = None
        self.torch = torch
        self.schedule = schedule
        self.step_num = 0
        self.range_start = None
        self.range_name = None
        self.group = torch.distributed.new_group(backend='gloo')
        self.graph_events = (GraphEvents(torch, self.lib)
                             if not self.disabled and os.environ.get('CUPTI_GRAPH_EVENTS') == '1' else None)

    def start(self):
        self._advance()

    def step(self):
        self.step_num += 1
        self._advance()

    def _close_range(self):
        if self.range_start is not None:
            self.lib.gpu_trace_range(self.range_name, self.range_start, self.lib.gpu_trace_timestamp())
            self.range_start = None

    def _advance(self):
        if self.disabled:
            return
        self._close_range()
        action = self.schedule(self.step_num)
        recording = action in (
            self.torch.profiler.ProfilerAction.WARMUP,
            self.torch.profiler.ProfilerAction.RECORD,
            self.torch.profiler.ProfilerAction.RECORD_AND_SAVE,
        )
        if recording and not type(self).active:
            self.torch.cuda.synchronize()
            type(self).initialize_backend()
            if self.graph_events is not None:
                self.graph_events.start()
            self.torch.distributed.monitored_barrier(group=self.group)
            self.lib.gpu_trace_start()
            self.torch.distributed.monitored_barrier(group=self.group)
            type(self).active = True
            print(f'PROFILE_EFFECTIVE mode=cupti capture_start_step={self.step_num}', flush=True)
        elif not recording and type(self).active:
            self.stop()
        if type(self).active:
            self.range_name = f'iteration_{self.step_num}'.encode()
            self.range_start = self.lib.gpu_trace_timestamp()

    def stop(self):
        self._close_range()
        if type(self).active:
            type(self).active = False
            self.torch.cuda.synchronize()
            if self.graph_events is not None:
                self.graph_events.finish()
            root = Path(os.environ['CUPTI_OUTPUT_DIR'])
            root.mkdir(parents=True, exist_ok=True)
            self.lib.gpu_trace_stop(str(root / f'rank-{os.environ["RANK"]}.json').encode())
            print(f'PROFILE_EFFECTIVE mode=cupti capture_stop_step={self.step_num}', flush=True)
