"""Targeted CPU and CUDA-stream operation ranges without CUPTI instrumentation.

GPU ranges include dependency and launch gaps within an operation. They are not
kernel busy time, and nested ranges must not be added together.
"""

import functools
import inspect
import json
import os
import threading
import time
from pathlib import Path

import torch


class CudaEventCapture:
    current = None
    installed = False

    @classmethod
    def initialize(cls):
        pass

    @classmethod
    def wrap(cls, owner, attribute, label):
        descriptor = inspect.getattr_static(owner, attribute)
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            capture = cls.current
            if capture is None or not capture.active:
                return original(*args, **kwargs)
            assert not torch.cuda.is_current_stream_capturing(), label
            index = len(capture.records)
            assert index < len(capture.pool), 'CUDA-event pool exhausted'
            begin, end = capture.pool[index]
            record = {'name': label, 'iteration': capture.step_num,
                      'thread': threading.get_native_id(), 'index': index}
            capture.records.append(record)
            stream = torch.cuda.current_stream()
            record['stream'] = stream.cuda_stream
            begin.record(stream)
            wall, cpu = time.time_ns(), time.thread_time_ns()
            try:
                return original(*args, **kwargs)
            finally:
                record.update(cpu_start_ns=wall, cpu_end_ns=time.time_ns(),
                              thread_cpu_ns=time.thread_time_ns()-cpu)
                end.record(stream)

        if isinstance(descriptor, staticmethod):
            measured = staticmethod(measured)
        elif isinstance(descriptor, classmethod):
            raise TypeError('Explicit classmethod handling required')
        setattr(owner, attribute, measured)

    @classmethod
    def install_stage_labels(cls):
        if cls.installed:
            return
        from transformer_engine.pytorch.module.grouped_linear import _GroupedLinear
        from megatron.core.optimizer.optimizer import ChainedOptimizer
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
        from megatron.core.distributed.nonuniform_ep import NonuniformEPNCCLParamAndGradBucketGroup
        from cuda_collective_capture import install_collective_capture

        install_collective_capture(cls)

        for owner, method, label in [
            (_GroupedLinear, 'forward', 'expert.forward'),
            (_GroupedLinear, 'backward', 'expert.backward'),
            (ChainedOptimizer, 'step', 'optimizer.step'),
            (DistributedOptimizer, '_copy_main_params_to_model_params', 'optimizer.copy_params'),
            (NonuniformEPNCCLParamAndGradBucketGroup,
             '_copy_nep_nccl_contexts_to_distopt_grads', 'nep.stage_grads'),
            (NonuniformEPNCCLParamAndGradBucketGroup,
             '_scatter_nep_distopt_owner_params_to_physical_holders', 'nep.redistribute_params'),
        ]:
            cls.wrap(owner, method, label)
        cls.installed = True

    def __init__(self, *, schedule, execution_trace_observer=None, **kwargs):
        assert execution_trace_observer is None
        self.execution_trace_observer = None
        self.schedule, self.step_num = schedule, 0
        self.active = False
        self.disabled = os.environ.get('CUPTI_COLLECTION_DISABLED') == '1'
        self.records, self.ranges = [], []
        self.range_start = None
        type(self).current = self
        mode = 'control_no_cupti' if self.disabled else 'cupti backend=cuda_events'
        print(f'PROFILE_EFFECTIVE mode={mode}', flush=True)

    def start(self):
        self._advance()

    def step(self):
        self.step_num += 1
        self._advance()

    def _close_range(self):
        if self.range_start is not None:
            iteration, start = self.range_start
            self.ranges.append((iteration, start, time.time_ns()))
            self.range_start = None

    def _advance(self):
        if self.disabled:
            return
        self._close_range()
        recording = self.schedule(self.step_num) in (
            torch.profiler.ProfilerAction.WARMUP,
            torch.profiler.ProfilerAction.RECORD,
            torch.profiler.ProfilerAction.RECORD_AND_SAVE,
        )
        if recording and not self.active:
            # Event creation/materialization and clock calibration are outside the
            # measured iterations. There are no event queries inside the window.
            pool_size = int(os.environ.get('CUDA_EVENT_POOL_SIZE', '4096'))
            assert pool_size > 0
            self.pool = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                         for _ in range(pool_size)]
            for pair in self.pool:
                for event in pair:
                    event.record()
            torch.cuda.synchronize()
            samples = []
            for _ in range(5):
                anchor = torch.cuda.Event(enable_timing=True)
                anchor.record()
                anchor.synchronize()
                before = time.time_ns()
                anchor.record()
                anchor.synchronize()
                after = time.time_ns()
                samples.append((after-before, anchor, (after+before)//2))
            width, self.anchor, self.epoch = min(samples, key=lambda x: x[0])
            assert width < 400000, ('Clock calibration uncertainty', width)
            self.clock_error_ns = width//2
            self.active = True
        elif not recording and self.active:
            self.stop()
        if self.active:
            self.range_start = self.step_num, time.time_ns()

    def stop(self):
        self._close_range()
        if not self.active:
            return
        self.active = False
        torch.cuda.synchronize()
        events = []
        origin = self.ranges[0][1]
        for record in self.records:
            begin, end = self.pool[record['index']]
            a = self.epoch + round(self.anchor.elapsed_time(begin)*1e6)
            b = self.epoch + round(self.anchor.elapsed_time(end)*1e6)
            assert b >= a, record
            events.append(dict(name=record['name'], cat='gpu_operation_range', ph='X',
                               ts=(a-origin)/1000, dur=(b-a)/1000,
                               pid=int(os.environ.get('RANK', '0')), tid=str(record['stream']),
                               args={'iteration': record['iteration'], 'stream': record['stream'],
                                     **({'collective': record['collective'],
                                         'completion_stream': record['completion_stream']}
                                        if 'collective' in record else {})}))
            events.append(dict(name=record['name'], cat='cpu_operation_range', ph='X',
                               ts=(record['cpu_start_ns']-origin)/1000,
                               dur=(record['cpu_end_ns']-record['cpu_start_ns'])/1000,
                               pid='CPU', tid=record['thread'],
                               args={'iteration': record['iteration'],
                                     'thread_cpu_us': record['thread_cpu_ns']/1000,
                                     **({'collective': record['collective'],
                                         'observer_enqueue_cpu_us': record['observer_enqueue_cpu_ns']/1000}
                                        if 'collective' in record else {})}))
        for iteration, a, b in self.ranges:
            events.append(dict(name=f'iteration_{iteration}', cat='iteration', ph='X',
                               ts=(a-origin)/1000, dur=(b-a)/1000, pid='CPU', tid='iteration'))
        root = Path(os.environ['CUPTI_OUTPUT_DIR'])
        root.mkdir(parents=True, exist_ok=True)
        result = dict(backend='cuda_events', collective_schema_version=1, origin_ns=origin,
                      clock_error_bound_ns=self.clock_error_ns, traceEvents=events,
                      note='Operation elapsed ranges, not kernel busy time; nested ranges overlap.')
        (root/f'rank-{os.environ.get("RANK", "0")}.json').write_text(json.dumps(result)+'\n')
        print(f'GPU_EVENT_TRACE_RESULT operations={len(self.records)} '
              f'clock_error_ns={self.clock_error_ns}', flush=True)
