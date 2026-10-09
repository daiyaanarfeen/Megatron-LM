"""Drive Nsight from Megatron's existing profiler step calls, without Kineto."""


class NsightCapture:
    def __init__(self, *, schedule, execution_trace_observer=None, **kwargs):
        import nvtx
        import torch
        import faulthandler

        assert execution_trace_observer is None
        self.execution_trace_observer = None
        self.schedule = schedule
        self.torch = torch
        self.faulthandler = faulthandler
        self.nvtx = nvtx
        self.step_num = 0
        self.active = False
        self.range_open = False
        self.capture_group = torch.distributed.new_group(backend='gloo')

    def start(self):
        self._advance()

    def step(self):
        self.step_num += 1
        self._advance()

    def _advance(self):
        if self.range_open:
            self.nvtx.pop_range(domain='nep_capture')
            self.range_open = False
        action = self.schedule(self.step_num)
        recording = action in (
            self.torch.profiler.ProfilerAction.WARMUP,
            self.torch.profiler.ProfilerAction.RECORD,
            self.torch.profiler.ProfilerAction.RECORD_AND_SAVE,
        )
        if recording and not self.active:
            # Capture activation is slow and rank-asymmetric. Keep it outside
            # measured steps, with no HybridEP work in flight during activation.
            self.torch.cuda.synchronize()
            self.torch.distributed.monitored_barrier(group=self.capture_group)
            self.torch.cuda.check_error(self.torch.cuda.cudart().cudaProfilerStart())
            self.torch.distributed.monitored_barrier(group=self.capture_group)
            self.active = True
            self.faulthandler.dump_traceback_later(45)
            print(f'PROFILE_EFFECTIVE mode=nsys capture_start_step={self.step_num}', flush=True)
        elif not recording and self.active:
            self.stop()
        if self.active:
            self.nvtx.push_range(f'iteration_{self.step_num}', domain='nep_capture')
            self.range_open = True

    def stop(self):
        if self.range_open:
            self.nvtx.pop_range(domain='nep_capture')
            self.range_open = False
        if self.active:
            self.faulthandler.cancel_dump_traceback_later()
            self.torch.cuda.check_error(self.torch.cuda.cudart().cudaProfilerStop())
            self.active = False
            print(f'PROFILE_EFFECTIVE mode=nsys capture_stop_step={self.step_num}', flush=True)
