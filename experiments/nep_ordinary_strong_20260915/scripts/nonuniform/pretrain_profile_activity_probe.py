"""Run the trusted entrypoint with an explicit profiler activity selection."""

import os
import runpy
import sys
from pathlib import Path

def install_profile_mode(mode):
    if mode == 'cupti':
        if os.environ.get('CUPTI_EVENTS_ONLY') == '1':
            from cuda_event_capture_adapter import CudaEventCapture as CuptiCapture
        else:
            from cupti_capture_adapter import CuptiCapture

        CuptiCapture.initialize()
        import torch

        torch.profiler.profile = CuptiCapture
        return
    import torch

    if mode == 'nsys':
        from nsys_capture_adapter import NsightCapture

        torch.profiler.profile = NsightCapture
        return
    activities = {
        'cpu_cuda': [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        'cuda': [torch.profiler.ProfilerActivity.CUDA],
    }[mode]
    original_profile = torch.profiler.profile

    class ActivityProfile(original_profile):
        def __init__(self, *args, **kwargs):
            kwargs['activities'] = activities
            super().__init__(*args, **kwargs)
            assert self.activities == set(activities)

        def prepare_trace(self):
            super().prepare_trace()
            assert self.profiler.use_cpu == (mode == 'cpu_cuda')
            print(
                f'PROFILE_EFFECTIVE mode={mode} activities={self.activities} '
                f'use_cpu={self.profiler.use_cpu}',
                flush=True,
            )

    torch.profiler.profile = ActivityProfile


def main():
    install_profile_mode(os.environ['PROFILE_ACTIVITY_MODE'])
    repo = Path(os.environ['CODE_REPO'])
    sys.path.insert(0, str(repo))
    # The NEP trampoline execv's this entrypoint, discarding Python patches.
    os.environ['MEGATRON_NONUNIFORM_EP_ENTRYPOINT'] = '1'
    if os.environ['PROFILE_ACTIVITY_MODE'] == 'cupti':
        if os.environ.get('CUPTI_EVENTS_ONLY') == '1':
            from cuda_event_capture_adapter import CudaEventCapture as CuptiCapture
        else:
            from cupti_capture_adapter import CuptiCapture

        CuptiCapture.install_stage_labels()
    runpy.run_path(str(repo / 'pretrain_hybrid.py'), run_name='__main__')


if __name__ == '__main__':
    main()
