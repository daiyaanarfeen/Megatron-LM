"""Diagnostic-only graph-node Nsight capture plus existing all-rank event ranges."""

import functools
import importlib
import inspect
import os
import sys
from pathlib import Path


def main():
    sys.path.insert(0, os.environ['CODE_REPO'])
    import nvtx

    import cuda_collective_capture as collectives
    import pretrain_profile_activity_probe as probe
    from cuda_event_capture_adapter import CudaEventCapture
    from nsys_capture_adapter import NsightCapture
    from pretrain_a8b_native_moe_graph import main as train

    os.environ['CUPTI_OUTPUT_DIR'] = str(Path(os.environ['NSYS_OUTPUT_DIR']).parent / 'cupti_profile')
    original_wrap = CudaEventCapture.wrap

    @classmethod
    def wrap(cls, owner, attribute, label):
        original_wrap(owner, attribute, label)
        descriptor = inspect.getattr_static(owner, attribute)
        measured = getattr(owner, attribute)

        @functools.wraps(measured)
        def tagged(*args, **kwargs):
            active = cls.current is not None and cls.current.active
            if active:
                nvtx.push_range(label, domain='nep_capture')
            try:
                return measured(*args, **kwargs)
            finally:
                if active:
                    nvtx.pop_range(domain='nep_capture')

        setattr(owner, attribute, staticmethod(tagged) if isinstance(descriptor, staticmethod) else tagged)

    CudaEventCapture.wrap = wrap
    original_begin, original_finish = collectives._begin, collectives._finish

    def begin(capture, metadata, operations):
        record = original_begin(capture, metadata, operations)
        nvtx.push_range(f"{record['name']}:{metadata['group_desc']}", domain='nep_capture')
        return record

    def finish(*args, **kwargs):
        try:
            return original_finish(*args, **kwargs)
        finally:
            nvtx.pop_range(domain='nep_capture')

    collectives._begin, collectives._finish = begin, finish

    class CombinedCapture:
        def __init__(self, **kwargs):
            self.execution_trace_observer = None
            self.events = CudaEventCapture(**kwargs)
            self.nsight = NsightCapture(**kwargs)

        def start(self):
            self.events.start()
            self.nsight.start()

        def step(self):
            self.events.step()
            self.nsight.step()

        def stop(self):
            self.events.stop()
            self.nsight.stop()

    def install(mode):
        import torch

        assert mode == 'nsys'
        torch.profiler.profile = CombinedCapture

    probe.install_profile_mode = install
    from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
    from megatron.core.distributed.nonuniform_ep import NonuniformEPDistributedDataParallel
    from megatron.core.optimizer.optimizer import ChainedOptimizer
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    finalize = importlib.import_module('megatron.core.distributed.finalize_model_grads')
    for owner, method, label in [
        (finalize, 'finalize_model_grads', 'finalize.grads'),
        (DistributedDataParallel, 'finish_grad_sync', 'ddp.finish_grad_sync'),
        (DistributedDataParallel, 'scale_gradients', 'ddp.scale_gradients'),
        (NonuniformEPDistributedDataParallel, 'scale_gradients', 'nep.scale_gradients'),
        (ChainedOptimizer, 'prepare_grads', 'optimizer.prepare_grads'),
        (ChainedOptimizer, 'get_grad_norm', 'optimizer.grad_norm'),
        (ChainedOptimizer, 'step_with_ready_grads', 'optimizer.adam_step'),
        (DistributedOptimizer, '_copy_model_grads_to_main_grads', 'optimizer.copy_grads'),
    ]:
        CudaEventCapture.wrap(owner, method, label)
    train()


if __name__ == '__main__':
    main()
