"""Remove all broad profiling, optionally retaining only final-sync milestones."""

import contextlib
import functools
import json
import os
import runpy
import sys
import time
from pathlib import Path

import pretrain_nep_collective_probe as probe
import pretrain_profile_activity_probe as entry


class TailMarkers:
    """Workload-specific diagnostic; no collective injection or in-window query."""

    def __init__(self):
        import torch

        self.torch = torch
        self.active = False
        self.initialized = False
        self.records = []
        self.context = None
        self.counts = {}
        self.groups = {}

    def initialize(self):
        torch = self.torch
        self.rank = torch.distributed.get_rank()
        self.nep = torch.distributed.get_world_size() == 14
        self.expert_count = (0 if self.rank in (6, 7) else 8) if self.nep else 7
        self.pool = [torch.cuda.Event(enable_timing=True) for _ in range(256)]
        self.observers = {kind: torch.cuda.Stream() for kind in ('expert', 'dense')}
        for event in self.pool:
            event.record()
        torch.cuda.synchronize()
        self.origin = torch.cuda.Event(enable_timing=True)
        before = time.time_ns()
        self.origin.record()
        self.origin.synchronize()
        after = time.time_ns()
        self.origin_ns = (before + after) // 2
        self.clock_error_bound_ns = (after - before) // 2
        self.initialized = True

    def mark(self, name, metadata=None):
        event = self.pool[len(self.records)]
        event.record()
        self.records.append((self.step, name, metadata, event))

    def install(self):
        import megatron.core.distributed.param_and_grad_buffer as buffers
        import megatron.training.training as training
        from megatron.core.distributed.nonuniform_ep import NonuniformEPNCCLParamAndGradBucketGroup
        from megatron.core.optimizer.optimizer import ChainedOptimizer
        from transformer_engine.pytorch.module.grouped_linear import _GroupedLinear

        original_step = training.train_step

        @functools.wraps(original_step)
        def train_step(*args, **kwargs):
            if not self.initialized:
                self.initialize()
            self.step = kwargs['iteration'] + 1
            self.active = 40 <= self.step < 50
            self.counts = dict(backward=0, stage=0, expert=0, dense=0)
            result = original_step(*args, **kwargs)
            if self.active:
                assert self.counts == dict(
                    backward=90, stage=self.expert_count if self.nep else 0,
                    expert=self.expert_count, dense=2,
                ), (self.rank, self.step, self.counts)
            self.active = False
            return result

        training.train_step = train_step
        original_backward = _GroupedLinear.backward

        @functools.wraps(original_backward)
        def backward(*args, **kwargs):
            if not self.active:
                return original_backward(*args, **kwargs)
            self.counts['backward'] += 1
            index = self.counts['backward']
            if index == 61:
                self.mark('final_backward_start')
            result = original_backward(*args, **kwargs)
            if index == 90:
                self.mark('final_backward_end')
            return result

        _GroupedLinear.backward = staticmethod(backward)
        original_stage = NonuniformEPNCCLParamAndGradBucketGroup._copy_nep_nccl_contexts_to_distopt_grads

        @functools.wraps(original_stage)
        def stage(group, *args, **kwargs):
            selected = False
            if self.active:
                self.counts['stage'] += 1
                selected = self.counts['stage'] == self.expert_count
            if selected:
                self.mark('last_stage_start')
            result = original_stage(group, *args, **kwargs)
            if selected:
                self.mark('last_stage_end')
            return result

        NonuniformEPNCCLParamAndGradBucketGroup._copy_nep_nccl_contexts_to_distopt_grads = stage
        original_grad = buffers._ParamAndGradBucketGroup.start_grad_sync

        @functools.wraps(original_grad)
        def grad(group, *args, **kwargs):
            if not self.active:
                return original_grad(group, *args, **kwargs)
            pg = group.intra_distributed_optimizer_instance_group
            desc = pg.group_desc.lower()
            kind = 'expert' if desc == 'ep_dp' or 'expert_data_parallel' in desc else 'dense'
            self.counts[kind] += 1
            selected = self.counts[kind] == (self.expert_count if kind == 'expert' else 2)
            previous = self.context
            self.context = kind if selected else None
            try:
                return original_grad(group, *args, **kwargs)
            finally:
                self.context = previous

        buffers._ParamAndGradBucketGroup.start_grad_sync = grad
        original_manager = buffers._coalescing_manager

        @contextlib.contextmanager
        def manager(group=None, device=None, async_ops=False):
            label = self.context
            selected = self.active and label is not None
            if selected and group not in self.groups:
                self.groups[group] = dict(
                    desc=group.group_desc, ranks=self.torch.distributed.get_process_group_ranks(group)
                )
            with original_manager(group=group, device=device, async_ops=async_ops) as work:
                yield work
                if selected:
                    self.mark(label + '_rs_ready', self.groups[group])
            if selected:
                assert async_ops
                # Only an observer dependency; the training streams never wait on it.
                with self.torch.cuda.stream(self.observers[label]):
                    work.wait()
                    self.mark(label + '_rs_done', self.groups[group])

        buffers._coalescing_manager = manager
        original_optimizer = ChainedOptimizer.step

        @functools.wraps(original_optimizer)
        def optimizer(*args, **kwargs):
            if self.active:
                self.mark('optimizer_start')
            return original_optimizer(*args, **kwargs)

        ChainedOptimizer.step = optimizer

    def dump(self):
        self.torch.cuda.synchronize()
        output = Path(os.environ['CUPTI_OUTPUT_DIR']).parent / 'tail_markers'
        output.mkdir(exist_ok=True)
        events = [dict(iteration=i, name=n, metadata=m, ms=self.origin.elapsed_time(e))
                  for i, n, m, e in self.records]
        (output / f'rank-{self.rank}.json').write_text(json.dumps(dict(
            origin_ns=self.origin_ns, clock_error_bound_ns=self.clock_error_bound_ns,
            events=events, note='Sparse event milestones, not kernel profiling.'
        ), indent=2) + '\n')
        print('TAIL_MARKERS_RESULT', self.rank, len(events), flush=True)


def raw_main():
    repo = Path(os.environ['CODE_REPO'])
    sys.path.insert(0, str(repo))
    os.environ['MEGATRON_NONUNIFORM_EP_ENTRYPOINT'] = '1'
    markers = None
    if os.environ['NEP_TAIL_MODE'] == 'sparse':
        markers = TailMarkers()
        markers.install()
    assert 'cuda_event_capture_adapter' not in sys.modules
    assert 'cuda_collective_capture' not in sys.modules
    assert '--profile' not in sys.argv
    print('PROFILE_EFFECTIVE mode=control_no_cupti instrumentation=' + os.environ['NEP_TAIL_MODE'], flush=True)
    print('TAIL_CONTROL_NO_BROAD_PROFILER', os.environ['NEP_TAIL_MODE'], flush=True)
    runpy.run_path(str(repo / 'pretrain_hybrid.py'), run_name='__main__')
    if markers is not None:
        markers.dump()


def change_options(before, variant):
    after = probe.change_options(before, variant)
    assert after.count('--profile') == 1
    after.remove('--profile')
    return after


if __name__ == '__main__':
    assert os.environ['NEP_TAIL_MODE'] in ('none', 'sparse')
    probe.base.EXPECTED_SHA = os.environ['NEP_PREFETCH_EXPECTED_SHA']
    probe.base.change_options = change_options
    probe.base.CHANGES.update(probe.CHANGES)
    entry.main = raw_main
    probe.base.main()
