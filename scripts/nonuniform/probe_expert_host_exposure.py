"""Isolate native expert host submission using identical eager GPU operations.

This is a local expert microbenchmark, not a distributed parity measurement.
A stream delay BEFORE the measured interval lets the CPU pre-submit exactly
the same work. It must not be counted as a training optimization.
"""

import argparse
import inspect
import json
import statistics
import time
from pathlib import Path

import torch
import transformer_engine.pytorch as te

from megatron.core.fusions.fused_weighted_squared_relu import weighted_squared_relu_impl


class Experts(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        options = dict(
            num_gemms=count,
            bias=False,
            params_dtype=torch.bfloat16,
            fuse_wgrad_accumulation=True,
            parallel_mode=None,
            tp_size=1,
        )
        self.fc1 = te.GroupedLinear(in_features=1152, out_features=3072, **options)
        self.fc2 = te.GroupedLinear(in_features=3072, out_features=1152, **options)
        for parameter in self.parameters():
            parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)

    def forward(self, x, splits, probs):
        return self.fc2(weighted_squared_relu_impl(self.fc1(x, splits), probs), splits)


def run(output, samples):
    assert str(output.resolve()).startswith('/lustre/')
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    (output / 'te_grouped_linear_source.py').write_text(inspect.getsource(te.GroupedLinear))
    report = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cases': []}
    for sequence in (8192, 16384):
        for count in (32, 42, 43):
            # ETP1, TP2, MBS2, top-k6, 512 experts. Proportional local batch.
            ep = 16 if count == 32 else 12
            tokens = 2 * sequence // 2 * ep * 6 // 512
            splits = [tokens] * count
            model = Experts(count).cuda()
            x = (
                torch.randn(sum(splits), 1152, device='cuda', dtype=torch.bfloat16) * 0.01
            ).requires_grad_()
            grad = torch.randn_like(x) * 0.01
            x.grad = torch.zeros_like(x)
            probs = torch.full(
                (sum(splits), 1), 1 / 6, device='cuda', dtype=torch.float32, requires_grad=True
            )
            probs.grad = torch.zeros_like(probs)

            def zero():
                x.grad.zero_()
                probs.grad.zero_()
                for parameter in model.parameters():
                    parameter.main_grad.zero_()

            def work():
                y = model(x, splits, probs)
                y.backward(grad)
                return y

            for _ in range(15):
                zero()
                work()
            torch.cuda.synchronize()
            # Calibrate a ~50ms head start on this GPU; exclude it from timing.
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            torch.cuda._sleep(10_000_000)
            end.record()
            end.synchronize()
            cycles = int(10_000_000 * 50 / start.elapsed_time(end))
            markers = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            for marker in markers:
                marker.record()
            torch.cuda.synchronize()
            rows = []
            reference = None

            def trial(prequeued, check=False):
                nonlocal reference
                zero()
                torch.cuda.synchronize()
                anchor, begin, middle, finish = markers
                anchor.record()
                anchor.synchronize()
                t0 = time.perf_counter_ns()
                if prequeued:
                    torch.cuda._sleep(cycles)
                begin.record()
                fcpu = time.thread_time_ns()
                y = model(x, splits, probs)
                fcpu = (time.thread_time_ns() - fcpu) / 1e6
                middle.record()
                bcpu = time.thread_time_ns()
                y.backward(grad)
                bcpu = (time.thread_time_ns() - bcpu) / 1e6
                finish.record()
                submitted_ms = (time.perf_counter_ns() - t0) / 1e6
                finish.synchronize()
                result = dict(
                    prequeued=prequeued,
                    forward_ms=begin.elapsed_time(middle),
                    backward_ms=middle.elapsed_time(finish),
                    total_ms=begin.elapsed_time(finish),
                    forward_cpu_ms=fcpu,
                    backward_cpu_ms=bcpu,
                    submission_ms=submitted_ms,
                    gpu_start_delay_ms=anchor.elapsed_time(begin),
                )
                if check:
                    tensors = [y, x.grad, probs.grad] + [p.main_grad for p in model.parameters()]
                    if reference is None:
                        reference = [t.clone() for t in tensors]
                    else:
                        for actual, expected in zip(tensors, reference):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        reference = None
                return result

            trial(False, check=True)
            trial(True, check=True)
            for index in range(samples):
                # Alternate order to reduce monotonic drift confounding.
                for prequeued in (False, True) if index % 2 == 0 else (True, False):
                    rows.append(trial(prequeued))
            diagnostic = []
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                for prequeued in (False, True, True, False):
                    with torch.profiler.record_function('prequeued' if prequeued else 'normal'):
                        diagnostic.append(trial(prequeued))
            prof.export_chrome_trace(str(output / f'seq{sequence}_experts{count}.json'))
            case = dict(
                sequence=sequence,
                experts=count,
                ep=ep,
                tokens_per_expert=tokens,
                rows=rows,
                diagnostic_profile_rows=diagnostic,
                correctness='bitwise equal',
            )
            report['cases'].append(case)
            (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
            for mode in (False, True):
                values = [r for r in rows if r['prequeued'] == mode]
                print(
                    'RESULT',
                    sequence,
                    count,
                    'prequeued',
                    mode,
                    {
                        k: round(statistics.mean(r[k] for r in values), 4)
                        for k in values[0]
                        if k != 'prequeued'
                    },
                    flush=True,
                )
            del model, x, grad, probs
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=20)
    args = parser.parse_args()
    run(args.output, args.samples)
