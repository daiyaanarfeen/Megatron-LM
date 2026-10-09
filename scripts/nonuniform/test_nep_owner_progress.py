"""CPU-only regression tests for profiler accounting, runnable with unittest."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analyze_nep_closeout import scan as scan_recurring
from analyze_nep_closeout import scan_host
from analyze_nep_owner_progress import classify, exclusive, normalize_dense_labels, scan
from analyze_nep_scope_progress import scan as scan_scopes


class ProgressTests(unittest.TestCase):
    def test_recurring_window_and_host_submission_bounds(self):
        events = [
            {'cat': 'user_annotation', 'name': 'ProfilerStep#1', 'ts': 0, 'dur': 9000},
            {'cat': 'cpu_op', 'name': '_GroupedLinear', 'ts': 1000, 'dur': 4000},
            {
                'cat': 'cuda_runtime',
                'name': 'cudaLaunchKernel',
                'ts': 2000,
                'dur': 10,
                'args': {'correlation': 1},
            },
            {
                'cat': 'kernel',
                'name': 'compute',
                'ts': 2500,
                'dur': 1000,
                'args': {'correlation': 1},
            },
            {'cat': 'kernel', 'name': 'compute2', 'ts': 3000, 'dur': 1000},
        ]
        cache = {
            'rank': 0,
            'steps': [
                {
                    'step': 'ProfilerStep#1',
                    'backward_microbatches_ms': [[i, i + 1] for i in range(8)],
                }
            ],
        }
        with patch('analyze_nep_closeout.iter_trace_events', side_effect=lambda _: iter(events)):
            measured = scan_recurring(Path('unused'), cache)['windows'][0]
            host = scan_host(Path('unused'), cache)['windows'][0]
        self.assertEqual(measured['span_ms'], 7)
        self.assertEqual(measured['states_ms']['other_kernel'], 1.5)
        self.assertEqual(measured['states_ms']['idle'], 5.5)
        self.assertEqual(host['idle_before_next_launch_ms'], 1)
        self.assertEqual(host['idle_before_next_launch_in_grouped_cpu_ms'], 1)
        self.assertEqual(host['idle_in_grouped_cpu_ms'], 2.5)

    def test_label_correction_preserves_timing_and_exact_op(self):
        original = {
            'steps': [
                {
                    'span_ms': 10.0,
                    'phases_ms': {'through_backward': 10.0},
                    'operations': {'dense_grad_rs': {'residency_ms': 3.0}},
                    'collectives': [
                        {'category': 'dense_grad_rs', 'op': 'allreduce', 'elements': 1}
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case' / 'nep' / 'rank-0.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(original))
            normalize_dense_labels(Path(directory))
            corrected = json.loads(path.read_text())
        self.assertEqual(corrected['steps'][0]['span_ms'], 10.0)
        self.assertEqual(corrected['steps'][0]['phases_ms'], original['steps'][0]['phases_ms'])
        self.assertEqual(corrected['steps'][0]['operations']['dense_dp_sync']['residency_ms'], 3.0)
        self.assertEqual(corrected['steps'][0]['collectives'][0]['op'], 'allreduce')
        self.assertEqual(corrected['steps'][0]['collectives'][0]['category'], 'dense_dp_sync')

    def test_scope_idle_and_late_host_submission(self):
        def event(name, cat, start, end, correlation=None):
            return {
                'name': name,
                'cat': cat,
                'ts': start,
                'dur': end - start,
                'args': {'correlation': correlation},
            }

        events = [
            event('ProfilerStep#1', 'user_annotation', 0, 200),
            event('nep_distopt_param_scatter', 'gpu_user_annotation', 0, 100),
            event('cudaMemcpyAsync', 'cuda_runtime', 15, 17, 1),
            event('cudaMemcpyAsync', 'cuda_runtime', 55, 57, 2),
            event('Memcpy DtoD', 'gpu_memcpy', 20, 30, 1),
            event('Memcpy DtoD', 'gpu_memcpy', 60, 70, 2),
        ]
        with patch(
            'analyze_nep_scope_progress.iter_trace_events', side_effect=lambda _: iter(events)
        ):
            result = scan_scopes(Path('unused'))
        self.assertAlmostEqual(
            result['scope_states_mean_ms']['nep_distopt_param_scatter']['idle'], 0.080
        )
        self.assertAlmostEqual(
            result['idle_before_next_host_launch_mean_ms']['nep_distopt_param_scatter'], 0.040
        )
        self.assertEqual(result['scoped_gpu_events'][0]['calls_per_step'], 2)

    def test_overlap_counts_shared_time_once(self):
        self.assertEqual(
            exclusive({'a': [(0, 10)], 'b': [(5, 15)]}, 0, 20),
            {'a': 0.005, 'a+b': 0.005, 'b': 0.005, 'idle': 0.005},
        )

    def test_gradient_parameter_split(self):
        event = {
            'name': 'ncclKernel',
            'cat': 'kernel',
            'ts': 0,
            'args': {
                'Process Group Description': 'ep_dp',
                'Collective name': 'allgather_into_tensor_coalesced',
            },
        }
        self.assertEqual(classify(event, [], []), 'expert_param_ag')
        event['args']['Collective name'] = 'reduce_scatter_tensor_coalesced'
        self.assertEqual(classify(event, [], []), 'expert_grad_rs')
        event['args'].update(
            {'Process Group Description': 'owner_transfer_0', 'Collective name': 'broadcast'}
        )
        self.assertEqual(classify(event, [], []), 'param_redistribution')

    def test_gpu_optimizer_boundary_and_hook_exclusion(self):
        def event(name, cat, start, end, external=None):
            return {
                'name': name,
                'cat': cat,
                'ts': start,
                'dur': end - start,
                'tid': 1,
                'pid': 1,
                'args': {'External id': external, 'stream': 7},
            }

        events = [
            event('ProfilerStep#1', 'user_annotation', 0, 200),
            event('autograd::engine::evaluate_function: ModelBackward', 'cpu_op', 10, 12, 1),
            event('model_gradient', 'kernel', 20, 30, 1),
            event(
                'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad',
                'cpu_op',
                13,
                14,
                2,
            ),
            event('gradient_add', 'kernel', 90, 100, 2),
            event('Optimizer.step#FusedAdam.step', 'user_annotation', 40, 45),
            event('Optimizer.step#FusedAdam.step', 'gpu_user_annotation', 120, 125),
        ]
        with patch(
            'analyze_nep_owner_progress.iter_trace_events', side_effect=lambda _: iter(events)
        ):
            result = scan(Path('unused'), 1)[0]
        self.assertEqual(
            result['phases_ms'],
            {
                'through_backward': 0.030,
                'gradient_drain': 0.090,
                'adam': 0.005,
                'after_adam': 0.075,
            },
        )
        self.assertAlmostEqual(result['drain_states_ms']['gradient_accumulation'], 0.010)


if __name__ == '__main__':
    unittest.main()
