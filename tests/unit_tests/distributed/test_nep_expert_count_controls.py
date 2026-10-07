"""CPU-only preflight for routing and native expert-buffer layout in the count sweep."""

import ast
import json
import math
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from nep_balanced_routing import balanced_routes


class ExpertCountControlsTest(unittest.TestCase):
    def test_exact_routing_at_each_count(self):
        for experts in (128, 256, 384):
            rows = balanced_routes(96, experts, 8, 101)
            self.assertTrue(all(len(set(row)) == 8 for row in rows))
            counts = Counter(expert for row in rows for expert in row)
            self.assertEqual(len(counts), experts)
            self.assertEqual(set(counts.values()), {96 * 8 // experts})
            self.assertEqual(8448 % 96, 0)

    @staticmethod
    def native_layout():
        # Execute the actual dependency-free native method, not a reimplementation.
        source = (
            Path(__file__).resolve().parents[3] / 'megatron/core/optimizer/distrib_optimizer.py'
        )
        tree = ast.parse(source.read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'DistributedOptimizer'
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == '_compute_per_buffer_param_layout'
        )
        method.decorator_list = []
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        env = dict(
            pad_param_start=lambda n: math.ceil(n / 64) * 64,
            pad_bucket_end=lambda n, dp, high: math.ceil(n / math.lcm(dp, 128)) * math.lcm(dp, 128),
            PerBufferParamLayout=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        exec(compile(module, str(source), 'exec'), env)
        return env['_compute_per_buffer_param_layout']

    def test_native_layout_keeps_three_layers_per_bucket(self):
        native = self.native_layout()

        class Param:
            def __init__(self, layer):
                self.layer = layer
                self.data = SimpleNamespace(nelement=lambda: 1152 * 3072)

        for experts in (128, 256, 384):
            for ep in (16, 12):
                for local_experts in {experts // ep, math.ceil(experts / ep)}:
                    params = [Param(layer) for layer in range(15) for _ in range(2 * local_experts)]
                    total = len(params) * 1152 * 3072
                    layout = native(
                        params,
                        total // 5,
                        2,
                        SimpleNamespace(pad_buckets_for_high_nccl_busbw=False),
                    )
                    layers = [set() for _ in layout.bucket_indices]
                    for param, (_, _, bucket) in layout.param_index_map.items():
                        layers[bucket].add(param.layer)
                    self.assertEqual(
                        layers, [set(range(start, start + 3)) for start in (12, 9, 6, 3, 0)]
                    )

    def test_dense_boundaries_using_captured_model_shapes(self):
        source = Path(
            '/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3263307/controlled_actual_H256_r0_healthy_moe/controls/initialization-0.json'
        )
        manifest = json.loads(source.read_text())
        expected = [bucket['names'] for bucket in manifest['dense_buckets']]
        native = self.native_layout()

        class Param:
            def __init__(self, name, experts):
                self.name = name
                shape = manifest['parameters'][name]['shape'].copy()
                if '.router.' in name:
                    assert shape[0] == 256
                    shape[0] = experts
                self.data = SimpleNamespace(nelement=lambda: math.prod(shape))

        for experts in (128, 256, 384):
            params = [Param(name, experts) for bucket in expected for name in bucket][::-1]
            for dp in (14, 16):
                config = SimpleNamespace(pad_buckets_for_high_nccl_busbw=False)
                layout = native(params, 692000000, dp, config)
                groups = [[] for _ in layout.bucket_indices]
                for param, (_, _, bucket) in layout.param_index_map.items():
                    groups[bucket].append(param.name)
                self.assertEqual(groups, expected)
                if experts != 128:
                    previous = native(params, 697279864, dp, config)
                    self.assertEqual(layout.param_index_map, previous.param_index_map)
                    self.assertEqual(layout.bucket_indices, previous.bucket_indices)


if __name__ == '__main__':
    unittest.main()
