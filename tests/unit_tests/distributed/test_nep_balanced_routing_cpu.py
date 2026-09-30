"""CPU-only routing preflight, executable without importing Megatron or Torch."""

import ast
import math
import os
import typing
import unittest
from collections import Counter
from pathlib import Path

from nep_balanced_routing import balanced_routes, identity_seed, route_summary


def placement_functions():
    """Exercise the actual pure placement functions without GPU dependencies."""
    source = Path(__file__).resolve().parents[3] / 'megatron/core/distributed/nonuniform_common.py'
    names = {
        'compute_nonuniform_ep_owner_expert_slots',
        'compute_nonuniform_ep_expert_placement',
        'build_expert_to_ep_rank_map',
    }
    module = ast.Module(
        body=[
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ],
        type_ignores=[],
    )
    scope = dict(vars(typing), math=math, os=os)
    exec(compile(module, str(source), 'exec'), scope)
    return scope['compute_nonuniform_ep_expert_placement']


class BalancedRoutingTest(unittest.TestCase):
    def test_degrees_and_seed(self):
        for tokens, experts, topk in [(1024, 512, 6), (512, 128, 6), (6, 9, 3), (1, 4, 4)]:
            rows = balanced_routes(tokens, experts, topk, 101)
            self.assertTrue(all(len(set(row)) == topk for row in rows))
            self.assertEqual(
                set(Counter(expert for row in rows for expert in row).values()),
                {tokens * topk // experts},
            )
            self.assertEqual(rows, balanced_routes(tokens, experts, topk, 101))
        self.assertNotEqual(balanced_routes(1024, 512, 6, 101), balanced_routes(1024, 512, 6, 202))

    def test_invalid(self):
        for args in [(0, 4, 2), (4, 4, 5), (3, 4, 2)]:
            with self.assertRaises(ValueError):
                balanced_routes(*args, 101)

    def test_placement_counts_and_no_adjacency_bias(self):
        placement = placement_functions()
        for seed in (101, 202, 303):
            rows = balanced_routes(1024, 512, 6, identity_seed('route:0', seed))
            for ep, owners in [(8, 6), (16, 12)]:
                summaries = []
                for policy in ('compact', 'spread'):
                    os.environ['NEP_ABLATION_PLACEMENT'] = policy
                    summary = route_summary(rows, placement(512, ep, owners, 1))
                    self.assertEqual(set(summary['expert_counts']), {12})
                    self.assertEqual(set(summary['peer_assignments']), {6144 // ep})
                    self.assertLess(summary['adjacent_expert_pairs'], 150)
                    summaries.append(summary)
                self.assertEqual(summaries[0]['route_sha256'], summaries[1]['route_sha256'])
                print(
                    'ROUTING_PREFLIGHT',
                    seed,
                    ep,
                    [row['total_token_peer_pairs'] for row in summaries],
                    flush=True,
                )


if __name__ == '__main__':
    unittest.main()
