"""CPU-only checks for the expert-bucket experiment's control normalization."""

import copy
import unittest

from nep_actual_ablation import rebucketed_fingerprint
from validate_nep_randomized_controls import expert_slots


def records(expert_sizes):
    def call(group, ranks, size):
        return dict(
            group_desc=group,
            ranks=ranks,
            operations=[
                dict(op='all_gather', input_bytes=size, output_bytes=2 * size, dtype='bf16')
            ],
        )

    return {
        '0:25': [call('dp_cp', [0, 2], 32)]
        + [call('ep_dp', [0, 16], size) for size in expert_sizes]
    }


class RebucketControlsTest(unittest.TestCase):
    def test_boundaries_only(self):
        self.assertEqual(
            rebucketed_fingerprint(records([60])),
            rebucketed_fingerprint(records([10, 20, 30])),
        )
        a = [dict(slots=['a', 'b'], numels=[10, 20])]
        b = [dict(slots=['a'], numels=[10]), dict(slots=['b'], numels=[20])]
        self.assertEqual(expert_slots(a), expert_slots(b))
        self.assertNotEqual(expert_slots(a), expert_slots(list(reversed(b))))
        self.assertNotEqual(expert_slots(a), expert_slots(b[:1]))
        with self.assertRaises(ValueError):
            expert_slots([dict(slots=['a'], numels=[])])

    def test_payload_participants_and_dtype_remain_invariant(self):
        original = records([60])
        expected = rebucketed_fingerprint(original)
        for index in (0, 1):
            for key, value in [('input_bytes', 61), ('output_bytes', 121), ('dtype', 'fp32')]:
                changed = copy.deepcopy(original)
                changed['0:25'][index]['operations'][0][key] = value
                self.assertNotEqual(expected, rebucketed_fingerprint(changed))
            changed = copy.deepcopy(original)
            changed['0:25'][index]['ranks'][1] += 1
            self.assertNotEqual(expected, rebucketed_fingerprint(changed))

    def test_dense_boundaries_and_followers(self):
        original = records([60])
        changed = copy.deepcopy(original)
        changed['0:25'].insert(0, copy.deepcopy(changed['0:25'][0]))
        self.assertNotEqual(rebucketed_fingerprint(original), rebucketed_fingerprint(changed))
        self.assertEqual(rebucketed_fingerprint(records([]))['0:25']['expert'], {})
        with self.assertRaises(AssertionError):
            rebucketed_fingerprint({'0:25': original['0:25'][1:]})


if __name__ == '__main__':
    unittest.main()
