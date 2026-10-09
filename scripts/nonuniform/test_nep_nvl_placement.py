"""CPU-only tests for the fail-closed launch placement guard."""

import unittest

from nep_nvl_placement import fabric_fields, validate


class PlacementTests(unittest.TestCase):
    def rows(self):
        return [
            {
                'rank': r,
                'gpu_uuid': f'gpu{r}',
                'host': f'host{r // 4}',
                'domain': 'a' if r < 8 else 'b',
                'gpu_name': 'GB300',
                'packages': {'torch': 'same'},
                'fabric': {
                    'State': 'Completed',
                    'Status': 'Success',
                    'Summary': 'Healthy',
                    'Bandwidth': 'Full',
                    'ClusterUUID': 'a' if r < 8 else 'b',
                    'CliqueId': '32766',
                },
            }
            for r in range(14)
        ]

    def test_valid_replicas(self):
        result = validate(self.rows(), [8, 6])
        self.assertEqual([r['domain'] for r in result['replicas']], ['a', 'b'])

    def test_same_domain_rejected(self):
        rows = self.rows()
        for row in rows:
            row['domain'] = 'a'
        with self.assertRaisesRegex(ValueError, 'share an NVLink domain'):
            validate(rows, [8, 6])

    def test_split_replica_rejected(self):
        rows = self.rows()
        rows[7]['domain'] = 'b'
        with self.assertRaisesRegex(ValueError, 'spans domains'):
            validate(rows, [8, 6])

    def test_missing_duplicate_and_unhealthy_rejected(self):
        for kind in ('missing', 'duplicate_gpu', 'health', 'package', 'hardware', 'fabric'):
            rows = self.rows()
            if kind == 'missing':
                rows.pop()
            elif kind == 'duplicate_gpu':
                rows[1]['gpu_uuid'] = rows[0]['gpu_uuid']
            elif kind == 'health':
                rows[1]['fabric']['Bandwidth'] = 'Degraded'
            elif kind == 'package':
                rows[1]['packages']['torch'] = 'different'
            elif kind == 'hardware':
                rows[1]['gpu_name'] = 'GB200'
            else:
                for row in rows:
                    row['fabric']['ClusterUUID'] = 'same'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate(rows, [8, 6])

    def test_fabric_parser(self):
        text = '    Fabric\n        State : Completed\n        Status : Success\n        Health\n            Summary : Healthy\n            Bandwidth : Full\n    Processes\n        State : unrelated\n'
        self.assertEqual(fabric_fields(text)['State'], 'Completed')
        self.assertEqual(fabric_fields(text)['Bandwidth'], 'Full')
        with self.assertRaises(ValueError):
            fabric_fields('No fabric section')


if __name__ == '__main__':
    unittest.main()
