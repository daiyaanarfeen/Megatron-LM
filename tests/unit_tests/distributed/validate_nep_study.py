"""Fail closed between study cases, using saved controls and all-rank GPU records."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from validate_nep_randomized_controls import validate


def audit(run: Path, full: int, reduced: int, mode: str, profiled: bool) -> dict:
    """Check completion, native manifests, and every parameter-transfer split."""
    replay = mode == 'replay'
    world = full + reduced
    if replay:
        for rank in range(world):
            record = json.loads((run / f'replay-{rank}.json').read_text())
            assert len(record['timings']) == 8 and record['verified_parameters'] > 0
            assert all(row['elapsed_ms'] > 0 for row in record['timings'])
    if not replay:
        log = (run / 'driver.log').read_text()
        statuses = [
            tuple(map(int, row))
            for row in re.findall(
                r'iteration\s+(\d+)/\s*(\d+).*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)',
                log,
            )
        ][-25:]
        assert statuses == [(i, 25, 0, 0) for i in range(1, 26)], (run, statuses)
    if not profiled:
        return dict(profiled=False)
    steps = range(8) if replay else range(15, 21)
    traffic = defaultdict(lambda: defaultdict(list))
    source, dest, native = defaultdict(int), defaultdict(int), {}
    for rank in range(world):
        assert (run / f'nsys_profile/rank-{rank}.sqlite').is_file(), (run, rank)
        events = json.loads((run / f'cupti_profile/rank-{rank}.json').read_text())['traceEvents']
        for step in steps:
            gpu = [
                e
                for e in events
                if e.get('cat') == 'gpu_operation_range' and e['args']['iteration'] == step
            ]
            names = {e['name'] for e in gpu}
            if replay:
                assert 'replay.parameter_transfers' in names
                assert not names.intersection(
                    {
                        'collective.all_gather',
                        'collective.reduce_scatter',
                        'microbatch.forward',
                        'microbatch.backward',
                    }
                )
            else:
                assert {
                    'collective.all_gather',
                    'collective.reduce_scatter',
                    'nep.redistribute_params',
                } <= names
            native[f'{rank}:{step}'] = []
            for event in gpu:
                meta = event['args'].get('collective')
                if not meta:
                    continue
                assert 'completion_stream' in event['args'], (run, rank, event['name'])
                if not meta['group_desc'].startswith('nep_owner'):
                    native[f'{rank}:{step}'].append(
                        [meta['group_desc'], meta['ranks'], meta['operations']]
                    )
                if meta['group_desc'] != 'nep_owner_transfer':
                    continue
                (op,) = meta['operations']
                assert op['op'] == 'all_to_all' and not (op['input_bytes'] and op['output_bytes'])
                traffic[step, meta['group_id'], tuple(meta['ranks'])][rank].append(op)
                source[step] += op['input_bytes']
                dest[step] += op['output_bytes']
    assert source == dest and set(source) == set(steps) and len(set(source.values())) == 1
    for (_, _, members), calls in traffic.items():
        assert set(calls) == set(members)
        assert len({len(calls[r]) for r in members}) == 1
        for index in range(len(calls[members[0]])):
            for i, sender in enumerate(members):
                for j, receiver in enumerate(members):
                    assert (
                        calls[sender][index]['input_splits'][j]
                        == calls[receiver][index]['output_splits'][i]
                    )
    return dict(profiled=True, native=native, bytes_per_step=next(iter(source.values())))


if __name__ == '__main__':
    left, right = map(Path, sys.argv[1:3])
    full, reduced = map(int, sys.argv[3:5])
    mode, profile = sys.argv[5:7]
    controls = validate(left, right, full, reduced, require_routing=mode != 'replay')
    before = audit(left, full, reduced, mode, profile == 'on')
    after = audit(right, full, reduced, mode, profile == 'on')
    assert before == after, 'Native collectives or aggregate parameter payload changed'
    (right / 'study-validation.json').write_text(
        json.dumps(dict(status='PASS', controls=controls, audit=after), indent=2) + '\n'
    )
    print('STUDY_VALIDATION_PASS', right, flush=True)
