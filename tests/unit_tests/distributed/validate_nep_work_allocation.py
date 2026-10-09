"""Fail closed on memory, batch, initialization, and all-rank profile controls."""

import json
import re
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def validate(run):
    cell = run.name.split('_')[2]
    healthy = cell.startswith('H')
    world = 32 if healthy else 28
    mbs = 2 if cell.endswith('2') else 1
    ga = 2 if mbs == 2 else 4
    reduced_ga = {'R': 3, 'O': 5}.get(cell, ga)
    expected_gbs = 32 + (32 if healthy else 6 * mbs * reduced_ga)
    profiled = '_off_' not in run.name
    log = (run / 'driver.log').read_text()
    statuses = [
        tuple(map(int, row))
        for row in re.findall(
            r'iteration\s+(\d+)/\s*(\d+).*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)',
            log,
        )
    ][-25:]
    assert statuses == [(i, 25, 0, 0) for i in range(1, 26)], (run, statuses)
    batches = {
        int(row[0]): tuple(map(int, row[1:]))
        for row in re.findall(
            r'BATCH_SCHEDULE rank=(\d+) replica=(\d+) mbs=(\d+) ga=(\d+) calculator_gbs=(\d+) true_gbs=(\d+)',
            log,
        )
    }
    assert set(batches) == set(range(world))
    memory, initial, routing, native = {}, {}, {}, {}
    for rank in range(world):
        local_ga = reduced_ga if rank >= 16 else ga
        assert batches[rank] == (
            rank // 16,
            mbs,
            local_ga,
            world // 2 * mbs * local_ga,
            expected_gbs,
        )
        memory[rank] = read(run / f'memory-{rank}.json')
        mem = memory[rank]
        # Leave substantial space beyond the caching allocator and NCCL pools.
        assert mem['total'] - mem['peak_reserved'] > 16 * 1024**3, (rank, mem)
        assert mem['end_free'] > 8 * 1024**3, (rank, mem)
        initial[rank] = read(run / f'controls/initialization-{rank}.json')
        files = list((run / 'controls').glob(f'routing-{rank}-*.json'))
        assert files, (rank, 'missing routing')
        routing[rank] = [read(path) for path in files]
        for route in routing[rank]:
            assert len(set(route['expert_counts'])) == 1
            assert route['experts'] == 512 and route['topk'] == 6
        if not profiled:
            continue
        assert (run / f'nsys_profile/rank-{rank}.sqlite').is_file()
        events = read(run / f'cupti_profile/rank-{rank}.json')['traceEvents']
        for step in range(15, 21):
            gpu = [
                e
                for e in events
                if e.get('cat') == 'gpu_operation_range' and e['args']['iteration'] == step
            ]
            names = [e['name'] for e in gpu]
            assert names.count('microbatch.forward') == local_ga, (
                rank,
                step,
                names.count('microbatch.forward'),
                local_ga,
            )
            assert names.count('microbatch.backward') == local_ga
            assert {'collective.all_gather', 'collective.reduce_scatter'} <= set(names)
            if not healthy:
                assert 'nep.redistribute_params' in names
            native[f'{rank}:{step}'] = []
            for event in gpu:
                if event['name'] not in ('collective.all_gather', 'collective.reduce_scatter'):
                    continue
                assert 'completion_stream' in event['args']
                native[f'{rank}:{step}'].append(event['args']['collective'])
    # Compare immutable startup state against all earlier completed cases.
    for previous in run.parent.glob('controlled_work_*/work-validation.json'):
        other = previous.parent
        other_cell = other.name.split('_')[2]
        same_topology = other_cell.startswith('H') == healthy
        for rank in range(min(world, 32 if other_cell.startswith('H') else 28)):
            before = read(other / f'controls/initialization-{rank}.json')
            now = initial[rank]
            if same_topology:
                assert before == now, (other, run, rank, 'initialization/bucket drift')
            else:
                shared = set(before['parameters']) & set(now['parameters'])
                for name in shared:
                    assert before['parameters'][name] == now['parameters'][name], (
                        rank,
                        name,
                        'weight drift',
                    )
            prior_routes = [read(p) for p in (other / 'controls').glob(f'routing-{rank}-*.json')]
            assert {r['unit_route_sha256'] for r in prior_routes} == {
                r['unit_route_sha256'] for r in routing[rank]
            }
    result = dict(
        status='PASS',
        cell=cell,
        profiled=profiled,
        true_gbs=expected_gbs,
        memory=memory,
        native_collectives=native,
    )
    (run / 'work-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(
        'WORK_VALIDATION_PASS',
        run,
        'max_peak_GiB',
        max(m['peak_allocated'] for m in memory.values()) / 1024**3,
        flush=True,
    )


if __name__ == '__main__':
    validate(Path(sys.argv[1]))
