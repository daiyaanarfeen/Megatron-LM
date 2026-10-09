"""Audit exact startup identities, native payloads, and NCCL transport."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

OPTIMIZER_GROUPS = {
    'dp_cp', 'ep_dp', 'DATA_PARALLEL_GROUP_WITH_CP', 'EXPERT_DATA_PARALLEL_GROUP'
}


def audit(root, run_pattern='controlled_work_*', allow_expert_rebucketing=False):
    pattern = re.compile(r'comm (0x[0-9a-f]+) rank (\d+) nranks (\d+).*commId (0x[0-9a-f]+) - Init START')
    identities, manifests, results = {}, {}, {}
    for run in sorted(root.glob(run_pattern)):
        gate = run / 'work-validation.json'
        if not gate.is_file():
            continue
        validation = json.loads(gate.read_text())
        healthy = validation['cell'].startswith('H')
        world = 32 if healthy else 28
        bucket_count = None
        if allow_expert_rebucketing:
            # The experiment's per-case validator checks total bytes across
            # counts; retain exact collective fingerprints within each count.
            bucket_count = json.loads((run / 'actual-spec.json').read_text())['expert_bucket_groups']
        for rank in range(world):
            initial = json.loads((run / f'controls/initialization-{rank}.json').read_text())
            for name, value in initial['parameters'].items():
                key = (None if '.experts.' in name else rank % 2, name)
                if key in identities:
                    assert identities[key] == value, (run, rank, name, 'canonical weight mismatch')
                identities[key] = value
        if validation['profiled']:
            for rank_step, records in validation['native_collectives'].items():
                rank, step = map(int, rank_step.split(':'))
                fingerprint = sorted(json.dumps({k: rec[k] for k in ('group_desc', 'ranks', 'operations')}, sort_keys=True)
                                     for rec in records if rec['group_desc'] in OPTIMIZER_GROUPS)
                assert fingerprint, (run, rank, step, 'missing native optimizer collectives')
                key = healthy, rank, bucket_count
                if key in manifests:
                    assert manifests[key] == fingerprint, (run, rank, step, 'native AG/RS payload drift')
                manifests[key] = fingerprint
        comms = defaultdict(list)
        for path in (run / 'transport').glob('*.log'):
            global_rank, current = None, None
            for line in path.read_text().splitlines():
                match = pattern.search(line)
                if match:
                    if current:
                        comms[current['cid']].append(current)
                    _, rank, size, cid = match.groups()
                    rank, size = int(rank), int(size)
                    if size == world:
                        global_rank = rank
                    assert global_rank is not None
                    current = dict(rank=global_rank, n=size, cid=cid, routes=[])
                elif current and ' via ' in line:
                    current['routes'].append(line.split(' via ', 1)[1])
            if current:
                comms[current['cid']].append(current)
        cross, local = [], []
        for rows in comms.values():
            members = sorted({r['rank'] for r in rows})
            routes = [route for row in rows for route in row['routes']]
            if len(members) == 2 and min(members) < 16 <= max(members):
                assert all(r['n'] == 2 for r in rows)
                assert routes and all('NET/' in r and ('RDMA' in r or 'IB' in r) for r in routes), (run, members, routes[:3])
                cross.append(members)
            elif 2 <= len(members) <= 5 and max(members) < 16:
                assert routes and all('P2P/' in r for r in routes), (run, members, routes[:3])
                local.append(members)
        assert len(cross) == (16 if healthy else 12), (run, cross)
        if not healthy:
            assert local
        results[run.name] = dict(cross_replica_pairs_rdma=cross, intra_replica_groups_p2p=local)
        print('AUDIT_PASS', run.name, 'cross-replica RDMA pairs', len(cross), flush=True)
    assert results
    output = dict(status='PASS', cases=results, canonical_tensors=len(identities))
    (root / 'work-controls-audit.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    audit(Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab') / sys.argv[1])
