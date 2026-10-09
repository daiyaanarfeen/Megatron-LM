#!/usr/bin/env python3
"""Startup-only actual rank/fabric guard; nothing runs during training steps."""

import argparse
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path


def topology_map(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        fields = dict(field.split('=', 1) for field in line.split())
        hosts = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', fields['Nodes']], text=True
        ).split()
        result.update({host: fields['BlockName'] for host in hosts})
    return result


def fabric_fields(text: str) -> dict[str, str]:
    match = re.search(r'^    Fabric\n(.*?)(?=^    \S|\Z)', text, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError('Missing NVIDIA fabric status')
    return {
        m[1].strip(): m[2].strip()
        for m in re.finditer(r'^\s+([^:\n]+):\s*([^\n]+)', match[1], re.MULTILINE)
    }


def validate(rows: list[dict], replica_sizes: list[int]) -> dict:
    world = sum(replica_sizes)
    ranks = {row['rank']: row for row in rows}
    if len(rows) != world or set(ranks) != set(range(world)):
        raise ValueError('Missing or duplicate rank records')
    if len({r['gpu_uuid'] for r in rows}) != world:
        raise ValueError('Multiple training ranks selected the same physical GPU')
    for row in rows:
        fabric = row['fabric']
        for key, expected in [
            ('State', 'Completed'),
            ('Status', 'Success'),
            ('Summary', 'Healthy'),
            ('Bandwidth', 'Full'),
        ]:
            if fabric.get(key) != expected:
                raise ValueError(f"rank{row['rank']}: fabric {key}={fabric.get(key)}")
        if not fabric.get('ClusterUUID') or fabric['ClusterUUID'] == 'N/A':
            raise ValueError('Missing fabric UUID')
    replicas, offset = [], 0
    for size in replica_sizes:
        local = [ranks[r] for r in range(offset, offset + size)]
        domains = {r['domain'] for r in local}
        fabrics = {(r['fabric']['ClusterUUID'], r['fabric']['CliqueId']) for r in local}
        if len(domains) != 1 or len(fabrics) != 1:
            raise ValueError(
                f'Replica starting rank{offset} spans domains/fabrics: {domains}, {fabrics}'
            )
        replicas.append(
            {
                'first_rank': offset,
                'ranks': size,
                'domain': next(iter(domains)),
                'fabric': list(next(iter(fabrics))),
                'hosts': sorted({r['host'] for r in local}),
            }
        )
        offset += size
    if len({r['domain'] for r in replicas}) != len(replicas):
        raise ValueError('Different replicas share an NVLink domain')
    if len({tuple(r['fabric']) for r in replicas}) != len(replicas):
        raise ValueError('Different replicas share the same reported NVLink fabric')
    if len({r['gpu_name'] for r in rows}) != 1:
        raise ValueError('Mixed GPU hardware in one comparison')
    if len({json.dumps(r['packages'], sort_keys=True) for r in rows}) != 1:
        raise ValueError('Package version mismatch between ranks')
    return {
        'validated': True,
        'replicas': replicas,
        'gpu_name': rows[0]['gpu_name'],
        'packages': rows[0]['packages'],
    }


def probe(directory: Path, domain_file: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rank, gpu = int(os.environ['RANK']), os.environ['CUDA_VISIBLE_DEVICES']
    host = socket.gethostname().split('.')[0]
    domains = json.loads(domain_file.read_text())
    raw = subprocess.check_output(['nvidia-smi', '-q', '-i', gpu], text=True, timeout=30)
    (directory / f'gpu-{rank}.txt').write_text(raw)
    identity = (
        subprocess.check_output(
            ['nvidia-smi', '-i', gpu, '--query-gpu=uuid,name', '--format=csv,noheader'],
            text=True,
            timeout=30,
        )
        .strip()
        .split(',', 1)
    )
    packages = {}
    for name in ('torch', 'transformer_engine', 'mamba_ssm', 'causal_conv1d', 'flash-attn'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not-in-metadata'
    row = {
        'rank': rank,
        'host': host,
        'local_gpu': gpu,
        'domain': domains[host],
        'gpu_uuid': identity[0],
        'gpu_name': identity[1].strip(),
        'fabric': fabric_fields(raw),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'packages': packages,
        'environment': {
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                ('NCCL_', 'TORCH_NCCL_', 'MEGATRON_NONUNIFORM_', 'NVTE_', 'SLURM_CPU_BIND')
            )
            or k in ('CUDA_DEVICE_MAX_CONNECTIONS', 'OMP_NUM_THREADS')
        },
    }
    temporary = directory / f'rank-{rank}.tmp'
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(directory / f'rank-{rank}.json')
    deadline = time.monotonic() + 120
    ready, error = directory / 'validated.json', directory / 'error.txt'
    world = int(os.environ['WORLD_SIZE'])
    while time.monotonic() < deadline:
        if error.exists():
            raise RuntimeError(error.read_text())
        if rank == 0 and len(list(directory.glob('rank-*.json'))) == world:
            try:
                axis = int(os.environ['TENSOR_MODEL_PARALLEL_SIZE']) * int(
                    os.environ['CONTEXT_PARALLEL_SIZE']
                )
                sizes = [int(n) * axis for n in os.environ['NONUNIFORM_EP_TOPOLOGY'].split()]
                summary = validate(
                    [json.loads(p.read_text()) for p in directory.glob('rank-*.json')], sizes
                )
                temporary = directory / 'validated.tmp'
                temporary.write_text(json.dumps(summary, indent=2) + '\n')
                temporary.replace(ready)
            except Exception as exc:
                error.write_text(str(exc))
                raise
            print('NVL placement validated:', summary, flush=True)
        if ready.exists():
            return
        time.sleep(0.2)
    raise TimeoutError('NVL startup validation did not complete in120s')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--domains', type=Path)
    args = parser.parse_args()
    if args.prepare:
        text = subprocess.check_output(['scontrol', 'show', 'topology'], text=True)
        args.prepare.write_text(json.dumps(topology_map(text), indent=2) + '\n')
    else:
        probe(args.probe, args.domains)


if __name__ == '__main__':
    main()
