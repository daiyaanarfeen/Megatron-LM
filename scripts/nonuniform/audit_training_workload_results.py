#!/usr/bin/env python3
"""Inventory historical workload confounders; unknown provenance is not a pass."""

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from training_scripts_nep_matrix import discover, extract_options


def runtime_args(path: Path) -> dict[str, list[str]]:
    found = defaultdict(set)
    for line in path.read_text(errors='replace').splitlines():
        match = re.match(r'^\s{2}(\w+)\s+\.{3,}\s+(.*?)\s*$', line)
        if match:
            found[match[1]].add(match[2])
    return {k: sorted(v) for k, v in found.items()}


def source_args(tokens: list[str]) -> dict[str, str]:
    result = {}
    for i, token in enumerate(tokens):
        if token.startswith('--'):
            value = (
                tokens[i + 1]
                if i + 1 < len(tokens) and not tokens[i + 1].startswith('--')
                else 'True'
            )
            result[token[2:].replace('-', '_')] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    domains = json.loads((args.audit / 'domains.json').read_text())
    jobs = {}
    for line in (args.audit / 'sacct.txt').read_text().splitlines():
        job, state, nodes, partition, submit = line.split('|', 4)
        hosts = subprocess.check_output(['scontrol', 'show', 'hostnames', nodes], text=True).split()
        jobs[job] = {'state': state, 'nodes': hosts, 'partition': partition, 'submit': submit}
    workloads = {w.name: w for w in discover(Path.cwd())}
    rows = []
    for path in sorted(args.root.rglob('comparison_*.json')):
        data = json.loads(path.read_text())
        name = data['workload']['name']
        workload = workloads[name]
        source = source_args(extract_options(Path(workload.source)))
        job = str(data.get('job_id', path.stem.split('_')[-1]))
        alloc = jobs[job]
        problems, limitations, changes = [], [], []
        runtime, placement = {}, {}
        for mode in ('healthy', 'nep'):
            record = data[mode]
            nodes = alloc['nodes'][: record['nodes']]
            counts = Counter(domains[n] for n in nodes)
            axis = workload.tensor_parallel * workload.context_parallel
            replicas = [int(x) * axis for x in record['topology']]
            # Historical default Slurm block layout is inferred, not an observed rank manifest.
            world = record['world_size']
            base, extra = divmod(world, len(nodes))
            rank_hosts = [n for i, n in enumerate(nodes) for _ in range(base + (i < extra))]
            offset, replica_domains = 0, []
            for count in replicas:
                replica_domains.append(
                    sorted({domains[n] for n in rank_hosts[offset : offset + count]})
                )
                offset += count
            assert offset == world
            valid = all(len(d) == 1 for d in replica_domains) and len(
                {d[0] for d in replica_domains}
            ) == len(replica_domains)
            placement[mode] = {
                'domain_node_counts': dict(counts),
                'inferred_replica_domains': replica_domains,
                'policy_pass_under_default_mapping': valid,
                'rank_mapping_provenance': 'inferred default contiguous Slurm block mapping',
            }
            if len(counts) == 1:
                problems.append(
                    f'{mode}: both replicas share one NVL domain (allocation proves this)'
                )
            elif not valid:
                problems.append(f'{mode}: inferred replica crosses NVL domains or shares a domain')
            else:
                limitations.append(
                    f'{mode}: domain placement consistent, but no actual startup rank/fabric manifest'
                )
            log = path.parent / mode / job / 'driver.log'
            if not log.exists():
                problems.append(f'{mode}: missing driver log')
                continue
            actual = runtime_args(log)
            runtime[mode] = actual
            for key in (
                'hidden_size',
                'ffn_hidden_size',
                'moe_ffn_hidden_size',
                'num_experts',
                'num_layers',
                'moe_router_topk',
                'seq_length',
                'expert_tensor_parallel_size',
                'tensor_model_parallel_size',
                'context_parallel_size',
                'optimizer',
                'cuda_graph_impl',
            ):
                expected = source.get(key)
                if expected is not None and actual.get(key) != [expected]:
                    problems.append(f'{mode}: source {key}={expected}, runtime={actual.get(key)}')
            for key in ('te_op_fuser', 'use_te_op_fuser', 'use_transformer_engine_op_fuser'):
                if actual.get(key) == ['True'] and source.get(key) != 'True':
                    problems.append(f'{mode}: non-source {key} enabled')
            for key in ('log_energy', 'check_for_nan_in_loss_and_grad'):
                if actual.get(key) == ['True']:
                    problems.append(f'{mode}: {key} enabled')
            if actual.get('moe_router_force_load_balancing') != ['True']:
                problems.append(f'{mode}: random-logit forced balancing not confirmed')
            if actual.get('use_distributed_optimizer') != ['True'] and actual.get(
                'use_layer_wise_distributed_optimizer'
            ) != ['True']:
                problems.append(f'{mode}: distributed optimizer not confirmed')
            if actual.get('moe_token_dispatcher_type') != ['flex']:
                problems.append(f'{mode}: Flex dispatcher not confirmed')
            if actual.get('moe_flex_dispatcher_backend') != ['hybridep']:
                problems.append(f'{mode}: HybridEP backend not confirmed')
            batch = data.get('batch_configuration', {}).get(mode)
            if batch:
                mbs, ga = batch['replica_micro_batch_sizes'], batch['replica_num_microbatches']
            else:
                launched = re.findall(
                    r'rank=\d+ replica=(\d+) mbs=(\d+) num_microbatches=(\d+)',
                    log.read_text(errors='replace'),
                )
                settings = {int(r): (int(m), int(g)) for r, m, g in launched}
                mbs = [settings[r][0] for r in sorted(settings)]
                ga = [settings[r][1] for r in sorted(settings)]
                if len(settings) != 2:
                    limitations.append(f'{mode}: replica-local batch metadata incomplete')
            if mbs and (
                mbs != [workload.micro_batch_size] * 2
                or ga != [workload.grad_accumulation_steps] * 2
            ):
                changes.append(f'{mode}: intentional MBS/GA tuning {mbs}/{ga}')
            samples = record['elapsed_ms']
            if len(samples) < 10:
                limitations.append(f'{mode}: only {len(samples)} timing samples')
            cv = record.get('std_elapsed_ms', 0) / record['mean_elapsed_ms']
            if cv > 0.03:
                problems.append(f'{mode}: iteration CV {100 * cv:.2f}% exceeds3%')
            timed = set(record.get('timing_iterations', []))
            starts, ends = actual.get('profile_step_start', []), actual.get('profile_step_end', [])
            if len(starts) == len(ends) == 1:
                profiled = set(range(int(starts[0]), int(ends[0])))
                if timed and timed != profiled:
                    limitations.append(
                        f'{mode}: logged timing and profiler windows differ; do not mix decompositions'
                    )
            if actual.get('moe_expert_rank_capacity_factor') == ['None']:
                limitations.append(
                    f'{mode}: dynamic HybridEP sizing; host synchronization is part of workload'
                )
        for key in (
            'hidden_size',
            'moe_ffn_hidden_size',
            'seq_length',
            'moe_router_topk',
            'num_experts',
            'expert_tensor_parallel_size',
            'cuda_graph_impl',
            'cuda_graph_modules',
            'te_op_fuser',
        ):
            if key in runtime.get('healthy', {}) and runtime['healthy'][key] != runtime.get(
                'nep', {}
            ).get(key):
                problems.append(f'H/N runtime mismatch: {key}')
        parity = data['comparison']['owner_iteration_time_parity_percent']
        if parity > 100:
            problems.append(
                f'apparent speedup {parity:.2f}% requires replication; not accepted as intrinsic NEP gain'
            )
        limitations.extend(
            [
                'no immutable per-rank package/container fingerprint',
                'historical GPU clocks, thermal state and fabric background traffic not fully characterized',
                'forced random logits do not guarantee identical expert or peer split counts',
                'same DDP bucket target does not guarantee equal realized expert buckets',
                'no synchronized cross-host clock proof for precise participant-arrival subtraction',
            ]
        )
        rows.append(
            {
                'workload': name,
                'job_id': job,
                'artifact': str(path),
                'allocation': alloc,
                'placement': placement,
                'runtime': runtime,
                'confirmed_or_suspect_confounders': problems,
                'limitations': limitations,
                'intended_changes': changes,
                'parity_percent': parity,
            }
        )
    (args.audit / 'all_comparisons.json').write_text(json.dumps(rows, indent=2) + '\n')
    text = [
        '# Workload result audit: 2026-09-09',
        '',
        'Policy: each replica fits one NVLink domain; different replicas occupy different domains.',
        'Scope: every successful comparison artifact in the training-script matrix, including tuning and superseded runs.',
        'A consistent inferred mapping is not a verified live rank/fabric manifest. Common unknowns are recorded per run in JSON.',
        '',
        '| Workload | Job | GPU | Domain node counts (allocation) | Parity | Placement | Other detected issues |',
        '| --- | --- | --- | --- | ---: | --- | --- |',
    ]
    for row in rows:
        counts = Counter(domains[n] for n in row['allocation']['nodes'])
        ok = all(x['policy_pass_under_default_mapping'] for x in row['placement'].values())
        other = [x for x in row['confirmed_or_suspect_confounders'] if 'NVL domain' not in x]
        text.append(
            f"| {row['workload']} | {row['job_id']} | {row['allocation']['partition']} | {dict(counts)} | {row['parity_percent']:.2f}% | {'consistent, unverified' if ok else 'FAIL'} | {'; '.join(other) or 'see common limitations'} |"
        )
    (args.audit / 'audit.md').write_text('\n'.join(text) + '\n')
    print(
        'Audited',
        len(rows),
        'comparisons;',
        sum(
            not all(x['policy_pass_under_default_mapping'] for x in r['placement'].values())
            for r in rows
        ),
        'fail placement policy',
    )
    for name in sorted({r['workload'] for r in rows}):
        subset = [r for r in rows if r['workload'] == name]
        print(
            name,
            len(subset),
            'runs;',
            sum(
                not all(x['policy_pass_under_default_mapping'] for x in r['placement'].values())
                for r in subset
            ),
            'placement failures',
        )


if __name__ == '__main__':
    main()
