"""Disjoint main-stream operation ranges, not kernel busy-time attribution."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_a8b_current_overheads import span, stats


def phase_windows(row):
    """Use the same expert anchors as the existing iteration partition."""
    micros = row['micros']
    opt = next(e for e in row['events'] if e['name'] == 'optimizer.step')
    windows = [('startup', row['start_ms'], micros[0]['f_start_ms'])]
    for mb, micro in enumerate(micros):
        windows.extend(
            [
                (
                    'first_forward' if mb == 0 else 'later_forward',
                    micro['f_start_ms'],
                    micro['f_end_ms'],
                ),
                ('forward_backward_transition', micro['f_end_ms'], micro['b_start_ms']),
                (
                    'last_backward' if mb == len(micros) - 1 else 'earlier_backward',
                    micro['b_start_ms'],
                    micro['b_end_ms'],
                ),
            ]
        )
        if mb:
            windows.append(
                ('between_microbatches', micros[mb - 1]['b_end_ms'], micro['f_start_ms'])
            )
    windows.extend(
        [
            ('backward_to_optimizer', micros[-1]['b_end_ms'], span(opt)[0]),
            ('optimizer', *span(opt)),
            ('after_optimizer', span(opt)[1], row['end_ms']),
        ]
    )
    assert abs(sum(b - a for _, a, b in windows) - row['iteration_ms']) < 1e-5
    return windows


def selected_scope(name):
    if name.startswith(('expert.', 'hybridep.', 'moe.', 'graph.')):
        return name.replace('moe.metadata.native', 'moe.metadata').replace(
            'moe.metadata.nep', 'moe.metadata'
        )
    return None


def partition(start, end, events):
    """Assign each elapsed segment to its innermost labelled operation."""
    boundaries = [(start, 0, -1), (end, 0, -1)]
    active = {}
    for index, (a, b, name) in enumerate(events):
        left, right = max(a, start), min(b, end)
        if right > left:
            boundaries.extend([(left, 1, index), (right, -1, index)])
    output = defaultdict(float)
    previous = start
    for point, action, index in sorted(boundaries):
        if point > previous:
            # The shortest enclosing scope is the child when scopes are nested.
            name = min(active.values(), key=lambda e: e[1] - e[0])[2] if active else 'unscoped'
            output[name] += point - previous
        if action == 1:
            active[index] = events[index]
        elif action == -1:
            active.pop(index)
        previous = point
    assert abs(sum(output.values()) - (end - start)) < 1e-5
    return dict(output)


def rank_rows(data):
    output = {}
    for step, row in data['steps'].items():
        main = Counter(
            e['tid'] for e in row['events'] if e['name'] == 'expert.backward'
        ).most_common(1)[0][0]
        events = [
            (*span(e), selected_scope(e['name']))
            for e in row['events']
            if e['tid'] == main and selected_scope(e['name'])
        ]
        phases = defaultdict(lambda: defaultdict(float))
        for phase, start, end in phase_windows(row):
            for name, duration in partition(start, end, events).items():
                phases[phase][name] += duration
        cpu = defaultdict(lambda: defaultdict(float))
        for e in row['cpu']:
            name = selected_scope(e['name'])
            if name:
                cpu[name]['elapsed_ms'] += e['dur'] / 1000
                cpu[name]['thread_ms'] += e['args']['thread_cpu_us'] / 1000
                cpu[name]['count'] += 1
        output[step] = dict(
            phases={p: dict(v) for p, v in phases.items()},
            cpu_inclusive={k: dict(v) for k, v in cpu.items()},
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache_dir', type=Path)
    args = parser.parse_args()
    assert str(args.cache_dir.resolve()).startswith(
        '/lustre/'
    ), 'Generated output belongs on Lustre'
    result = {
        'note': 'Disjoint elapsed main-stream scopes, NOT pure compute or transfer service. '
        'Unscoped does NOT mean GPU idle. CPU totals are inclusive and non-additive.'
    }
    for case, size in [('healthy', 32), ('nep', 28)]:
        ranks = {}
        for rank in range(size):
            path = args.cache_dir / f'{case}-rank-{rank}.json'
            ranks[rank] = rank_rows(json.loads(path.read_text()))
        result[case] = {}
        for role, members in [
            ('owners', range(12)),
            ('followers', range(12, 16)),
            ('second', range(16, size)),
        ]:
            role_output = {}
            for kind in ['phases', 'cpu_inclusive']:
                keys = set().union(*(ranks[r][s][kind] for r in members for s in ranks[r]))
                aggregates = {}
                for key in sorted(keys):
                    names = set().union(
                        *(ranks[r][s][kind].get(key, {}) for r in members for s in ranks[r])
                    )
                    aggregates[key] = {
                        name: stats(
                            [
                                statistics.mean(
                                    ranks[r][str(s)][kind].get(key, {}).get(name, 0)
                                    for r in members
                                )
                                for s in range(25, 35)
                            ]
                        )
                        for name in sorted(names)
                    }
                role_output[kind] = aggregates
            result[case][role] = role_output
    path = args.cache_dir / 'operation_partition.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    for role in ['owners', 'second']:
        print(role)
        for phase in result['healthy'][role]['phases']:
            h = result['healthy'][role]['phases'][phase]
            n = result['nep'][role]['phases'][phase]
            print(
                phase,
                {
                    k: round(n.get(k, {'mean': 0})['mean'] - h.get(k, {'mean': 0})['mean'], 3)
                    for k in sorted(set(h) | set(n))
                },
            )
    print(path)


if __name__ == '__main__':
    main()
