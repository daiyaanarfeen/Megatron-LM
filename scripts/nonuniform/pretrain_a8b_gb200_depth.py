"""Reduce only complete a8b layer repeats to fit GB200, with matched profiling."""

import hashlib
import json
import os
import sys
from pathlib import Path

ORIGINAL_PATTERN = 'MEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEME/*E/*E'


def change_options(options: list[str]) -> list[str]:
    for flag, expected in (
        ('--hidden-size', '4608'),
        ('--ffn-hidden-size', '3072'),
        ('--moe-latent-size', '1152'),
        ('--num-experts', '512'),
        ('--moe-router-topk', '6'),
        ('--seq-length', '8192'),
        ('--ddp-num-buckets', '8'),
        ('--hybrid-layer-pattern', ORIGINAL_PATTERN),
    ):
        assert options.count(flag) == 1, flag
        assert options[options.index(flag) + 1] == expected, flag
    sections = ORIGINAL_PATTERN.split('/')
    blocks = sections[0].split('*')
    assert blocks[1:-1] == ['EMEMEMEMEM'] * 4
    sections[0] = '*'.join(blocks[:2] + blocks[-1:])
    pattern = '/'.join(sections)
    assert pattern.count('E') == 15
    after = options.copy()
    after[after.index('--hybrid-layer-pattern') + 1] = pattern
    assert sum(a != b for a, b in zip(options, after)) == 1
    return after


def main() -> None:
    before = sys.argv[1:]
    after = change_options(before)
    assert os.environ['REPLICA_MICRO_BATCH_SIZES'] == '2 2'
    assert os.environ['REPLICA_NUM_MICROBATCHES'] == '8 8'
    assert os.environ['CUPTI_EVENTS_ONLY'] == '1'
    assert os.environ['CUPTI_COLLECTION_DISABLED'] == '0'
    runtime = Path(os.environ['CODE_REPO']) / 'megatron/core/distributed/nonuniform_ep.py'
    sha = hashlib.sha256(runtime.read_bytes()).hexdigest()
    assert sha == os.environ['A8B_DEPTH_EXPECTED_SHA']
    output = Path(os.environ['CUPTI_OUTPUT_DIR']).parent / 'launch_manifest'
    output.mkdir(exist_ok=True)
    (output / f"rank-{os.environ['RANK']}.json").write_text(json.dumps(dict(
        runtime_sha256=sha, options_before=before, options=after,
        variant=os.environ['SOFT_CLIQUE_TAG'],
        replica_mbs=os.environ['REPLICA_MICRO_BATCH_SIZES'],
        replica_ga=os.environ['REPLICA_NUM_MICROBATCHES'],
    ), indent=2) + '\n')
    sys.argv[1:] = after
    from pretrain_profile_activity_probe import main as profiled_main

    profiled_main()


if __name__ == '__main__':
    main()
