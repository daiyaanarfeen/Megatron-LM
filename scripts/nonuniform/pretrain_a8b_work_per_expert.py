"""Increase only sequence length in the matched depth-15 a8b diagnostic."""

import os

import pretrain_a8b_gb200_depth as depth
from pretrain_a8b_remaining_overheads import main as profiled_main

original_change_options = depth.change_options


def change_options(options: list[str]) -> list[str]:
    after = original_change_options(options)
    sequence_length = int(os.environ.get('A8B_SEQUENCE_LENGTH', '12288'))
    assert sequence_length in (8192, 10240, 12288, 16384)
    for flag in ('--seq-length', '--max-position-embeddings'):
        assert after.count(flag) == 1
        index = after.index(flag) + 1
        assert after[index] == '8192', flag
        after[index] = str(sequence_length)
    return after


if __name__ == '__main__':
    depth.change_options = change_options
    profiled_main()
