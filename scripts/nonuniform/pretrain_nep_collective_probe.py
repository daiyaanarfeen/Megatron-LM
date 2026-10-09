"""Replay the E512/S2048 case with AG/RS-capable instrumentation."""

import pretrain_nep_sync_workload as base

validate_anchor = base.change_options
CHANGES = {'e512_seq2048': (('--num-experts', '128', '512'), ('--seq-length', '8192', '2048'))}


def change_options(before: list[str], variant: str) -> list[str]:
    after = validate_anchor(before, 'anchor')
    for flag, old, new in CHANGES[variant]:
        index = after.index(flag) + 1
        assert after[index] == old
        after[index] = new
    assert sum(a != b for a, b in zip(before, after)) == len(CHANGES[variant])
    return after


if __name__ == '__main__':
    base.change_options = change_options
    base.CHANGES.update(CHANGES)
    base.main()
