"""Stronger workload-only perturbations using the accepted profiling harness."""

import pretrain_nep_sync_workload as base

CHANGES = {
    'anchor': (),
    'expert_f3072': (('--ffn-hidden-size', '768', '3072'),),
    'expert_f3072_seq2048': (
        ('--ffn-hidden-size', '768', '3072'),
        ('--seq-length', '8192', '2048'),
    ),
}
validate_anchor = base.change_options


def change_options(before: list[str], variant: str) -> list[str]:
    """Keep all independent settings except the named one or two fixed."""
    after = validate_anchor(before, 'anchor')
    for flag, old, new in CHANGES[variant]:
        index = after.index(flag) + 1
        assert after[index] == old
        after[index] = new
    assert sum(a != b for a, b in zip(before, after)) == len(CHANGES[variant])
    return after


if __name__ == '__main__':
    base.change_options = change_options
    base.CHANGES = dict(CHANGES, anchor=None)
    base.main()
