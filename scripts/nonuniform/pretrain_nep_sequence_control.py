"""Change only sequence length between pinned E512 workload comparisons."""

import os

import pretrain_nep_collective_probe as probe


probe.CHANGES['e512_seq8192'] = (('--num-experts', '128', '512'),)


if __name__ == '__main__':
    probe.base.EXPECTED_SHA = os.environ['NEP_PREFETCH_EXPECTED_SHA']
    probe.base.change_options = probe.change_options
    probe.base.CHANGES.update(probe.CHANGES)
    probe.base.main()
