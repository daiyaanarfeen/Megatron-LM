"""S8192 workload with no profiler or measurement hooks installed."""

import os
import runpy

import pretrain_nep_collective_probe as probe


if __name__ == '__main__':
    assert os.environ['NEP_TAIL_MODE'] == 'none'
    probe.CHANGES['e512_seq8192'] = (('--num-experts', '128', '512'),)
    runpy.run_module('pretrain_nep_tail_control', run_name='__main__')
