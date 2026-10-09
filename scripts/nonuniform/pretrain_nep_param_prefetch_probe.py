"""Same E512 workload and profiler; select a pinned runtime and optional shadow check."""

import os
import sys
from pathlib import Path

import pretrain_nep_collective_probe as probe


if __name__ == '__main__':
    probe.base.EXPECTED_SHA = os.environ['NEP_PREFETCH_EXPECTED_SHA']
    probe.base.change_options = probe.change_options
    probe.base.CHANGES.update(probe.CHANGES)
    if os.environ.get('NEP_PREFETCH_VALIDATE') == '1':
        root = Path(os.environ['CODE_REPO'])
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / 'tests/unit_tests/distributed'))
        from nep_sync_shadow_entrypoint import install_shadow

        install_shadow()
    probe.base.main()
