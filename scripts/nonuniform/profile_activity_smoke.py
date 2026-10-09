"""Verify device-only capture before launching distributed training."""

import json
import os
from collections import Counter
from pathlib import Path

import torch

from pretrain_profile_activity_probe import install_profile_mode


def main():
    root = Path(os.environ['PROFILE_SMOKE_ROOT'])
    root.mkdir(parents=True, exist_ok=True)
    original = torch.profiler.profile
    a = torch.randn(1024, 1024, device='cuda')
    for _ in range(10):
        torch.mm(a, a)
    torch.cuda.synchronize()
    for mode in ('cpu_cuda', 'cuda'):
        torch.profiler.profile = original
        install_profile_mode(mode)
        with torch.profiler.profile(record_shapes=False, with_stack=False) as p:
            for _ in range(10):
                torch.mm(a, a)
            torch.cuda.synchronize()
        path = root / f'{mode}.json'
        p.export_chrome_trace(str(path))
        counts = Counter(e.get('cat') for e in json.loads(path.read_text())['traceEvents'])
        assert counts['kernel'] == 10, counts
        assert (counts['cpu_op'] > 0) == (mode == 'cpu_cuda'), counts
        print(f'SMOKE_PASS mode={mode} counts={dict(counts)}', flush=True)


if __name__ == '__main__':
    main()
