"""Keep the existing hard-workload profiler and add untimed all-rank memory peaks."""

import atexit
import json
import os
from pathlib import Path


def main():
    import torch
    from pretrain_a8b_graph_attribution import main as train

    def memory():
        root = Path(os.environ['NSYS_OUTPUT_DIR']).parent / 'memory_peaks'
        root.mkdir(exist_ok=True)
        values = dict(
            rank=int(os.environ['RANK']),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            allocated_bytes=torch.cuda.memory_allocated(),
            backward_sm_margin=os.environ['NVTE_BWD_LAYERNORM_SM_MARGIN'],
        )
        (root / f"rank-{values['rank']}.json").write_text(json.dumps(values, indent=2) + '\n')

    atexit.register(memory)
    train()


if __name__ == '__main__':
    main()
