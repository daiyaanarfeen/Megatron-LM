"""Change exactly one workload argument, recording the effective rank launch."""

import json
import os
import shlex
from pathlib import Path


PERTURBATIONS = {
    'anchor': (),
    'e512': (('--num-experts', '128', '512'),),
    'e512_seq2048': (('--num-experts', '128', '512'), ('--seq-length', '8192', '2048')),
}


def main():
    variant = os.environ['SENSITIVITY_VARIANT']
    before = shlex.split(os.environ['options'])
    after = before.copy()
    change = PERTURBATIONS[variant]
    for name, value in [('--hidden-size', '1280'), ('--ffn-hidden-size', '768'),
                        ('--num-experts', '128'), ('--seq-length', '8192'),
                        ('--moe-router-topk', '6'), ('--ddp-num-buckets', '8')]:
        assert before.count(name) == 1 and before[before.index(name) + 1] == value
    for flag, old, new in change:
        index = after.index(flag) + 1
        assert after[index] == old
        after[index] = new
    assert sum(a != b for a, b in zip(before, after)) == len(change)
    mbs = 2
    assert os.environ['REPLICA_MICRO_BATCH_SIZES'] == f'{mbs} {mbs}'
    assert os.environ['REPLICA_NUM_MICROBATCHES'] == '3 3'
    assert int(os.environ['TRUE_GLOBAL_BATCH_SIZE']) == int(os.environ['WORLD_SIZE']) * mbs * 3
    assert os.environ['CUPTI_EVENTS_ONLY'] == '1'
    assert os.environ.get('CUPTI_COLLECTION_DISABLED', '0') == '0'
    root = Path(os.environ['CUPTI_OUTPUT_DIR']).parent / 'launch_manifest'
    root.mkdir(exist_ok=True)
    manifest = dict(variant=variant, change=change, micro_batch_size=mbs, grad_accumulation=3,
                    batch_change=([2, 1] if variant == 'mbs1' else None),
                    options_before=before, options=after,
                    environment={key: value for key, value in os.environ.items()
                                 if key.startswith(('NCCL_', 'NVTE_', 'MEGATRON_', 'CUPTI_',
                                                    'REPLICA_', 'SOFT_', 'SLURM_', 'OMP_',
                                                    'TORCH_', 'CUDA_', 'PYTORCH_'))
                                 or key in ('RANK', 'WORLD_SIZE', 'TRUE_GLOBAL_BATCH_SIZE',
                                            'NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN')})
    (root / f"rank-{os.environ['RANK']}.json").write_text(json.dumps(manifest, indent=2) + '\n')
    assert all(not any(c.isspace() for c in token) for token in after)
    os.environ['options'] = ' '.join(after)
    os.execv('/bin/bash', ['bash', str(Path(__file__).with_name('run_lyris_training_scripts_nep_rank.sh'))])


if __name__ == '__main__':
    main()
