"""Apply one architecture perturbation before the unchanged profiled entrypoint."""

import hashlib
import json
import os
import sys
from pathlib import Path

EXPECTED_SHA = '653a7efd3da53c0ace84c9e32d65eb2faa1f9b17f44b1daf43ace136a3eae0c7'
CHANGES = {
    'anchor': None,
    'expert_f1536': ('--ffn-hidden-size', '768', '1536'),
    'shared_f3072': ('--moe-shared-expert-intermediate-size', '1536', '3072'),
}


def change_options(before: list[str], variant: str) -> list[str]:
    """Validate the anchor and change exactly one independent argument."""
    for flag, value in (
        ('--hidden-size', '1280'),
        ('--ffn-hidden-size', '768'),
        ('--moe-shared-expert-intermediate-size', '1536'),
        ('--num-experts', '128'),
        ('--moe-router-topk', '6'),
        ('--seq-length', '8192'),
        ('--ddp-num-buckets', '8'),
    ):
        assert before.count(flag) == 1 and before[before.index(flag) + 1] == value
    after = before.copy()
    change = CHANGES[variant]
    if change is not None:
        flag, old, new = change
        index = after.index(flag) + 1
        assert after[index] == old
        after[index] = new
    assert sum(a != b for a, b in zip(before, after)) == int(change is not None)
    return after


def main() -> None:
    variant = os.environ['SYNC_WORKLOAD_VARIANT']
    before = sys.argv[1:]
    after = change_options(before, variant)
    assert os.environ['REPLICA_MICRO_BATCH_SIZES'] == '2 2'
    assert os.environ['REPLICA_NUM_MICROBATCHES'] == '3 3'
    assert os.environ['CUPTI_EVENTS_ONLY'] == '1'
    assert os.environ['CUPTI_COLLECTION_DISABLED'] == '0'
    runtime = Path(os.environ['CODE_REPO']) / 'megatron/core/distributed/nonuniform_ep.py'
    sha = hashlib.sha256(runtime.read_bytes()).hexdigest()
    assert sha == EXPECTED_SHA, 'Do not mix runtime versions in a workload ablation'
    output = Path(os.environ['CUPTI_OUTPUT_DIR']).parent / 'launch_manifest'
    output.mkdir(exist_ok=True)
    manifest = dict(
        variant=variant,
        change=CHANGES[variant],
        runtime_sha256=sha,
        options_before=before,
        options=after,
        environment={
            key: value
            for key, value in os.environ.items()
            if key.startswith(
                (
                    'NCCL_',
                    'NVTE_',
                    'MEGATRON_',
                    'CUPTI_',
                    'REPLICA_',
                    'SOFT_',
                    'SLURM_',
                    'OMP_',
                    'TORCH_',
                    'CUDA_',
                    'PYTORCH_',
                )
            )
            or key
            in (
                'RANK',
                'WORLD_SIZE',
                'TRUE_GLOBAL_BATCH_SIZE',
                'NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN',
            )
        },
    )
    (output / f"rank-{os.environ['RANK']}.json").write_text(json.dumps(manifest, indent=2) + '\n')
    sys.argv[1:] = after
    from pretrain_profile_activity_probe import main as profiled_main

    profiled_main()


if __name__ == '__main__':
    main()
