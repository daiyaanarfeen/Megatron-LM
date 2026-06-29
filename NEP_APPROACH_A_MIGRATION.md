# NEP Approach A Migration Notes

## Branch and remote

- Working branch: `nonuniform-approach-a-training-scripts`
- Intended push remote: `git@github.com:daiyaanarfeen/Megatron-LM.git`
- Do not push to `NVIDIA/Megatron-LM` or the `dnarayanan/training_scripts` branch.

## Implementation state

- Current implementation is Approach A for nonuniform EP gradient sync.
- NEP NCCL reshard gather/scatter now uses the dedicated `nep_transfer_group`, not the regular MoE EP communicator.
- This fixed the EP `4 2` smoke hang where the regular MoE token-dispatch `ALLTOALL_BASE` timed out after NEP collectives interleaved on the EP group.
- Key code is in:
  - `megatron/core/distributed/nonuniform_ep.py`
  - `megatron/core/distributed/nonuniform_common.py`
  - `examples/nonuniform/pretrain_gpt_nonuniform.py`

## Validated runs

- Failed before communicator isolation:
  - Job: `3599151` / `nep_ep4_2_masked`
  - Run dir: `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/darfeen/training_scripts_dp1_dummy_runs/nonuniform_ep_task_sched_ep4_2_masked_batch_profile`
  - Result: completed iteration 1, then NCCL watchdog timeout in regular EP `ALLTOALL_BASE`.

- Passed after communicator isolation:
  - Job: `3640099` / `nep_ep4_2_xfergrp`
  - Run dir: `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/darfeen/training_scripts_dp1_dummy_runs/nonuniform_ep_task_sched_ep4_2_transfer_group_batch_profile`
  - Result: completed all 6 iterations and wrote profiler traces for ranks 0, 2, and 4.

## Benchmarks submitted

These were submitted for the first small slowdown comparison. Check their latest state with `squeue`/`sacct` before taking next action.

- Healthy baseline:
  - Job: `3640490` / `nep_bench_ep8healthy`
  - Topology: `8 8`
  - Nodes/ranks: 4 nodes x 4 GPUs = 16 ranks
  - GBS/MBS: `16` / `1`
  - Run dir: `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/darfeen/training_scripts_dp1_dummy_runs/nonuniform_ep_bench_healthy_ep8_ep8_h1024_l8_s1024`

- NEP comparison:
  - Job: `3640685` / `nep_bench_ep8ep4`
  - Topology: `8 4`
  - Nodes/ranks: 3 nodes x 4 GPUs = 12 ranks
  - GBS/MBS: `12` / `1`
  - Run dir: `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/darfeen/training_scripts_dp1_dummy_runs/nonuniform_ep_bench_nep_ep8_ep4_h1024_l8_s1024`

Both benchmark submissions used:

- `TRAIN_ITERS=30`
- `NUM_EXPERTS=8`
- `NUM_LAYERS=8`
- `HIDDEN_SIZE=1024`
- `FFN_HIDDEN_SIZE=4096`
- `NUM_ATTENTION_HEADS=16`
- `SEQ_LENGTH=1024`
- debug/profiler disabled

## Local-only context

- There is a git stash named `nep-a2a-pre-hook-scheduler-snapshot` in the original worktree. The source rollback snapshot is also copied under `.codex_rollback/` and is committed for migration.
- Runtime outputs under `/lustre/.../training_scripts_dp1_dummy_runs/` are not part of git. Copy the run directories above if the new machine cannot see the same Lustre filesystem.
