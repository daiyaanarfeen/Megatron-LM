# NEP Approach A Journal

Append a dated entry whenever we do something new: code changes, job submissions, benchmark results, trace analysis, or decisions that change the next step. Keep entries factual and include job IDs, run dirs, and commits when available.

## 2026-06-30

### Journal and remote sync

- Started this append-only journal so future NEP Approach A actions are recorded as dated entries.
- Prepared a remote-sync commit containing this journal and the smoke-script `CUDA_DEVICE_MAX_CONNECTIONS` override needed by the queued overlap diagnostic.

### Queued overlap diagnostic

- Submitted job `3672204` / `nep_overlap_diag_ep4ep2`.
- Purpose: determine whether a correct NEP implementation can overlap owner-transfer/all-reduce/scatter with later backward compute, using a smaller EP `4 2` diagnostic instead of the larger training-script workloads.
- Configuration:
  - Topology: `NONUNIFORM_EP_TOPOLOGY="4 2"`
  - World: 6 ranks via `--nodes=2`, `--gpus-per-node=4`, `NPROC_PER_NODE=3`
  - Model: `NUM_LAYERS=16`, `HIDDEN_SIZE=1024`, `FFN_HIDDEN_SIZE=4096`, `SEQ_LENGTH=1024`, `NUM_EXPERTS=4`
  - Batches: `GLOBAL_BATCH_SIZE=6`, `MICRO_BATCH_SIZE=1`, `TRAIN_ITERS=16`
  - Profiling: steps 8-12, ranks `0 1 2 3 4 5`
  - Overlap probes: `CUDA_DEVICE_MAX_CONNECTIONS=32`, `MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16`
- Run dir: `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/darfeen/training_scripts_dp1_dummy_runs/nonuniform_ep_overlap_diag_ep4_ep2_l16_h1024_s1024_w16_cmc32_profile`
- Current state when submitted/polled: `PENDING`, no start estimate, no artifacts yet.
- Local code change made for this diagnostic: `examples/training_scripts/nonuniform_ep_approach_a_smoke.sh` now lets `CUDA_DEVICE_MAX_CONNECTIONS` be overridden from the batch environment while preserving default `1`.

## 2026-06-29

### Current-code EP8/4 validation

- Validated the fixed NEP NCCL path on EP `8 4`.
- Job `3645757` / `nep_ep8ep4_paramslots` completed 4/4 iterations with debug enabled.
- Job `3645970` / `nep_ep8ep4_paramslots_l4` completed 8/8 iterations on the 4-layer validation case; steady-state iterations 3-8 averaged about `66.3 ms`.
- Key correctness signal: healthy owner and reduced owner ranks used matching `chunk_size=2097152`, fixing the earlier EDP all-reduce size mismatch.

### Current-code healthy versus NEP benchmark

- Submitted and completed a fair current-code comparison:
  - Healthy job `3648896` / `nep_bench_ep8healthy_ps`, topology EP `8 8`, 16 GPUs, GBS 16.
  - NEP job `3648922` / `nep_bench_ep8ep4_ps`, topology EP `8 4`, 12 GPUs, GBS 12.
- Both jobs completed cleanly.
- Warmup-excluded iteration averages, iterations 3-30:
  - Healthy EP `8 8`: `114.179 ms`, `8.758 samples/s/GPU`.
  - NEP EP `8 4`: `111.268 ms`, `8.987 samples/s/GPU`.
- Initial read: NEP was close/slightly faster per GPU in this unprofiled run, but the logs did not emit MFU/TFLOPs because `log_throughput=False`.

### Torch profiler comparison

- Submitted and completed profiled current-code comparison:
  - Healthy profile job `3651734` / `nep_prof_ep8healthy_ps`.
  - NEP profile job `3651735` / `nep_prof_ep8ep4_ps`.
- Trace dirs:
  - `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/darfeen/training_scripts_dp1_dummy_runs/nonuniform_ep_bench_healthy_ep8_ep8_h1024_l8_s1024_paramslots_profile/torch_profile`
  - `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/darfeen/training_scripts_dp1_dummy_runs/nonuniform_ep_bench_nep_ep8_ep4_h1024_l8_s1024_paramslots_profile/torch_profile`
- Correctness from trace:
  - NEP owner-transfer uses `Process Group Description=nep_owner_transfer`, `Collective name=all_to_allv`, dtype `Float`.
  - Reduced owner all-reduce uses `Process Group Description=ep_dp`.
  - Regular MoE token dispatch remains separate `ep` all-to-all, dtype `BFloat16`.
  - Extra healthy ranks participate in owner-transfer but not `ep_dp` all-reduce; reduced ranks participate in `ep_dp` all-reduce but not owner-transfer.
- Performance from trace:
  - NEP reshard/all-reduce was effectively 0% overlapped with non-NCCL GPU kernels in the profiled toy run.
  - The profiled toy has only about 7-8 ms of non-NCCL GPU work per step on sampled ranks, while NEP reshard/all-reduce work is much larger, so 100% overlap would be impossible for that exact workload.
  - The stronger concern is implementation-level exposure: the async buffer-slot path can call CPU `work.wait()` before later backward compute launches when the async window is too small.

### Implementation fixes pushed before today

- Commit `fe452dadb` (`Fix NEP NCCL owner reshard ordering`) fixed:
  - Per-owner source subgroups for NEP reshard all-to-alls.
  - Rank-invariant per-parameter expert slot ordering.
  - Correct owner/min-rank all-reduce participant matching.
- Commit `fd6245b71` (`Record NEP EP8 EP4 validation result`) recorded the EP8/4 validation result.

## 2026-06-25 to 2026-06-29

### Approach A stabilization

- Moved focus to Approach A only after reading `NONUNIFORM_PARALLELISM_BRANCHES.md`.
- Rebased local work from `dnarayanan/training_scripts` without changing that remote branch.
- Preserved rollback state before reworking the implementation.
- Fixed the EP `4 2` path after earlier hangs by isolating NEP reshard traffic from the regular MoE EP communicator.
- Introduced a safe ordered scheduler for NEP NCCL owner tasks to avoid rank-dependent collective ordering.
- Changed the reshard path to all-to-all based gather/scatter around the owner/min-rank `ep_dp` all-reduce.

## 2026-06-04 to 2026-06-11

### Early profiler runs and performance investigation

- Collected early profiler traces for small EP `4 2` smoke cases and larger `a8b_120b_latentmoe_1t` NEP/baseline runs.
- Early large-workload traces included:
  - `a8b_120b_latentmoe_1t_nep_denseguard_profile25`
  - `a8b_120b_latentmoe_1t_nep_revorder_profile25`
  - `a8b_120b_latentmoe_1t_baseline_profile25_rerun`
- These early runs established that the old NEP overlap path was much slower than the healthy baseline and motivated the later ordered scheduler plus all-to-all reshard work.

## Earlier Benchmarking Context

### Training-script baselines and tuning

- Checked out and examined `dnarayanan/training_scripts`.
- Ran baseline training scripts on dummy data with forced load balancing, one DP replica where applicable, and GBS scaled to the selected parallelism.
- Focused on `8b_1t`, `a8b_120b_latentmoe_1t`, `a3b_30b_moe_1t`, and `a3b_30b_transformer_moe_1t`, then pulled and ran the Nemotron3 nano/super/ultra scripts.
- Tuned allowed knobs only: decrease TP / increase EP or DP, and adjust MBS/GBS.
- Settled on the best known baseline settings for the seven runs before switching focus to NEP Approach A correctness and performance.
