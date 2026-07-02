# NEP Approach A Journal

Append a dated entry whenever we do something new: code changes, job submissions, benchmark results, trace analysis, or decisions that change the next step. Keep entries factual and include job IDs, run dirs, and commits when available.

## 2026-07-01

### a3b training-script Approach-A baseline versus NEP submission

- Started the next comparison on the real `examples/training_scripts` workload, using `a3b_30b_moe_1t` as the smallest practical MoE baseline.
- Added `scripts/nonuniform/run_dlcluster_a3b_30b_moe_approach_a.sh`, a dlcluster GB200 wrapper for the `a3b_30b_moe_1t_approach_a_nccl.sh` model configuration with dummy data, forced MoE load balancing, no distributed optimizer, and `--nonuniform-skip-optimizer-step`.
- Submitted a fair no-optimizer Approach-A pair on `gb200nvl72`, account `blackwell`:
  - Healthy comparison job `1437115` / `dl_a3b_moe_h16_16`, topology `16 16`, TP `2`, nominal EP `32`, 64 active GPUs, GBS `32`, 20 train iterations.
  - NEP comparison job `1437135` / `dl_a3b_moe_nep16_8`, topology `16 8`, TP `2`, nominal EP `32`, 48 active GPUs, GBS `24`, 20 train iterations.
- Slurm state immediately after submission: both jobs pending on priority. `squeue --start` estimated job `1437135` for `2026-07-02T12:20:00`; job `1437115` had no start estimate yet.
- Replaced the initial EP32 pair with smaller queued shapes after Slurm estimates were several hours to a day out:
  - EP16 pair (`1437165`/`1437166`) was submitted then canceled after the healthy estimate moved later.
  - Final active pair is EP8 to minimize GPU count while keeping the same a3b training-script model architecture:
    - Healthy comparison job `1437187` / `dl_a3b_moe_ep8_h4_4_30m`, topology `4 4`, TP `2`, nominal EP `8`, 16 active GPUs, GBS `8`, 20 train iterations, 30-minute walltime.
    - NEP comparison job `1437188` / `dl_a3b_moe_ep8_nep4_2_30m`, topology `4 2`, TP `2`, nominal EP `8`, 12 active GPUs, GBS `6`, 20 train iterations, 30-minute walltime.
  - `squeue --start` estimated NEP `1437188` for `2026-07-01T20:52:34` and healthy `1437187` for `2026-07-01T22:25:06`.
- NEP job `1437188` started first on `gb-nvl-147-compute[04,08-09]`.
  - Startup spent about 9 minutes staging the 29.6 GB sqsh image and creating/using the enroot container under `/tmp/darfeen_mep_a3b`.
  - The training process then failed before iteration 0 with `AssertionError: Parameter hashes not matching across DP replicas`.
  - Root cause: the wrapper inherited `--check-weight-hash-across-dp-replicas-interval 20000`; Megatron checks this unconditionally before iteration 0 when the flag is present. That cross-check is invalid for reduced-EP NEP replicas because expert parameter shards are intentionally partitioned differently.
  - Updated `scripts/nonuniform/run_dlcluster_a3b_30b_moe_approach_a.sh` so the weight-hash check is opt-in via `CHECK_WEIGHT_HASH_ACROSS_DP_REPLICAS_INTERVAL` and omitted by default.
  - Canceled failed job `1437188` and submitted fixed NEP job `1439079` / `dl_a3b_moe_ep8_nep4_2_nohash`, topology `4 2`, TP `2`, nominal EP `8`, 12 active GPUs, GBS `6`, 20 train iterations, 30-minute walltime.
  - `squeue --start` estimated healthy `1437187` for `2026-07-01T20:52:34` and fixed NEP `1439079` for `2026-07-01T23:34:51`.
- Healthy EP8 job `1437187` later started and failed during staging before training:
  - Exit status `12`, run dir `/home/scratch.darfeen_gpu/dlcluster_runs/dl_a3b_30b_moe_ep8_h4_4_gbs8_t20_30m/1437187/`.
  - Root cause from `driver_1437187.log`: several compute-node rsyncs from `dlcluster-login-01` failed with `exec request failed on channel 0` / rsync protocol code `12`.
  - Updated `scripts/nonuniform/run_dlcluster_a3b_30b_moe_approach_a.sh` to retry image and repo rsyncs up to five times with increasing backoff.
- Submitted smaller EP4 pairs to reduce GPU count and improve queue time:
  - Initial 20-iteration EP4 pair (`1439361` healthy topology `2 2`, `1439367` NEP topology `2 1`) was submitted then canceled after estimates moved later.
  - Active pair is now 10 iterations:
    - Healthy job `1439376` / `dl_a3b_moe_ep4_h2_2_t10`, topology `2 2`, TP `2`, nominal EP `4`, 8 active GPUs, GBS `4`.
    - NEP job `1439385` / `dl_a3b_moe_ep4_nep2_1_t10`, topology `2 1`, TP `2`, nominal EP `4`, 6 active GPUs, GBS `3`.
  - Immediate `squeue --start` after submission estimated healthy `1439376` for `2026-07-02T03:26:00`; NEP `1439385` had no start estimate yet.
- Jobs `1439376` and `1439385` both completed successfully with exit `0`.
  - Healthy `1439376` ran on `gb-nvl-147-compute[05,09]`, elapsed `00:09:06`, log tarball `/home/scratch.darfeen_gpu/dlcluster_runs/dl_a3b_30b_moe_ep4_h2_2_gbs4_t10_retry/1439376/logs_1439376.tgz`.
  - NEP `1439385` ran on `gb-nvl-147-compute[05,09]`, elapsed `00:02:43`, log tarball `/home/scratch.darfeen_gpu/dlcluster_runs/dl_a3b_30b_moe_ep4_nep2_1_gbs3_t10_retry/1439385/logs_1439385.tgz`.
  - Both logs had finite losses and zero skipped/nan iterations.
  - Warmup-excluded averages over iterations 3-10:
    - Healthy topology `2 2`: `710.263 ms/iter`, `128.5 TFLOP/s/GPU`, `0.704 samples/s/GPU`.
    - NEP topology `2 1`: `1949.438 ms/iter`, `46.825 TFLOP/s/GPU`, `0.256 samples/s/GPU`.
  - This EP4/EP2 NEP configuration is much slower than the healthy EP4 pair; the reduced replica is extremely small, so NEP reshard overhead dominates this tiny real-workload run.

## 2026-06-30

### dlcluster GB200 minimal healthy versus NEP overlap benchmark

- User rejected the earlier 32-node A8B submission direction and asked for the minimal job where overlap should be visible.
- Added/updated launch and analysis support:
  - `scripts/nonuniform/run_dlcluster_nonuniform_ep_overlap_diag.sh` now defaults to `gb200nvl72` and accepts model/topology/profiler settings via environment.
  - `examples/training_scripts/nonuniform_ep_approach_a_smoke.sh` accepts `EXTRA_MEGATRON_ARGS`.
  - `examples/nonuniform/pretrain_gpt_nonuniform.py` now mirrors the hybrid entrypoint's `--nonuniform-skip-optimizer-step` no-op optimizer hook for performance-only GPT validation.
  - `scripts/nonuniform/analyze_dlcluster_min_nep.py` parses iteration metrics, NEP overlap debug records, and compressed PyTorch trace JSON.
- Submitted a two-job minimal comparison on `gb200nvl72`, account `blackwell`:
  - Healthy job `1430778` / `dl_min_healthy_ep4_ep4`, topology `4 4`, 8 active GPUs, GBS 8.
  - NEP job `1430779` / `dl_min_nep_ep4_ep2`, topology `4 2`, 6 active GPUs, GBS 6.
  - Shared model: 16 layers, hidden 2048, FFN 8192, sequence 1024, 4 experts, MBS 1, 20 train iterations, dummy data with forced MoE load balancing, PyTorch profiler steps 10-12, no-op optimizer step.
  - Both jobs completed successfully with exit `0`.
  - Results were copied under `/home/scratch.darfeen_gpu/dlcluster_runs/` and unpacked under `slurm_runs/dlcluster_min_results/`.
- Performance using post-profiler iterations 13-20:
  - Healthy EP `4 4`: `225.975 ms/iter`, `23.04 TFLOP/s/GPU`, `4.425 samples/s/GPU`.
  - NEP EP `4 2`: `195.650 ms/iter`, `26.60 TFLOP/s/GPU`, `5.111 samples/s/GPU`.
  - NEP was about `15.5%` faster per GPU on this minimal no-op-optimizer benchmark.
- Overlap/correctness signals:
  - Both logs had finite losses and zero skipped/nan iterations.
  - NEP debug records: 2337 task groups, average comm `2.365 ms`, average final drain wait `0.269 ms`, hidden-by-drain metric `1 - sum(wait)/sum(comm) = 88.6%`.
  - Healthy debug records: 3154 task groups, hidden-by-drain metric `61.1%`.
  - PyTorch traces show NEP owner ranks have `640` CPU `nccl:all_to_all` events versus healthy `384`, matching extra NEP reshard all-to-alls.
  - Same-GPU NCCL-kernel overlap with non-NCCL GPU kernels was low: about `4.5%` across NEP ranks and about `5.2%` on NEP owner ranks. Interpretation: the reshard work is launched during backward and mostly not exposed at the final drain, but the NCCL kernels mostly run in gaps/serialization rather than concurrently with compute kernels.

### dlcluster GB300 segmented smoke runs

- Ported the HSG-style nonuniform EP wrapper to dlcluster as `scripts/nonuniform/run_dlcluster_hsg_a8b_ep64_ep32_nep.sh`.
- `gb300nvl72_preprod` accepts segmented multi-node requests; `scontrol show job` reports `SegmentSize=4` for `sbatch --segment=4`.
- 1-node tiny validation:
  - Job `1428803`, partition `gb200nvl4`, topology `2 2`, completed successfully.
- 2-node GB300 first attempt:
  - Job `1429125`, partition `gb300nvl72_preprod`, topology `6 2`, nodes `gb300-nvl-022-compute[08,14]`.
  - Failed because the staged node-local source tree did not include the compiled dataset extension `megatron.core.datasets.helpers_cpp`; `MockGPTDataset` raised `ModuleNotFoundError`.
  - Pending 4-node job `1429156` was cancelled before it could hit the same packaging failure.
- Wrapper fix:
  - During per-node staging, create/reuse the named enroot container and compile `helpers_cpp` inside that container before launching training.
  - This uses `python -c "from megatron.core.datasets.utils import compile_helpers; compile_helpers()"` inside the mounted node-local repo.
  - The stage archive `/tmp/megatron_ep_stage.tar.gz` must be refreshed after wrapper changes because dlcluster jobs pull that archive, not the live checkout.
- 2-node GB300 fixed validation:
  - Job `1429317`, partition `gb300nvl72_preprod`, topology `6 2`, nodes `gb300-nvl-022-compute[08,13]`, completed successfully with exit `0`.
  - Result tarball: `/home/scratch.darfeen_gpu/dlcluster_runs/dlcluster_hsg_path_tiny_2n_gb300_topo6_2_t2_fix_helpers/1429317/logs_1429317.tgz`.
  - Both nodes compiled `helpers_cpp`; rank 0 then reported `make: Nothing to be done for 'default'` during dataset-index builder setup.
  - Two training iterations completed. Iteration logs reported `lm loss` around `1.0436E+01`.
  - Full job elapsed time was `14:29`; most of it was first-time image staging on fresh node `gb300-nvl-022-compute13`.
- Enroot/cache finding and fix:
  - Pyxis documents that `--container-name=NAME` reuses an existing named container and skips import.
  - The wrapper already used `--container-name`, but previously copied the `29.6 GB` `.sqsh` before checking whether the named enroot container existed.
  - The wrapper now sets `ENROOT_CACHE_PATH`, `ENROOT_DATA_PATH`, and `ENROOT_RUNTIME_PATH` before image sync, checks `enroot list`, and skips image sync when the named container already exists.
  - This avoids repeated image copies on nodes that keep `/tmp/darfeen_mep_hsg_a8b/enroot-data` between jobs. First use on a fresh node still requires one full image copy unless the image/container is pre-staged on that node.
- 4-node GB300 fixed validation:
  - Job `1429469`, partition `gb300nvl72_preprod`, topology `12 4`, submitted with `--segment=4`.
  - Nodes: `gb300-nvl-022-compute[03,08,13,16]`.
  - Completed successfully with exit `0`.
  - Result tarball: `/home/scratch.darfeen_gpu/dlcluster_runs/dlcluster_hsg_path_tiny_4n_gb300_topo12_4_t2_fix_helpers_cache/1429469/logs_1429469.tgz`.
  - Full job elapsed time was `12:57`; the main distributed step elapsed `3:01`.
  - Cache behavior matched the intended fix: reused nodes `compute08` and `compute13` skipped image sync, while fresh nodes `compute03` and `compute16` still paid the one-time `29.6 GB` image copy and enroot extraction.
  - Two training iterations completed. Iteration logs:
    - Iteration 1/2: elapsed `124123.1 ms`, `lm loss: 1.043616E+01`.
    - Iteration 2/2: elapsed `62098.2 ms`, `lm loss: 1.044082E+01`.
  - The log ended with `[dlcluster-hsg] exit status: 0`.

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

### dlcluster NEP overlap diagnostic

- Added `scripts/nonuniform/run_dlcluster_nonuniform_ep_overlap_diag.sh` for a 2-node GB300 overlap diagnostic using topology `4 2`, 6 ranks, 16 layers, hidden size 1024, sequence length 1024, GBS 6, and 16 train iterations.
- The wrapper stages the repo/image to node-local `/tmp`, reuses the warmed enroot cache under `/tmp/darfeen_mep_hsg_a8b`, and gathers node-local run artifacts back to `/home/scratch.darfeen_gpu/dlcluster_runs/dlcluster_nep_overlap_diag_ep4_ep2_l16_h1024_s1024/<jobid>/`.
- Job `1429925` completed the workload with PyTorch profiler enabled, but the traces had no CUDA kernel records. The log reported `CUPTI_ERROR_INVALID_DEVICE`, so the PyTorch profiler data was CPU/NVTX-only and could not answer GPU overlap.
- Added an `ENABLE_NSYS_PROFILE` mode to `examples/training_scripts/nonuniform_ep_approach_a_smoke.sh`.
- Job `1430076` used `nsys --capture-range=cudaProfilerApi`; Megatron's CUDA profiler start/stop happened in child rank processes and nsys reported `No reports were generated`.
- Job `1430118` used full-run nsys capture around `torchrun`; it produced `.nsys-rep` and `.sqlite` files on both nodes, but the SQLite exports contained only NVTX tables and no CUDA runtime/kernel tables.
- Job `1430182` used direct Slurm rank launch with one Python rank per task and one nsys wrapper per rank. It produced per-rank nsys reports, but those SQLite exports also contained only NVTX tables. This confirms CUDA activity capture is blocked in this dlcluster/container stack rather than being only a torchrun child-process issue.
- Added diagnostic-only CUDA-event logging behind `MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=1` in `megatron/core/distributed/nonuniform_ep.py`.
- Job `1430321` ran the CUDA-event diagnostic without nsys and completed successfully.
- Parsed `1430321` logs:
  - 1845 overlap records across ranks 0-5.
  - `comm_ms`: min `0.002`, p50 `0.859`, p90 `4.233`, p99 `5.826`, max `9.888`, mean `1.725`.
  - `ready_since_comm_start_ms`: min `2.677`, p50 `29.388`, p90 `47.492`, p99 `75.806`, max `86.611`, mean `29.556`.
  - `finish_wait_ms`: min `0.002`, p50 `0.039`, p90 `0.058`, p99 `1.769`, max `8.039`, mean `0.092`.
  - `cpu_drain_ms`: min `0.006`, p50 `0.018`, p90 `0.029`, p99 `0.073`, max `0.095`, mean `0.020`.
  - There were zero records where `comm_ms > ready_since_comm_start_ms`; the comm stream always had enough later default-stream work available to hide the measured comm window in this diagnostic.
  - Only 29 of 1845 records had `finish_wait_ms > 1 ms`, mostly early group-0 cases. Median and p90 exposed wait were tiny.
- Current conclusion: in the larger diagnostic where overlap should be possible, NEP NCCL reshard work is being overlapped with later CUDA work. No implementation fix is indicated from this test. The remaining profiler limitation is that CUPTI CUDA activity collection is unavailable on this dlcluster/container path, so CUDA-kernel timeline proof requires either a different container/permissions setup or lower-level instrumentation.

## 2026-07-02

### Larger-MBS overlap sweep

- Submitted a follow-up `a3b_30b_moe_1t` EP8 sweep to test whether increasing useful non-NCCL work per gradient sync can hide NEP reshard:
  - Healthy MBS2/GBS16: job `1448087`, topology `4 4`, `--ddp-num-buckets 16`.
  - NEP MBS2/GBS12: job `1448089`, topology `4 2`, `--ddp-num-buckets 16`, `MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=128`.
  - Healthy MBS4/GBS32: job `1448091`, topology `4 4`, `--ddp-num-buckets 16`.
  - NEP MBS4/GBS24: job `1448096`, topology `4 2`, `--ddp-num-buckets 16`, `MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=128`.
- All four were initially pending with `ReqNodeNotAvail`.
- Follow-up queue investigation showed the `gb200nvl72` partition was not currently available to `darfeen`:
  - `gb-nvl-081/082/147` nodes are in reservation `DLR-5450-2` until `2027-07-02T14:00:00`, with allowed users `dlfwadmin,jvaddi`.
  - `gb-nvl-115` nodes are in reservation `DLR-3954` until `2026-12-20T03:14:12`, with a different allowed-user list.
  - The remaining visible `gb200nvl72` nodes were draining.
- Current status: the sweep jobs are submitted but will not start unless reservation access changes, the jobs are moved to an accessible GB200/NVL72 allocation, or the partition state changes.

### dlcluster training-script EP8 profile comparison

- Submitted a profiled `a3b_30b_moe_1t` training-script comparison on `dlcluster` / `gb200nvl72`:
  - Healthy baseline job `1440214`, topology `4 4`, EP size 8, 16 GPUs, GBS 8, `PROFILE_RANKS=0`.
  - NEP job `1440215`, topology `4 2`, EP size 8, 12 GPUs, GBS 6, `PROFILE_RANKS=0`.
- NEP job `1440215` completed successfully with zero skipped/nan iterations.
- NEP post-profile iterations 8-12 averaged `2780.02 ms`, `32.9 TFLOP/s/GPU`, and `0.180 samples/s/GPU`.
- NEP overlap debug reported tiny finish waits:
  - 1474 debug records.
  - Average `comm_ms=194.83`, max `comm_ms=3751.28`.
  - Average `finish_wait_ms=0.052`, max `finish_wait_ms=0.197`.
  - Derived finish-wait exposure fraction was about `0.027%`.
- Rank-0 PyTorch trace tells a different story about useful GPU overlap:
  - Trace span about `5664 ms`.
  - NCCL kernel union about `4014 ms`.
  - Non-NCCL kernel union about `625 ms`.
  - NCCL/non-NCCL interval overlap about `7.7 ms` (`0.19%` of NCCL union).
- Interpretation so far: `finish_wait_ms` only shows that async NEP work completed before `finish_grad_sync`; it does not prove the NCCL kernels overlapped with later useful compute. In this workload/profile, the NEP reshard traffic appears mostly exposed on rank 0. Healthy job `1440214` was still pending on priority with estimated start `2026-07-02T05:13:29Z` at the time of this entry.
- Follow-up hypothesis test submitted as job `1441370`: same NEP topology `4 2`, same GBS 6 and profiler settings, but `MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=128` instead of the wrapper default `16`. This tests whether launch-path slot/drain waits are contributing to the exposed reshard time.
- Healthy baseline job `1440214` completed successfully after a long staging phase caused by one node missing the cached enroot container and syncing the sqsh image.
- Healthy post-profile iterations 8-12 averaged `798.94 ms`, `117.64 TFLOP/s/GPU`, and `0.626 samples/s/GPU`.
- Direct comparison, iterations 8-12:
  - Healthy `4 4`: `798.94 ms`, `117.64 TFLOP/s/GPU`, `0.626 samples/s/GPU`.
  - NEP `4 2`: `2780.02 ms`, `32.90 TFLOP/s/GPU`, `0.180 samples/s/GPU`.
  - NEP is about `3.48x` slower by iteration time and about `28%` of healthy per-GPU TFLOP/s.
- Rank-0 trace comparison:
  - Healthy trace span about `1904 ms`, NCCL union about `184.5 ms`, non-NCCL kernel union about `357.6 ms`, NCCL/non-NCCL overlap about `5.0 ms`.
  - NEP trace span about `5664 ms`, NCCL union about `4014.2 ms`, non-NCCL kernel union about `624.5 ms`, NCCL/non-NCCL overlap about `7.7 ms`.
  - Healthy had no NCCL interval over `20 ms`; NEP had repeated long send/recv intervals around `34-37 ms`.
- Conclusion from this comparison: the slowdown is real reshard exposure in this training-script workload, not a profiler artifact. The internal `finish_wait_ms` metric remains misleading because it only proves the async work is complete by final drain, while the PyTorch trace shows the NEP NCCL kernels are mostly serialized/exposed relative to useful GPU compute.
- Follow-up job `1441370` failed before training during the initial Slurm `srun` staging step: `Unable to confirm allocation for job 1441370: Socket timed out on send/recv operation`. No performance data was collected from that job.

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
