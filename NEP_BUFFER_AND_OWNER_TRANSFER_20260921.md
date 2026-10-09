# NEP gradient buffers, owner transfers, and cooperative-kernel stalls

## Scope and rollback

- Trusted baseline: `nep-graph-launch-order-20260921`, commit `8abb29fca` (unchanged).
- Buffer cleanup: `nep-buffer-cleanup-20260921`, commit `ee28f09ff`.
- Parallel owner-transfer experiment: `nep-param-owner-parallel-20260921`, commit `f57cc9a65`.
- Only `megatron/core/distributed/nonuniform_ep.py` changes in production code.
  Additional changes are tests, isolated experiment launchers, analysis, and this record.
  No changes to native Megatron, Transformer Engine, dispatcher, or optimizer algorithms.

## Buffer cleanup

Physical expert gradients still reset before accumulation. Logical-owner gradients are
fully overwritten by staging before native reduce-scatter, so only their alignment padding
is reset. Cached padding offsets refer to live buffers, not stale tensor views. After sync,
only dense and logical-owner gradients are scaled: obsolete physical expert copies are not
optimizer inputs. The non-distributed optimizer retains its parent implementation.

## Parameter dependency

A layer must wait for its own updated parameters. Unrelated owner broadcasts do not need
to serialize on a follower. The experiment gives each owner a persistent transfer stream
and separate bounded staging storage, submits every owner, then joins their completion at
the existing bucket parameter-ready boundary. Buffer reuse is ordered within each owner.
This costs more temporary memory; training performance and memory must be checked before
acceptance. Force-sync and non-overlap modes still establish readiness on the caller stream.

## Backward stall evidence

Nsight in GB200 diagnostic job `3122071` associates `rmsnorm_bwd_general_kernel` with
`cudaLaunchCooperativeKernel`. With no reserved SMs its grid is 456 CTAs (three per SM).
RMSNorm forward and the preceding memset proceed, but the cooperative backward launch
waits for an unrelated resident NCCL kernel. There is no Gather-to-graph device dependency
in this test. Changing TE's existing backward SM margin removes the stall:

| Backward SM margin | No Gather | Gather CTA limit 4 | Gather CTA limit 32 |
|---|---:|---:|---:|
| 0 | 0.203 ms | 14.732 ms | 14.632 ms |
| 8 | 0.205 ms | 0.244 ms | 14.560 ms |
| 32 | 0.225 ms | 0.264 ms | 0.268 ms |

Values are median rank-0 graph elapsed times; the sender delay was deliberately injected.
They are NOT training performance or transfer service times. The original hard benchmark
already used a margin of 16, not zero; its normalization grid was 408 CTAs. Its Gather
kernel had 32 CTAs, exceeding that margin. A training A/B must validate the proposed margin
of 32 and check healthy performance as well. Graph-node `launchType=0` is insufficient to
identify launch mode; uncaptured warmup API/kernel correlation identifies the cooperative
launch. TE documents the SM-margin control in its [environment-variable reference](https://nvidia.github.io/TransformerEngine/envvars.html).

## Correctness validation

- Job `3122017`, GB200: 79 unit tests and the existing four-GPU native AG/RS plus parameter
  prefetch smoke passed on the cleanup branch.
- Job `3122082`, GB300: 80 unit tests, new shared-follower/multiple-owner test, and existing
  AG/RS/prefetch test passed. New tests cover live values, two-slot reuse, forced sync,
  non-overlap, and submitting all owners before joining streams.
- Shared follower rank 1, measured test iteration 2: owner-2 broadcast ran at 9.963-10.057 ms;
  delayed owner-0 broadcast remained active at 9.766-14.490 ms. Independent progress is real.
- These tests are not a substitute for model performance acceptance or a complete
  graph-versus-eager numerical equivalence proof.

## Earlier benchmark attempts

EP8/6 training comparisons initially failed. Jobs `3122068`, `3122093`, `3122109`, and
`3122131` are not accepted measurements: a checkout-selection error was corrected, followed
by repeated RDMA memory-registration failures in the unmodified reference's native dense RS.
Restoring the trusted allocator configuration in `3122146` did NOT resolve the failure, so
the allocator is not an established cause. Job `3122176` returns to the previously working
four-MoE-only-layer smoke (MBS2, GA2, sequence 1024), keeping all-rank profiling. No transport
downgrade was applied. The hard-workload launcher retains its original architecture/settings;
no hard comparison is accepted until the small implementation checks pass. Artifacts live under
`/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/` and
`/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_buffer_cleanup/`.

The journal for the previously successful small run (`3121169`) revealed the missing
launch-time `NCCL_PXN_DISABLE=1` override. The same native dense-RS error 524 was already
documented there. It is restored for every small A/B case, not for the original EP16/12
hard comparison. The allocator change was not the fix. No result from the failed retries
is used as a performance measurement.

## EP8/6 accepted timing experiment

Job `3122197`, GB200, physical domains `nvlblk08` and `nvlblk09`, completed all four
25-iteration cases. EEEE, H4608/F3072/latent1152/E512/top-k6, sequence1024,
MBS2/GA2, TP2/CP1, GBS28; graph-enabled Flex/DistOpt. Every case uses the same
PXN setting and all-rank Nsight plus CUDA-event metadata. Means below are full-replica
owners 0-5 over matched measured iterations 15-20, after 14 warmup iterations.

| NEP variant | Iteration | First F | Other F | Nonfinal B | Final B | Outside F/B |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 167.012 | 44.449 | 4.535 | 7.808 | 54.675 | 55.546 |
| Buffer cleanup | 162.080 | 46.151 | 4.534 | 7.829 | 55.849 | 47.716 |
| Cleanup + parallel owners | 159.430 | 43.632 | 4.538 | 7.794 | 55.588 | 47.878 |
| Above + backward SM margin32 | 157.220 | 43.594 | 4.542 | 7.828 | 17.327 | 83.930 |

All values ms; this table compares NEP implementations, not healthy/NEP parity. Small
baseline cases intentionally use backward margin0; the hard-workload reference uses16.
Cleanup reduces gradient-scaling scope 11.145 ->6.579 ms, startup 2.411 ->1.160 ms,
and optimizer grad-norm scope 3.006 ->1.433 ms. Some first-forward elapsed time shifts
across the earlier startup boundary; startup-plus-first-forward is 46.860 ->47.311 ms.
Parallel transfers then reduce first forward by2.519 ms and iteration by2.650 ms.
With SM reservation, final-backward time falls38.261 ms but outside time rises36.052 ms:
communication still dominates this compute-light toy, so the NET iteration gain is2.210 ms,
not38 ms. Ordinary forward/backward remain essentially unchanged.

Rank-0 logged means (measured/post-profile) are166.800/163.300,161.817/158.475,
159.183/155.550,157.017/153.825 ms. Instrumentation adds approximately3.2-3.6 ms;
post-profile timings are a distortion check, not replacements for the measured rows.
All100 iterations were finite with no skips. First two loss/gradient-norm log values
match across variants; later graph-enabled values are not bitwise identical, as with
the prior trusted graph experiment. Do not interpret this as full training numerical
equivalence. The explicit parameter-value/reuse and gradient-buffer tests are the
targeted correctness evidence.

## Hard EP16/12 result: keep cleanup, reject parallel transfers

Job `3122295` completed six cases on GB300, eight nodes/segment4. Each replica stayed
within one physical NVL domain; the two replicas occupied different domains. Original
15-MoE-layer hybrid workload, H4608/F3072/latent1152/E512/top-k6, MBS1/GA2/sequence8192,
TP2/CP1, healthy EP16/16 and NEP EP16/12, proportional local batches16/12. Original
transport settings retained. Native DDP requested8; actual expert buckets healthy5
versus NEP8, as in the original hard comparison. This is not an iso-bucket comparison.
All35 iterations per case completed with finite values/no skips. Required all-rank
profiles validated. Means below are owner ranks0-11, measured iterations20-27 after
19 warmup iterations, not post-profile timings.

| Variant | Iteration | First F | Other F | Nonfinal B | Final B | Outside F/B |
|---|---:|---:|---:|---:|---:|---:|
| Healthy, margin16 | 486.492 | 95.540 | 78.773 | 119.181 | 134.391 | 58.607 |
| Original NEP, margin16 | 582.952 | 133.330 | 78.928 | 118.651 | 166.856 | 85.188 |
| Cleanup, margin16 | 569.738 | 133.747 | 79.087 | 118.781 | 166.789 | 71.333 |
| Parallel owners, margin16 | 587.212 | 146.901 | 79.005 | 119.632 | 169.250 | 72.425 |
| Parallel owners, margin32 | 580.678 | 147.219 | 78.999 | 119.732 | 163.265 | 71.464 |
| Healthy, margin32 | 486.386 | 95.219 | 78.819 | 119.410 | 134.357 | 58.581 |

All values ms. Cleanup saves13.214 ms, primarily outside F/B (13.855 ms). Owner parity
improves83.45% ->85.39%. Parallel owners regress17.474 ms versus cleanup, mostly first
forward (+13.154 ms), and are REJECTED despite improving the small toy. The experimental
branch stays isolated; the trusted branch is untouched. Margin32 on the parallel branch
cannot establish acceptance on the cleanup branch.

Why parallel loses: follower rank15 broadcast summed residency rises89.824 ->260.697 ms
and native scan-kernel residency2.099 ->34.876 ms. Full-owner rank0 HybridEP device-sync
kernel residency rises3.255 ->38.623 ms; rank11 similarly3.409 ->38.506 ms. Rank11 native
AG-NVL residency rises5.999 ->19.034 ms. These are summed first-forward kernel residencies,
NOT additive E2E costs; independent broadcasts overlap each other. They show that faster
redistribution scopes do not imply faster native model progress. Resource interference
with native kernels is supported; NVLink bandwidth saturation specifically is NOT proven
without resource counters. Follower host redistribution scopes also rise19.554 ->25.158 ms.

The final-backward norm diagnostic on ranks2/11 finds mean summed preceding-memset-to-
norm gaps20.622 ms for cleanup, versus8.777 ms healthy. A representative cleanup gap is
12.234 ms, with Gather resident throughout and native expert/dense RS also active.
These gap sums are diagnostics, not additive overhead attribution. Parallel+margin32
reduces its corresponding gap sum18.588 ->8.019 ms, consistent with launch admission,
but it retains the rejected forward regression.

Rank0 logged measured/post-profile iteration means: healthy486.163/475.171,
reference582.600/568.571, cleanup569.425/555.143, parallel586.888/570.043,
parallel+margin580.375/561.814, healthy+margin486.050/472.800 ms. Instrumentation still
affects NEP more; these windows are not substituted for each other.

An attempted seventh-case append did not execute: the inherited postprocessor exits1
because it expects random forced balancing, while this workload uses exact-uniform
routing. The outer wrapper intentionally validates the actual uniform workload instead,
but the inner shell exited before the appended command. Moved cleanup_margin into the
outer case list, removed the append. Follow-up job `3122515` started immediately on GB200
at18:56:48 PDT: healthy16, cleanup16, cleanup32, healthy32, all identical resource/settings
within this allocation. No parallel-transfer code is used. Final results follow below.

Artifacts: Lustre `slurm_runs/soft_clique_baseline/3122295/`, including
`cleanup_phase_analysis.json`, `norm_admission_gaps.json`, and
`forward_kernel_residencies.json`. Untimed all-rank memory peaks are in each case's
`memory_peaks/` directory. Sampled cleanup/parallel high-water marks are identical on
ranks0/11/15/16/27 (114.218/112.636/60.932/116.394/114.615 GiB), which does not imply the
parallel implementation allocates no additional transient storage.

## Independent backward mitigation: GB200 follow-up

Job `3122515` completed successfully in19m32s. Physical domains12 and16, same hard
workload and instrumentation as above; all140 iterations finite/no skips, all120 rank
profile sets and memory records validated. Each case again uses measured iterations20-27
after19 warmups. SM16/SM32 refer to `NVTE_BWD_LAYERNORM_SM_MARGIN` only. No production
code changed after cleanup commitee28f09ff; no parallel-owner streams in these runs.

| Variant | Iteration | First F | Other F | Nonfinal B | Final B | Outside F/B |
|---|---:|---:|---:|---:|---:|---:|
| Healthy SM16 | 513.373 | 99.291 | 83.252 | 128.584 | 142.344 | 59.902 |
| Cleanup NEP SM16 | 594.782 | 137.808 | 83.270 | 128.853 | 172.181 | 72.669 |
| Cleanup NEP SM32 | 584.405 | 137.452 | 82.965 | 128.892 | 163.472 | 71.624 |
| Healthy SM32 | 513.329 | 99.843 | 82.622 | 127.826 | 142.762 | 60.276 |

All values ms. The independent SM-margin mitigation saves10.377 ms; final backward
saves8.709 ms, without moving the savings into the outside tail. Healthy iteration is
unchanged within variation. Owner iteration parity86.31% ->87.84%. Iteration standard
deviations1.065/3.509/2.072/2.263 ms in table order. This does not establish total parity:
first-forward delivery and final synchronization still contribute substantial overhead.

The sampled ranks2/11 final-backward preceding-memset-to-RMSNorm gap sum falls
18.932 ->7.568 ms, versus healthy8.346/8.426 ms at SM16/SM32. Cooperative grid shrinks
408 ->360 CTAs. Together with the dependency-free isolated reproduction, this supports
resource/admission stalls rather than an intended gradient-sync-to-backward dependency.
Individual gaps remain (largest cleanup+SM32 sampled gap8.070 ms with Gather, expert RS,
and dense RS active), so margin32 is a mitigation, not a guarantee against every stall.
Graph-node adjacency and these gap sums alone are not a full graph-DAG or additive
critical-path attribution. Do not label them pure NCCL transfer time.

Logged measured/post-profile means in table order:513.075/499.871,
594.475/580.286,584.138/570.000,513.025/499.400 ms. Instrumentation-window differences
remain approximately13-14 ms, with nearly identical distortion across the two NEP cases.
Peak allocated memory is unchanged by SM margin: healthy60.707 GiB; NEP ranges60.932
to116.394 GiB across ranks. All ranks confirm the requested margin in memory records.

Disposition: retain cleanup `ee28f09ff` and use the verified native SM32 setting for
this workload. Reject parallel transfer `f57cc9a65`. No claim of a universal optimal SM
margin, full graph numerical equivalence, or resolution of all remaining NEP overhead.
Trusted checkpoint remains unchanged and both experimental worktrees are clean.

Reproduction: `HARD_CASES='healthy cleanup cleanup_margin healthy_margin' sbatch
--time=00:25:00 scripts/nonuniform/run_nep_hard_cleanup_ab.sh` (one shell line).
Artifacts and analysis: Lustre `slurm_runs/soft_clique_baseline/3122515/`.
