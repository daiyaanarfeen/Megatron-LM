# Controlled Single-Knob NEP Sensitivity

## Final Status: Both Accepted-Runtime Matrices Complete

Updated2026-09-14 after17:57 PDT. No blocked commands or outstanding jobs owned
by this study. A3057161 COMPLETED0:0 in21m50s; B3057162 COMPLETED0:0 in28m06s.
All14 cases trained60 iterations; all210 rank profiles /2100 rank-iterations
passed targeted-event counts/schema/clock and finite-loss/no-skip/no-NaN checks.
Frozen accepted93d313df4 runtime, ignored helper and harness hashes unchanged
after both jobs. No training failures or retries on the accepted runtime.

Owner means average ranks0-5 within each iteration, then average iterations40-49;
SD is across those ten iteration means, not across ranks. No post-profile mixing.

| Matrix / case | Healthy mean +/- SD ms | NEP mean +/- SD ms | Owner parity | Pre / post logged parity |
| --- | --- | --- | --- | --- |
| A anchor start | 709.93 +/-8.13 | 747.84 +/-13.45 | 94.93% | 94.17 /95.69% |
| A MBS1 | 452.39 +/-0.78 | 460.95 +/-5.08 | 98.14% | 94.17 /92.12% |
| A anchor end | 714.31 +/-9.08 | 736.37 +/-2.47 | 97.00% | 95.40 /96.45% |
| B anchor start | 704.86 +/-3.38 | 736.07 +/-7.32 | 95.76% | 95.55 /96.00% |
| B sequence4096 | 434.31 +/-4.13 | 471.20 +/-6.13 | 92.17% | 92.58 /93.01% |
| B topk2 | 621.93 +/-2.48 | 639.45 +/-4.65 | 97.26% | 96.98 /97.41% |
| B anchor end | 707.24 +/-11.46 | 730.65 +/-3.98 | 96.80% | 95.99 /96.12% |

Sequence4096 is below both B anchors across recorded and surrounding windows.
Top-k2 does not reduce parity; its small improvement over the final anchor is
not a strong mechanism claim given anchor drift. MBS1 recorded parity cannot be
interpreted as normal-runtime improvement: healthy changed regime before the
profile window and NEP changed regime after it (see detailed chronology below).
These are unresolved instrumentation/runtime nonstationarity, not evidence of
failed training, and not cleanly separable by subtracting pre/post means.

### Ordinary Microbatch Findings

Elapsed span from first to last grouped-expert GPU operation per microbatch.
Includes intervening layers/dependencies, not just expert GEMM kernels and not
necessarily the entire model-forward/backward boundary. LaterF averages MB1/2;
earlierB averages MB0/1 (zero-based). H and N-full average ranks0-5; N-reduced
averages ranks8-13. SD across10 per-iteration phase means.

| Case | LaterF H / N-full / N-reduced ms | EarlierB H / N-full / N-reduced ms |
| --- | --- | --- |
| A anchor start | 84.43 /88.91 /81.18 | 97.14 /97.18 /96.18 |
| A MBS1 | 64.37 /60.32 /57.33 | 52.34 /52.84 /52.57 |
| A anchor end | 85.83 /86.20 /80.58 | 97.18 /97.22 /96.18 |
| B anchor start | 82.14 /86.47 /82.77 | 97.14 /97.21 /95.73 |
| B sequence4096 | 58.95 /56.12 /62.90 | 51.76 /51.84 /51.01 |
| B topk2 | 72.45 /73.63 /73.68 | 78.95 /79.11 /79.51 |
| B anchor end | 81.13 /84.43 /85.50 | 97.19 /97.21 /95.80 |

1. The large earlier-backward reduced-replica lag from older full-Kineto traces
   is NOT reproduced here. Except confounded MBS1, full-owner earlierB differs
   from H by only0.04-0.16ms. Reduced earlierB is mostly shorter; topk2 is0.56ms
   longer than H, not a large bottleneck. All sampled earlierB grouped-expert
   CPU calls complete before their GPU range starts (calibration margin applied).
   CPU-to-GPU start lag on reduced ranks remains~21ms for seq4096,22ms forMBS1,
   54ms fortopk2, and62-63ms foranchors. Thus smaller work reduces submission
   headroom, but does not exhaust it in these backward phases.
2. Seq4096 does show slower reduced ordinary forward:62.90 +/-1.22ms versus
   full56.12 +/-0.47 and H58.95 +/-0.54. Reduced-minus-full6.78ms exceeds both
   B anchor differences (-3.70,+1.07ms). However reduced-minus-healthy3.95ms is
   not larger than final-anchor4.37ms; do not attribute all iteration degradation
   to a newly created ordinary-forward penalty from this comparison alone.
3. In seq4096 laterF, summed grouped-expert CPU wall time is H14.49 /full13.58 /
   reduced17.39ms; corresponding GPU elapsed-range sums14.99 /14.10 /17.93ms.
   Those forward calls have essentially no CPU submission completed ahead of
   their GPU-range start. This is consistent with host-paced grouped-forward
   execution, but event ranges alone cannot prove exact GPU idle or isolate
   individual GEMM execution, native communication, or peer polling.
4. Fewer tokens per expert is not a sufficient predictor of low parity:
   seq4096 reduces work in shared/Mamba/attention as well as experts; topk2
   reduces expert work and token dispatch traffic while retaining the other
   computation. The latter retains~54ms backward submission lead and high
   parity. A pure GEMM-efficiency or late-peer HybridEP causal claim is not
   supported by these targeted event ranges.

All cases retain90 grouped-forward and90 grouped-backward calls/iteration on
each rank. There are30 grouped calls/direction/microbatch. Physical experts per
GPU:16 full,22 on reduced ranks8/9 and21 on reduced10-13. No per-kernel launch
counts or pure GEMM FLOP/s were measured; do not equate the grouped-call count
with individual kernel launches. No runtime, GA, H/F or expert-count changes.

### Controls and Artifacts

Actual rank0 bucket initialization signatures are byte-identical across knobs
within each H/N type:3 dense buckets,4 physical expert buckets; NEP additionally
uses8 logical-owner expert sync groups. This is the inherited baseline layout,
NOT a claim that healthy and NEP have equal EDP bucket counts. Parameter sizes,
bucket grouping, graph scopes and all unrelated model arguments remained fixed.

Every case passed fabric health/full-bandwidth status and software-clique rank
placement. Each tag has28 rank logs showing two-rank cross-replica communicators
with cliqueSize1, and1344 RDMA channel records. Thus cross-replica EDP uses RDMA,
not the colocated domain's NVLink. NVLink-domain containment as recorded above.

Full results and all-rank phase samples, one directory per job:
/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/JOB/sensitivity_analysis/
summary.json, TAG_{healthy,nep}_ranks.json and final_validation.json.
Full source/hash manifest:
/home/darfeen/Megatron-LM-nep-sensitivity-accepted-20260914/sensitivity_manifest.json.

Reproduce offline analysis (no GPU jobs):
`python3 scripts/nonuniform/analyze_nep_single_knob.py 3057161 --tags anchor_start mbs1 anchor_end`
and `python3 scripts/nonuniform/analyze_nep_single_knob.py 3057162`.
Parent owns separate firstF/finalB analysis and figures; no plots or Markdown
uploaded by this agent. Unfinished validation: MBS1 nonstationarity attribution,
full-kernel/peer causal attribution, and statistical replication beyond one
allocation per matrix. No additional jobs launched to address these limits.

## 2026-09-14: Resumed On Accepted CPU-Optimized Runtime

Current study supersedes historical entries below. GA3 throughout: GA6 removed.
Submitted2026-09-14 17:28 PDT: matrix A job3057161 (30min), matrix B job3057162
(35min), four nodes each, GB200/GB300 backfill. Initially pending Priority with
no start estimate. All accepted snapshot/runtime/helper/harness hashes verified
before submission. Eight mocked H/N variant launches and shell syntax passed.

Both allocated17:29:07 PDT on GB200. A: lyris[0172-0175], nvlblk10;
B: lyris[0002-0005], nvlblk01. Each replica fits its domain; distinct software
cliques1000/1001 retained. First anchor pairs completed~17:38 PDT.

Initial accepted-runtime anchor results, measured40-49, six full-owner ranks:

| Matrix | Healthy mean +/- SD ms | NEP mean +/- SD ms | Owner parity |
| --- | --- | --- | --- |
| A3057161 | 709.93 +/-8.13 | 747.84 +/-13.45 | 94.93% |
| B3057162 | 704.86 +/-3.38 | 736.07 +/-7.32 | 95.76% |

Logged pre/recorded/post parity A94.17/94.93/95.69%, B95.55/95.75/96.00%.
There is within-process drift, particularly A NEP (754.22 ->747.63 ->735.81ms),
so do not interpret this as a stationary no-overhead profiler calibration.
Both anchor H/N pairs passed full-rank event/schema/count and finite-loss,
no-skipped/no-NaN validation. Repeated anchors pending. No knob conclusion yet.
Interim machine-readable summaries: JOB/sensitivity_analysis/summary.json;
full per-rank rows in anchor_start_{healthy,nep}_ranks.json.

17:44 PDT: MBS1 and sequence4096 pairs completed, unchanged actual bucket
signatures and full-rank trace/count/loss checks passed.

| Case | Healthy mean +/- SD ms | NEP mean +/- SD ms | Owner parity | Pre / post logged parity |
| --- | --- | --- | --- | --- |
| A MBS1 | 452.39 +/-0.78 | 460.95 +/-5.08 | 98.14% | 94.17 /92.12% |
| B seq4096 | 434.31 +/-4.13 | 471.20 +/-6.13 | 92.17% | 92.58 /93.01% |

Important: MBS1's measured/post gap is large and the recorded parity cannot be
treated as representative unrecorded performance. Report instrumentation/drift
confounding rather than a clean parity improvement. Sequence4096's lower parity
is consistent across all windows. No new runs added to resolve the MBS1 issue;
complete the authorized matrices and report the limitation. Parent notified.

17:55 PDT update: A3057161 COMPLETED0:0 in21m50s; all six cases ran60 iterations.
Its final anchor H714.31 +/-9.08ms / N736.37 +/-2.47ms, owner97.00%.
Pre/post95.40/96.45%. Initial-to-final recorded anchor shifts2.07 percentage
points; retain both anchors instead of selecting the favorable one.
B topk2 completed H621.93 +/-2.48ms / N639.45 +/-4.65ms, owner97.26%,
pre/post96.98/97.41%. Lower top-k did NOT degrade parity in this allocation.
B final reversed-order anchor still running. Actual bucket signatures match
within each matrix/case type. No training failures in accepted-runtime jobs.

MBS1 discrepancy is not cleanly attributable to profiler overhead: healthy
jumps418-422 ->450ms already at37, before recording initialization39; NEP
jumps448 ->500ms at53 after recording. These abrupt nonstationary changes
prevent estimating profiler overhead by simply subtracting surrounding means.
Goal: characterize remaining overhead, not manufacture a parity regression.
This agent owns ordinary later-forward/earlier-backward ablation and launches.
Parent owns first-forward/final-backward analysis using the same profiles, plus
separate expert/shared-F experiments. No duplicate MBS/sequence/top-k jobs.

Runtime source: /home/darfeen/Megatron-LM/.worktrees/nep-sync-cpu-20260914,
clean commit93d313df4d5588795ea628592668b188ed352a66. Accepted parameter metadata
cache, singleton direct copies and parameter foreach; NO gradient-stage changes.
nonuniform_ep.py SHA653a7efd3da53c0ace84c9e32d65eb2faa1f9b17f44b1daf43ace136a3eae0c7.
Frozen snapshot: /home/darfeen/Megatron-LM-nep-sensitivity-accepted-20260914.
No runtime edits; ignored prebuilt dataset helper included in snapshot manifest.

Matrix A: anchor_start H/N, mbs1 N/H, anchor_end H/N (six cases).
Matrix B: anchor_start H/N, seq4096 N/H, topk2 H/N, anchor_end N/H (eight cases).
Each matrix uses one four-node segment4 allocation, fixed nodes/hardware within
matrix. GB200/GB300 backfill, 30-minute A and35-minute B, per-case6min watchdog.
All60 iterations retained; CPU/CUDA-event profiles40-49 on all ranks;
29-38 and51-60 are drift/profiling diagnostics only. No profiler-free cases.

All cases: H1280, expertF768, sharedF1536,128 real experts,15MoE layers;
TP/CP/ETP/PP1, Flex/HybridEP and DistOpt, unchanged graph scopes and buckets.
Anchor MBS2, sequence8192, topk6. H EP8/8 GBS96, N EP8/6 GBS84;
replica batches48/48 healthy and48/36 NEP. MBS1 changes GBS to48/42,
replica batches24/24 and24/18; GA3 unchanged. Seq4096 changes tokens, not sample
batch/GBS; topk2 changes routed token-expert assignments. No compounded knobs.

Artifacts: soft_clique_baseline/JOB/TAG/cupti/ep8/
scaling_ladder__a770m_4b_moe_215b/{healthy,nep}/JOB/
with driver.log, launch_manifest/rank-N.json and cupti_profile/rank-N.json.
TAGs as above. Schema unchanged: backend=cuda_events, origin_ns,
clock_error_bound_ns, traceEvents containing cpu_operation_range,
gpu_operation_range and iteration. Operation labels expert.forward,
expert.backward, nep.stage_grads, nep.redistribute_params, optimizer.step,
optimizer.copy_params. These are elapsed ranges, NOT kernel busy time.
Parent can consume artifacts as each case completes. No Markdown uploads.

Old3054530 exited1 from intentional pause guard after anchor pair and seq4096
NEP completed. Old-runtime anchor95.79% is historical, not this study's comparator.

## 2026-09-14: Paused By Owner

Owner prioritizes implementation fixes. No further sensitivity cases or jobs
will be launched. Job3054530 completed anchor healthy and NEP; sequence4096 NEP
was already running when the pause arrived (~12:18 PDT). Let that case finish.
The isolated harness now blocks subsequent case-fields/options requests and
new wrapper invocations, so sequence4096 healthy, topk2 and anchor_end will not
run. Only isolated benchmark harness scripts changed for this stop guard;
trusted runtime, CPU-optimization worktree, and snapshot runtime remain untouched.
Original harness hashes describe the pre-pause run, not these stop guards.
Artifacts remain under soft_clique_baseline/3054530 on Lustre. The sequence
case alone will not constitute a valid H/N comparison. No queued study jobs.

## 2026-09-14: Experiment Prepared

Status: replacement job3054530 submitted. First job3054479 allocated at12:01:52
PDT on GB200-backfill lyris[0113,0118,0121,0123], segment4, within nvlblk07,
but failed before training after4m13s: the snapshot lacked the ignored prebuilt
dataset helper. No benchmark result. Monitoring continues for the replacement.

Owner: this agent, end-to-end; no monitoring agent or tight scheduler polling.
Poll pending/running jobs every 300 seconds until terminal. Kill demonstrated
hangs after diagnostics; each distributed case has a six-minute timeout.

Anchor: a770m_4b_moe_215b, healthy EP8/8 versus NEP EP8/6; H1280, expert F768,
shared F1536, 128 true experts, top-k6, sequence8192, 15 MoE layers. All other
parallel axes1. MBS2, GA3 on both replicas; healthy GBS96, NEP GBS84;
full/reduced batches48/36, proportional. DistOpt and Flex/HybridEP retained.

Matrix, sequential in one four-node allocation:

1. Anchor, healthy then NEP.
2. Sequence8192 -> 4096 only, NEP then healthy. Sample batch sizes unchanged;
   token counts halve. Parameter payload and bucket configuration unchanged.
   max-position-embeddings remains8192.
3. Top-k6 -> 2 only, healthy then NEP. Tokens, model weights, batch and bucket
   settings unchanged; routed token-expert assignments fall by threefold.
4. Anchor repeat, NEP then healthy, to assess order/allocation drift.

Expert-F perturbation deferred: it also changes gradient payload and can
recompute native bucket boundaries. The first two independent knobs are a
cleaner bounded initial test. No compounded perturbations or runtime changes.

Profiling: same accepted targeted CPU/CUDA-event instrumentation as3007029,
every rank, every case, iterations40-49; 60 total iterations. Surrounding
29-38 and51-60 diagnose recording distortion/drift. No independent profiler-off
case and no mixing post-profile parity with mechanism timings. Operation ranges
are elapsed envelopes, not pure GEMM/NCCL residency or exact GPU idle.

Topology: four-node segment, GB200 or GB300 selected by first available backfill
allocation; fixed hardware/nodes within the matrix. Both replicas must fit an
NVL domain. Software MNNVL cliques1000/1001 force cross-replica RDMA even if
physically colocated. Placement and NCCL transport logs recorded per case.

Runtime source: /home/darfeen/Megatron-LM-nep-main-20260817, including existing
trusted dirty runtime, nonuniform_ep.py SHA256
4676f164a95255bdcc7e2531a06ca92c41a2583bf6f9a7a85b3b74b64f26f2d1.
Snapshot: /home/darfeen/Megatron-LM-nep-sensitivity-20260914.
Harness: snapshot/sensitivity_harness. Full source hashes in
sensitivity_manifest.json. No CPU-cleanup implementation mixed in.

Scripts: scripts/nonuniform/prepare_nep_single_knob.py,
run_nep_single_knob_rank.py, run_nep_single_knob_20260914.sh.
Artifacts: /lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/JOB/{anchor_start,seq4096,topk2,anchor_end}.

Success criterion: reproduced high-parity anchor, then reproducible degradation
of approximately five percentage points or more, preferably below90%. Null
results, profiling distortion, rebucketing, failed/NaN/skipped iterations and
other validity failures must be reported rather than counted as slowdowns.

Preflight: all tracked runtime and frozen-harness hashes verified; all shell
syntax checks passed; mocked rank launches verify exactly zero/one changed
argument for anchor/perturbations and preserve the hybrid pattern unquoted.
Trusted runtime and shared launchers remain unchanged. Snapshot commit
99ae923603b295c72372af85bc1bffe5c39bf8b2 plus trusted dirty files, preserved byte
for byte. The parameter-CPU-cleanup agent branch is not used.

Packaging correction before3054530: copied the exact trusted
megatron/core/datasets/helpers_cpp.cpython-312-aarch64-linux-gnu.so into the
snapshot. SHA256 b36e44b4754ae1e6e6ef1ee9721aa32ced4b015454cc87c14edf3b9e7a705314.
This supplement is not in the original tracked-source manifest (the extension
is ignored by git); future snapshot preparation includes it explicitly. No
source, runtime logic, batch, profiler or matrix setting changed for the retry.
