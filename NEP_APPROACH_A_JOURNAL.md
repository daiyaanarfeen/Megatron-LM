# NEP Approach A Journal

Append a dated entry whenever we do something new: code changes, job submissions, benchmark results, trace analysis, or decisions that change the next step. Keep entries factual and include job IDs, run dirs, and commits when available.

## 2026-07-06 - NCCL CTA and copy-engine overlap investigation

### EDP owner-readiness gate

- Rechecked the EP8/4 A/B trace from job `2293060` after confirming the batch split. The topology uses TP2/ETP1, so full ranks 0-7 form EP8 and reduced ranks 8-11 form EP4. Four versus two attention-DP lanes already assign 32 versus 16 samples per iteration.
- The paired EDP traces identify the actual skew: full owner rank 0 launched its first `ep_dp` all-reduce 261.6 ms before reduced owner rank 8. Rank 0's matching 32-CTA kernel stayed resident for 259.7 ms while rank 8's took 1.35 ms. Across both profiled iterations, rank-0 EDP kernels accumulated 1654 ms versus 146 ms on rank 8.
- Reduced EP4 ranks own 32 experts each versus 16 on EP8. They receive comparable total routed tokens but execute twice as many smaller expert problems and expose more autograd/launch overhead. Before the first EDP launch, rank 8 had 3652 AccumulateGrad events versus 2148 on rank 0, 224 ms versus 121 ms of grouped-linear backward CPU time, and 466 ms versus 89 ms of GPU idle time.
- First implementation used a separate zero-CTA `nep_edp_ready` communicator with a one-element symmetric-memory AllGather after local Gather/accumulation. Smoke `2294499` passed Black and all seven focused tests, created the process groups and model successfully, then made no progress in its first step for over five minutes versus 7.5 seconds in the prior smoke. It was canceled together with dependent a3b job `2294562`.
- The initial hypothesis was that the zero-CTA readiness operation did not release its stream dependency across the full/reduced placement boundary. A later gate-off control showed that this was not sufficient evidence: the TP1 topology-8/4 toy itself stalls at delayed dense synchronization, independent of the gate.
- Added focused tests for nonblocking readiness-slot reuse and Gather/readiness/all-reduce/Scatter operation order. Login-node syntax and diff checks passed; Black and pytest are unavailable there and run in the smoke preflight.
- Replaced the copy-engine gate with a one-CTA NCCL readiness communicator. Each owner enqueues a one-element all-reduce after local Gather/accumulation; the normal 32-CTA EDP all-reduce inherits that stream dependency. An early owner now occupies at most one CTA while waiting instead of 32, with no symmetric-memory or topology requirement.
- Readiness gating is independent of zero-SM owner Gather/Scatter and defaults on for nonuniform NCCL Approach-A topologies; `MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=0` remains an A/B escape hatch. P2P NEP does not create the extra communicator.
- Submitted fast profiled EP8/4 one-CTA smoke `2294682` with eight experts and rank 0/8 debug output on `lyris[0135-0137]`.
- Queued exact a3b gate-on job `2294704` and gate-off control `2294705` on the same warmed nodes, each TP2/EP8 topology 4/2, MBS4/GBS48, zero-SM owner transfers, and rank 0/8 profiles.
- Job `2294682` and a gate-off repeat `2294740` both used TP1 topology 8/4. Both reached delayed dense synchronization without completing the first step, so they were canceled; the dependent exact jobs were also canceled. These runs do not distinguish readiness-gate behavior because their DP/process-group mapping differs from the real a3b TP2/ETP1 layout.
- Corrected the smoke to topology 4/2 with TP2/ETP1, which produces the intended EP8/EP4 layout on 12 ranks and DP6. Gate-off job `2294774` passed Black and all seven focused tests, then failed in the first forward pass because the generic smoke launcher omitted Megatron's required `--sequence-parallel` flag for TP greater than one. Dependent gate-on job `2294776` was canceled automatically.
- Updated the generic smoke launcher to add `--sequence-parallel` whenever TP is greater than one. Login-node shell syntax, Python compile, and diff checks pass; the next container preflight will rerun Black and the focused test suite.
- Submitted corrected profiled smoke pair `2294883` (gate off) and dependent `2294886` (gate on), both topology 4/2 with TP2/ETP1, eight experts, MBS1/GBS6, zero-SM owner transfers, and profiles on owner ranks 0 and 8.
- Scoped readiness-group creation explicitly to NCCL Approach A in both `examples/nonuniform/pretrain_gpt_nonuniform.py` and `pretrain_hybrid.py`; P2P NEP retains its existing process-group set.
- Control `2294883` reached the first owner EDP all-reduce on both ranks 0 and 8 but stopped advancing with all GPUs at 0% utilization. After cancellation, NHC put `lyris0082` and `lyris0086` into reboot for a PCLK-speed fault, so this allocation is invalid as a code diagnostic; dependent `2294886` was canceled.
- Submitted a two-microbatch repeat with follower rank 4 also profiled/debugged. Initially pinned jobs `2294948`/`2294950` were canceled when the warmed nodes remained in reboot/completing state.
- Resubmitted unpinned jobs `2294955` (gate off) and dependent `2294958` (gate on): topology 4/2, TP2/ETP1, MBS1/GBS12, zero-SM owner transfers, and profiles on ranks 0, 4, and 8.
- Control `2294955` allocated healthy `lyris[0280-0281,0286]` in NVL block 16. Updated dependent `2294958` to require the same nodes so it reuses the enroot image cache and provides a same-node A/B comparison.
- Control `2294955` reproduced the first-batch stall before NHC intervened. Ranks 0 and 8 both entered owner-0 EDP all-reduce, while follower rank 4 had already enqueued its owner-0 Scatter and advanced through owner 1-3 Gather calls; all 12 GPUs then remained at 0% utilization. The control and dependent were canceled.
- Tested the hypothesis that the first-batch `async_op=False` special case caused the stall. The experiment used the ordered async Gather/EDP/Scatter pipeline while retaining the synchronous drain in `finish_grad_sync`; jobs `2295049`/`2295052` were submitted on healthy `lyris[0110,0112,0117]`.
- Job `2295049` showed `async_op=True` on ranks 0, 4, and 8, but ranks 0 and 8 still blocked inside the first owner-0 `dist.all_reduce` call before it returned a Work handle. Deferred Scatter therefore cannot be the cause of this toy stall. The job and dependent were canceled after more than three minutes at 0% GPU utilization.
- Reverted the first-batch async experiment and its regression test. The readiness gate remains limited to normal async backward-hook launches. Next validation uses the regular all-to-all reshard path to avoid the toy zero-SM first-call interaction, followed by the known-good real a3b zero-SM workload.
- Three-node regular-all-to-all pairs `2295188`/`2295190` (priority) and `2295200`/`2295202` (backfill) were initially submitted, but both controls received start estimates after `21:00 PDT`.
- Submitted smaller six-rank jobs `2295206` (gate off) and dependent `2295207` (gate on): two nodes, TP1, EP4/EP2 topology 4/2, MBS1/GBS12, regular all-to-all reshard, and profiles on paired EDP owner ranks 0 and 4. Control estimated start is `19:24 PDT`.
- Canceled the four superseded three-node fallback jobs. The small pair validates ordering and readiness-gate mechanics only; the subsequent real a3b run remains TP2 EP8/EP4 with zero-SM reshard.
- The priority estimate later slipped to `20:52 PDT`. Backfill duplicates `2295221`/`2295226` received the same estimate and were canceled, leaving only `2295206`/`2295207` to monitor.
- Added `scripts/nonuniform/run_lyris_a3b_edp_ready_ab.sh`, a thin same-allocation wrapper that runs exact a3b gate-off then gate-on cases on the same three nodes with TP2 EP8/EP4, MBS4/GBS48, zero-SM reshard, 12 iterations, and rank 0/4/8 profiles.
- Submitted the exact A/B wrapper as job `2295237`. It is pending for resources with no estimated start; the smaller smoke remains queued independently.
- Both allocated at about `19:27 PDT`: smoke control `2295206` on healthy `lyris[0204-0205]` and exact A/B `2295237` on healthy `lyris[0253,0255-0256]`. Pinned dependent smoke `2295207` to the control nodes for image-cache reuse and same-node traces.
- Small all-to-all runs `2295206`/`2295207` both completed 10/10 finite iterations with zero skips/nans. Gate-on emitted 64 `nep_edp_ready` all-reduces per profiled rank, all with a one-CTA grid.
- Gate-off EDP launch skew between rank 0 and reduced owner rank 4 had `4.42 ms` median and `90.64 ms` maximum absolute skew. Rank-0 32-CTA EDP kernels accumulated `461.36 ms` with `90.78 ms` maximum residency, versus `9.38 ms` total on rank 4.
- Gate-on reduced paired EDP launch skew to `0.263 ms` median and `0.484 ms` maximum. Rank-0 EDP residency fell to `26.34 ms` total and `0.63 ms` maximum; rank 4 remained `9.36 ms` total.
- The wait moved to the bounded readiness communicator: rank 0 accumulated `472.45 ms` of one-CTA readiness kernels while rank 4 accumulated `0.55 ms`. Clean iterations 7-10 averaged `407.2 ms` gate-off versus `441.0 ms` gate-on in this small host/communication-dominated smoke, so the real a3b same-allocation result is required to judge net performance.
- Exact same-allocation a3b job `2295237` completed both cases with identical finite losses and zero skips/nans. Clean iterations 8-10 and 12 averaged `1470.1 ms` / `496.6 TFLOP/s/GPU` gate-off and `1470.2 ms` / `496.6 TFLOP/s/GPU` gate-on: effectively exact throughput parity.
- Exact gate-off rank-0 EDP kernels accumulated `1195.19 ms` with `108.59 ms` maximum residency, versus `171.98 ms` total on reduced owner rank 8. Paired launch skew had `6.72 ms` median and `106.96 ms` maximum absolute skew.
- Exact gate-on emitted 92 one-CTA readiness all-reduces and 92 EDP all-reduces on both owner ranks. Every readiness kernel completed before its paired EDP kernel, and the readiness communicator participants were exactly ranks `[0, 8]`.
- With the gate, paired EDP launch skew fell to `0.009 ms` median and `6.65 ms` maximum. Rank-0 32-CTA EDP residency fell to `153.31 ms` total and `2.86 ms` maximum; rank 8 was `184.63 ms` total and `8.16 ms` maximum.
- Rank 0 spent `1190.50 ms` in the one-CTA readiness kernels, 53.0% concurrent with non-NCCL GPU kernels. The gate therefore moves the unavoidable owner wait from a 32-CTA EDP kernel to one CTA and preserves exact workload throughput.
- Removed temporary per-task debug formatting and the explicit profiler record wrapper from the final hot path; `record_param_comms` retains the `nep_edp_ready` process-group trace metadata. Moved the new optional initializer flag to the end of the signature to preserve positional-call compatibility.
- Final validation job `2295339` stopped before tests because the freshly imported NeMo 26.06 image carried Black `26.3.1`, while this repository requires Black major version `24`; this was a tooling-version failure, not a source/test failure.
- Resubmitted Black, compile, and focused pytest validation as job `2295366` using NeMo 25.09, whose toolchain predates that mismatch.
- Job `2295366` reached Black 24 and reported formatting changes were required, so job `2295398` reran Black in write mode, verified Black check and compile, and passed all seven focused tests.
- Reverted unrelated whole-file Black churn in the two entrypoints and reapplied only their readiness-gate keywords. The core implementation and focused test retain Black 24 formatting; login-node compile and `git diff --check` pass for the scoped final tree.
- Committed the validated implementation as `7f2189347` (`Gate NEP EDP reduction on owner readiness`) and pushed `nonuniform-approach-a-training-scripts` to the `daiyaanarfeen/Megatron-LM` fork.

### Backward-order scheduler experiment

- Re-examined the remaining large/late host batches in the fused zero-SM path. `build_nonuniform_ep_nccl_bucket_groups` sorted normalized parameter names in reverse lexicographic order, while the shared scheduler only launches a ready prefix. For multi-digit layers this puts layer 9 ahead of layers 45-10 and creates head-of-line blocking until late backward hooks.
- The prior EP4/2 16-layer zero-SM trace from job `2291922` confirms the effect: one AccumulateGrad hook in each profiled iteration submitted 13 native gathers and occupied about 13.3-13.5 ms, while the other gather-containing hooks normally submitted one gather.
- Changed NCCL expert-slot grouping to preserve first occurrence in the Megatron grad buffer, whose default layout is already constructed in backprop order. The global task sequence and per-communicator collective order remain deterministic; only the incorrect name-based reordering is removed.
- Added a focused regression test with backward-order layer keys `45, 44, 10, 9`. A local `py_compile` check passed; repository `uv` is unavailable on the login node, so required isort and pytest run inside the submitted container smoke.
- Submitted identical profiled two-node EP4/2, 16-layer, zero-SM smoke jobs using public `nvcr.io/nvidia/nemo:26.06` and the enroot/pyxis image cache:
  - Priority partition job `2292439`, estimated start `2026-07-06T16:01:00-07:00` at submission.
  - Backfill job `2292443`; use whichever starts first and cancel the duplicate before allocation.
- Queued exact profiled a3b NEP benchmark `2292458` behind `afterok:2292439`: TP2/EP8, topology 4/2, MBS4/GBS48, 16 scheduler slots, zero-SM Gather/Scatter, low-priority communication groups, and the same NeMo 26.06 image used by job `2291941`.
- A static replay of the 52-layer a3b hybrid pattern reproduced the trace exactly: reverse lexicographic ordering releases ready-prefix batches of 21 and 19 expert-slot groups at layers 6 and 3, matching the pre-fix 21/19 gather bursts.
- EP4/2 smoke `2292439` completed 10/10 finite iterations, zero skips/nans, and all five focused tests passed. Duplicate `2292443` was canceled after both allocated.
- Smoke trace comparison kept 128 native calls but reduced maximum calls per AccumulateGrad hook from 26 to 2 and maximum hook duration from 13.50 to 1.52 ms. The tiny smoke still had 0% CE/EDP overlap because individual transfers complete between compute kernels.
- Full a3b job `2292458` completed 12/12 finite iterations with zero skips/nans. Clean iterations 8-10 and 12 averaged `1653.3 ms` and `441.6 TFLOP/s/GPU`, slower than pre-fix job `2291941` (`1517.1 ms`, `481.2 TFLOP/s/GPU`).
- Its rank-0 trace nevertheless shows the scheduler fix working: maximum native calls per hook fell from 41 to 2, longest hook from 38.28 to 1.98 ms, and total reshard-hook CPU from 162.34 to 157.92 ms. Rank-0 profiled steps improved by 30-34 ms; CE union fell `128.54 -> 104.10 ms`, EDP union fell `192.36 -> 179.62 ms`, and EDP/non-NCCL overlap rose `44.1% -> 58.1%`.
- The clean-step regression appears after the profiler window and coincides with much lower global average GPU power, while the sampled rank improves. This is consistent with an unsampled node/rank straggler, so the single node-set throughput result is not sufficient to accept or reject the scheduler change.
- Same-allocation A/B job `2293060` completed successfully on `lyris0167-0169`, with identical finite losses and zero skips/nans. Legacy clean iterations 8-10 and 12 averaged `1721.5 ms` / `424.3 TFLOP/s/GPU`; backward order averaged `1633.3 ms` / `447.25 TFLOP/s/GPU`, a `5.4%` throughput gain on the same nodes.
- Multi-rank traces confirm the mechanism. Rank 0 maximum calls per hook fell `41 -> 2`, longest hook `39.33 -> 2.09 ms`, and profiled steps improved `2284.4 -> 2098.4 ms`. Rank 0 CE overlap rose `12.7% -> 29.2%` and EDP overlap `14.6% -> 44.0%`; rank 4 CE overlap rose `10.4% -> 24.8%`. Native operation counts were unchanged on every sampled rank.
- Removed the temporary legacy-order environment switch and A/B wrapper after collecting the comparison; only backward grad-buffer ordering and its regression test remain.
- Required formatting/test job `2293064` completed: `uv run isort` fixed the test import block, targeted Black check passed, and all five focused tests passed.

- Focused the next optimization cycle on physical reshard overlap. The final phase-pipeline trace from job `2263612` still used 32-CTA `ncclDevKernel_SendRecv` kernels for all 184 `nep_owner_transfer` operations and 32-CTA all-reduce kernels for all 92 owner `ep_dp` operations.
- Checked NVIDIA's NCCL zero-CTA requirements. Copy-engine zero-CTA requires NCCL 2.28 or newer, a zero-CTA communicator, symmetrically registered NCCL-window buffers, and a native supported collective. NCCL 2.29 supports AlltoAll, AllGather, Gather, and Scatter within one NVL/MNNVL domain; variable-split SendRecv is not supported.
- Probed public 26.04 containers on Lyris GB200:
  - PyTorch job `2291298`: NCCL `2.29.7`; `ProcessGroupNCCL.Options.config.cta_policy` is exposed.
  - NeMo job `2291299`: NCCL `2.29.2`; CTA policy is exposed.
  - Corrected NeMo dependency job `2291325`: Mamba `2.3.1`, causal-conv `1.6.1`, and Transformer Engine `2.14.1` import successfully; standalone `grouped_gemm` is missing. The image reports `NCCL_CTA_POLICY_ZERO=2`.
- Added `scripts/nonuniform/probe_nccl_zero_cta.py` to profile registered-buffer default, efficiency-policy, one-CTA, and zero-CTA collective variants on four GPUs.
- First collective probe job `2291355` established two constraints before failing on the intentionally tested zero-policy AlltoAll:
  - PyTorch's equal-split `all_to_all_single` maps to `ncclDevKernel_SendRecv`, not native `ncclAlltoAll`.
  - `NCCL_CTA_POLICY_EFFICIENCY` did not reduce that SendRecv launch; both default and efficiency traces used a 32-CTA grid. Efficiency was slower in this short sample.
  - Forcing zero policy on that unsupported SendRecv path raised an NCCL unhandled-CUDA error. This confirms that the current variable-split NEP path cannot become copy-engine-only by changing the communicator policy alone.
- Revised the probe to isolate risky cases, added a `max_ctas=min_ctas=1` communicator control, and moved known-supported zero-CTA AllGather ahead of unsupported SendRecv tests.
- One-node rerun `2291380` completed the default, efficiency, and one-CTA controls before failing in the 26.04/NCCL 2.29.3 copy-engine AllGather path:
  - Default and efficiency variable SendRecv used 11 CTAs in this one-peer pattern and took about `0.104 ms` and `0.054 ms` of sampled GPU time respectively.
  - The one-CTA variable path used one CTA but took about `1.662 ms`, roughly 16x the default sampled duration. This agrees with the earlier CTA-4/8 model runs: reducing CTA count stretches communication enough to hurt throughput.
  - The zero-policy AllGather reached NCCL's `Init CE` path, then failed in `ce_coll.cc:411` with `Cuda failure 'invalid argument'`. The node had driver `13.1`, NCCL detected the four GPUs in one node, and symmetric-window setup completed, so the next check targets a newer NCCL CE implementation rather than another CTA limit.
- Submitted isolated zero-AllGather job `2291407` with public `nvcr.io/nvidia/nemo:26.06`, which carries NCCL 2.30.4 and the 2.30 CE-collective fixes. The job was pending on priority at submission.
- Job `2291407` completed successfully on one GB200 node. The container's PyTorch build reports NCCL `2.29.7`; registered-buffer zero-CTA AllGather passed on four ranks, and NCCL logged `Init CE` without the `ce_coll.cc` failure seen in the 26.04 image.
- The rank-0 trace for `2291407` contains `Memcpy DtoD` and `Memcpy PtoP` GPU activity and no `ncclDevKernel`, confirming that the collective ran on copy engines rather than consuming SMs.
- Submitted two-rank job `2291447` on the same public NeMo 26.06 image to test zero-CTA equal-split AlltoAll. Current PyTorch can call native `ncclAllToAll` without a custom C++ binding, but code inspection corrected an important topology assumption: round-robin follower placement makes a3b EP8/4 owner groups size 5 and a8b EP64/32 groups size 9, so fixed AlltoAll would multiply traffic by the group size and is only a control, not the preferred integration.
- The better-fit operation is native zero-CTA Gather/Scatter. Each follower in an owner group contributes the same dense payload while the owner can retain its larger local contribution directly; only the owner's dummy follower-sized slice is padding. Submitted job `2291453` to check whether NVIDIA's PyTorch 26.06 build already maps `dist.gather` to native `ncclGather` before implementing a small direct binding.
- Jobs `2291447` and `2291453` both completed on two GB200 GPUs. Equal-split `all_to_all_single` used only copy-engine `Memcpy DtoD/PtoP` events with no NCCL kernel, while `dist.gather` still emitted `ncclDevKernel_SendRecv`; PyTorch therefore exposes native `ncclAllToAll` but not native `ncclGather` in this image.
- Added `scripts/nonuniform/probe_nccl_native_gather.py`, a small experimental binding that calls the stable `ncclGather`/`ncclScatter` C API through `ctypes` using `ProcessGroupNCCL._comm_ptr()`. This avoids a compiled extension and keeps the experiment isolated from NEP until validated.
- Submitted native Gather/Scatter correctness and trace job `2291467` on two GB200 GPUs with symmetrically registered NCCL-window buffers and a zero-CTA communicator.
- Job `2291467` completed successfully. Native Gather and Scatter both passed BF16 correctness using the raw communicator pointer; each rank-0 trace contains only `Memcpy DtoD/PtoP` copy-engine activity and no `ncclDevKernel` event. The measured two-active-step GPU copy-event sums were about `0.112 ms` for each direction at 2 MiB per rank.
- Decided to integrate this path behind an explicit opt-in. The existing ordered backward-hook task scheduler and slot CUDA streams remain unchanged; the integration only replaces variable SendRecv AlltoAll with native Gather/Scatter and preallocates symmetrically registered staging buffers.
- PyTorch's ProcessGroupNCCL exposes its `ncclMemAlloc` allocator directly as `backend.mem_allocator`. Updated the probe to use that allocator with `torch.cuda.MemPool`, eliminating Megatron's inline allocator compilation from this path, and submitted validation job `2291478`.
- Job `2291478` completed successfully with the built-in ProcessGroup allocator. Native zero-CTA Gather/Scatter again passed and entered NCCL's CE path, now without compiling the Megatron NCCL allocator extension.
- Added an opt-in core implementation selected by `MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=1`:
  - Only `nep_owner_transfer` communicators receive `NCCL_CTA_POLICY_ZERO`.
  - Persistent small/large staging buffers are allocated from `ProcessGroupNCCL.mem_allocator`, registered symmetrically with each local owner group, and shared by the existing bounded slot scheduler.
  - The owner-transfer phases call native `ncclGather`/`ncclScatter` through a lazy `ctypes` helper. The existing variable-split AlltoAll path remains the default fallback.
- Submitted two-node EP4/2 smoke job `2291495` with 16 layers, four staging slots, ten iterations, and a rank-0 PyTorch profile on NeMo 26.06. It was pending on priority at submission.
- Job `2291495` completed successfully on two nodes. All ten iterations had finite losses and zero skipped/nan iterations; clean iterations 7-10 were `178-185 ms`. NCCL logged `Init CE` for both owner communicators, confirming the integrated path was active. The wrapper had overwritten the requested four-slot setting with 16, which was corrected after the run.
- Exact trace correlation found 64 Gather and 64 Scatter calls over two profiled iterations. Their 448 correlated GPU copy events moved about 4.29 GB, had an 8.03 ms union, and contained no owner-transfer NCCL kernel. Physical overlap with non-NCCL kernels on other streams was 0% in this small workload.
- The no-overlap explanation is launch granularity, not SM contention: native NCCL CPU markers averaged about `99 us` for Gather and `105 us` for Scatter, while their GPU annotations averaged `73 us` and `84 us`. Each short copy finishes before autograd has submitted the next backward kernel. The exposed transfer span is nevertheless only about `4 ms/iteration` here.
- Removed whole-buffer memsets from the native path. Only unequal-payload padding is now zeroed, and zero-payload chunk collectives are skipped. The smoke wrapper now respects an explicitly configured async window.
- Submitted an exact NeMo 26.06 comparison after that optimization:
  - Zero-SM job `2291559`, EP4/2, 12 iterations, rank-0 profile.
  - Existing variable-AlltoAll control `2291561`, with identical model, topology, profiler, and 16-slot window.
  - Both jobs were pending on priority at submission.
- The exact small comparison completed with finite losses and zero skipped/nan iterations:
  - Fallback `2291561`: clean iterations 7-10 averaged `191.2 ms`; 128 owner-transfer SendRecv kernels totaled `8.10 ms` over two profiled iterations.
  - Memset-removal zero-SM `2291559`: clean iterations averaged `230.7 ms`, a regression. Its native host markers rose to roughly `133/139 us` for Gather/Scatter from `99/105 us` in the original integrated run, while GPU annotations rose to `79/91 us` from `73/84 us`.
  - The original zero-SM smoke `2291495` averaged `181.3 ms`, about 5.2% faster than the exact fallback on its node pair. Because the memset-removal variant worsened launch latency, the whole-buffer staging writes were restored; only the zero-payload guard remains.
- Updated the real a3b wrapper to support a named enroot container on NeMo 26.06 and to rely on Transformer Engine's grouped-GEMM implementation instead of requiring the absent standalone `grouped_gemm` package during preflight.
- Submitted a real a3b EP8/4 MBS4/GBS48 pair, 12 iterations with rank-0 profiles and low-priority communicators:
  - Zero-SM job `2291612` on `lyris[0073-0075]`.
  - Variable-AlltoAll control `2291617` on `lyris[0076-0078]`.
  - Both jobs entered RUNNING immediately.
- Both real a3b jobs stalled in the first backward pass before iteration 1. Every sampled GPU remained at 100% utilization with no memory-controller activity, consistent with NCCL kernels spinning rather than useful model compute. Zero-SM job `2291612` eventually failed with exit code 143 after `28:26`; fallback job `2291617` was canceled after reproducing the same state.
- NCCL RAS queries identified the same mismatch in both jobs: a six-rank communicator spanning three nodes had four ranks at AllReduce operation 4 while communicator ranks 4 and 5, both on the reduced-replica node, remained at operation 2. All other queried communicator groups were healthy.
- The affected size-six group is a dense data-parallel communicator, not an owner-transfer communicator. The identical zero-SM and fallback failure therefore isolates a shared cross-communicator launch-order problem in this NeMo 26.06/NCCL 2.30 real-workload configuration rather than a native Gather/Scatter correctness failure.
- Next diagnostic: run the smallest a3b EP8/4 first-backward case with `CUDA_DEVICE_MAX_CONNECTIONS=1` and `NCCL_LAUNCH_ORDER_IMPLICIT=1`. This restores Megatron's sequence-parallel launch-order requirement and asks NCCL 2.30 to impose host launch order across communicators before resuming the full zero-SM/fallback profile comparison.
- Submitted that diagnostic as job `2291773`: a3b EP8/4 topology `4 2`, TP2, MBS1/GBS12, two iterations, no profiler, fallback owner transfers, and a 20-minute limit on three GB200 nodes.
- Job `2291773` reproduced the same first-backward stall even with `CUDA_DEVICE_MAX_CONNECTIONS=1` and `NCCL_LAUNCH_ORDER_IMPLICIT=1`. NCCL RAS again reported both six-rank dense-DP communicators at operation counts `4` on four healthy-replica ranks versus `2` on the two reduced-replica ranks. The job was canceled after the live diagnosis.
- Code inspection found the first-batch ordering bug: `_start_delayed_dense_grad_syncs()` skipped every first-batch dense bucket group, leaving `DistributedDataParallel.finish_grad_sync()` to start and wait groups sequentially. Rank-dependent bucket grouping can then leave reduced ranks waiting after two dense AllReduces while healthy ranks attempt to launch four.
- Removed the first-batch skip. The NEP wrapper now submits every ready dense bucket group before the parent finish loop waits; Megatron's existing first-batch `grad_reduce_handle` guard makes the parent start calls no-ops. Added opt-in per-bucket launch markers under `MEGATRON_NONUNIFORM_EP_DEBUG`.
- Submitted fixed fallback validation job `2291798`: a3b EP8/4, MBS1/GBS12, two iterations, original `CUDA_DEVICE_MAX_CONNECTIONS=32`/implicit-order disabled settings, with debug output restricted to representative healthy and reduced ranks.
- Job `2291798` reached the new dense prelaunch markers and exposed the underlying invalid layout. Healthy ranks had four dense groups with `299,945,984`, `302,388,192`, `302,388,192`, and `200,870,240` elements; reduced ranks had two with `531,236,512` and `574,356,096` elements. Both layouts total exactly `1,105,592,608` elements, but collective counts and boundaries differ. The job was canceled after this proof.
- Root cause is `--ddp-num-buckets 16`: training derives `bucket_size` from each rank's total local parameter count. Reduced-EP ranks hold more local expert parameters, so they derive a larger bucket size even though the dense parameter sequence is identical across replicas.
- Added a pre-DDP synchronization for NEP runs when `num_buckets` is configured. Each dense DP group takes the maximum locally derived bucket size before constructing buffers, producing identical dense bucket boundaries while retaining bucketing. Corrected first-batch readiness validation to use complete local parameter coverage because golden ready-count maps are populated only after the first reset.
- Submitted job `2291821` with the same minimal a3b EP8/4 MBS1/GBS12 fallback configuration to validate synchronized bucket construction and two complete iterations.
- Job `2291821` completed both iterations with finite loss and zero skipped/nan iterations. The local bucket-size derivations were `298,590,226` elements on healthy ranks and `528,080,914` on reduced ranks; synchronization selected `528,080,914` everywhere, and all sampled ranks built the identical two dense buckets of `531,236,512` and `574,356,096` elements. This removed the NCCL operation-count mismatch.
- The validation's `91.7 s` first and `46.5 s` second iteration times are not performance data: rank-filtered debug still emitted thousands of per-group lines. The full comparison disables all NEP debug output.
- Submitted the full corrected a3b EP8/4 TP2 MBS4/GBS48, 12-iteration profiled pair on NeMo 26.06:
  - Native zero-SM Gather/Scatter job `2291854`.
  - Variable-AlltoAll fallback job `2291856`.
  - Both use 16 scheduler slots, rank-0 profiler steps 5-7, low-priority communication groups, `CUDA_DEVICE_MAX_CONNECTIONS=32`, and implicit NCCL launch ordering disabled.
- Both jobs completed with finite losses and zero skipped/nan iterations. Excluding warmup, profiled iterations, and the manual-GC outlier, iterations 8-10 and 12 averaged:
  - Zero-SM `2291854`: `1514.7 ms`, `482.0 TFLOP/s/GPU`.
  - Fallback `2291856`: `1582.6 ms`, `461.3 TFLOP/s/GPU`.
  - Native Gather/Scatter improved iteration time and throughput by about `4.5%`.
- Rank-0 trace analysis over profiler steps 5-6:
  - Zero-SM contained 92 Gather and 92 Scatter GPU annotations, 644 correlated copy-engine events, about 44.06 GB moved, 77.07 ms copy-event union, and no `nep_owner_transfer` NCCL kernel.
  - Fallback contained 184 `nep_owner_transfer` SendRecv kernels with 115.05 ms union.
  - Literal copy-engine/non-NCCL-kernel overlap in zero-SM remained only 0.82 ms, or `1.1%`; fallback owner-transfer overlap was `0.35%`.
  - Zero-SM EDP all-reduce union was 131.91 ms with 10.5% compute overlap; fallback EDP was 114.43 ms with 4.9% overlap. Dense-DP all-reduce was about 23.35 ms in both.
- Host trace analysis explained the remaining lack of overlap. Two late AccumulateGrad hooks per profiled iteration each batched 38-41 native calls and occupied the autograd thread for 85-98 ms. Each such hook contained roughly 2,700 tensor-slice dispatches, 900 copies, 300 adds, and 300 scales, so copy-engine operations usually finished before autograd could submit the next compute kernel.
- Implemented a bounded-complexity host-path optimization: cache stable source-segment metadata and tensor views, then use PyTorch foreach copy/add/multiply operations for owner packing, source packing, gather accumulation, scatter packing, and local unpacking. The loop fallback remains for PyTorch builds without the foreach APIs.
- Submitted two-node EP4/2 zero-SM job `2291922` to run the focused unit tests and validate the fused-view path for ten finite profiled iterations before repeating the a3b measurement.
- Job `2291922` completed successfully: all four focused unit tests passed, all ten iterations had finite losses with zero skips/nans, and clean iterations 8-10 averaged `183.4 ms` versus `181.3 ms` in the original zero-SM smoke on a different node pair.
- Exact smoke trace comparison showed the host-path optimization worked as intended:
  - Reshard-hook slice operations fell from 1,344 to 448.
  - Individual `copy_`, `add_`, and `mul_` calls fell from 320/128/128 to zero and were replaced by 208/128/64 foreach calls.
  - Total CPU time in hooks containing native reshard calls fell from 73.61 to 63.48 ms (`13.8%`).
  - Copy-engine union remained about 8.05 ms and had zero useful-compute overlap in both toy traces because each transfer is too short.
- Submitted the full fused-view a3b MBS4/GBS48 profiled pair:
  - Zero-SM job `2291941`.
  - Variable-AlltoAll fallback job `2291942`.
- Zero-SM fused-view job `2291941` completed successfully. Clean iterations 8-10 and 12 averaged `1517.1 ms` and `481.2 TFLOP/s/GPU`, statistically unchanged from the pre-fusion zero-SM run (`1514.7 ms`, `482.0 TFLOP/s/GPU`).
- Its trace nevertheless confirms substantially better physical overlap:
  - Native copy-engine/non-NCCL-kernel overlap increased from `1.1%` to `31.2%` (`40.12 ms` overlapped out of `128.54 ms` copy union).
  - CPU time in reshard-containing AccumulateGrad hooks fell from `366.50 ms` to `162.34 ms` over two profiled iterations; the two largest hooks per iteration fell from roughly 85-98 ms to 34-38 ms.
  - Native copy union stretched from `77.07 ms` to `128.54 ms` while overlapping compute, showing memory/fabric contention offsets the reduced host exposure in steady-state throughput.
- Submitted zero-SM window sweep jobs to reduce concurrent copy-engine pressure while retaining fused host dispatch:
  - Eight slots: `2291974`.
  - Four slots: `2291975`.
  - Both were pending on priority with no start estimate at submission.
- Fused-view fallback job `2291942` also completed with finite losses and no skips/nans. Clean iterations 8-10 and 12 averaged `1611.5 ms` and `453.1 TFLOP/s/GPU`, about 1.8% slower than the pre-fusion fallback result on a different node set.
- Its trace shows the same tradeoff as zero-SM: long reshard-hook CPU time fell from `385.54 ms` to `160.20 ms`, but the 184 owner-transfer SendRecv kernels stretched to `152.45 ms` summed duration under increased concurrent memory traffic. The foreach optimization should therefore remain scoped to the zero-SM opt-in path rather than changing default fallback behavior.
- Scoped foreach copy/add/multiply launches to `zero_sm_reshard=True`; fallback keeps its original per-segment CUDA operations while retaining cached metadata/views.
- Window jobs `2291974` and `2291975` completed with finite losses and no skips/nans. Clean iterations 8-10 and 12:
  - 16 slots (`2291941`): `1517.1 ms`, `481.2 TFLOP/s/GPU`.
  - 8 slots (`2291974`): `1523.9 ms`, `479.0 TFLOP/s/GPU`.
  - 4 slots (`2291975`): `1527.8 ms`, `477.9 TFLOP/s/GPU`.
- Trace tradeoff across 16/8/4 slots:
  - Copy-engine overlap: `31.2%` / `27.7%` / `11.5%`.
  - Copy union: `128.54` / `129.89` / `83.14 ms`.
  - EDP all-reduce overlap: `44.1%` / `42.5%` / `19.1%`.
  - Four slots shorten individual communication but serialize the cross-task pipeline enough to make profiled iterations about 67 ms slower. Sixteen slots remains the selected setting.
- Submitted current same-code/container healthy control `2292010`: topology `4 4`, TP2/EP8, 16 GPUs, MBS4/GBS64, 12 iterations with profiler steps 5-7 on NeMo 26.06. It started immediately on four GB200 nodes.
- Healthy control `2292010` completed with finite losses and no skipped/nan iterations. Clean iterations 8-10 and 12 averaged `1484.8 ms` and `491.7 TFLOP/s/GPU`.
- Final current comparison:
  - Healthy `4 4`: `1484.8 ms`, `491.7 TFLOP/s/GPU`.
  - Zero-SM NEP `4 2`: `1517.1 ms`, `481.2 TFLOP/s/GPU`.
  - NEP is 2.2% slower by step time and reaches `97.9%` of healthy per-GPU throughput, with a `32.3 ms` step gap.
- Healthy versus NEP rank-0 traces show the remaining cost is dominated by owner-layout EDP reduction. Over two profiled iterations, healthy `ep_dp` union was `56.47 ms` with effectively no compute overlap; NEP was `192.36 ms` but overlapped `44.1%`. NEP's zero-SM owner copies additionally overlapped `31.2%` of their `128.54 ms` copy union.
- Submitted final containerized lint/test job `2292060` to run required `uv run isort`, targeted `black --check`, and the focused NEP unit suite on the edited Python files.
- Job `2292060` completed required `uv run isort` but stopped at `black --check`, which identified four files needing formatting; pytest did not run in that job.
- Follow-up job `2292099` applied targeted Black formatting, reran required `uv run isort`, passed Black check on all five edited Python files, and passed all four focused tests in `tests/unit_tests/distributed/test_nonuniform_ep.py`.

## 2026-07-02 - Lyris NEP overlap investigation

- Completed corrected eight-bucket Lyris comparisons using nvcr.io/nvidia/nemo:25.09: healthy MBS2 job 2261389, NEP MBS2 job 2261369, healthy MBS4 job 2261390, and NEP MBS4 job 2261391.
- Clean iterations 8-10:
  - MBS2 healthy 242.03 TFLOP/s/GPU versus NEP 66.00 (27.3% of healthy).
  - MBS4 healthy 440.60 TFLOP/s/GPU versus NEP 128.73 (29.2% of healthy).
- Process-group-specific rank-0 trace analysis showed exactly 0 ms overlap between non-NCCL kernels and all 184 nep_owner_transfer kernels at both MBS2 and MBS4. MBS4 contained about 3513 ms of owner-transfer NCCL and 104 ms of owner ep_dp all-reduce NCCL.
- NEP debug events showed owner-transfer launches beginning roughly two seconds before backward completed. Therefore late launch was not the primary problem.
- One serialization source identified in NonuniformEPNCCLParamAndGradBucketGroup: every bounded buffer-slot reuse called Work.wait() from the autograd hook. With 184 tasks and 16 slots, these host waits repeatedly stopped submission of later backward kernels.
- Replaced slot-reuse host waits with Work.block_current_stream() ordering on the dedicated NEP communication stream. This preserves buffer safety and gather/all-reduce/scatter ordering while allowing the autograd thread to keep submitting later compute.
- Added focused coverage in tests/unit_tests/distributed/test_nonuniform_ep.py and a Lyris profiled smoke wrapper at scripts/nonuniform/run_lyris_nonuniform_ep_overlap_smoke.sh.
- Container-side focused checks passed in jobs 2261947 and 2261949; their distributed steps did not start because an initial wrapper version contained a malformed srun continuation. The wrapper was corrected.
- Corrected EP4/2 smoke job 2262080 completed 10/10 iterations with finite losses. Its owner-transfer kernels were too short to provide a meaningful large-workload overlap test.
- MBS4 jobs 2262099 (healthy) and 2262100 (NEP after nonblocking slot reuse) completed:
  - Healthy: 806.00 ms and 452.90 TFLOP/s/GPU.
  - NEP: 2833.53 ms and 128.83 TFLOP/s/GPU, effectively unchanged from the original full-layout path.
  - Owner-transfer overlap increased only from 0% to 2.16%, so host slot waits were not the dominant large-workload cost.
- CTA-8, CTA-4, CGA-disabled, and high-priority CTA-8 owner-transfer communicator experiments did not improve overlap or throughput. Reducing CTAs increased communication duration enough to make end-to-end performance worse.
- The dominant algorithmic issue was zero-padded transfer volume. For EP8/4, each of four follower ranks sent a full 32-expert owner-layout chunk while holding only four useful expert slices, and the owner sent the same full chunk back to every follower.
- Reimplemented owner gather/scatter with dense variable-split all-to-all payloads. Owners still reconstruct the same full layout before the EDP all-reduce, but followers send and receive only their actual expert slices.
- Dense correctness smoke job 2262211 completed 10/10 iterations and halved EP4/2 owner-transfer message size and kernel time.
- Dense MBS4 NEP job 2262239 completed successfully:
  - 1219.03 ms and 299.43 TFLOP/s/GPU.
  - NEP improved from 28.4% to 66.1% of the 452.90 TFLOP/s/GPU healthy result.
  - Owner-transfer NCCL union fell from 3738.88 ms to 776.91 ms; overlap rose from 2.16% to 10.00%.
  - The dense trace reduced the rank-0 owner-transfer message from 638,582,784 to 79,822,848 elements, an 8x reduction.
- Container job 2262356 ran required isort and the focused tests; both nonblocking slot reuse and dense gather/scatter round-trip tests passed.
- Chunk-size sweep jobs 2262739 (256 MiB), 2262740 (128 MiB), and 2262741 (64 MiB) showed that splitting the dense owner payload further does not improve overlap. Clean iteration averages were about 1214, 1320, and 1345 ms respectively, versus 1219 ms for the single-chunk dense run.
- Trace/code correlation identified an additional scheduler serialization: all owner tasks shared one auxiliary CUDA stream, so gather, owner EDP all-reduce, scatter, and the next independent owner task were chained on that stream despite the bounded buffer window.
- Replaced the single auxiliary stream with a bounded per-buffer-slot stream pool. Dependencies and buffer reuse remain ordered within a slot, while independent owner tasks can progress concurrently.
- Focused test job 2262843 passed 3/3 tests, including new stream-slot coverage. Its batch status was nonzero only because whole-file `black --check` requested formatting of the modified implementation file.
- Multi-stream EP8/4 smoke job 2262844 completed 10/10 finite iterations with zero skipped/nan iterations.
- Multi-stream a3b NEP job 2262845 completed successfully:
  - Clean iterations 8-10 averaged about 914 ms and 399 TFLOP/s/GPU.
  - Throughput improved from 66% to about 88% of the existing healthy MBS4 result.
  - Rank-0 owner-transfer GPU time fell from about 777 ms to 81 ms over the two profiled iterations because independent owner collectives no longer suffered cross-rank launch skew.
  - Direct owner-transfer/non-NCCL kernel overlap remained low at about 5%; the improvement primarily came from shortening owner-transfer completion time, not hiding it behind compute.
- Disabling high-priority EP/EDP communicators in job 2262893 improved clean iterations 8-10 to about 886 ms and 412 TFLOP/s/GPU, about 91% of the existing healthy result. Direct communication/non-NCCL overlap remained low, so a matching healthy control was submitted as job 2262899.
- MBS8 NEP job 2262892 OOMed at roughly 182 GiB used during the vocabulary cross-entropy allocation. MBS6 job 2262901 fit at roughly 142 GiB and completed at about 1151 ms and 476 TFLOP/s/GPU; direct owner-transfer plus EDP overlap increased to about 24%. The matching 45-minute healthy request was canceled when essential controls were resubmitted with accurate 15-minute limits.
- Exact low-priority MBS4 healthy control job 2263540 completed at 769.1 ms and 474.6 TFLOP/s/GPU over clean iterations 8-10. The pre-phase-pipeline NEP result was therefore 86.8% of exact healthy throughput, with a 117 ms step gap.
- Exact two-microbatch controls completed:
  - Healthy job 2263541: 1433.8 ms and 509.2 TFLOP/s/GPU.
  - Pre-phase-pipeline NEP job 2263542: 1577.0 ms and 462.9 TFLOP/s/GPU, or 90.9% of healthy throughput.
  - Trace overlap remained about 1.7%, confirming that accumulation amortized exposed synchronization rather than hiding it.
- Added bounded cross-layer phase pipelining. Each slot now keeps gather and owner EDP reduce in flight; scatter is deferred until slot reuse or the final drain. Deferred scatters are enqueued on their original slot streams, and reset asserts that no deferred scatter survived the iteration.
- Phase-pipeline validation:
  - Focused test job 2263584 passed 3/3 tests.
  - EP8/4 smoke job 2263573 completed 10/10 iterations with finite losses and zero skipped/nan iterations.
  - Full one-microbatch NEP job 2263585 completed at 858.7 ms and 425.1 TFLOP/s/GPU, improving exact parity from 86.8% to 89.6%.
  - Phase pipelining reduced profiled owner-transfer plus EDP union from about 225 ms to 189 ms; literal non-NCCL overlap remained low.
- MBS7 healthy job 2263610 completed at 1173.5 ms and 544.3 TFLOP/s/GPU. Phase-pipeline NEP job 2263611 OOMed because the bounded pipeline staging buffers pushed memory above the GB200 limit; MBS7 is not a viable NEP setting with a 16-slot window.
- Final near-parity result uses two MBS4 microbatches:
  - Phase-pipeline NEP job 2263612: 1517.1 ms and 481.2 TFLOP/s/GPU over iterations 8-10.
  - Exact healthy job 2263541: 1433.8 ms and 509.2 TFLOP/s/GPU.
  - NEP reaches 94.5% of healthy per-GPU throughput with an 83.3 ms step gap; all measured iterations had finite losses and zero skipped/nan iterations.
  - Rank-0 trace still shows only 3.7% combined owner-transfer/EDP overlap with non-NCCL kernels. Near parity comes from reduced communication span plus amortizing the remaining fixed cost over two microbatches, not from mostly hiding reshard behind backward compute.

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
