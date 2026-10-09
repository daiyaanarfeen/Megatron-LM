# Final Backward and First Forward Overheads

Date: 2026-09-14. Existing-data/code analysis only; no runtime changes or jobs.

## Verdict

These are workload-dependent exposures of a mixture of necessary data movement
and implementation-specific work. They are not fixed, unavoidable latency taxes.
At fixed model, topology and bucket layout, much of the payload and operation
count is fixed per optimizer update, but its elapsed or exposed time is not.

The conclusions below apply to the trusted DistOpt + Flex/HybridEP implementation,
including nondivisible experts. Do not substitute the older ordinary-optimizer
gradient-Scatter lifecycle or mix in the unaccepted CPU-cleanup implementation.

## Actual Lifecycle

Final accumulation microbatch:

1. AccumulateGrad makes an expert group ready; ordered callbacks submit Gather.
2. Gather gathers/accumulates physical gradients into logical owner layout.
3. Per-expert/parameter copies stage that layout into native DistOpt grad views.
4. Native DDP starts expert reduce-scatter. Remaining backward can overlap it.
5. Final gradient synchronization joins outstanding reductions before optimizer
   use. Only owner optimizer shards update these expert parameters.

Next first forward:

1. Native owner parameter all-gather reconstructs updated logical owner weights.
2. NEP packs owner weights, broadcasts to physical holders, and unpacks into
   model parameter tensors. Singleton transfers still pack/unpack in this version.
3. Forward can use each group's weights after they are ready. Later accumulated
   microbatches reuse them; redistribution is not repeated until another update.

DistOpt does NOT scatter the reduced gradients back to every physical holder:
`_start_nep_nccl_owner_task_scatter` returns after bookkeeping in that mode.
Parameter redistribution is a different operation in the subsequent forward.

## Best Low-Distortion Evidence

Job 3007029, a770, EP8/8 healthy versus EP8/6 NEP, GA3, sequence8192,
H1280/F768, 128 experts/top-k6. MBS1 versus MBS2 changes the amount of work;
parameter payload, eight expert groups and implementation remain unchanged.
Targeted CPU/CUDA-event profiling, iterations40-49, full-owner ranks0-5.
See NEP_A770_LOW_OVERHEAD_MBS_RESULTS.md for calibration and topology controls.

| NEP minus healthy, ms | MBS1 | MBS2 |
| --- | ---: | ---: |
| Iteration start through last expert forward of first microbatch | 28.426 | 19.788 |
| Final microbatch: first expert backward start to last expert backward end | 4.316 | 4.545 |
| Last expert backward end through iteration end | 16.278 | 2.552 |
| Whole iteration | 56.08 | 31.12 |

These rows are not an additive partition: intervening work is omitted. The first
row includes preparation; the backward row is an expert-bounded execution span,
not all model backward kernels; the tail includes remaining model work, sync,
optimizer and finalization. None is a pure collective-service measurement.

The tail itself is healthy12.986/NEP29.264ms at MBS1, versus
healthy20.577/NEP23.129ms at MBS2. Its penalty shrinks13.726ms, but NEP only
shortens6.135ms: healthy's tail also grows7.591ms. Do not call the whole difference
newly hidden NEP communication. The final backward span penalty does not shrink.

### First Forward: Direct Host-Work and Launch-Headroom Evidence

Recomputed from the existing cuda_event_analysis/summary.json:

| Parameter redistribution | MBS1 | MBS2 |
| --- | ---: | ---: |
| Full owners CPU wall / thread execution, summed ms | 19.443 / 19.224 | 16.655 / 16.201 |
| Reduced owners CPU wall / thread execution, summed ms | 19.735 / 19.495 | 19.081 / 18.661 |
| Full owners: CPU call finished before GPU range began | 1.7% | 54.6% |
| Reduced owners: CPU call finished before GPU range began | 0% | 40.0% |
| Full owners GPU operation-range union, ms | 16.703 | 11.615 |
| Reduced owners GPU operation-range union, ms | 14.059 | 5.243 |

CPU thread time nearly equals wall time: long host calls here are predominantly
executing on CPU, not simply being descheduled for most of their lifetime.
This includes library/runtime/instrumentation work, not just Python.
Almost unchanged reduced-owner CPU work is submitted farther ahead of the GPU
at MBS2. More compute provides scheduling headroom; less parameter traffic is
not the explanation. CUDA-event ranges include stream dependencies and host
spacing; their shortening is not an exact GPU-idle or bandwidth measurement.

### Final Backward: Do Not Reuse the First-Forward Explanation Uncritically

Full-owner gradient staging CPU wall time is9.733/9.935ms at MBS1/2; CPU thread
time is9.736/9.931ms (tiny instrumentation/timestamp differences). Already98.3%
of staging calls finish on CPU before the GPU range begins at MBS1, versus100%
at MBS2. Mean CPU-to-GPU start lag grows14.287->59.723ms.

Thus staging is real implementation work, but the trace does not show it to be
generally submission-starved in the same way as first-forward redistribution.
Its host execution can still delay later host submissions. The final bucket
has less independent backward left to hide Gather, staging and reduce-scatter.
The optimizer requires completed gradients; exposed late readiness is possible.
These ranges cannot partition the remaining4.3-4.5ms backward penalty into
Gather, staging, reduce-scatter, native-EP interference and host stalls exactly.

## Which Costs Are Avoidable?

| Component | Necessary part | Implementation/workload dependence |
| --- | --- | --- |
| Gradient owner reshard | Correct expert contributions must reach optimizer owners | Chosen owner layout, padding, packing, streams and group readiness determine bytes, launch count and exposure |
| Native expert reduce-scatter | DistOpt needs globally combined gradient shards | Healthy also does this; NEP changes owner payload/distribution. Bucket readiness and peer progress affect exposure |
| Owner-gradient staging | Native DistOpt must see the correct gradient layout | A separate scalar copy per parameter is a layout/implementation choice, not an algorithmic requirement |
| Native parameter all-gather | Needed by the current sharded optimizer to reconstruct owner weights | Healthy also pays this; NEP payload and prefetch schedule differ |
| Parameter owner-to-holder redistribution | Remote physical holders need their updated weights | Network movement is required for remote holders, but redundant singleton staging, repeated metadata work and per-parameter launch granularity are not |
| Final sync / first weight-use dependencies | Cannot update before gradients or compute with stale weights | Correct dependency is mandatory; its entire duration need not be exposed or a host-blocking wait |

At fixed weights/dtype/placement, changing MBS/sequence does not change gradient
or parameter payload. Changing H/F, experts, layers or EP does. Therefore neither
communication bytes nor CPU operation count is a universal model-independent tax.

Native parameter all-gather prefetch is also modified by the integration:
NEP calls native finish with skip_next_bucket_dispatch=True, submits the current
redistribution, then starts the next group. This adds host-submission ordering
relative to native DDP, which starts the next gather inside finish. It is a
potential avoidable limitation, not a measured attribution or a recommendation
to reorder NCCL without correctness/order testing. A synchronous-looking CUDA
distributed API call must not automatically be counted as a CPU wait until all
network work completes.

## Isolated Implementation Experiment Already Available

Separate unaccepted branch nep/param-redistribution-cpu-20260911 caches static
transfer metadata and removes one of two copies for singleton transfers.
No changed ordinary expert execution, prefetch or collective order.

Same-allocation NEP-only A/B job3007909, GB300, not combined with job3007029:

| Measurement | Original | Modified |
| --- | ---: | ---: |
| MBS1 owner iteration, ms | 470.15 | 442.58 |
| MBS2 owner iteration, ms | 718.54 | 709.20 |
| MBS1 reduced-owner redistribution CPU, ms | 19.086 | 8.248 |
| MBS2 reduced-owner redistribution CPU, ms | 17.194 | 7.767 |

This supports an implementation-specific, reducible cost, especially at MBS1.
It is NOT a new healthy-relative parity result or proof every saved millisecond
was first-forward GPU starvation. MBS2's timing gain is small versus variation.
Mapping tests preserve values/storage/gradients, but independent training losses
later diverge and broader distributed numerical validation remains unfinished.
Do not treat the branch as accepted or merge it on this evidence alone.

## Workload Knobs and Limits

- More MBS/sequence gives more forward and final-backward compute per update's
  fixed synchronization payload. It can hide host submission and communication,
  but memory, native token traffic and reduced-replica compute also increase.
- More GA amortizes first-forward delivery and final-backward synchronization
  over more ordinary microbatches. It does not itself enlarge the final backward
  microbatch's overlap window or remove first-weight-use dependencies.
- Larger H/F raises compute but also weight/gradient bytes; more layers repeats
  both launch work and arithmetic. Neither guarantees less absolute overhead.
- More experts/parameter tensors and more buckets raise metadata/copy/collective
  launch count; fewer buckets reduce launch count but make gradients ready later.
- EP degree and nondivisibility change owner/holder fanout, local expert counts,
  singleton opportunities and padding. Placement and peer readiness also matter.

## Full-Profile Cross-Workload Check and Uncertainty

The existing microbatch figure qualitatively locates costs at these special
microbatches, but its full-profiler timings must not replace the low-overhead
figures above. Example: a770 MBS1 old full-profile final-backward GPU span is
healthy84.330/NEP143.592ms; low-overhead expert-bounded delta is only4.316ms.
The runs and endpoints differ, so the difference cannot be assigned entirely
to the profiler; it is nevertheless not valid evidence of a fixed60ms tax.

a8b's large first-forward excess in the full profile is likewise not a clean
production latency estimate. Existing evidence supports fixed-count-per-update
work and GA amortization, not a universal fixed-millisecond reshard overhead.

High-confidence conclusions: phase lifecycle; redundant scalar-copy/metadata
work in code; nearly unchanged CPU work but more launch headroom at larger MBS;
substantial variability of exposure; no DistOpt backward gradient Scatter.
Not established: exact unavoidable latency floor, exact per-collective share of
the low-overhead backward delta, or a universally safe optimized schedule.

## Code and Evidence References

Trusted checkout: /home/darfeen/Megatron-LM-nep-main-20260817.
megatron/core/distributed/nonuniform_ep.py:
- 1336: stage owner Gather output into native DistOpt gradient views.
- 1449: invoke native expert gradient sync after staging.
- 1503: final native gradient synchronization.
- 1629: owner-to-physical parameter packing/broadcast/unpacking.
- 1688/1722: native all-gather plus NEP parameter-sync lifecycle.
- 1749: skip backward gradient Scatter under DistOpt.
- 1760: Gather event dependency onto ordered EDP stream.
- 2926: AccumulateGrad callback installation.
Native next-group prefetch: param_and_grad_buffer.py:538.

Primary data: /lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/
soft_clique_baseline/3007029/cuda_event_analysis/summary.json.
Optimization report: /home/darfeen/Megatron-LM-nep-param-cpu-20260911/
NEP_PARAM_TRANSFER_CPU_REPORT.md.
Full-profile methodology: NEP_MICROBATCH_PHASES_20260914.md.
