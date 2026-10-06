# Backward Normalization Headroom Investigation

Parent: `nep-expert-buckets-20261006`, commit `a6f90f29e`. Training code unchanged.

Hypothesis: concurrent gradient Gather and expert RS consume more SM resources
than the 32-SM backward normalization margin allows, delaying cooperative
RMSNorm admission and propagating waits into native TP/model-EP.

## Controlled Test

- Preserve previous a8b 15-MoE workload, EP16/12, TP2, CP1, MBS2/GA8/S8192,
  DistOpt, Flex/HybridEP, graphs, routing, dense buckets, topology and profiling.
- Fix NEP expert buckets at five; healthy retains five native expert buckets.
- Only vary native `NVTE_BWD_LAYERNORM_SM_MARGIN`: 32 versus 64. Apply each
  to healthy and NEP. Forward normalization margin remains16.
- Two reversed-order repeats; all-rank profiles plus previously authorized
  timing companions; 35 iterations, matched measured steps25-30.
- GB300 only: the previous GB200 case failed unchanged memory-headroom guards.
  Retain segment4 and audit local NVL versus cross-replica RDMA.
- Archive installed TE source and verify instantiated RMSNorm margin before
  training. Confirm actual norm grid changes in each rank's trace afterwards.

## Interpretation

Evidence for causality requires the changed grid to reduce norm launch gaps
and downstream TP/model-EP waits, with early host submission unchanged.
Compare final and ordinary backward, healthy as well as NEP, and total owner
iteration times. Norm runtime can increase with fewer SMs; report that cost.
Do not infer gains by adding residency across ranks or overlapping operations.
Healthy speedup alone is not a NEP parity improvement. If effects are mixed,
report that the margin does not fully explain the remaining overhead.

No production fix will be applied as part of the investigation. If supported,
propose the existing environment setting rather than invasive scheduling code.

## Results

Job3253300 completed16/16 cases, exit0, elapsed1:25:12 on GB300. Replicas were
in nvlblk18/23; all cross-replica optimizer pairs verified RDMA. Runtime
checkpoint557777453; `megatron/` is identical to the preceding bucket study.
Installed TE2.16.0+b9d690e0, NCCL2.30.4,152SMs. The installed Python sources and
upstream C++ sources at those exact versions are archived in the job output.

All-rank traces verify that margin32 uses360 norm CTAs and margin64 uses264,
with128 threads and167 registers/thread in both cases. The exact-version
general RMSNorm launcher uses cooperative launch when a row spans more than
one CTA; H4608 with the observed HIDDEN_SIZE4096 specialization meets this
condition. The PyTorch extension subtracts the margin from physical SM count.

### Mechanism

The intervention supports insufficient cooperative-launch headroom as a real
contributor, not a whole-collective completion fence or late host submission:

| Final-backward measurement, full-replica ranks0-15 | Margin32 | Margin64 |
| --- | ---: | ---: |
| Mean summed no-native-work gaps before RMSNorm per rank/step | 3.245ms | 0.311ms |
| Individual graph gaps greater than1ms, both repeats | 193 | 0 |
| Largest individual graph gap | 7.422ms | 0.159ms |
| Mean RMSNorm kernel residency per rank/step | 8.205ms | 10.995ms |

These gap sums are rank-local diagnostics, not additive e2e savings. Healthy
also improves its small norm gaps, from0.403 to0.244ms. Ordinary-backward norm
gaps remain approximately0.25ms; the large improvement is specific to final
backward when gradient communication is active.

Concrete paired example: repeat0/rank10/step28/backward block10 has a6.223ms
norm launch gap with margin32, versus0.0099ms with64. The graph was submitted
349/351ms ahead, respectively. During the32-margin gap, Gather runs32CTAs and
RS8CTAs. Gather starts2.410ms before the gap, ends6.481ms after its beginning;
RS starts1.187ms before, ends22.213ms after. Norm starts at+6.223ms, before
either full collective completes. A360-CTA norm grid uses the occupancy
budget of120SMs;32+8 large NCCL CTAs can exceed the32-SM headroom. This is a
resource-budget explanation supported by the intervention, not a direct
hardware SM-occupancy counter measurement.

Check against a timing-shift-only explanation: in margin64 repeat0, ranks
0/4/8/10 contain5/11/19/25 norm kernels respectively whose entire execution
is inside simultaneous Gather+RS kernel residency totaling more than32CTAs.
For rank0 at Nsight timestamp3115222183ns, the264-CTA norm starts after a
0.015488ms native-work gap and runs0.3752ms;32-CTA Gather spans -0.133120 to
+2.032704ms relative to norm start and24-CTA RS spans -7.169024 to+5.298624ms.
Thus norm really can execute concurrently with the56-CTA communication
combination, not merely start after communication finishes. Rank4 has another
example with32-CTA Gather plus24- and8-CTA RS kernels, totaling64CTAs.
The selected paired block10 at margin64 itself has no concurrent collective;
do not use that individual pair alone as proof of simultaneous admission.

### End-To-End Tradeoff

Owner ranks0-11, matched steps25-30, means over both profiled repeats:

| Measurement | Margin32 | Margin64 | Change |
| --- | ---: | ---: | ---: |
| Healthy native iteration | 3213.652ms | 3237.236ms | +23.584ms |
| NEP native iteration | 3279.179ms | 3286.869ms | +7.690ms |
| NEP owner parity | 98.002% | 98.490% | +0.488pp |
| NEP final backward | 271.213ms | 266.006ms | -5.207ms |
| NEP other backwards, summed | 1613.416ms | 1629.686ms | +16.270ms |

Do NOT adopt margin64 on the strength of the higher parity: it slows healthy
more and also slows absolute NEP iteration time in both profiled repeats.
Reducing norm resources increases norm kernel duration in every microbatch.
The remaining final-backward overhead versus healthy is23.885ms at64,
compared with31.788ms at32; this hypothesis does not explain the entire gap.

Timing companions are not consistent enough to claim a net speedup:

| Margin | Healthy repeat0/1 ms | NEP repeat0/1 ms |
| --- | ---: | ---: |
| 32 | 3168.267 / 3163.394 | 3232.971 / 3278.499 |
| 64 | 3181.473 / 3183.331 | 3240.709 / 3235.739 |

The second32-margin NEP companion has a45.527ms shift, present across all six
measured iterations (SD5.508ms), not one outlier. Its cause is unresolved;
there is no low-level profile for a timing companion. Retain it rather than
discarding or substituting a post-profile window. The raw pooled companion
parities97.239/98.276% are descriptive, not evidence of a reliable speedup.
All gates passed, with no skipped/NaN iterations. Dummy-training values diverge
later even across same-setting runs; do not claim bitwise trajectory identity.

## Proposed Fix, Not Applied

Keep normalization margin32. Reduce only NEP gradient Gather's SM footprint
using existing Megatron per-process-group NCCL options, initially:

```yaml
nep_owner_gather:
  max_ctas: 16
```

Pass the file via `--nccl-communicator-config-path`. No runtime Python edits
are necessary: `nonuniform_ep.py` creates the separate `nep_owner_gather`
group and uses it for gradient Gather; `nonuniform_common.py` already calls
native `parallel_state.get_nccl_options`, which supports `max_ctas`.
The separate parameter-delivery group `nep_owner_transfer`, expert/dense RS,
TP and model-EP remain unchanged. NCCL2.30.4 P2P channel construction explicitly
honors communicator maxCTAs. No extra host hooks, queues, waits or ordering
changes are needed. This is configurable, not a hardcoded universal cap for
all EP degrees or devices.

For the observed32-Gather+8-RS case,16+8 fits inside the existing32-SM margin
without slowing normalization in every microbatch. The risk is reduced Gather
bandwidth; the net gain is unproven. Test baseline versus Gather16 (optionally8)
with identical healthy controls/profiles, verify actual grids and unchanged
payload/ordering/numerics, then require reduced norm gaps and improved absolute
owner iteration time without first-forward or ordinary-backward regression.
Do not accept an apparent parity improvement caused by a slower healthy run.
Some traced windows have BOTH24-CTA dense RS and8-CTA expert RS as well as
Gather. A Gather-only cap cannot guarantee that their combined resource budget
fits32SMs. This proposal is the least-invasive first test, not a guarantee of
eliminating every stall. If those windows remain limiting, consult before
changing native RS communicator caps, which can also affect parameter AG on
the same groups. Do not silently expand the scope to those collectives.

## Evidence

Job root: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3253300`.
`norm-headroom-summary.json`: timing, phases, all-rank norm grids and native
graph statistics. `norm_headroom/`: per-rank/per-repeat graph records.
`actual-summary.json`, `work-controls-audit.json`, and per-case validation
preserve all16 cases. `normalization_source/` holds pinned source evidence.

- TE cooperative launcher: https://github.com/NVIDIA/TransformerEngine/blob/b9d690e0/transformer_engine/common/normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
- TE SM-margin plumbing: https://github.com/NVIDIA/TransformerEngine/blob/b9d690e0/transformer_engine/pytorch/csrc/extensions/normalization.cpp
- NCCL P2P channel cap: https://github.com/NVIDIA/nccl/blob/v2.30.4-1/src/graph/paths.cc
