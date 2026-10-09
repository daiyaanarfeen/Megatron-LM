# Native expert host-exposure experiment

## Question and falsifiable test

Does unhidden native per-expert host submission cause the slower ordinary
backward on reduced EP? Higher whole-iteration parity alone is not evidence.

Use the a8b expert dimensions (latent1152, F3072, weighted squared-ReLU,
BF16, ETP1, fused FP32 weight-gradient accumulation), same native TE
GroupedLinear implementation, 32 experts for EP16 and 42/43 for EP12.
At MBS2, TP2, sequence8192, top-k6, 512 experts, balanced expected counts
are1536 tokens/expert on EP16 and1152 on EP12. EP12 has eight ranks with43
experts and four with42; use their weighted mean. Sequence16384 doubles
token counts with expert count and invocation count unchanged.

This is an isolated expert subworkload, not full distributed training.
It excludes token dispatch, NEP, attention/Mamba, shared experts, and optimizer
communication. Counts are deterministic uniform expectations, not the exact
random routing vectors from the earlier workload. It retains native TE expert
FC1/FC2 and Megatron's fused activation. This cannot alone quantify the exact
number of critical-path milliseconds in the earlier full workload.

For each input compare identical eager execution in two conditions:

- Normal: CPU submits recompute-forward and backward as the GPU progresses.
- Prequeued: a calibrated ~50ms stream delay BEFORE the measured interval
  provides host launch headroom. Same math, shapes and native calls. This is
  an attribution intervention, NOT a proposed optimization: the omitted
  artificial delay would make real training slower.

Check outputs, input/probability gradients and all FP32 parameter gradients
bitwise. Alternate condition order over20samples, after15warmups. Record CUDA
events plus host thread times for every sample. Additionally collect four
CPU/CUDA kernel-profiled diagnostic samples per configuration, reporting their
instrumentation distortion separately. Do not combine their timings with the
lower-overhead samples. Inspect kernel inventories/durations to guard against
algorithm changes or GPU clock/cache effects from the head start.

The difference of differences is the attribution target:
`(EP12 normal - EP16 normal) - (EP12 prequeued - EP16 prequeued)`.
It isolates how much of the reduced-EP penalty is sensitive to launch headroom,
subject to validation that all calls really finish before the GPU starts and
the GPU work is unchanged. If the penalty persists, host submission is not a
sufficient explanation. No production runtime changes.

## Existing distributed sequence evidence

All healthy second-replica16ranks and NEP reduced12ranks; steps25-34,
ordinary backward microbatches2-7 (one-based). These are elapsed expert
envelopes, not sums of compute kernels.

| GB300 sequence | Healthy second B, ms | NEP reduced B, ms | Extra B, ms | Healthy expert CPU, ms | Reduced expert CPU, ms |
|---|---:|---:|---:|---:|---:|
| 8192, job3091888 | 356.712 | 372.629 | 15.917 | 78.404 | 97.928 |
| 16384, job3092074 | 535.198 | 563.201 | 28.003 | 78.997 | 99.503 |

Thus the prior16K parity improvement does NOT demonstrate a smaller absolute
ordinary-backward penalty. Different allocations remain a confounder; the new
microbenchmark keeps the GPU fixed. Full16K job3092074 completed0:0 in24:07,
all60rank profiles,45iterations/case. Measuredhealthy7919.98/NEP8157.17ms,
parity97.09%; post7856.73/8065.28ms,parity97.41%. Both replicas within one
physicalNVLdomain but separate MNNVLcliques; cross-replica RDMA verified.

## Jobs

- 3119924: stopped during container preparation after discovering the draft
  probe used SwiGLU rather than a8b's squared-ReLU. No measurements accepted.
- 3119944: corrected probe, oneGB200node lyris0018,10minlimit, started
  2026-09-21 09:49:47PDT. Completed0:0 in3:07. GB200/GB300 both eligible.
- Code: `scripts/nonuniform/probe_expert_host_exposure.py` and
  `scripts/nonuniform/run_expert_host_exposure.sh`.
- Artifacts: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/expert_host_exposure/3119944/`.

## Results: local mechanism demonstrated

All six configurations pass bitwise output/input/probability/FP32-weight-gradient
equality. All120prequeued timing samples finish CPU submission at least39ms
before GPU execution begins. Four diagnostic traces per shape have identical
kernel names, counts and grid/block geometry across normal/prequeued modes.
Each phase below is a CUDA-event elapsed interval, measured20times/condition.

EP12 uses the actual8:4 weighting of43-expert and42-expert ranks:

| Tokens equivalent to sequence | Phase | EP12 minus EP16 normal, ms/layer | EP12 minus EP16 prequeued, ms/layer | Difference removed |
|---|---|---:|---:|---:|
| 8K | Backward only | 0.8284 | 0.2305 | 0.5979 (72.2%) |
| 8K | Recompute forward + backward | 1.1791 | 0.2823 | 0.8968 (76.1%) |
| 16K | Backward only | 0.3627 | 0.1975 | 0.1652 (45.5%) |
| 16K | Recompute forward + backward | 0.5832 | 0.2460 | 0.3372 (57.8%) |

For8K, actual backward means are EP16normal2.2227/prequeued1.5754ms versus
EP12weightednormal3.0512/prequeued1.8060ms. For16K, EP16normal2.7869/
prequeued2.5319ms versusEP12weightednormal3.1496/prequeued2.7294ms.
Doubling tokens reduces the NORMAL backward penalty56.2%, and the portion
removed by host headroom72.4%. This is an absolute gap reduction, not merely
amortization in the isolated test. The remaining prequeued penalty must not
be called exposed CPU submission; kernel execution/scheduling differs between
the32-expert and42/43-expert shapes.

### Kernel and host evidence

8K43experts execute258GEMM kernels plus4activation/gradient-add kernels,
unchanged normal/prequeued (32experts execute192GEMMs+4other kernels).
Across the two diagnostic samples per mode:

- No-kernel gaps inside first-to-last work-kernel interval shrink from2.774ms
  normal to0.073ms prequeued; this is UNION over all observed compute streams.
  Excludes the artificial delay and gradient zeroing, both before work.
- Summed GEMM kernel duration does NOT improve:2.535ms normal versus2.626ms
  prequeued. Therefore the large elapsed gain is not faster GEMM service.
  Kernel busy union2.644->2.567ms changes modestly with scheduling/overlap.
- In the first normal diagnostic, relative to its outer annotation, the GPU
  becomes idle at3.775ms. The next GEMM's `cuLaunchKernelEx` is only entered
  at4.413ms, and the next kernel starts at4.428ms:~653us gap, mostly BEFORE
  the launch API, inside native `_GroupedLinearBackward` (4.038-5.509ms).
  Another native backward gap starts6.121ms; its launch arrives6.378ms.
- Native `_GroupedLinearBackward` CPU elapsed sum stays~2.95ms normal and
  ~2.90ms prequeued for43experts: the host work was hidden, not eliminated.

### Instrumentation and interpretation limits

Kernel profiling inflates normal total times:~17-45% versus CUDA-event samples;
prequeued~2-4%. Therefore diagnostic gap sizes above are NOT subtracted from
lower-overhead timing and NOT extrapolated to the older training trace. Kernel
identities and launch ordering corroborate the mechanism; the headline gap
removal uses the separate CUDA-event-profiled samples only. The event samples
are not zero-overhead but use identical fixed instrumentation in both modes.

`backward_cpu_ms` in raw event rows measures the CALLING thread, not the
autograd worker on which native backward runs. Do not use it as total backward
host work. The native backward CPU evidence comes from the worker's profiled
`_GroupedLinearBackward` scopes instead. Forward and backward event timing
do include all work: `backward()` returns after submission, and the final
CUDA event is synchronized before reading times/gradients.

No claim that72.2% of full distributed slowdown is explained. This local test
demonstrates a substantial host-submission-sensitive expert penalty with the
right dimensions, and improvement with more GPU work. It does NOT reproduce
dynamic routing, interlayer dependencies or concurrent model/gradient traffic.
The earlier full16K experiment still has a larger absolute ordinary-backward
gap than8K; its better iteration parity cannot be cited as confirming this
local hiding effect. Synthetic uniform splitting and a large artificial head
start are deliberate controls, not a production optimization. Sleep may affect
GPU state; unchanged kernel identities and non-improving GEMM service at8K
argue against faster hardware service explaining the elapsed gain.

Analysis: `scripts/nonuniform/analyze_expert_host_exposure.py <artifact-root>`.
It asserts correctness, headroom and identical kernel geometry and writes
`analysis.json`. No distributed jobs or production implementation edits were
needed for this causal isolation. No jobs remain from this experiment.
