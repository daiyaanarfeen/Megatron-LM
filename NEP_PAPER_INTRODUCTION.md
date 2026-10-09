# Nonuniform Expert Parallelism for Efficient MoE Training

## 1. Introduction

Mixture-of-experts (MoE) models separate model capacity from the computation
performed on each token: a model may contain hundreds of experts while
activating only a small subset for each input. Expert parallelism distributes
these experts across GPUs, while replicated training requires corresponding
expert parameters to remain synchronized. Conventional uniform layouts couple
these two tasks: replicas use the same expert-parallel degree, and corresponding
ranks hold matching expert partitions. This arrangement simplifies training,
but restricts how a job can use uneven GPU allocations. If one replica has
fewer available GPUs, retaining a uniform layout can require giving up usable
resources elsewhere or reorganizing the entire job.

We study a different question: **can replicas with different expert-parallel
degrees train the same MoE model while preserving the iteration latency of a
full-sized replica?** Such a capability could allow training systems to use
uneven allocations or retain more resources after a capacity reduction. The
challenge is not simply to place more experts on fewer GPUs. Each replica must
still execute the same logical model, synchronize matching gradients, and
deliver updated parameters to the ranks that execute them. Moreover, the
reduced replica must receive an appropriate share of training work; otherwise,
its progress determines the iteration time of every replica.

MoE introduces constraints that make this problem more than an application of
tensor resharding. Experts have distinct identities and token-dependent
execution, whereas synchronization buffers and optimizer shards are organized
by memory layout. With different expert-parallel degrees, identical local
offsets need not represent the same expert. Expert counts may also be
nondivisible by one replica's degree, while fused dispatchers expect regular
rank-and-slot layouts. Finally, a sharded optimizer creates a second placement
problem: the ranks that own optimizer state need not be the ranks that execute
the corresponding experts. A correct solution must reconcile all three
layouts without changing the router's expert set or replacing optimized
dispatch and optimizer machinery.

We present **nonuniform expert parallelism (NEP)**, which separates physical
expert placement from logical synchronization and optimizer ownership.
Physical placement assigns each real expert to a GPU within each replica.
A canonical owner layout identifies matching expert parameters across
replicas, independently of their physical locations. Gradients are gathered
into this layout before native cross-replica synchronization. With a
distributed optimizer, native reduce-scatter, sharded updates, and parameter
all-gather operate on owner buffers; updated parameters are then redistributed
to their physical execution locations. This design builds on the general
reshard--synchronize principle of nonuniform parallelism rather than proposing
a new collective or optimizer algorithm.

NEP preserves the true expert set even when expert counts are nondivisible.
It assigns unequal numbers of real experts to ranks and maps logical expert
identities into a padded dispatch-slot representation. Empty slots satisfy
backend layout requirements without introducing trainable dummy experts or
changing routing probabilities over real experts. We integrate this mapping
with Megatron's Flex/HybridEP dispatcher and connect owner buffers to native
gradient synchronization and distributed-optimizer machinery. The resulting
implementation supports the combined use of nondivisible expert placement,
fused dispatch, sharded optimization, and native CUDA graphs.

Correctness alone does not make nonuniform execution efficient. NEP adds
gradient gathering and parameter redistribution to an already communication-
intensive training schedule. We therefore organize transfers around gradient
readiness and parameter-consumption dependencies, using native parameter
prefetch to make updated experts available before their forward computation.
Our evaluation asks which placement and scheduling choices actually help,
when reduced replicas become stragglers, and how workload shape determines
whether additional communication is hidden or merely amortized. These are
distinct effects: an unchanged exposed startup cost can become inexpensive
relative to a longer iteration without becoming any more overlapped.

Experiments on GB200 and GB300 systems demonstrate near-healthy iteration
latency for representative configurations while retaining fused dispatch and
distributed optimization. In an EP16/EP12 comparison against uniform EP16/EP16,
a 512-expert workload reaches **97.35% owner iteration parity** with proportional
replica-local batches, microbatch size two, and two gradient-accumulation steps.
Reducing sequence length from 24K to 8K lowers parity to **91.34%**: first-forward
and final-backward penalties remain similar in absolute duration but occupy a
larger fraction of the shorter iteration. A complementary expert-count sweep
increases synchronization payload: at two accumulation steps, increasing
experts from 128 to 384 reduces parity from **97.55% to 95.73%**; at eight steps,
the corresponding endpoint change is only **98.97% to 98.70%**. These results
identify workload-dependent limits rather than a universal constant overhead.
They also show why neither increasing transfer concurrency nor changing expert
placement alone is sufficient to explain high parity.

Throughout, iteration parity means healthy iteration time divided by NEP
full-replica owner iteration time. It is not a claim of equal aggregate token
throughput with fewer GPUs: proportional work allocation reduces the global
batch processed by the smaller allocation. We evaluate useful-token
throughput separately and distinguish profiled execution from matched timing
companions. Our broader workload evaluation achieves **[X--Y%] iteration
parity across [N] configurations**, with **[Z%] normalized per-GPU throughput**.

In summary, this paper makes three contributions:

1. **Expert-aware nonuniform execution and synchronization.** We decouple
   physical expert placement from canonical gradient and optimizer ownership,
   allowing replicas with different expert-parallel degrees to train the same
   logical model, including nondivisible expert counts.
2. **A fused-dispatch and sharded-optimizer-compatible system.** We reconcile
   irregular real-expert placement with regular dispatch slots and native
   optimizer buffers, and schedule gradient gathering and parameter delivery
   through existing training dependencies rather than replacing the underlying
   dispatcher or optimizer.
3. **An empirical characterization of NEP's efficiency envelope.** Controlled
   placement, transfer-scheduling, work-allocation, microbatch, sequence-length,
   and expert-count ablations distinguish replica imbalance, communication
   hiding, and amortization, explaining when NEP approaches healthy iteration
   latency and when its overhead becomes significant.

---

## Draft Notes (Not Part of the Introduction)

- Bracketed aggregate results are placeholders, not measured claims. Define
  the normalized-throughput denominator when consolidating the workload table.
- Add verified MoE, Megatron, fused-dispatch, and nonuniform tensor-parallelism
  citations before submission. Do not claim novelty for reshard/synchronize
  itself or describe this implementation as a demonstrated failure-recovery
  system; the experiments concern steady-state nonuniform training.
- Quoted parities use authorized timing companions on matched measured
  iterations, after warmup. All-rank profiles are available separately.
- Sequence results: jobs3274487,3275099,3275392; see
  `.worktrees/nep-seq32k-ga2-20261008/NEP_SEQ8K_GA2_20261008.md`.
- Expert-count results: jobs3265011,3265012,3263820; see
  `.worktrees/nep-expert-count-ga-20261007/NEP_EXPERT_COUNT_GA_20261007.md`.
  The expert-count and sequence studies use different controlled
  configurations; do not combine them into one compute/payload scaling curve.
- The expert-count companion endpoint loss at GA2 is1.82 percentage points;
  it is weaker than the sequence-length effect. Intermediate GA4/GA8 points
  are not monotonic and must remain visible in the evaluation.
- Do not claim complete causal attribution of the sequence effect. An
  additional outside-forward/backward gap remains unresolved. Parameter
  payload is held fixed in the sequence sweep, but token communication and
  compute shapes change; expert count also changes tokens per expert.
- Scope excludes claims of FSDP, checkpoint, arbitrary Flex-backend support,
  convergence equivalence across different batch sizes, or universal parity.
