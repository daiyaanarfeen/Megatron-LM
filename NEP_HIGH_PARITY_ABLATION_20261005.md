# High-Parity Compute-Window Sensitivity

User-approved priority: run the experiments most likely to change parity by
more than 2 percentage points (3+ is a strong effect). Do not spend another
full sweep establishing sub-point scheduling effects first.

Start from the previously 97.203% companion-parity a8b case: GB300, EP16/12,
TP2/CP1, H4608/latent1152/expert F3072, 512 experts/topk6, 15 MoE layers,
MBS2/GA8, sequence8192, DistOpt/HybridEP/native graphs, early redistribution,
depth1 lookahead, one delivery lane. Healthy EP16/16. Replica local batches
128/96 NEP and128/128 healthy; true GBS224 and256 respectively.

Only independent workload setting changed: sequence8192/4096/2048. Keep
parameter buckets, actual parameter bytes, expert identity, routing algorithm,
placement, optimizer, graph mode, scheduling and work-allocation ratio fixed.
Token-dispatch bytes and compute shapes necessarily change. This is a workload
sensitivity, not a perfectly isolated causal compute-only intervention.

No production implementation edits. Separate branch from11e2388c6. Only add
sequence2048 to the harness and counterbalance cases. Largest-memory8192 NEP
profile runs first. Reuse existing numerical and fail-closed training gates.
Eight GB300 nodes, segment4, each replica inside its own NVLink domain;
cross-replica RDMA verified. Do not substitute GB200 and pool results.

Three repeats, six cells, each with all-rank Nsight/custom AG/RS/Gather/delivery
profiling and a separate previously authorized timing companion:36cases.
35iterations, matched25-30 after24warmup. Orders:
- r0: N8K,H8K,N4K,H4K,N2K,H2K; profile then companion.
- r1: H2K,N2K,H4K,N4K,H8K,N8K; companion then profile.
- r2: N4K,H4K,H8K,N8K,N2K,H2K; profile then companion.

Primary parity is healthy owner iteration / NEP full-owner iteration for the
same sequence,repeat,and measurement mode. Report every repeat, mean and range;
do not count ranks or adjacent steps as independent experimental repeats.
Call >2pp consistently across repeats meaningful; retain smaller effects but
do not describe them as explaining high parity. Three repeats do not establish
population-level significance across allocations.

Measure absolute first-forward demand-to-ready waiting, native AG dependency
prefix, redistribution exposure and native compute/model-EP changes. Report
full phase excesses including final backward and optimizer/tail. A lower
parity with unchanged absolute overhead is amortization, not evidence of lost
hiding. Flag phase speedups or instrumentation-induced reversals rather than
discarding inconvenient iterations. No post-profile timing substitution.

All outputs and analysis on Lustre; final plots published to Talos. Poll
queued/running job every300s and stop on validated hangs or audit failures.
