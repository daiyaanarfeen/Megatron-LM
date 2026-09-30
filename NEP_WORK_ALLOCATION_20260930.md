# Work Allocation Study

Approved five-case EP16/12 experiment, isolated from the completed placement study.
Runtime is unchanged from 4142d0bcb. Only benchmark harness/control code changes.

Four latent MoE layers: H4608, latent1152, F3072, E512, topk6, S1024, TP2/CP1,
DistOpt, HybridEP, full MoE graphs, compact/waves/one transfer lane.
Healthy EP16/16 has32 GPUs; NEP EP16/12 has28 active GPUs. Segment4 and replica
software cliques require within-domain replicas and cross-replica RDMA.

| Cell | Mode | MBS | GA full/reduced | Replica batches | True GBS |
|---|---|---|---|---|---|
| H1 | healthy | 1 | 4/4 | 32/32 | 64 |
| H2 | healthy | 2 | 2/2 | 32/32 | 64 |
| N1 | NEP | 1 | 4/4 | 32/24 | 56 |
| N2 | NEP | 2 | 2/2 | 32/24 | 56 |
| R | NEP | 1 | 4/3 | 32/18 | 50 |

Memory gate: run N2 then H2 with all-rank profiles and25 iterations, including
optimizer-state initialization and graph capture. Require more than16GiB beyond
peak reserved allocator memory and8GiB end-of-run free memory on every rank.
An OOM/hang/control mismatch stops the job before remaining cases. This reduces
risk; it is not a guarantee about future allocator/driver behavior.

MBS1/2 route maps repeat the same512-token exact-balanced randomized unit, so
rebatching does not change per-token expert choices. Initialization remains
canonical by global expert ID, including the native healthy DDP control.
Rank-local surrogate GBS selects GA; existing calculate-per-token-loss handles
actual global token weighting. Report true GBS, not surrogate calculator GBS.

Profiling stays enabled; approved separate profiler-off companions provide
performance timing. Measured steps15-20, counterbalanced repeats, no post-window
substitution. Validate all-rank native AG/RS and NEP transfer completions.
Report owner parity and aggregate tokens/sec, not parity alone for R.

## Memory Gate

Job3201156 completed0:0 in9m44s on GB200. Full replica nvlblk07, reduced nvlblk16.
N2 then H2 completed25iterations each and passed all-rank checks. Maximum peak
allocated memory30.136GiB NEP and11.897GiB healthy; every rank passes the headroom
gate. Profiled matched rank0 means89.383/69.583ms are a single-order preflight,
not the final counterbalanced performance result. No production code changes.
