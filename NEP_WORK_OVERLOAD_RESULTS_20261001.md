# Reduced-Replica Overload Results

Job3208136 completed0:0 in31m44s on GB300. All12 cases passed validation:
three configurations, two reversed-order repeats, each with an all-rank profile
and a separately approved profiler-off timing companion. No production changes.

## Result

Four latent-MoE layers,H4608/latent1152/F3072,E512,topk6,S1024,MBS1,
TP2/CP1/ETP1,DistOpt,HybridEP,full MoE graphs. Full-replica GA remains4.
Measured steps15-20 after14 warmup steps, all six samples retained per repeat.

| Configuration | EP | Full/second GA | Replica batches | GBS | Iteration ms | Owner parity | ktokens/s |
|---|---|---|---|---:|---:|---:|---:|
| Healthy H1 | 16/16 | 4/4 | 32/32 | 64 | 74.400 | 100% | 880.861 |
| Proportional N1 | 16/12 | 4/4 | 32/24 | 56 | 94.933 | 78.371% | 604.045 |
| Overloaded O | 16/12 | 4/5 | 32/30 | 62 | 102.608 | 72.509% | 618.744 |

Table uses profiler-off companions, not post-profile windows. Parity is healthy
rank0 iteration time divided by NEP owner-rank0 iteration time. Throughput uses
the actual summed replica batch, not the rank-local surrogate calculator GBS.

- Reduced work increases25%; aggregate tokens increase10.714%.
- Iteration increases7.675ms,or8.085%; token throughput increases2.433%.
- Repeat0 N1/O=94.917/102.817ms; throughput gain2.207%.
- Repeat1 N1/O=94.950/102.400ms; throughput gain2.659%.
- Within-repeat iteration SD:N1=0.256/0.259ms;O=0.422/0.424ms.
- Thus extra work is not free: there is a small throughput benefit at the cost
  of worse latency parity. This measures tokens/second, not convergence quality.

## Profiles And Interpretation

All ranks retain Nsight kernels and native AG/RS plus NEP GPU operation ranges,
payloads,process groups,host submissions and GPU completions.

| Profile mode | Healthy ms | N1 ms | O ms | O throughput change vs N1 |
|---|---:|---:|---:|---:|
| Off companion | 74.400 | 94.933 | 102.608 | +2.433% |
| On | 94.842 | 121.342 | 134.842 | -0.370% |

Instrumentation adds20.442/26.408/32.233ms to H1/N1/O respectively. Do not
subtract a universal correction, combine profile phases with unprofiled totals,
or present the profiled throughput result as undistorted performance.

Paired expert-RS GPU input-ready reduced-minus-full means,by repeat:
N1=+1.506/-4.677ms;O=+14.110/+14.473ms. Reduced arrives>0.2ms late for
91.7%/100% of O bucket/pair/step observations,versus38.9%/20.8% for N1.
Paired completion absolute differences average0.025-0.036ms; per-rank GPU/host
mapping error bounds are under10us. Completion alignment is a consistency
check,not an independent cross-node clock calibration. These profiled ranges
include dependencies and are not pure transfer times or unprofiled readiness.

Owner-cohort profiled phase means:

| Phase ms | H1 | N1 | O |
|---|---:|---:|---:|
| First forward | 17.459 | 23.045 | 23.218 |
| Other forwards | 14.816 | 16.580 | 16.324 |
| Nonfinal backwards | 18.488 | 21.160 | 20.493 |
| Final backward | 8.129 | 24.484 | 23.297 |
| Outside F/B | 36.172 | 36.332 | 51.758 |
| Iteration | 95.063 | 121.600 | 135.089 |

Under profiling the added owner delay appears outside F/B,consistent with
later reduced readiness being paid in the completion tail. This supports the
critical-path interpretation but is not an exact decomposition of the7.675ms
unprofiled latency increase. N1 does not have universally early reduced arrival.

## Controls

- Full nodes theia0110/0112/0113/0114 in nvlblk23; second nodes
  theia0145/0147/0149/0155 in nvlblk25. Segment4; cliques1000/1001.
- All16 healthy and12 NEP cross-replica pairs audited as RDMA. Local NEP
  transfer groups audited as P2P. Both repeats on the same GB300 allocation.
- Canonical initialization,route-row statistics,native AG/RS payloads and
  bucket layouts match within topology. Dense buckets1/1,expert2/3(H/NEP);
  unchanged across GA,not an iso-expert-bucket experiment.
- Exact-balanced randomized512-token route units,seed101. Routing controls
  statistics,not bitwise sample/route association or equal training trajectories.
- Max allocated memory30.117GiB NEP,11.879GiB healthy; every rank passes
  existing free/reserved headroom gates. All25steps/case have zero skips/NaNs.
- Runtime at0c8ab3e39 unchanged from the prior study. Only overload launcher
  case,validator GA expectation,and offline analyzer extended. Archived runner
  and per-case implementation diffs capture benchmark changes.

Artifacts:
/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3208136/
including work-allocation-summary.json,work-controls-audit.json and per-case
profiles,logs,memory,initialization,routing,transport and validation records.

Do not compare absolute GB300 timings with the previous GB200 study to claim
an overload effect; the fresh within-allocation N1/H1 controls establish it.
