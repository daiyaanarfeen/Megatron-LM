# Actual-Workload Ablations

Start with the saved high-parity a8b latent workload from3120733, not the toy:
15 MoE layers in MEMEMEM*EMEMEMEMEM*EMEMEMEME/*E/*E, H4608/latent1152/F3072,
E512/topk6,sharedF6144,S8192,vocab131072,MBS2,GA8,TP2/CP1/ETP1/PP1.
This is the previously memory-limited depth, not the original full-depth120B model.
Retain attention,Mamba,MTP,native mamba/attn/moe graphs,capacity1,
Flex/HybridEP,DistOpt,DDP request8 and native model/optimizer arguments.
HealthyEP16/16 versus NEP16/12, local128/128 versus128/96, GBS256/224.
The restored gate's actual bucket counts are dense3/3 and expert5/3
(healthy/NEP),not the historical5/8 expert layout. Retained DDP request8 does
not imply identical expert grouping across runtime versions. Freeze and verify
these actual layouts within this ablation; do not claim an iso-bucket comparison.

Current production runtime is unchanged from the controlled placement study.
Use its canonical expert initialization and randomized exact-balanced route
controls in every case. This routing/runtime is not identical to the historical
run; establish fresh healthy/proportional parity before policy conclusions.

## Matrix

Placement A-F: compact ascending1, compact disjoint-first1, compact ascending2,
compact disjoint-first2, spread ascending1, spread ascending2; same useful bytes,
preallocated resources, fixed backward owner order, routing and native buckets.
Fresh healthy H in each comparison block.

Work allocation uses compact disjoint-first1:
B=MBS2/GA8:8; R=MBS2/GA8:6; O=MBS2/GA8:10;
S=MBS1/GA16:16 with its own healthy H1=MBS1/GA16:16.
Full local batch128 throughout; reduced96/72/120/96. TrueGBS224/200/248/224.
Healthy GBS256. Report latency parity separately from actual token throughput.

## Gates And Protocol

- Begin O/H/B on the real workload with profiles and timing companions. O has
  the largest reduced schedule; GA does not retain all microbatches' activations
  at once with PP1. Two transfer lanes are preallocated in every policy.
- 35iterations; measured25-30 after24 warmup steps. All-rank Nsight plus native
  AG/RS,Gather,parameter-transfer ranges,PGs,payloads,submission/completion.
  Keep separately approved profiler-off companions, no post-window substitution.
- Require existing per-rank memory headroom gates, exact initialization/routing/
  bucket checks, pairwise transfer-split agreement and fixed native AG/RS payloads.
- Segment4, whole replica within NVLink domain, separate replica cliques and
  audited RDMA EDP. Both GB200 and GB300 eligible; only within-hardware matched
  controls accepted. No overlapping jobs in an allocation.
- After the gate, counterbalanced repeats of each policy and batch cell with
  healthy/proportional anchors. Stop on a new confounder; never silently shrink
  depth,sequence length,expert dimensions or memory gates to get a pass.
- All logs,traces,validation and large generated output live on Lustre.

This first matrix covers a8b, not broad cross-model generalization. Do not imply
the existing smaller a3b/nemotron workloads have been retested by this study.
