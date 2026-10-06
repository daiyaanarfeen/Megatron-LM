# Gather-Only CTA Cap

Parent `nep-norm-headroom-20261006` at be5dcb32f. No training runtime edits.

User approved trying the proposed16-CTA cap on `nep_owner_gather` only.
Native `--nccl-communicator-config-path` loads the two-line YAML; no global
NCCL resource variables, native RS/AG settings, or scheduling changes.

## Controls

- Healthy EP16/16; reference/capped NEP EP16/12; TP2, CP1, ETP1.
- Saved memory-limited a8b15-MoE hybrid/MTP, H4608/latent1152/F3072,
  E512/topk6, MBS2/GA8/S8192. Healthy localBS128/128, NEP128/96.
- DistOpt, Flex/HybridEP, native MoE graphs, exact randomized balanced routing.
- Five expert buckets, three dense buckets, norm margins forward16/backward32,
  compact placement, ascending transfer order, one stream, prefetch1.
- Eight GB300 nodes, segment4, each replica in one NVL domain, cross-replica
  optimizer transport audited as RDMA. GB200 failed prior memory-headroom gate.
- Repeat0 GBASE/HGC/G16, profile then timing companion; repeat1 reverse order,
  companion then profile. Twelve cases,35iterations, matched measured25-30.
- Preserve all-rank AG/RS/Gather/parameter redistribution profiles, payloads,
  host submission and GPU completion. Do not substitute post-profile timing.

## Acceptance

Verify actual Gather kernel grids shrink from32 to at most16 while norm grids
stay360, and unrelated process-group settings/payloads remain unchanged.
Compare owner iteration times, phase overheads, norm admission gaps, propagated
model-EP/TP waits and Gather residency. Residency includes peer waits and
queueing, not just transfer service. Reject a local improvement that worsens
absolute iteration time. Preserve all repeat results, including anomalies.

Gather16 may still exceed the32-SM headroom when both expert and dense RS run.
Do not change native RS caps without consulting the user. No default cap is
promoted unless evidence supports it; original branches remain rollback points.
