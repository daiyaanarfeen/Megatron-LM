# Heterogeneous Expert Parallelism — Implementation Record

## Overview

This document records the complete implementation of heterogeneous expert parallelism (het EP) gradient synchronization in Megatron-LM. The work spans three approaches (A, B, C) for syncing expert gradients across MoE replicas with different EP sizes, plus an interleaved expert placement algorithm that removes constraints on EP size ratios.

**Branch**: `heterogeneous-rank-generator`
**Key files**:
- `megatron/core/parallel_state.py` — group creation, NVSHMEM init, expert placement
- `megatron/core/distributed/pipelined_reshard_collective.py` — Approach B (NVSHMEM pipeline)
- `megatron/core/distributed/param_and_grad_buffer.py` — Approach A & C (NCCL-based)
- `megatron/core/distributed/distributed_data_parallel.py` — DDP wiring
- `megatron/core/distributed/distributed_data_parallel_config.py` — config flags
- `megatron/core/transformer/moe/moe_layer.py` — expert index assignment
- `megatron/core/transformer/moe/token_dispatcher.py` — non-contiguous index support

---

## Problem Statement

MoE training with heterogeneous replicas: different replicas have different numbers of tp*cp units (k_i), giving different EP sizes. Expert gradient buffers differ in size across replicas, so a standard allreduce fails. Solution: reshard → allreduce → unreshard (NTP pattern).

**Running example**: k=[2,4], tp=2, cp=1, etp=2, num_experts=8, 12 GPUs
- Replica 0 (4 ranks): ep=2, each rank holds 4 experts
- Replica 1 (8 ranks): ep=4, each rank holds 2 experts
- min_ep=2: after resharding, the reshard replica's leaders each hold 4 experts (matching replica 0)

---

## Approach A — NCCL-Only (Baseline)

**Implementation**: `param_and_grad_buffer.py:_start_heterogeneous_ep_grad_sync()`

Per bucket, three NCCL collectives:
1. `all_gather` on ep_group — gathers all ep ranks' grads to leaders
2. `all_reduce` on edp_group — cross-replica sync
3. `all_reduce(SUM)` on ep_group — scatter back (leader keeps data, others have zeros)

**Verification**: Unit tests fill grads with 1.0, sync, check result = num_replicas/dp_size. Cross-rank consistency test verifies matching grads across replicas.

**Test files**: `tests/unit_tests/distributed/test_heterogeneous_ep_grad_sync.py`
**Job scripts**: `run_het_ep_tests_12gpu.sh`, `run_het_ep_nvshmem_12gpu.sh`

---

## Approach B — NVSHMEM Pipeline

### Architecture

Completely removes NCCL from the gradient sync loop. All communication via NVSHMEM put/signal_op/signal_wait. Three stages per chunk:
1. **Gather**: NVSHMEM put from ep peers to leaders
2. **Ring allreduce**: NVSHMEM-based ring reduce-scatter + allgather across replicas
3. **Scatter**: NVSHMEM put from leaders back to ep peers

### Multi-Leader Design

`num_leaders = min_ep_size`. Each leader gathers from its sub-group, ring-allreduces with cross-replica leaders, scatters back. For k=[2,4] with min_ep=2: 2 leaders, each handling half the experts.

**Key config fields** (in `_HETEROGENEOUS_EP_CONFIG`):
- `is_b_leader`: True for ep_rank < min_ep
- `expert_placement`: interleaved expert-to-rank assignment
- `expert_gather_map`: routing map for per-expert gather/scatter
- `b_edp_peer_pes`: ring peer PEs for cross-replica allreduce

### NVSHMEM Buffer Allocation (parallel_state.py)

All allocated collectively on the symmetric heap during `initialize_heterogeneous_model_parallel()`:

| Buffer | Size | Purpose |
|--------|------|---------|
| `gather_slots[2][max_ep]` | chunk_size each | Double-buffered per-parity gather destinations |
| `local_slots[2]` | chunk_size each | Double-buffered scatter destinations + gather staging |
| `exchange_bufs[2]` | chunk_size/N each | Double-buffered ring exchange |
| `gather_signals[max_ep]` | 8 bytes each | Per-peer gather data-ready signals |
| `scatter_signal` | 8 bytes | Scatter data-ready signal |
| `gather_states[2]` | 8 bytes each | Epoch handshake for gather slot reuse |
| `scatter_states[2]` | 8 bytes each | Cumulative ADD ack for scatter slot reuse |
| `exchange_signals[2]` | 8 bytes each | Ring step data signals |
| `exchange_acks[2]` | 8 bytes each | Ring buffer reuse acks |

**Slot size**: configurable via `MEGATRON_NVSHMEM_SLOT_MB` env var (default 32MB). Larger slots reduce K (number of chunks) and improve throughput. 256MB recommended for production.

**Signal zeroing**: `nvshmem.core.buffer(8)` does NOT zero memory. All signals zeroed via put-to-self at init using a zeroed bytetensor.

### Signal Protocol

**Gather handshake** (double-buffered by parity):
- DATA READY (many→one): per-peer `gather_signals[sub_rank]` with SIGNAL_SET. Leader waits on each peer's signal.
- SLOT FREE (one→many): `gather_states[parity]` with SIGNAL_SET epoch values. Leader sets after assembly, peers wait before next put.

**Scatter handshake** (double-buffered by parity):
- DATA READY (one→many): `scatter_signal` per peer with SIGNAL_SET. Leader sets after put, peer waits.
- SLOT FREE (many→one): `scatter_states[parity]` with SIGNAL_ADD. Each peer adds 1 after copy-out. Leader waits for >= epoch * num_peers.

**Ring exchange**:
- Data signals: `exchange_signals[parity]` with SIGNAL_SET per step.
- Reuse acks: `exchange_acks[parity]` with SIGNAL_SET. Receiver acks after reading, sender waits before next same-parity put. Fixes N>2 race where step s+2 overwrites exchange_buf before step s reads it.

**Persistent signal_base**: stored in `het_ep_config['_signal_base']` to survive across model re-creations (PipelinedReshardCollective instances). Prevents stale signal values from previous training iterations.

### Ring Allreduce (`_ring_allreduce`)

GPU-pipelined ring reduce-scatter + allgather:
- **No per-step quiet()**: all operations enqueued on nv_stream without host blocking
- **Double-buffered send staging**: `local_slots[0]` and `local_slots[1]` alternate per step
- **Double-buffered exchange**: `exchange_bufs[0]` and `exchange_bufs[1]` alternate per step
- **Ack protocol**: prevents N>2 buffer reuse race
- **ring_signal_base parameter**: ensures all ring members use matching signal values regardless of pre-ring work (gather phase)
- One `quiet()` at the end of the ring

Sub-chunk handling: data split into N sub-chunks with remainder distributed to first sub-chunks.

### Per-Expert Ring Pipeline (`_execute_interleaved`)

For interleaved expert placement. All ring members process experts in the same order (0..experts_per_leader-1):
- **Locally-held experts**: ring proceeds immediately
- **Offloaded experts**: wait for that expert's gather signal, then ring
- **Followers**: batch all gather puts upfront. Scatter waits after.

The gather of expert E+1 overlaps with ring of expert E because followers send early.

Signal values: `epoch * MAX_ROUTES + route_idx` for gather, `epoch * MAX_ROUTES + local_idx` for scatter. `epoch * 100000 + expert_idx * (ring_steps + 1)` for ring.

### Host Sync Elimination

Original code had `torch.cuda.synchronize()` and `nv_torch.synchronize()` in the pipeline loop — 25-35% slowdown. Replaced with CUDA event-based stream dependencies:
- Copy-in: `ev_grad_ready.record(default_stream)` → `nv_torch.wait_event(ev_grad_ready)` → copy on nv_stream
- Copy-out: `ev_scatter_done.record(nv_torch)` → `default_stream.wait_event(ev_scatter_done)` → copy on default stream

### Chunk-Based Pipeline (`_execute_chunked`)

For topologies where ep % min_ep == 0 (integer ratio sub-groups). Processes K fixed-size chunks through gather → ring → scatter. Non-leaders interleave gather puts with scatter waits (depth-2 pipeline). Uses all the double-buffered handshake infrastructure.

**Per-chunk math**:
```
slot_elems = chunk_size / element_size
max_ratio = max_ep / min_ep
effective_ar_chunk = (slot_elems // max_ratio) * max_ratio
per_member_elems = effective_ar_chunk // ratio
K = ceil(per_rank_numel / per_member_elems)
```

### Benchmarks (12 GPUs, k=[2,4], 256MB slot)

| hidden | A (NCCL) | B (NVSHMEM) | B/A |
|--------|----------|-------------|-----|
| 1024 | 1.02ms | 1.45ms | 0.71x |
| 2048 | 3.43ms | 1.87ms | 1.84x |
| 4096 | 12.29ms | 7.22ms | 1.70x |
| 8192 | 47.39ms | 28.74ms | 1.65x |

### Benchmarks (28 GPUs, k=[4,4,6], 256MB slot, per-expert pipeline)

| hidden | A (NCCL) | B (NVSHMEM) | B/A |
|--------|----------|-------------|-----|
| 1024 | 1.62ms | 2.70ms | 0.60x |
| 2048 | 5.71ms | 2.85ms | 2.00x |
| 4096 | 21.44ms | 3.71ms | 5.78x |

---

## Approach C — Phased NCCL with all_to_all

**Implementation**: `param_and_grad_buffer.py` methods `_setup_phased_splits`, `_start_phased_gather`, `_start_phased_allreduce`, `_finish_phased_scatter_all`

**Design**: Separates the three stages for optimal overlap:
1. **Gather** (per bucket, during backward): `torch.distributed.all_to_all_single` on ep_group
2. **Allreduce** (per bucket, after gather): `torch.distributed.all_reduce` on edp_group
3. **Scatter** (ALL buckets, in finish_grad_sync): reverse `all_to_all_single` on ep_group

**Split sizes**: precomputed at init from `expert_placement`. For each (src_rank, dst_rank) pair: count how many expert params src sends to dst based on which experts belong to which leader's range.

**Key detail**: follower ranks with recv_total=0 must create zero-size output tensors (not size-1), otherwise `all_to_all_single` fails with "Split sizes doesn't match total dim 0 size".

**Config flag**: `use_phased_ep_reshard=True` in `DistributedDataParallelConfig`

**Verification**: 12 GPUs k=[2,4] — all three approaches produce bitwise-identical losses over 10 training steps.

---

## Interleaved Expert Placement

### Problem

The original sub-group model required `local_ep % min_ep == 0` (integer ratio). This excluded topologies like k=[4,4,6] where ep=6 and min_ep=4 (ratio=1.5).

### Solution: NTP-Style Interleaved Assignment

`compute_expert_placement(num_experts, local_ep_size, min_ep_size)` in `parallel_state.py`:

1. **Leaders** (ep_rank 0..min_ep-1): keep first `E/ep` experts from their contiguous range [l*E/min_ep, (l+1)*E/min_ep)
2. **Offloaded experts**: remaining experts distributed round-robin across followers
3. After gather, each leader has the same contiguous expert range as the min-ep replica

**Example** (ep=6, min_ep=4, E=12):
```
Leader 0: keeps [0,1], offloads expert 2 → follower 0
Leader 1: keeps [3,4], offloads expert 5 → follower 1
Leader 2: keeps [6,7], offloads expert 8 → follower 0
Leader 3: keeps [9,10], offloads expert 11 → follower 1
Follower 0 (ep_rank 4): experts [2, 8]
Follower 1 (ep_rank 5): experts [5, 11]
```

### Token Dispatcher Changes

`token_dispatcher.py` assumed contiguous expert indices. Fixed:
1. **Removed contiguity assert** (lines 392-395)
2. **Replaced slice indexing** with `torch.index_select` for `local_map`, `local_probs`, `num_global_tokens_per_local_expert`
3. **Fixed `input_splits`**: uses `scatter_add_` with explicit `expert_to_ep_rank` mapping instead of reshape-based grouping
4. **Sort/restore permutation**: builds explicitly for non-contiguous indices

`get_expert_to_ep_rank_map()` in `parallel_state.py` returns the global expert→ep_rank mapping.

### MoE Layer Changes

`moe_layer.py` line ~123: checks `parallel_state.get_heterogeneous_ep_config()` for `local_expert_indices`. If present, uses it instead of contiguous `ep_rank * num_local_experts + i`.

---

## Testing Infrastructure

### Unit Tests

`tests/unit_tests/distributed/test_heterogeneous_ep_grad_sync.py`:
- Fill grads with 1.0, sync, check result = num_replicas/dp_size
- Cross-rank consistency: compare rank 0's experts with reshard rank's experts
- Topologies: 8 GPU (k=[1,3]), 12 GPU (k=[2,4]), 24 GPU (k=[4,8])
- Both Approach A and B tested

### Integration Training Tests

`tests/unit_tests/distributed/test_heterogeneous_ep_training.py`:
- Creates MoE model with DDP, runs forward + backward + SGD step for N iterations
- Compares losses across approaches A, B, C — must match exactly
- Topologies: 12, 24, 28, 32 GPUs
- All three approaches produce bitwise-identical results

### Benchmark

`tests/unit_tests/distributed/bench_heterogeneous_ep_grad_sync.py`:
- Measures pure grad sync latency (no backward overlap)
- Sweeps hidden sizes, reports avg/p50/min times
- Supports `--hidden` and `--bucket-sizes` args
- `MEGATRON_NVSHMEM_SLOT_MB` env var controls slot size

### Standalone Diagnostics

Various `check_*.py` scripts in the repo root:
- `check_ring.py`: standalone NVSHMEM ring allreduce test
- `check_placement.py`: expert placement + forward/backward + grad sync test
- `check_put_signal.py`: NVSHMEM put_signal/signal_wait verification
- `check_combined_exact.py`: combined NVSHMEM+NCCL pattern test

### Job Scripts

- `run_het_ep_nvshmem_12gpu.sh`: correctness tests + benchmark (12 GPU)
- `run_integration_12gpu.sh`, `run_integration_28gpu.sh`, `run_integration_32gpu.sh`: training integration
- `run_bench_28gpu.sh`, `run_bench_nvshmem_12gpu.sh`: benchmarks
- `run_bench_slot_sweep.sh`: slot size sweep
- `run_check_placement.sh`, `run_check_ring.sh`: standalone tests

---

## Key Bugs Found and Fixed

### NVSHMEM Signal Buffers Not Zeroed
`nvshmem.core.buffer(8)` allocates but does NOT zero memory. Caused stale signal values that made `signal_wait` return immediately (reading garbage data). Fix: zero all signals via put-to-self at init.

### N>2 Ring Exchange Buffer Race
With 2 exchange buffers (ping-pong), steps 0 and 2 share the same buffer. For N>2, the remote's step 2 put could overwrite data before the local step 0's add_ finishes reading. Fix: ack-based protocol — receiver signals sender after reading, sender waits before reuse.

### Ring Timing Mismatch (Interleaved Path)
Min-ep ranks entered the ring immediately while reshard leaders were still gathering. The ack protocol caused a deadlock because min-ep ranks waited for acks from reshard leaders who hadn't entered the ring yet. Fix: per-expert ring pipeline where all ring members process the same expert at each step.

### Per-Step quiet() Serialization
Each ring step had a `quiet()` that blocked the host until the put completed. With small per-expert chunks, this made the ring latency-bound. Fix: remove per-step quiet, use double-buffered send staging, one quiet at the end.

### Signal Base Divergence
Leaders and non-leaders incremented `_signal_base` differently due to gather/ring/scatter phases having different signal counts. Fix: epoch-based signal values computed deterministically from the static routing map, separate `ring_signal_base` parameter.

### Exchange Buffer Size for N=3
`chunk_size / 3` = 89478485.33 bytes — not divisible by 4 (float32 alignment). Fix: ceil division + 8-byte alignment.

### all_to_all Zero-Size Output
Followers with recv_total=0 created output tensors of size 1 (via `max(0, 1)`), but `all_to_all_single` requires `sum(output_split_sizes) == output.numel()`. Fix: create truly empty tensors (size 0).

---

## Configuration Reference

### DDP Config Flags

| Flag | Default | Description |
|------|---------|-------------|
| `use_pipelined_ep_reshard` | False | Enable Approach B (NVSHMEM pipeline) |
| `num_ep_reshard_pipeline_chunks` | 4 | K for chunk-based pipeline (Approach B) |
| `use_phased_ep_reshard` | False | Enable Approach C (phased all_to_all) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEGATRON_NVSHMEM_SLOT_MB` | 32 | NVSHMEM slot size in MB. Larger = fewer chunks = better throughput. 256 recommended. |
| `NVSHMEM_DISABLE_NVLS` | (unset) | Set to 1 to work around NVLS transport bug |
| `NVSHMEM_MAX_TEAMS` | (unset) | Set to 512 for team operations |
| `NCCL_NVLS_ENABLE` | (unset) | Set to 0 to disable NCCL NVLS |

### Topologies Tested

| GPUs | k | Replicas | min_ep | max_ep | Ring N | Notes |
|------|---|----------|--------|--------|--------|-------|
| 12 | [2,4] | 2 | 2 | 4 | 2 | Primary dev/test topology |
| 28 | [4,4,6] | 3 | 4 | 6 | 3 | Non-integer ratio, interleaved placement |
| 32 | [4,4,8] | 3 | 4 | 8 | 3 | Integer ratio, 3 replicas |

---

## Future Work / Known Limitations

### Performance Optimizations
- **Symmetric grad buffer**: allocate expert grad buffer on NVSHMEM symmetric heap to eliminate copy-in/copy-out. Estimated ~5-7% improvement.
- **Persistent/device-initiated kernels**: GPU-driven ring loop without host involvement. Eliminates all per-step Python overhead. Estimated ~10% at moderate K.
- **Batch locally-held experts**: ring-allreduce contiguous local experts in one call instead of per-expert. Reduces ring call count.
- **Auto-tune slot size**: set `MEGATRON_NVSHMEM_SLOT_MB` based on expert buffer size to keep K in optimal range (1-4).

### Approach C Overlap
- Current implementation is synchronous (`overlap_grad_reduce=False`). For full benefit, need to wire gather into backward hooks so it fires per-bucket as grads become ready.
- The allreduce should overlap across buckets on a communication stream.
- Scatter batched in `finish_grad_sync` after all allreduces.

### Approach B Chunk-Based + Interleaved
- The chunk-based pipeline (`_execute_chunked`) still uses the old sub-group model. It works for ep % min_ep == 0 topologies. The per-expert pipeline (`_execute_interleaved`) handles all cases but is slower at small model sizes due to per-expert ring overhead.
- Could unify the two paths or add expert batching to the interleaved path.

### `execute_ep1` Path
- Handles ep=1 ranks (min_ep=1). Not tested since min_ep=1 isn't a target config. Could be removed — ep=1 ranks can go through `execute()` with ratio=1.

### NVSHMEM Team Operations
- `team_split_strided` crashes on the test cluster. Workaround: use individual signal_op/signal_wait instead of team barriers. If teams are fixed, could use `nvshmem.core.allreduce` on leader teams.
