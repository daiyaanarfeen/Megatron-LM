# Nonuniform Expert Parallelism

This directory contains opt-in training entrypoints for nonuniform expert parallelism (NEP).
NEP allows expert-data-parallel replicas to use different expert-parallel sizes while preserving
one logical set of model experts.

## Public API

The highest-level library API is
[`NonuniformEPDistributedDataParallel`](../../megatron/core/distributed/nonuniform_ep.py), a
subclass of Megatron's native `DistributedDataParallel`. `NonuniformEPConfig` carries its NEP
configuration, and `initialize_nonuniform_ep_process_groups_from_args()` initializes the topology
before model construction.

The example entrypoints install a small `BenchmarkNonuniformEPDDP` subclass that binds parsed CLI
arguments to `NonuniformEPConfig`. `BenchmarkNonuniformEPDDP` is an example adapter, not a separate
core API.

## Native DDP Reuse

Native DDP constructs both `self.expert_parallel_buffers` and
`self.expert_parallel_bucket_groups`. NEP calls `DistributedDataParallel.__init__()` and retains
the native parameter and gradient buffers. After native initialization, it replaces only
`self.expert_parallel_bucket_groups` with NEP-aware groups. Dense buffers and dense bucket groups
remain native.

| DDP object | NEP behavior |
| --- | --- |
| Dense parameter and gradient buffers | Reused unchanged |
| Dense bucket groups | Reused unchanged |
| Expert parameter and gradient buffers | Reused as the physical local layout |
| Native expert bucket objects | Replaced with metadata views over the existing buffers |
| Native expert bucket groups | Replaced with NEP-aware synchronization groups |
| Native DDP gradient synchronization | Reused for EDP after gradients reach the owner layout |

The replacement expert buckets do not allocate another model-sized gradient buffer. Their
`param_data` and `grad_data` tensors are views into the native expert buffers.

Native expert buckets cannot be used directly because their boundaries are derived from each
rank's physical parameter order and bucket-size threshold. Replicas with different EP sizes own
different expert counts and global expert IDs, so corresponding native buckets need not contain
the same logical parameters or have matching shapes. NEP instead forms synchronization units from
canonical `(global expert ID, parameter slot)` entries and implements:

```text
physical replica layout -> owner layout -> native EDP synchronization -> physical replica layout
```

## Construction Call Stack

NEP has work both before and after native DDP initialization:

```text
initialize_model_parallel()
  -> initialize_nonuniform_ep_process_groups_from_args()
       -> construct nonuniform EP, transfer, and EDP process groups

NonuniformEPDistributedDataParallel.__init__()
  -> validate NEP configuration
  -> synchronize the native bucket-size threshold
  -> recompute the native distributed-optimizer layout when enabled
  -> DistributedDataParallel.__init__()
       -> allocate native dense and expert buffers
       -> construct native dense and expert bucket groups
       -> populate param_to_bucket_group
       -> register native AccumulateGrad hooks
       -> register native forward pre-hooks when parameter-gather overlap is enabled
  -> replace expert_parallel_bucket_groups
  -> update expert entries in param_to_bucket_group
  -> rebind parameter-ready callbacks to the replacement groups
  -> configure NEP scheduler and distributed-optimizer state
  -> associate NEP expert groups with their MoE backward boundaries
```

The native AccumulateGrad hooks are registered before the replacement, but they do not capture a
bucket group permanently. At runtime they look up `self.param_to_bucket_group[param]`, which NEP
has updated to reference the replacement expert group.

## Backward Call Stack

NEP does not register a second per-parameter autograd hook. It enters through native DDP's existing
backward post-hook:

```text
PyTorch AccumulateGrad
  -> DistributedDataParallel._make_backward_post_hook()
  -> self.param_to_bucket_group[param].register_grad_ready()
       dense parameter
         -> native _ParamAndGradBucketGroup.register_grad_ready()
       expert parameter
         -> NonuniformEPNCCLParamAndGradBucketGroup.register_grad_ready()
         -> launch ready canonical owner tasks
              -> Gather gradients into owner layout
              -> synchronize owner gradients through native DDP EDP machinery
              -> defer or complete Scatter back to physical holders
```

`_configure_nep_dispatch_boundary_hooks()` associates expert bucket groups with MoE modules, but
does not add another module-level autograd hook. The callback is reached from the replacement
bucket group's `register_grad_ready()` method after its local gradients become ready.

## Parameter Synchronization

`NonuniformEPDistributedDataParallel` inherits native DDP's `start_param_sync()` and forward
pre-hooks. Those methods iterate over the current bucket-group lists and therefore dispatch
polymorphically to the replacement expert groups. With the distributed optimizer enabled, the
expert groups gather the native owner parameters and redistribute them to each replica's physical
expert layout. Dense parameter synchronization remains native.

## Memory Model

The increase in EDP communication payload is not a complete estimate of NEP's incremental memory.
NEP retains the physical expert layout used by forward and backward while native DDP and the
distributed optimizer operate on a persistent logical owner layout. Separate Gather and parameter
redistribution staging buffers preserve asynchronous communication and adjacent-bucket overlap.

For the 128-expert EP32/EP28 a3b/30b workload, a healthy EP32 rank has four experts and an EP28
owner row has five. The healthy FP32 expert-gradient payload is 3.420 GiB and the NEP owner payload
is 4.275 GiB. Their 0.855 GiB difference is only the additional payload on the wire; it is not the
full-feature memory floor.

| Incremental owner-rank storage | GiB | Purpose |
| --- | ---: | --- |
| BF16 owner-layout parameters | 2.137 | Contiguous authoritative parameters for native DistOpt |
| FP32 owner-layout gradients | 4.275 | Native DDP EDP reduce-scatter input and result |
| Net additional sharded Adam state | 1.282 | Optimizer state for five owner experts instead of four physical experts |
| Native owner-state subtotal | 7.694 | Practical floor while retaining the current native DDP/DistOpt architecture |
| Persistent Gather scratch | 4.275 | Allows Gather to proceed separately from the native owner gradient buffer |
| Remote-expert Gather output | 0.855 | Receives the offloaded fifth expert |
| Two-slot BF16 parameter-transfer ring | 1.115 | Preserves adjacent-bucket parameter-redistribution concurrency |
| Predicted total | 13.939 | Current implementation |
| Observed total | 13.965 | 36,335.85 MiB NEP versus 22,035.98 MiB healthy |

Flex dispatch does not create dummy expert parameters for nondivisible EP. It uses fixed-width
virtual dispatch slots whose empty entries are metadata only. Nondivisible placement nevertheless
increases the owner width and requires redistribution of experts held by follower ranks. DistOpt is
therefore responsible for most of the persistent owner state, while overlap support is responsible
for most of the potentially reusable staging storage.

Approximately 7.7 GiB is a plausible lower bound under the current native DDP/DistOpt architecture,
not a validated low-memory configuration. Approaching it would require gathering directly into the
owner gradient buffer and redistributing parameters directly from owner storage. Those changes must
still demonstrate correct asynchronous lifetimes and no performance regression; the current
implementation intentionally retains separate staging to preserve the validated overlap behavior.

## End-of-Iteration Call Stack

NEP overrides `finish_grad_sync()` only to complete its outstanding reshard pipeline before using
the native finalization loop:

```text
NonuniformEPDistributedDataParallel.finish_grad_sync()
  -> finish pending NEP host phases
  -> submit and drain remaining deferred Scatter work
  -> DistributedDataParallel.finish_grad_sync()
       -> finish native dense bucket groups
       -> finish replacement NEP expert bucket groups
```

The other public lifecycle overrides are `zero_grad_buffer()` and `scale_gradients()`. Each first
uses the native implementation and then applies the same operation to NEP's persistent logical
owner buffers. Native `start_grad_sync()`, `start_param_sync()`, `no_sync()`, and forward pre-hook
machinery remain inherited and reach NEP through the replacement expert bucket groups.
