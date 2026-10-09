"""Add targeted operation ranges without changing a8b training or CUDA graphs."""

import atexit
import functools
import json
import os
import sys
from pathlib import Path


def install_scopes() -> None:
    """Keep existing expert/collective ranges and label previously unmeasured work."""
    sys.path.insert(0, os.environ['CODE_REPO'])
    from cuda_event_capture_adapter import CudaEventCapture

    from megatron.core.transformer.cuda_graphs import _CudagraphReplayNode
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.fused_a2a import HybridEPCombine, HybridEPDispatch
    from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher

    CudaEventCapture.install_stage_labels()
    for owner, method, label in [
        (HybridEPDispatch, 'forward', 'hybridep.dispatch.forward'),
        (HybridEPDispatch, 'backward', 'hybridep.dispatch.backward'),
        (HybridEPCombine, 'forward', 'hybridep.combine.forward'),
        (HybridEPCombine, 'backward', 'hybridep.combine.backward'),
        (MoEFlexTokenDispatcher, '_initialize_metadata', 'moe.metadata.native'),
        (MoEFlexTokenDispatcher, '_initialize_nonuniform_metadata', 'moe.metadata.nep'),
        (MoEFlexTokenDispatcher, 'dispatch_postprocess', 'moe.dispatch_postprocess'),
        (TEGroupedMLP, 'forward', 'moe.experts.forward'),
    ]:
        CudaEventCapture.wrap(owner, method, label)

    def name_graph(method: str) -> None:
        CudaEventCapture.wrap(_CudagraphReplayNode, method, 'graph.' + method)
        measured = getattr(_CudagraphReplayNode, method)

        @functools.wraps(measured)
        def named(*args, **kwargs):
            capture = CudaEventCapture.current
            if capture is None or not capture.active:
                return measured(*args, **kwargs)
            runner = args[1] if method == 'forward' else args[0].runner
            label = 'graph.' + type(runner.base_module).__name__ + '.' + method
            index = len(capture.records)
            try:
                return measured(*args, **kwargs)
            finally:
                capture.records[index]['name'] = label

        setattr(_CudagraphReplayNode, method, staticmethod(named))

    name_graph('forward')
    name_graph('backward')

    metadata = {}
    postprocess = MoEFlexTokenDispatcher.dispatch_postprocess

    @functools.wraps(postprocess)
    def describe_dispatch(self, *args, **kwargs):
        result = postprocess(self, *args, **kwargs)
        if not metadata:
            tokens = result[1]
            slots = getattr(self, '_active_local_expert_slots', None)
            metadata.update(
                tokens_per_expert_device=str(tokens.device),
                tokens_per_expert_shape=list(tokens.shape),
                active_slots_device=str(slots.device) if slots is not None else None,
                num_local_experts=self.num_local_experts,
                num_dispatch_slots=getattr(self, 'num_local_expert_slots', self.num_local_experts),
                local_expert_indices=list(self.local_expert_indices),
            )
        return result

    MoEFlexTokenDispatcher.dispatch_postprocess = describe_dispatch

    def save_metadata() -> None:
        root = Path(os.environ['CUPTI_OUTPUT_DIR']).parent / 'diagnostic_metadata'
        root.mkdir(exist_ok=True)
        (root / f"rank-{os.environ['RANK']}.json").write_text(
            json.dumps(metadata, indent=2) + '\n'
        )

    atexit.register(save_metadata)


def main() -> None:
    install_scopes()
    from pretrain_a8b_gb200_depth import main as run_matched_workload

    run_matched_workload()


if __name__ == '__main__':
    main()
