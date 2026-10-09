"""Workload-only full-MoE graph comparison with matched microbatch ranges."""

import hashlib
import json
import os
import sys
from pathlib import Path

from pretrain_a8b_gb200_depth import change_options as depth_options


def change_options(options: list[str], scope: str) -> list[str]:
    """Apply the same static routing workload to both native capture scopes."""
    assert scope in ("moe_router", "moe")
    after = depth_options(options)
    assert after.count("--moe-router-force-load-balancing") == 1
    after[after.index("--moe-router-force-load-balancing")] = "--moe-router-force-uniform-routing"
    index = after.index("--cuda-graph-modules")
    assert after[index + 1 : index + 4] == ["mamba", "attn", "moe_router"]
    after[index + 3] = scope
    for flag in ("--moe-expert-capacity-factor", "--moe-pad-expert-input-to-capacity"):
        assert flag not in after
    after += ["--moe-expert-capacity-factor", "1", "--moe-pad-expert-input-to-capacity"]
    return after


def main() -> None:
    """Keep production code intact; modify arguments and add profiling ranges."""
    before = sys.argv[1:]
    scope = os.environ["A8B_GRAPH_SCOPE"]
    after = change_options(before, scope)
    expected_mbs = os.environ.get("A8B_GRAPH_EXPECTED_MBS", "2")
    expected_ga = os.environ.get("A8B_GRAPH_EXPECTED_GA", "8")
    assert os.environ["REPLICA_MICRO_BATCH_SIZES"] == f"{expected_mbs} {expected_mbs}"
    assert os.environ["REPLICA_NUM_MICROBATCHES"] == f"{expected_ga} {expected_ga}"
    assert os.environ["CUPTI_EVENTS_ONLY"] == "1"
    assert os.environ["CUPTI_COLLECTION_DISABLED"] == "0"
    repo = Path(os.environ["CODE_REPO"])
    runtime = repo / "megatron/core/distributed/nonuniform_ep.py"
    sha = hashlib.sha256(runtime.read_bytes()).hexdigest()
    assert sha == os.environ["A8B_DEPTH_EXPECTED_SHA"]
    output = Path(os.environ["CUPTI_OUTPUT_DIR"]).parent / "launch_manifest"
    output.mkdir(exist_ok=True)
    (output / f"rank-{os.environ['RANK']}.json").write_text(
        json.dumps(
            dict(
                runtime_sha256=sha,
                options_before=before,
                options=after,
                graph_scope=scope,
                replica_mbs=os.environ["REPLICA_MICRO_BATCH_SIZES"],
                replica_ga=os.environ["REPLICA_NUM_MICROBATCHES"],
            ),
            indent=2,
        )
        + "\n"
    )
    sys.path.insert(0, str(repo))
    from pretrain_a8b_remaining_overheads import install_scopes

    install_scopes()
    from cuda_event_capture_adapter import CudaEventCapture
    from megatron.core.pipeline_parallel import schedules
    from pretrain_profile_activity_probe import main as profiled_main

    # Expert Python hooks disappear under full capture. These common boundaries
    # retain comparable forward/backward timings, including graph replay.
    CudaEventCapture.wrap(schedules, "forward_step", "microbatch.forward")
    CudaEventCapture.wrap(schedules, "backward_step", "microbatch.backward")
    sys.argv[1:] = after
    profiled_main()


if __name__ == "__main__":
    main()
