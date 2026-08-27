#!/usr/bin/env python3
"""Prepare and analyze short healthy/NEP runs of examples/training_scripts."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


REDUCED_EP = {8: 6, 16: 12, 32: 28, 64: 56}
GPUS_PER_NODE = 4

REMOVE_VALUE_OPTIONS = {
    "--check-weight-hash-across-dp-replicas-interval",
    "--context-parallel-size",
    "--data-cache-path",
    "--ddp-num-buckets",
    "--distributed-timeout-minutes",
    "--eval-interval",
    "--eval-iters",
    "--exit-duration-in-mins",
    "--expert-model-parallel-size",
    "--expert-tensor-parallel-size",
    "--global-batch-size",
    "--high-priority-stream-groups",
    "--load",
    "--log-interval",
    "--log-memory-interval",
    "--lr-decay-iters",
    "--lr-decay-samples",
    "--lr-warmup-iters",
    "--lr-warmup-samples",
    "--lr-wsd-decay-iters",
    "--lr-wsd-decay-samples",
    "--manual-gc-interval",
    "--micro-batch-size",
    "--moe-flex-dispatcher-backend",
    "--moe-expert-rank-capacity-factor",
    "--moe-token-dispatcher-type",
    "--num-workers",
    "--nonuniform-mode",
    "--per-split-data-args-path",
    "--phase-transition-iterations",
    "--pipeline-model-parallel-size",
    "--profile-step-end",
    "--profile-step-start",
    "--rerun-mode",
    "--result-rejected-tracker-filename",
    "--save",
    "--save-interval",
    "--save-retain-interval",
    "--straggler-minmax-count",
    "--te-precision-config-file",
    "--tensor-model-parallel-size",
    "--tensorboard-dir",
    "--timing-log-option",
    "--tiktoken-pattern",
    "--tokenizer-model",
    "--tokenizer-type",
    "--train-iters",
    "--train-samples",
    "--vocab-size",
}

REMOVE_FLAGS = {
    "--async-save",
    "--ckpt-assume-constant-structure",
    "--ckpt-fully-parallel-load",
    "--ckpt-fully-parallel-save",
    "--disable-gloo-process-groups",
    "--log-energy",
    "--log-num-zeros-in-grad",
    "--log-params-norm",
    "--log-progress",
    "--moe-router-enable-expert-bias",
    "--no-mmap-bin-files",
    "--profile",
    "--use-persistent-ckpt-worker",
    "--use-pytorch-profiler",
}


@dataclass(frozen=True)
class Workload:
    name: str
    source: str
    tensor_parallel: int
    context_parallel: int
    pipeline_parallel: int
    expert_parallel: int | None
    expert_tensor_parallel: int
    hybrid_ep_nvlink_domain_size: int | None
    micro_batch_size: int
    original_global_batch_size: int
    original_world_size: int
    grad_accumulation_steps: int
    ddp_num_buckets: int

    @property
    def group(self) -> str:
        return "dense" if self.expert_parallel is None else f"ep{self.expert_parallel}"

    @property
    def slug(self) -> str:
        return self.name.replace("/", "__")


def extract_options(path: Path) -> list[str]:
    text = path.read_text()
    start = text.index('options="') + len('options="')
    end = text.index("\n\nrun_cmd=", start)
    body = text[start:end].rstrip()
    if not body.endswith('"'):
        raise ValueError(f"{path}: options assignment has an unexpected ending")
    body = re.sub(r"\\\s*\n", " ", body[:-1])
    return shlex.split(body)


def option_value(tokens: list[str], option: str, default: int | None = None) -> int:
    try:
        return int(tokens[tokens.index(option) + 1])
    except ValueError:
        if default is None:
            raise ValueError(f"missing required option {option}") from None
        return default


def discover(repo: Path) -> list[Workload]:
    root = repo / "examples" / "training_scripts"
    workloads = []
    for path in sorted(root.rglob("*.sh")):
        tokens = extract_options(path)
        text = path.read_text()
        nodes = int(re.search(r"^#SBATCH\s+--nodes=(\d+)$", text, re.MULTILINE).group(1))
        tasks_per_node = int(
            re.search(r"^#SBATCH\s+--ntasks-per-node=(\d+)$", text, re.MULTILINE).group(1)
        )
        tp = option_value(tokens, "--tensor-model-parallel-size")
        cp = option_value(tokens, "--context-parallel-size", 1)
        pp = option_value(tokens, "--pipeline-model-parallel-size", 1)
        mbs = option_value(tokens, "--micro-batch-size")
        gbs = option_value(tokens, "--global-batch-size")
        world = nodes * tasks_per_node
        data_parallel = world // (tp * cp * pp)
        if world % (tp * cp * pp) or gbs % (mbs * data_parallel):
            raise ValueError(f"{path}: cannot derive integral gradient accumulation")
        ep = option_value(tokens, "--expert-model-parallel-size") if "--num-experts" in tokens else None
        etp = option_value(tokens, "--expert-tensor-parallel-size", 1)
        hybrid_ep_domain_match = re.search(
            r"^export\s+NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=(\d+)$",
            text,
            re.MULTILINE,
        )
        if ep is not None and ep not in REDUCED_EP:
            raise ValueError(f"{path}: no reduced-EP test mapping for EP{ep}")
        workloads.append(
            Workload(
                name=str(path.relative_to(root).with_suffix("")),
                source=str(path.relative_to(repo)),
                tensor_parallel=tp,
                context_parallel=cp,
                pipeline_parallel=pp,
                expert_parallel=ep,
                expert_tensor_parallel=etp,
                hybrid_ep_nvlink_domain_size=(
                    int(hybrid_ep_domain_match.group(1)) if hybrid_ep_domain_match else None
                ),
                micro_batch_size=mbs,
                original_global_batch_size=gbs,
                original_world_size=world,
                grad_accumulation_steps=gbs // (mbs * data_parallel),
                ddp_num_buckets=option_value(tokens, "--ddp-num-buckets"),
            )
        )
    return workloads


def workload_by_name(repo: Path, name: str) -> Workload:
    matches = [workload for workload in discover(repo) if workload.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown workload {name!r}")
    return matches[0]


def case_metadata(workload: Workload, case: str) -> dict[str, object]:
    if workload.expert_parallel is None:
        raise ValueError(f"{workload.name}: dense workload has no NEP case")
    ep = workload.expert_parallel
    reduced_ep = REDUCED_EP[ep]
    axis = workload.tensor_parallel * workload.context_parallel
    full_expert_ranks = ep * workload.expert_tensor_parallel
    reduced_expert_ranks = reduced_ep * workload.expert_tensor_parallel
    if full_expert_ranks % axis or reduced_expert_ranks % axis:
        raise ValueError(
            f"{workload.name}: EP*ETP degrees must be divisible by TP*CP={axis}"
        )
    full_units = full_expert_ranks // axis
    reduced_units = reduced_expert_ranks // axis
    topology = [full_units, full_units] if case == "healthy" else [full_units, reduced_units]
    mode = "none" if case == "healthy" else "ep"
    world = sum(topology) * axis
    gbs = workload.micro_batch_size * sum(topology) * workload.grad_accumulation_steps
    return {
        "case": case,
        "mode": mode,
        "topology": topology,
        "world_size": world,
        "nodes": math.ceil(world / GPUS_PER_NODE),
        "global_batch_size": gbs,
        "reduced_ep": reduced_ep,
        "segment_nodes": 4 if ep <= 16 else 16,
    }


def strip_runtime_options(tokens: list[str]) -> list[str]:
    output = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        base = token.split("=", 1)[0]
        if base in REMOVE_VALUE_OPTIONS:
            index += 1 if "=" in token else 2
            continue
        if token in REMOVE_FLAGS:
            index += 1
            continue
        output.append(token)
        index += 1
    return output


def append_flag_once(tokens: list[str], flag: str) -> None:
    if flag not in tokens:
        tokens.append(flag)


def replace_multi_value_option(tokens: list[str], option: str, values: list[str]) -> None:
    try:
        start = tokens.index(option)
    except ValueError:
        tokens.extend((option, *values))
        return
    end = start + 1
    while end < len(tokens) and not tokens[end].startswith("--"):
        end += 1
    tokens[start:end] = [option, *values]


def benchmark_options(
    repo: Path,
    workload: Workload,
    case: str,
    run_dir: Path,
    train_iters: int,
    profile_start: int,
    profile_end: int,
    cuda_graph_modules_override: list[str] | None = None,
) -> list[str]:
    metadata = case_metadata(workload, case)
    tokens = strip_runtime_options(extract_options(repo / workload.source))
    if cuda_graph_modules_override is not None:
        replace_multi_value_option(tokens, "--cuda-graph-modules", cuda_graph_modules_override)
    for flag in (
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
        "--calculate-per-token-loss",
        "--moe-router-force-load-balancing",
        "--no-check-for-nan-in-loss-and-grad",
        "--nonuniform-disable-nongrad-sync-collectives",
        "--manual-gc",
        "--profile",
        "--use-pytorch-profiler",
    ):
        append_flag_once(tokens, flag)
    tokens.extend(
        [
            "--moe-token-dispatcher-type",
            "flex",
            "--moe-flex-dispatcher-backend",
            "hybridep",
            "--mock-data",
            "--tokenizer-type",
            "NullTokenizer",
            "--vocab-size",
            "131072",
            "--num-workers",
            "1",
            "--train-iters",
            str(train_iters),
            "--lr-decay-iters",
            str(train_iters),
            "--lr-warmup-iters",
            "1",
            "--lr-wsd-decay-iters",
            "2",
            "--eval-interval",
            "1000",
            "--eval-iters",
            "0",
            "--tensor-model-parallel-size",
            str(workload.tensor_parallel),
            "--context-parallel-size",
            str(workload.context_parallel),
            "--pipeline-model-parallel-size",
            str(workload.pipeline_parallel),
            "--expert-model-parallel-size",
            str(workload.expert_parallel),
            "--expert-tensor-parallel-size",
            str(workload.expert_tensor_parallel),
            "--ddp-num-buckets",
            str(workload.ddp_num_buckets),
            "--high-priority-stream-groups",
            "ep",
            "--nonuniform-mode",
            str(metadata["mode"]),
            "--log-interval",
            "1",
            "--log-memory-interval",
            "1000",
            "--timing-log-option",
            "minmax",
            "--tensorboard-dir",
            str(run_dir / "tensorboard"),
            "--profile-step-start",
            str(profile_start),
            "--profile-step-end",
            str(profile_end),
            "--profile-ranks",
            *(str(rank) for rank in range(int(metadata["world_size"]))),
            "--manual-gc-interval",
            "1000",
            "--distributed-timeout-minutes",
            "10",
            "--rerun-mode",
            "disabled",
        ]
    )
    if case == "nep":
        tokens.extend(
            [
                "--nonuniform-ep-ddp-approach",
                "nccl",
                "--nonuniform-ep-num-tp-cp-per-replica",
                *(str(value) for value in metadata["topology"]),
            ]
        )
    unresolved = [token for token in tokens if "$" in token or "\n" in token]
    if unresolved:
        raise ValueError(f"{workload.name}: unresolved source tokens: {unresolved}")
    return tokens


TIMING_PATTERN = re.compile(
    r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
)
STATUS_PATTERN = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?number of skipped iterations:\s*(\d+).*?"
    r"number of nan iterations:\s*(\d+)"
)


def analyze_case(
    workload: Workload,
    case: str,
    run_dir: Path,
    train_iters: int,
    timing_start: int,
) -> dict[str, object]:
    metadata = case_metadata(workload, case)
    log = run_dir / "driver.log"
    if not log.is_file():
        raise RuntimeError(f"{case}: missing {log}")
    text = log.read_text(errors="replace")
    required = {
        "distributed optimizer": (
            r"(?:use_distributed_optimizer|use_layer_wise_distributed_optimizer)"
            r"\s+\.+\s+True"
        ),
        "overlap grad reduce": r"overlap_grad_reduce\s+\.+\s+True",
        "overlap param gather": r"overlap_param_gather\s+\.+\s+True",
        "Flex dispatcher": r"moe_token_dispatcher_type\s+\.+\s+flex",
        "HybridEP backend": r"moe_flex_dispatcher_backend\s+\.+\s+hybridep",
        "forced load balancing": r"moe_router_force_load_balancing\s+\.+\s+True",
        "NEP mode": rf"nonuniform_mode\s+\.+\s+{metadata['mode']}",
    }
    for description, pattern in required.items():
        if re.search(pattern, text) is None:
            raise RuntimeError(f"{case}: runtime did not confirm {description}")
    if "token drop during MoE token dispatch" in text or "rerunning forward-backward" in text:
        raise RuntimeError(f"{case}: HybridEP static capacity overflowed")
    statuses = [tuple(map(int, match.groups())) for match in STATUS_PATTERN.finditer(text)]
    statuses = statuses[-train_iters:]
    if [row[0] for row in statuses] != list(range(1, train_iters + 1)):
        raise RuntimeError(f"{case}: incomplete iterations: {statuses}")
    if any(total != train_iters or skipped or nan for _, total, skipped, nan in statuses):
        raise RuntimeError(f"{case}: invalid iteration status: {statuses}")
    rows_by_iteration = {}
    for iteration, elapsed_ms in TIMING_PATTERN.findall(text):
        iteration = int(iteration)
        if iteration >= timing_start:
            rows_by_iteration[iteration] = float(elapsed_ms)
    expected = list(range(timing_start, train_iters + 1))
    if sorted(rows_by_iteration) != expected:
        raise RuntimeError(f"{case}: expected timing iterations {expected}, got {rows_by_iteration}")
    values = [rows_by_iteration[iteration] for iteration in expected]
    traces = sorted((run_dir / "torch_profile").glob("rank-*.json.gz"))
    if len(traces) != metadata["world_size"]:
        raise RuntimeError(
            f"{case}: expected {metadata['world_size']} all-rank traces, found {len(traces)}"
        )
    return {
        **metadata,
        "run_dir": str(run_dir),
        "timing_iterations": expected,
        "elapsed_ms": values,
        "mean_elapsed_ms": statistics.mean(values),
        "std_elapsed_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "trace_count": len(traces),
    }


def analyze_pair(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    workload = workload_by_name(repo, args.workload)
    root = Path(args.root).resolve() / workload.slug
    result = {
        "workload": asdict(workload),
        "source_commit": args.source_commit,
        "precision_note": (
            "Nemotron Super/Ultra use built-in --fp4-recipe nvfp4; the source script's "
            "cluster-local te_quant.cfg is unavailable."
            if workload.name in {"nemotron3/super", "nemotron3/ultra"}
            else None
        ),
    }
    for case in ("healthy", "nep"):
        result[case] = analyze_case(
            workload,
            case,
            root / case / args.job_id,
            args.train_iters,
            args.timing_start,
        )
    healthy_ms = result["healthy"]["mean_elapsed_ms"]
    nep_ms = result["nep"]["mean_elapsed_ms"]
    result["comparison"] = {
        "latency_gap_ms": nep_ms - healthy_ms,
        "nep_slowdown_percent": 100.0 * (nep_ms / healthy_ms - 1.0),
        "owner_iteration_time_parity_percent": 100.0 * healthy_ms / nep_ms,
        "owner_relative_throughput_percent": 100.0 * healthy_ms / nep_ms,
    }
    output = root / f"comparison_{args.job_id}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["comparison"], sort_keys=True))
    print(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list")
    listing.add_argument("--group")
    listing.add_argument("--json", action="store_true")

    fields = subparsers.add_parser("case-fields")
    fields.add_argument("workload")
    fields.add_argument("case", choices=("healthy", "nep"))

    options = subparsers.add_parser("options")
    options.add_argument("workload")
    options.add_argument("case", choices=("healthy", "nep"))
    options.add_argument("--run-dir", required=True)
    options.add_argument("--train-iters", type=int, default=10)
    options.add_argument("--profile-start", type=int, default=5)
    options.add_argument("--profile-end", type=int, default=7)
    options.add_argument("--cuda-graph-modules-override")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("workload")
    analyze.add_argument("--root", required=True)
    analyze.add_argument("--job-id", required=True)
    analyze.add_argument("--train-iters", type=int, default=10)
    analyze.add_argument("--timing-start", type=int, default=8)
    analyze.add_argument("--source-commit", default="8643ab867049f71fb751121366c078ac58f17326")

    subparsers.add_parser("validate")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    if args.command == "list":
        workloads = discover(repo)
        if args.group:
            workloads = [workload for workload in workloads if workload.group == args.group]
        if args.json:
            print(json.dumps([asdict(workload) | {"group": workload.group} for workload in workloads], indent=2))
        else:
            print("\n".join(workload.name for workload in workloads))
        return
    if args.command == "validate":
        workloads = discover(repo)
        dense = [workload for workload in workloads if workload.expert_parallel is None]
        moe = [workload for workload in workloads if workload.expert_parallel is not None]
        if len(dense) != 2 or len(moe) != 13:
            raise RuntimeError(
                f"expected 2 dense and 13 MoE workloads, got {len(dense)} and {len(moe)}"
            )
        for workload in moe:
            rows = []
            for case in ("healthy", "nep"):
                metadata = case_metadata(workload, case)
                tokens = benchmark_options(
                    repo,
                    workload,
                    case,
                    Path("/tmp/training_scripts_nep_validate") / workload.slug / case,
                    10,
                    5,
                    7,
                )
                for option, value in (
                    ("--use-distributed-optimizer", None),
                    ("--overlap-grad-reduce", None),
                    ("--overlap-param-gather", None),
                    ("--moe-router-force-load-balancing", None),
                    ("--profile", None),
                    ("--use-pytorch-profiler", None),
                    ("--moe-token-dispatcher-type", "flex"),
                    ("--moe-flex-dispatcher-backend", "hybridep"),
                    ("--nonuniform-mode", metadata["mode"]),
                ):
                    if tokens.count(option) != 1:
                        raise RuntimeError(f"{workload.name}/{case}: expected one {option}")
                    if value is not None and tokens[tokens.index(option) + 1] != value:
                        raise RuntimeError(f"{workload.name}/{case}: wrong value for {option}")
                if tokens[tokens.index("--expert-tensor-parallel-size") + 1] != str(
                    workload.expert_tensor_parallel
                ):
                    raise RuntimeError(f"{workload.name}/{case}: source ETP was not preserved")
                forbidden = {
                    "--load",
                    "--save",
                    "--train-samples",
                    "--per-split-data-args-path",
                    "--te-precision-config-file",
                    "--moe-router-enable-expert-bias",
                    "--moe-expert-rank-capacity-factor",
                    "--use-transformer-engine-op-fuser",
                }
                present = forbidden.intersection(tokens)
                if present:
                    raise RuntimeError(f"{workload.name}/{case}: forbidden source options {present}")
                if case == "healthy":
                    expert_data_parallel = int(metadata["world_size"]) // (
                        workload.expert_parallel
                        * workload.expert_tensor_parallel
                        * workload.pipeline_parallel
                    )
                    if expert_data_parallel != 2:
                        raise RuntimeError(
                            f"{workload.name}: healthy case has expert DP "
                            f"{expert_data_parallel}, expected 2"
                        )
                rows.append(
                    f"{case}:world={metadata['world_size']},topology="
                    f"{','.join(map(str, metadata['topology']))},"
                    f"gbs={metadata['global_batch_size']}"
                )
            print(
                f"{workload.name}\tEP{workload.expert_parallel}/"
                f"EP{REDUCED_EP[workload.expert_parallel]}\tTP{workload.tensor_parallel}"
                f"\tMBS{workload.micro_batch_size}\tGA{workload.grad_accumulation_steps}"
                f"\t{' | '.join(rows)}"
            )
        print("dense/N-A\t" + ",".join(workload.name for workload in dense))
        print("PASS: 13 MoE workload pairs validated; 2 dense workloads are explicitly N/A for NEP")
        return
    workload = workload_by_name(repo, args.workload)
    if args.command == "case-fields":
        metadata = case_metadata(workload, args.case)
        print(
            "\t".join(
                str(value)
                for value in (
                    workload.tensor_parallel,
                    workload.context_parallel,
                    workload.expert_tensor_parallel,
                    workload.expert_parallel,
                    workload.micro_batch_size,
                    workload.grad_accumulation_steps,
                    " ".join(str(value) for value in metadata["topology"]),
                    metadata["mode"],
                    metadata["world_size"],
                    metadata["nodes"],
                    metadata["global_batch_size"],
                    workload.ddp_num_buckets,
                    metadata["segment_nodes"],
                    workload.hybrid_ep_nvlink_domain_size or "",
                )
            )
        )
        return
    if args.command == "options":
        cuda_graph_modules_override = None
        if args.cuda_graph_modules_override:
            cuda_graph_modules_override = args.cuda_graph_modules_override.split(",")
            if any(not module for module in cuda_graph_modules_override):
                raise ValueError("CUDA graph module override contains an empty module")
        tokens = benchmark_options(
            repo,
            workload,
            args.case,
            Path(args.run_dir),
            args.train_iters,
            args.profile_start,
            args.profile_end,
            cuda_graph_modules_override,
        )
        if any(re.search(r"\s", token) for token in tokens):
            raise ValueError("the direct launcher requires whitespace-free argument tokens")
        print(" ".join(tokens))
        return
    if args.command == "analyze":
        analyze_pair(args)
        return
    parser.error(f"unsupported command {args.command}")


if __name__ == "__main__":
    main()
