"""Compare matched all-rank microbatch and native expert host ranges."""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def average(values: list[float]) -> float:
    """Return a finite-sample mean; empty inputs indicate missing evidence."""
    assert values
    return statistics.mean(values)


def summarize_rank(path: Path, scope: str) -> list[dict]:
    """Analyze common schedule boundaries, never substitute kernel busy time."""
    data = json.loads(path.read_text())
    grouped = defaultdict(list)
    iterations = {}
    for event in data["traceEvents"]:
        if event["cat"] == "iteration":
            iterations[int(event["name"].split("_")[1])] = event
        else:
            grouped[event["args"]["iteration"]].append(event)
    rows = []
    # The training loop advances the profiler before the measured train step.
    # Validated against logged times: trace25 is log25; trace24 is warmup.
    for step in range(25, 35):
        events = grouped[step]
        gpu = [e for e in events if e["cat"] == "gpu_operation_range"]
        cpu = [e for e in events if e["cat"] == "cpu_operation_range"]

        def named(items: list[dict], name: str) -> list[dict]:
            return sorted((e for e in items if e["name"] == name), key=lambda e: e["ts"])

        forwards = named(gpu, "microbatch.forward")
        backwards = named(gpu, "microbatch.backward")
        cpu_backwards = named(cpu, "microbatch.backward")
        assert len(forwards) == len(backwards) == len(cpu_backwards) == 8
        graph_b = named(cpu, "graph.MoETransformerLayer.backward")
        expert_b = named(cpu, "expert.backward")
        if scope == "moe":
            assert len(graph_b) == 15 * 8 and not expert_b
            assert not named(cpu, "expert.forward")
        else:
            assert len(expert_b) == 2 * 15 * 8

        micro = []
        for forward, backward, cb in zip(forwards, backwards, cpu_backwards):
            enclosed = [e for e in cpu if cb["ts"] <= e["ts"] < cb["ts"] + cb["dur"]]

            def host_time(names: set[str]) -> float:
                return sum(e["args"]["thread_cpu_us"] for e in enclosed if e["name"] in names) / 1000

            micro.append(
                dict(
                    forward_ms=forward["dur"] / 1000,
                    backward_ms=backward["dur"] / 1000,
                    native_expert_backward_recompute_cpu_ms=host_time({"expert.forward", "expert.backward"}),
                    moe_graph_backward_cpu_ms=host_time({"graph.MoETransformerLayer.backward"}),
                    backward_start_epoch_ms=data["origin_ns"] / 1e6 + backward["ts"] / 1000,
                    backward_end_epoch_ms=data["origin_ns"] / 1e6 + (backward["ts"] + backward["dur"]) / 1000,
                )
            )
        collectives = [e for e in gpu if "collective" in e["args"]]
        assert all(e["args"]["collective"]["ranks"] and "completion_stream" in e["args"] for e in collectives)
        communication = defaultdict(float)
        for e in collectives:
            meta = e["args"]["collective"]
            for op in meta["operations"]:
                assert "input_bytes" in op and "output_bytes" in op
            communication[meta["group_desc"] + "." + e["name"]] += e["dur"] / 1000
        names = Counter(e["name"] for e in gpu)
        assert names["collective.all_gather"] and names["collective.reduce_scatter"]
        rows.append(
            dict(
                iteration=step,
                iteration_ms=iterations[step]["dur"] / 1000,
                micros=micro,
                collective_elapsed_sums_ms=dict(communication),
                operation_counts=dict(names),
            )
        )
    return rows


def main() -> None:
    """Validate profiles and report per-iteration rank-role means."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    args = parser.parse_args()
    root = Path("/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline") / args.job
    results = {}
    for scope in ("moe_router", "moe"):
        results[scope] = {}
        for case, size in (("healthy", 32), ("nep", 28)):
            run = root / f"graph_{scope}_{case}" / "cupti/ep16/a8b_120b_latentmoe_1t" / case / args.job
            files = list((run / "cupti_profile").glob("rank-*.json"))
            assert len(files) == size, (run, len(files), size)
            ranks = {r: summarize_rank(run / f"cupti_profile/rank-{r}.json", scope) for r in range(size)}
            roles = {"owners": range(12), "full": range(16), "second": range(16, size)}
            case_result = {}
            for role, members in roles.items():
                samples = []
                for index in range(10):
                    rows = [ranks[r][index] for r in members]
                    sample = {"iteration_ms": average([row["iteration_ms"] for row in rows])}
                    for label, indices in (("first", [0]), ("ordinary", list(range(1, 7))), ("final", [7])):
                        for metric in rows[0]["micros"][0]:
                            sample[label + "." + metric] = average([
                                average([row["micros"][mb][metric] for mb in indices]) for row in rows
                            ])
                    samples.append(sample)
                case_result[role] = {
                    "mean": {k: average([s[k] for s in samples]) for k in samples[0]},
                    "sd": {k: statistics.pstdev(s[k] for s in samples) for k in samples[0]},
                    "samples": samples,
                }
                print(scope, case, role, json.dumps(case_result[role]["mean"]), flush=True)
            log = (run / "driver.log").read_text()
            logged = {
                int(i): float(ms) for i, ms in re.findall(
                    r"iteration\s+(\d+)/\s*40.*?elapsed time per iteration \(ms\):\s*([\d.]+)", log
                )
            }
            assert len(logged) == 40
            case_result["logged_measured_ms"] = average([logged[i] for i in range(25, 35)])
            case_result["logged_pre_profile_ms"] = average([logged[i] for i in range(15, 24)])
            case_result["logged_post_profile_ms"] = average([logged[i] for i in range(36, 41)])
            results[scope][case] = case_result
    for scope in results:
        h, n = results[scope]["healthy"], results[scope]["nep"]
        results[scope]["owner_parity_pct"] = 100 * h["owners"]["mean"]["iteration_ms"] / n["owners"]["mean"]["iteration_ms"]
        results[scope]["logged_parity_pct"] = 100 * h["logged_measured_ms"] / n["logged_measured_ms"]
    output = root / "native_moe_graph_analysis.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
