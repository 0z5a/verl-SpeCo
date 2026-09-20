"""Summarize original PR10 runs without equating async submission with completion."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("log", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--tp", type=int, default=2)
args = parser.parse_args()
log = args.log.read_text()
metric_log = re.sub(
    r"\n\x1b\[36m\(SpecoTaskRunner pid=\d+\)\x1b\[0m (?!step:)", "", log
)
steps: list[dict[str, float]] = []
for line in metric_log.splitlines():
    match = re.search(r"\bstep:(\d+) - ", line)
    if match is None:
        continue
    metrics = {"step": float(match[1])}
    for key, value in re.findall(r"([\w/]+):(?:np\.\w+\()?([-+\d.eE]+)(?=[)\s])", line):
        if key.startswith(("actor/", "drafter/", "timing_s/")):
            metrics[key] = float(value)
    steps.append(metrics)

loads = re.findall(r"\[pr10 audit\] public_loader=(\S+) rank=(\d+) tensors=(\d+)", log)
load_counts = Counter(int(rank) for _, rank, count in loads if int(count) > 0)
retention = re.findall(
    r"\[pr10 audit\] after_target_sync rank=(\d+) latest_draft_retained=(True|False)",
    log,
)
retention_counts = Counter(int(rank) for rank, _ in retention)
published_hashes = re.findall(
    r"after_target_sync rank=(\d+) latest_draft_retained=\w+ expected=([a-f0-9]+)",
    log,
)
distinct_hash_counts = {
    rank: len(
        {
            value
            for observed_rank, value in published_hashes
            if int(observed_rank) == rank
        }
    )
    for rank in range(args.tp)
}
completed = [
    int(step)
    for step in re.findall(r"\[speco vllm draft update\] done global_steps=(\d+)", log)
]
exit_path = args.log.with_suffix(".exit")
exit_code = int(exit_path.read_text()) if exit_path.exists() else None
all_steps = [step["step"] for step in steps] == list(range(1, args.steps + 1))
all_loads = all(load_counts[rank] == args.steps for rank in range(args.tp))
gates = {
    "process_exit_zero": exit_code == 0,
    "all_steps": all_steps,
    "finite_metrics": bool(steps)
    and all(math.isfinite(value) for step in steps for value in step.values()),
    "actor_nonzero_gradient_observed": any(
        step.get("actor/grad_norm", 0) > 0 for step in steps
    ),
    "drafter_updates_every_step": bool(steps)
    and all(
        step.get("drafter/trained") == 1
        and step.get("drafter/train_successful_steps_max", 0) > 0
        for step in steps
    ),
    "public_loader_every_rank": all_loads,
    # Original PR10 waits pending RPCs before every rollout and in fit's finally.
    "publication_barriers_completed": exit_code == 0 and all_steps and all_loads,
    "later_target_sync_observed": all(
        retention_counts[rank] >= args.steps - 1 for rank in range(args.tp)
    ),
    "published_fc_retained": bool(retention)
    and all(value == "True" for _, value in retention),
    "published_fc_changed": all(count >= 2 for count in distinct_hash_counts.values()),
    "no_traceback": "Traceback (most recent call last)" not in log,
}
report = {
    "passed": all(gates.values()),
    "training_and_publication_passed": all(
        value for name, value in gates.items() if name != "published_fc_retained"
    ),
    "gates": gates,
    "exit_code": exit_code,
    "steps": steps,
    "public_loader_counts": dict(load_counts),
    "public_loader_classes": sorted({name for name, _, _ in loads}),
    "logged_adapter_completion_steps": completed,
    "publication_completion_evidence": "All-rank public loads plus complete training and clean exit through original PR10's pre-rollout and final pending-RPC barriers",
    "retention_counts": dict(retention_counts),
    "distinct_published_fc_hashes": distinct_hash_counts,
    "retention_failures": sum(value == "False" for _, value in retention),
    "scope": "FC hash samples private-weight retention; no complete derived-buffer oracle or controlled speed claim",
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {"passed": report["passed"], "steps": len(steps), "gates": gates}, indent=2
    )
)
