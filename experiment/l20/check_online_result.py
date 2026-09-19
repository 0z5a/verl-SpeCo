"""Require optimizer updates and all-rank commits, not async submission alone."""

import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("log", type=Path)
parser.add_argument("summary", type=Path)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--tp", type=int, default=2)
args = parser.parse_args()
log = args.log.read_text()
report = json.loads(args.summary.read_text())
steps = report["steps"]
assert not report["traceback_present"], "Training raised an exception"
assert [step["step"] for step in steps] == list(range(1, args.steps + 1))
for step in steps:
    assert step["drafter/collected_samples"] > 0, step["step"]
    assert step["drafter/trained"] == 1, step["step"]
    assert step["drafter/train_optimizer_step_max"] == step["step"], step["step"]

commits = re.findall(
    r"Worker_TP(\d+) pid=\d+\).*?committed online drafter revision=(\d+) loaded_params=(\d+)",
    log,
)
expected = {
    (rank, revision) for rank in range(args.tp) for revision in range(1, args.steps + 1)
}
observed = {
    (int(rank), int(revision)) for rank, revision, count in commits if int(count) > 0
}
assert observed == expected, f"Missing commits: {expected - observed}"
assert len(commits) == len(expected), "Duplicate revision commits"
restored = {
    (int(rank), int(revision))
    for rank, revision in re.findall(
        r"Worker_TP(\d+) pid=\d+\).*?drafter state restored after level-2 wake_up .*?revision=(\d+)",
        log,
    )
}
assert {
    (rank, revision) for rank in range(args.tp) for revision in range(1, args.steps)
} <= restored
print(
    f"PASS: {args.steps} optimizer updates, {len(commits)} rank commits, revisions 1–{args.steps - 1} restored"
)
