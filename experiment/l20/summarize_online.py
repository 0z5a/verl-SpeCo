"""Extract online step metrics without treating async submission as completion."""

import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("log", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
log = args.log.read_text()
# Ray can split a metric record mid-token and repeat the same actor prefix.
metric_log = re.sub(
    r"\n\x1b\[36m\(SpecoTaskRunner pid=\d+\)\x1b\[0m (?!step:)", "", log
)
steps = []
for line in metric_log.splitlines():
    step = re.search(r"\bstep:(\d+) - ", line)
    if step is None:
        continue
    metrics = {"step": int(step[1])}
    for key, value in re.findall(r"([\w/]+):(?:np\.\w+\()?([-+\d.eE]+)(?=[)\s])", line):
        if key.startswith(("drafter/", "timing_s/", "actor/", "critic/")):
            metrics[key] = float(value)
    steps.append(metrics)
revisions = [int(n) for n in re.findall(r"revision=(\d+)", log)]
commits = [int(n) for n in re.findall(r"committed online drafter revision=(\d+)", log)]
report = {
    "steps": steps,
    "committed_revision_counts": {
        str(n): commits.count(n) for n in sorted(set(commits))
    },
    "max_observed_revision": max(revisions, default=0),
    "traceback_present": "Traceback (most recent call last)" in log,
    "async_publication_metrics_require_rpc_and_later_rollout_confirmation": True,
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(f"steps={len(steps)} max_revision={report['max_observed_revision']}")
