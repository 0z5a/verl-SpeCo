"""Ray continuation lines must not drop successful-training metrics."""

import json
from pathlib import Path
import subprocess
import sys


def test_ray_splits_metric_mid_token(tmp_path):
    prefix = "\x1b[36m(SpecoTaskRunner pid=65789)\x1b[0m "
    log = tmp_path / "run.log"
    output = tmp_path / "summary.json"
    log.write_text(
        prefix
        + "step:1 - drafter/collected_samples:np.floa\n"
        + prefix
        + "t64(2.0) - drafter/trained:np.float64(1.0) "
        "- drafter/train_optimizer_step_max:np.int64(1) \n"
        + prefix
        + "step:2 - drafter/trained:np.float64(1.0) \n"
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("summarize_online.py")),
            str(log),
            str(output),
        ],
        check=True,
    )
    steps = json.loads(output.read_text())["steps"]
    assert steps == [
        {
            "step": 1,
            "drafter/collected_samples": 2.0,
            "drafter/trained": 1.0,
            "drafter/train_optimizer_step_max": 1.0,
        },
        {"step": 2, "drafter/trained": 1.0},
    ]
