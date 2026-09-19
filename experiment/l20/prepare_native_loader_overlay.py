"""Isolate one vLLM loader source change while reusing the native installation."""

import argparse
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
source = Path(str(distribution("vllm").locate_file("vllm"))).resolve()
relative = Path("model_executor/models/llama_eagle3.py")
original = (source / relative).read_text()
before = 'elif "lm_head" not in name:'
after = 'elif "lm_head" not in name and not name.startswith("model."):'
assert original.count(before) == 1
package = args.output / "vllm"
package.mkdir(parents=True, exist_ok=False)
for directory, next_component in (
    (Path(), "model_executor"),
    (Path("model_executor"), "models"),
    (Path("model_executor/models"), "llama_eagle3.py"),
):
    destination = package / directory
    destination.mkdir(exist_ok=True)
    for child in (source / directory).iterdir():
        if child.name not in {next_component, "__pycache__"}:
            (destination / child.name).symlink_to(child)
patched = original.replace(before, after)
(package / relative).write_text(patched)
(args.output / "manifest.json").write_text(
    json.dumps(
        {
            "native_package": str(source),
            "native_loader_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "candidate_loader_sha256": hashlib.sha256(patched.encode()).hexdigest(),
            "change": "Idempotent model. prefix only; native installation unchanged",
        },
        indent=2,
    )
    + "\n"
)
