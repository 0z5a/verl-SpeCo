"""Give the old trainer its architecture alias; serving uses the original config."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
args.destination.mkdir(exist_ok=True)
for path in args.source.iterdir():
    if path.name != "config.json":
        (args.destination / path.name).symlink_to(path.resolve())
config = json.loads((args.source / "config.json").read_text())
assert config["architectures"] == ["Eagle3LlamaForCausalLM"]
config["architectures"] = ["LlamaForCausalLMEagle3"]
(args.destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
