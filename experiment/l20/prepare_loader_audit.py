"""Copy PR10 into a test overlay and log completed public-loader calls."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
shutil.copytree(args.source / "verl_speco", args.destination / "verl_speco")
path = args.destination / "verl_speco/integration/vllm_runtime.py"
original = path.read_text()
needle = "        draft_model.load_weights(iter(translated_weights))\n"
assert original.count(needle) == 1
updated = original.replace(
    needle,
    needle
    + '        logger.warning("[pr10 audit] public_loader=%s rank=%s tensors=%s", '
    + "type(draft_model).__name__, torch.distributed.get_rank(), loaded_params)\n",
)
path.write_text(updated)
(args.destination / "manifest.json").write_text(
    json.dumps(
        {
            "source": str(args.source),
            "original_runtime_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "audited_runtime_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            "change": "Log after successful original public load_weights call; no loader logic changes",
        },
        indent=2,
    )
    + "\n"
)
