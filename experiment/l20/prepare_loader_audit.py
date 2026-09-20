"""Audit public-loader completion and draft retention across target sync."""

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
    + "        rank = torch.distributed.get_rank()\n"
    + "        _pr10_audit_published_fc[rank] = _pr10_audit_fingerprint(draft_model)\n"
    + '        logger.warning("[pr10 audit] public_loader=%s rank=%s tensors=%s", '
    + "type(draft_model).__name__, rank, loaded_params)\n",
)
assert updated.rstrip().endswith("return result")
prefix, suffix = updated.rsplit("        return result", 1)
updated = (
    prefix
    + "        _pr10_audit_after_target_sync(self)\n        return result"
    + suffix
)
updated += """

_pr10_audit_published_fc: dict[int, str] = {}
logger.warning("[pr10 audit] runtime_source=%s pid=%s", __file__, os.getpid())


def _pr10_audit_fingerprint(draft):
    import hashlib
    import torch

    tensor = draft.model.fc.weight.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(tensor.numpy()).hexdigest()


def _pr10_audit_after_target_sync(worker):
    import torch

    rank = torch.distributed.get_rank()
    expected = _pr10_audit_published_fc.get(rank)
    if expected is not None:
        draft, _ = worker._speco_resolve_draft_model()
        actual = _pr10_audit_fingerprint(draft)
        logger.warning(
            "[pr10 audit] after_target_sync rank=%s latest_draft_retained=%s expected=%s actual=%s",
            rank, actual == expected, expected, actual,
        )
"""
path.write_text(updated)
(args.destination / "manifest.json").write_text(
    json.dumps(
        {
            "source": str(args.source),
            "original_runtime_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "audited_runtime_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            "change": "Log completed public load and hash private fc after target sync; no loader logic changes",
        },
        indent=2,
    )
    + "\n"
)
