"""Use upstream pure-PyTorch padding helpers for the native SDPA experiment."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("verl_source", type=Path)
parser.add_argument("padding_source", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
shutil.copytree(args.verl_source, args.destination / "verl")
padding = args.padding_source.read_bytes()
(args.destination / "speco_upstream_bert_padding.py").write_bytes(padding)
path = args.destination / "verl/utils/attention_utils.py"
original = path.read_text()
needle = "from flash_attn.bert_padding import"
assert original.count(needle) == 1
path.write_text(original.replace(needle, "from speco_upstream_bert_padding import"))
(args.destination / "manifest.json").write_text(
    json.dumps(
        {
            "upstream": "Dao-AILab/flash-attention",
            "revision": "edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d",
            "file": "flash_attn/bert_padding.py",
            "sha256": hashlib.sha256(padding).hexdigest(),
            "change": "Redirect only verl padding import; unchanged upstream PyTorch helpers, no attention kernel replacement",
        },
        indent=2,
    )
    + "\n"
)
