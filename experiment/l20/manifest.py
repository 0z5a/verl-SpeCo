"""Record runtime identity and immutable local model files."""

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import torch

root = Path("/experiment")
packages = {
    name: metadata.version(name)
    for name in ("torch", "vllm", "verl", "transformers", "ray", "tensordict")
}
models = {}
for directory in sorted((root / "models").iterdir()):
    models[directory.name] = {}
    for path in sorted(directory.glob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            models[directory.name][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
record = {
    "packages": packages,
    "cuda_runtime": torch.version.cuda,
    "model_files": models,
    "visible_gpus": [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ],
}
(root / "evidence/l20-20260919/manifest.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
