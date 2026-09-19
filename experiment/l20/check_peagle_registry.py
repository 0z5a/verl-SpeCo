"""Record the installed runtime registry without claiming serving parity."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import vllm

root = Path(vllm.__file__).parent
registry_path = root / "model_executor/models/registry.py"
registry_source = registry_path.read_text()
registry = next(
    ast.literal_eval(node.value)
    for node in ast.parse(registry_source).body
    if isinstance(node, ast.Assign)
    and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id == "_SPECULATIVE_DECODING_MODELS"
)
architecture = "LlamaForCausalLMPeagle"
report = {
    "vllm_version": vllm.__version__,
    "speco_exported_architecture": architecture,
    "exported_architecture_registered": architecture in registry,
    "related_registered_architectures": {
        name: implementation
        for name, implementation in registry.items()
        if "peagle" in name.lower()
    },
    "source_sha256": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            registry_path,
            root / "model_executor/models/llama_eagle3.py",
            root / "config/speculative.py",
        )
    },
    "veomni_installed": importlib.util.find_spec("veomni") is not None,
    "parameter_and_logit_parity": "NOT_RUN",
}
Path("/experiment/evidence/l20-20260919/peagle-runtime-registry.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
