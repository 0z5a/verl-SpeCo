"""Fixed test environment: cached metadata and normal Ray worker priority."""

import importlib.metadata
import json
import os
from pathlib import Path

os.environ["RAY_worker_niceness"] = "0"

_mapping = json.loads(
    Path(__file__).with_name("package-distributions.json").read_text()
)


def _cached_packages_distributions():
    return {
        package: distributions.copy() for package, distributions in _mapping.items()
    }


importlib.metadata.packages_distributions = _cached_packages_distributions
