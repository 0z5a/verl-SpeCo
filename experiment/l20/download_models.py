"""Download pinned model files into this experiment, without remote code."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(sys.argv[1])


def download(item: tuple[str, Path]) -> None:
    url, destination = item
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-fLsS",
            "--retry",
            "2",
            "--connect-timeout",
            "15",
            "--max-time",
            "1800",
            "-C",
            "-",
            url,
            "-o",
            str(destination),
        ],
        check=True,
    )
    print(destination, destination.stat().st_size, flush=True)


if __name__ == "__main__":
    files: list[tuple[str, Path]] = []
    for name in ("target", "eagle", "dflash"):
        metadata = json.loads(
            (ROOT / f"evidence/l20-20260919/{name}-model.json").read_text()
        )
        for entry in metadata["siblings"]:
            filename = entry["rfilename"]
            if not filename.endswith((".json", ".safetensors", ".txt")):
                continue
            url = f"https://hf-mirror.com/{metadata['id']}/resolve/{metadata['sha']}/{filename}"
            files.append((url, ROOT / "models" / name / filename))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, files))
