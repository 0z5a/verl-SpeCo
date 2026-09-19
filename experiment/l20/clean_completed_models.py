from pathlib import Path
import hashlib
import json

root = Path("/experiment")
paths = list((root / "models/target").glob("*.safetensors")) + list(
    (root / "tiny-peagle").glob("*/*.safetensors")
)
for directory in (root / "evidence/l20-20260919").iterdir():
    if directory.is_dir() and directory.name.startswith(("veomni-", "partition-")):
        paths.extend(directory.rglob("*.safetensors"))
        paths.extend(directory.rglob("*.distcp"))
rows = []
for path in sorted(set(paths)):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    rows.append(
        {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    )
manifest = root / "evidence/l20-20260919/completed-model-cleanup.json"
manifest.write_text(
    json.dumps({"files": rows, "bytes": sum(row["bytes"] for row in rows)}, indent=2)
    + "\n"
)
for path in sorted(set(paths)):
    path.unlink()
assert all(not Path(row["path"]).exists() for row in rows)
print(
    json.dumps(
        {"removed_files": len(rows), "removed_bytes": sum(row["bytes"] for row in rows)}
    )
)
