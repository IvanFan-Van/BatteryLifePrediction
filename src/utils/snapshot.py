"""Freeze the importable source tree for queued experiments."""
import hashlib
from pathlib import Path
import shutil


def source_manifest(root):
    root = Path(root)
    files = sorted(path for folder in ("src", "scripts") for path in (root / folder).rglob("*.py"))
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def copy_source(root, destination, files):
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
