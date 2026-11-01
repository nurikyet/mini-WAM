#!/usr/bin/env python3
"""Copy sources into a Git-ready directory, without nested repos/venvs/weights.

Original files and nested Git histories are left intact. No git init/add/push.
Run with --dry-run first to inspect the exact omissions. Standard library only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


EXCLUDE_DIRS = {".git", ".git-history", ".cache", ".python", "__pycache__", ".pytest_cache",
                ".ipynb_checkpoints", "tmp", "runs", "logs", "data", "git-export", "build", "dist"}
EXCLUDE_SUFFIXES = {".pyc", ".pt", ".pth", ".safetensors", ".msgpack", ".zip", ".log"}


def inventory(source):
    included, omitted = [], []
    stack = [source]
    while stack:
        directory = stack.pop()
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(source).as_posix()
            if path.is_symlink():
                omitted.append({"path": relative, "reason": "symlink; avoid machine-specific/external paths"})
            elif path.is_dir():
                if path.name in EXCLUDE_DIRS or path.name.startswith(".venv") or path.name.endswith(".egg-info"):
                    omitted.append({"path": relative+"/", "reason": "local/generated directory or nested Git metadata"})
                else:
                    stack.append(path)
            elif (path.name == ".DS_Store" or path.name == ".env" or path.name.startswith(".env.")
                  or path.name.startswith("gangway-") or path.suffix == ".pem"
                  or path.suffix in EXCLUDE_SUFFIXES):
                omitted.append({"path": relative, "reason": "local file, credentials, checkpoint or archive"})
            elif path.stat().st_size >= 95*1024**2:
                omitted.append({"path": relative, "reason": "file >= 95 MiB; transfer data separately"})
            else:
                included.append(path)
    return sorted(included), omitted


def export(source, destination, dry_run=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Export destination must be outside the source folder")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    files, omitted = inventory(source)
    manifest = {"files": len(files), "bytes": sum(p.stat().st_size for p in files), "omitted": omitted,
                "note": "Sources of both upstream repos are vendored as ordinary files. No gitlinks/submodules."}
    if not dry_run:
        destination.mkdir(parents=True)
        for path in files:
            target = destination/path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (destination/"git-export-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = export(args.source, args.out, args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
