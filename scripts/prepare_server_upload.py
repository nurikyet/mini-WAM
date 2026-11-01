#!/usr/bin/env python3
"""Copy the standalone mini-WAM project into a local repository/upload folder.

Copies only the five project folders and four root files. Existing Git metadata
and other destination files are preserved. No GitHub access, commits or pushes.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


FOLDERS = ("miniwam", "configs", "scripts", "tests", "reports")
FILES = ("README.md", "pyproject.toml", "requirements.txt", ".gitignore")
SKIP = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".cache", ".DS_Store"}


def copy_project(source: Path, destination: Path):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or destination in source.parents:
        raise ValueError("Choose a separate destination folder")
    if any(destination == source/name or source/name in destination.parents for name in FOLDERS):
        raise ValueError("Destination cannot be inside a folder being copied")
    for name in (*FOLDERS, *FILES):
        if not (source/name).exists():
            raise FileNotFoundError(source/name)
    paths = [source/name for name in FILES]
    for name in FOLDERS:
        for current, directories, files in os.walk(source/name, followlinks=False):
            directory = Path(current)
            directories[:] = [entry for entry in directories if entry not in SKIP
                               and not entry.startswith(".venv") and not (directory/entry).is_symlink()]
            paths.extend(directory/entry for entry in files if entry not in SKIP
                         and not entry.endswith((".pyc", ".pyo")) and not (directory/entry).is_symlink())
    # Reject symlinks in existing output paths to keep all writes in destination.
    for path in paths:
        relative = path.relative_to(source)
        target = destination/relative
        for component in (target, *target.parents):
            if component == destination.parent:
                break
            if component.is_symlink():
                raise ValueError(f"Destination contains a symlink: {component}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = destination/path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return len(paths), sum(path.stat().st_size for path in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count, size = copy_project(args.source, args.out)
    print(f"Copied {count} files ({size/1024/1024:.2f} MiB) to {args.out.resolve()}")


if __name__ == "__main__":
    main()
