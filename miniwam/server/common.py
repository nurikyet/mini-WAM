from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
# Small state simulations use CPU JAX; NanoWM/PPO use the H100 via torch.
# No CUDA JAX wheel or competing GPU preallocation is needed for this curriculum.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, str(ROOT / "vendor/xland_minigrid"))
sys.path.insert(0, str(ROOT / "vendor/nanowm"))

import numpy as np
import torch

from miniwam.utils import file_hash, write_json

HF_REVISION = "430ac76a22b99d860c39944d6904d6d4cc2763e4"
BENCHMARK_SHA256 = "9c056812eaa6dbb3cd4dc85315c7173825b49f8d99f666802656094ef2c6df42"
BENCHMARK_FILE = "trivial_1m_v2"


def source_digest():
    h = hashlib.sha256()
    paths = sorted((ROOT / "miniwam").rglob("*.py"))
    paths += sorted((ROOT / "vendor").rglob("*.py"))
    for path in paths:
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def stage(out, config, resume):
    out = Path(out)
    metadata = {"config": config, "source_sha256": source_digest()}
    if out.exists():
        if not resume:
            raise FileExistsError(f"{out}: choose a new output or pass --resume")
        if not (out / "run.json").exists() or read_json(out / "run.json") != metadata:
            raise ValueError(f"Code/config/input changed for {out}; choose a new output")
    else:
        out.mkdir(parents=True)
        write_json(out / "run.json", metadata)
    return out


def read_json(path):
    return json.loads(Path(path).read_text())


def save_npz(path, **arrays):
    import tempfile
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_server():
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("Training is server-only: Linux and CUDA torch required. Run scripts/setup_server.sh on gangway.")
    if not str(Path.cwd().resolve()).startswith(str(Path.home().resolve()) + os.sep):
        raise RuntimeError("Run the cloned repository inside your persistent server HOME")
    torch.cuda.set_device(0)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    return torch.device("cuda:0")


def runtime():
    import importlib.metadata as md
    versions = {p: md.version(p) for p in ("torch", "torchvision", "numpy", "jax", "jaxlib", "flax", "diffusers", "timm")}
    return {"versions": versions, "python": platform.python_version(), "source_sha256": source_digest(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda, "jax_platform": os.environ["JAX_PLATFORMS"],
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()}


def verify_files(directory, hashes):
    for name, expected in hashes.items():
        if file_hash(Path(directory) / name) != expected:
            raise ValueError(f"Checksum mismatch: {directory}/{name}")


def completed(out, verify=True):
    path = Path(out) / "complete.json"
    if not path.exists():
        return False
    data = read_json(path)
    if verify:
        verify_files(out, data.get("files_sha256", {}))
    return True


def finish(out, names, **metadata):
    write_json(Path(out) / "complete.json", {**metadata, "files_sha256": {n: file_hash(Path(out) / n) for n in names}})


def seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)


def rng_state(rng):
    return {"numpy": rng.bit_generator.state, "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, rng):
    rng.bit_generator.state = state["numpy"]
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([a.cpu() for a in state["cuda"]])
