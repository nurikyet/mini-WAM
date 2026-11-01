from __future__ import annotations

import os
from pathlib import Path

from .common import completed, file_hash, finish, stage, write_json
from .dataset import load_arrays
import numpy as np
import torch

VAE_ID = "stabilityai/sd-vae-ft-mse"
VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
SCALE = 0.18215


def vae_source():
    return os.environ.get("MINIWAM_VAE", VAE_ID)


def vae_identity():
    source = vae_source()
    if Path(source).is_dir():
        files = {p.name: file_hash(p) for p in sorted(Path(source).iterdir())
                 if p.suffix in (".json", ".safetensors", ".bin")}
        if not files:
            raise ValueError("MINIWAM_VAE must be a Diffusers AutoencoderKL directory")
        return {"local_files_sha256": files}
    return {"repo_id": source, "revision": VAE_REVISION}


def load_vae(device):
    from diffusers import AutoencoderKL
    source = vae_source()
    kwargs = {"local_files_only": True} if Path(source).is_dir() else {"revision": VAE_REVISION}
    # Frozen VAE always FP32. No GradScaler or image augmentation in this cache.
    vae = AutoencoderKL.from_pretrained(source, **kwargs).to(device=device, dtype=torch.float32)
    return vae.eval().requires_grad_(False)


@torch.no_grad()
def encode(vae, rgb, device):
    x = torch.as_tensor(np.array(rgb), device=device).permute(0, 3, 1, 2).float()/127.5-1
    return vae.encode(x).latent_dist.mean * SCALE


@torch.no_grad()
def decode(vae, z):
    shape = z.shape
    x = vae.decode(z.flatten(0, len(shape)-4).float()/SCALE).sample
    return ((x.clamp(-1, 1)+1)/2).reshape(*shape[:-3], 3, 128, 128)


@torch.no_grad()
def cache_latents(dataset, out, device, resume=False, batch_size=32):
    dataset = Path(dataset)
    identity = vae_identity()
    out = stage(out, {"dataset_sha256": file_hash(dataset/"manifest.json"), "vae": identity,
                      "scale": SCALE, "posterior": "mean"}, resume)
    if completed(out):
        return out
    vae = load_vae(device)
    report, files = {}, ["vae.json", "reconstruction.json"]
    for split in ("train", "val", "test"):
        arrays = load_arrays(dataset, split)
        rgb = arrays["observations"]
        n, frames = rgb.shape[:2]
        latents = np.lib.format.open_memmap(out/f"{split}.npy", mode="w+", dtype=np.float16,
                                          shape=(n, frames, 4, 16, 16))
        flat = rgb.reshape(-1, 128, 128, 3)
        flat_z = latents.reshape(-1, 4, 16, 16)
        for start in range(0, len(flat), batch_size):
            flat_z[start:start+batch_size] = encode(vae, flat[start:start+batch_size], device).cpu().numpy()
        latents.flush()
        # Isolate representation failures before blaming diffusion or planning.
        sample_count = min(16, n)
        z = torch.as_tensor(np.array(latents[:sample_count]), device=device).float()
        reconstructed = decode(vae, z).cpu().numpy().transpose(0, 1, 3, 4, 2)
        truth = np.array(rgb[:sample_count]).astype(np.float32)/255
        error = (reconstructed-truth)**2
        changed = np.abs(truth[:, 1:]-truth[:, :-1]).max(-1) > 0.02
        board = np.zeros((128, 128), bool)
        board[:108, 10:118] = True
        mse = float(error.mean())
        report[split] = {"sample_episodes": sample_count, "rgb_mse": mse,
                         "psnr": float(-10*np.log10(max(mse, 1e-12))),
                         "changed_pixel_mse": float(error[:, 1:][changed].mean()) if changed.any() else 0.0,
                         "board_mse": float(error[:, :, board].mean())}
        from PIL import Image
        for i in range(min(4, sample_count)):
            montage = np.concatenate([truth[i, 0], reconstructed[i, 0], truth[i, -1], reconstructed[i, -1]], axis=1)
            name = f"{split}_reconstruction_{i}.png"
            Image.fromarray((montage.clip(0, 1)*255).astype(np.uint8)).save(out/name)
            files.append(name)
        files.append(f"{split}.npy")
        print(f"Cached {split}: {n} episodes, {latents.nbytes/2**20:.1f} MiB", flush=True)
    write_json(out / "vae.json", {**identity, "scale": SCALE})
    write_json(out / "reconstruction.json", report)
    finish(out, files)
    return out


class LatentWindows:
    """Only latents, actions, and masks; never imports a bank/state/SR label."""
    def __init__(self, dataset, cache, split, horizon):
        self.dataset, self.split = Path(dataset), split
        self.z = np.load(Path(cache)/f"{split}.npy", mmap_mode="r", allow_pickle=False)
        arrays = load_arrays(dataset, split)
        self.actions, self.valid, self.expert = (arrays[k] for k in ("actions", "valid", "expert_mask"))
        self.indices = np.argwhere(self.valid)
        self.horizon = horizon
        if horizon < 1 or horizon > self.actions.shape[1]:
            raise ValueError("horizon must be between 1 and the real episode depth; no fake horizons")

    def __len__(self):
        return len(self.indices)

    def batch(self, ids, device):
        episode, start = self.indices[np.asarray(ids)].T
        steps = start[:, None]+np.arange(self.horizon)
        depth = self.actions.shape[1]
        safe = steps.clip(max=depth-1)
        valid = (steps < depth) & self.valid[episode[:, None], safe]
        expert = valid & self.expert[episode[:, None], safe]
        arrays = (self.z[episode, start], self.actions[episode[:, None], safe].clip(min=0),
                  self.z[episode[:, None], (steps+1).clip(max=depth)], valid, expert)
        return tuple(torch.as_tensor(np.array(a), device=device).float() if i in (0, 2)
                     else torch.as_tensor(np.array(a), device=device) for i, a in enumerate(arrays))
