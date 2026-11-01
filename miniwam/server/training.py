from __future__ import annotations

import math
from pathlib import Path

from .common import (completed, file_hash, finish, restore_rng, rng_state,
                     runtime, seed, stage, write_json)
from .evaluation import closed_loop, counterfactual_metrics, offline
from .model import NanoWAM, autocast
from .vae import LatentWindows, load_vae
from miniwam.utils import save_checkpoint
import numpy as np
import torch

MODEL_KEYS = ("width", "layers", "heads", "patch_size", "action_width")


def build(mode, horizon, config, device):
    return NanoWAM(mode, horizon, **{k: config[k] for k in MODEL_KEYS}).to(device)


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = build(saved["mode"], saved["horizon"], saved["model_config"], device)
    model.load_state_dict(saved["model"])
    return model.eval(), saved


def train(bank_dir, dataset, cache, config, mode, horizon, model_seed, out, device, resume=False):
    bank_dir, dataset, cache = map(Path, (bank_dir, dataset, cache))
    out = stage(out, {"model": config, "mode": mode, "horizon": horizon, "seed": model_seed,
        "bank_sha256": file_hash(bank_dir/"manifest.json"), "dataset_sha256": file_hash(dataset/"manifest.json"),
        "latents_sha256": file_hash(cache/"complete.json")}, resume)
    if completed(out):
        return out / "best.pt"
    rng = seed(model_seed)
    train_windows, val_windows = [LatentWindows(dataset, cache, s, horizon) for s in ("train", "val")]
    model = build(mode, horizon, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    def lr_scale(step):
        if step < config["warmup"]:
            return (step+1)/max(1, config["warmup"])
        progress = (step-config["warmup"])/max(1, config["updates"]-config["warmup"])
        return 0.1+0.9*0.5*(1+math.cos(math.pi*min(1., progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    vae = load_vae(device)
    start, best, consecutive, history = 0, (-1., float("-inf")), 0, []
    if (out / "latest.pt").exists():
        saved = torch.load(out/"latest.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best, consecutive, history = saved["step"], tuple(saved["best"]), saved["consecutive"], saved["history"]
        restore_rng(saved["rng"], rng)
    step = start
    stop = False
    for step in range(start+1, config["updates"]+1) if consecutive < config["patience"] else ():
        model.train()
        batch = train_windows.batch(rng.integers(len(train_windows), size=config["batch_size"]), device)
        with autocast(device):
            loss, losses = model.loss(*batch, image_weight=config["image_weight"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {mode} loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Nonfinite {mode} gradients at step {step}")
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == 1:
            print(f"{mode} n={horizon} seed={model_seed} step={step}/{config['updates']} loss={float(loss):.5f} {losses}", flush=True)
        if step % config["eval_every"] == 0 or step == config["updates"]:
            model.eval()
            validation = offline(model, val_windows, device, config["val_windows"], config["ddim_steps"],
                vae, dataset, "val", out/"validation")
            control = closed_loop(model, vae, bank_dir, "val", device, config["ddim_steps"])
            key = (control["sr"], -validation["loss"])
            dynamics_ok = mode == "policy" or validation["latent_mse"] < validation["persistence_latent_mse"]
            consecutive = consecutive+1 if control["sr"] == 1. and dynamics_ok else 0
            record = {"step": step, "val_sr": control["sr"], "offline": validation}
            history.append(record)
            write_json(out/"validation/evaluation.json", {"closed_loop": control, "offline": validation})
            write_json(out/"metrics.json", history)
            if key >= best:
                best = key
                save_checkpoint(out/"best.pt", {"model": model.state_dict(), "mode": mode, "horizon": horizon,
                    "model_config": config, "seed": model_seed, "step": step, "selection": record})
            print(f"Validation {mode} n={horizon}: SR={control['sr']:.3f}, loss={validation['loss']:.5f}", flush=True)
            stop = consecutive >= config["patience"]
        if step % config["checkpoint_every"] == 0 or step % config["eval_every"] == 0 or step == config["updates"]:
            save_checkpoint(out/"latest.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step, "best": list(best), "consecutive": consecutive,
                "history": history, "rng": rng_state(rng)})
        if stop:
            break
    # Test is opened only after best checkpoint selection finishes.
    model, saved = load_model(out/"best.pt", device)
    for split in ("train", "val", "test"):
        control = closed_loop(model, vae, bank_dir, split, device, config["ddim_steps"], out=out/split)
        metrics = offline(model, LatentWindows(dataset, cache, split, horizon), device, config["val_windows"],
                          config["ddim_steps"], vae, dataset, split, out/split)
        result = {"closed_loop": control, "offline": metrics, "selected_step": saved["step"]}
        if mode != "policy":
            result["counterfactual"] = counterfactual_metrics(model, vae, LatentWindows(dataset,cache,split,horizon), device, config["ddim_steps"])
        if split == "test" and mode != "video":
            result["sampled_T1"] = closed_loop(model, vae, bank_dir, split, device, config["ddim_steps"], temperature=1.)
        write_json(out/split/"evaluation.json", result)
    finish(out, ["best.pt", "latest.pt", "metrics.json", "train/evaluation.json", "val/evaluation.json", "test/evaluation.json"],
           steps=step, parameters=sum(p.numel() for p in model.parameters()), runtime=runtime())
    return out/"best.pt"
