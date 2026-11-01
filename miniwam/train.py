from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import Windows, audit_dataset
from .model import MODES, PixelWAM
from .utils import file_hash, prepare_run, restore_rng, rng_state, runtime, save_checkpoint, write_json


@dataclass(frozen=True)
class ModelConfig:
    mode: str = "joint"
    horizon: int = 3
    width: int = 32
    updates: int = 1200
    batch_size: int = 128
    lr: float = 0.003
    image_weight: float = 1.0
    eval_every: int = 50
    patience: int = 3
    seed: int = 3407

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if min(self.horizon, self.width, self.updates, self.batch_size, self.eval_every, self.patience) < 1:
            raise ValueError("Model counts must be positive")
        if self.lr <= 0 or self.image_weight <= 0:
            raise ValueError("lr and image_weight must be positive")


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    model = PixelWAM(config.mode, config.width).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, checkpoint


def train_model(dataset, config: ModelConfig, out, device, resume=False):
    # Imported here to keep load/evaluation dependencies acyclic.
    from .evaluate import closed_loop, offline_metrics
    dataset = Path(dataset)
    audit_dataset(dataset)
    metadata = json.loads((dataset/"manifest.json").read_text())
    out = prepare_run(out, {"model": asdict(config), "dataset_manifest_sha256": file_hash(dataset/"manifest.json")}, resume)
    if (out/"complete.json").exists():
        return out/"best.pt"
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    train = Windows(dataset, "train", config.horizon, device)
    val = Windows(dataset, "val", config.horizon, device)
    model = PixelWAM(config.mode, config.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    start_update, best_key, consecutive, history = 0, (-1.0, float("-inf")), 0, []
    if (out/"latest.pt").exists():
        saved = torch.load(out/"latest.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_update, best_key, consecutive = saved["update"], tuple(saved["best_key"]), saved["consecutive"]
        history = saved["history"]
        restore_rng(saved["rng"], rng)
    started = time.perf_counter()
    initial = offline_metrics(model, val, config.image_weight)
    window_losses = []
    update = start_update
    pending = range(start_update+1, config.updates+1) if consecutive < config.patience else ()
    for update in pending:
        model.train()
        ids = torch.as_tensor(rng.integers(len(train), size=config.batch_size), device=device)
        loss, losses = model.loss(*train.batch(ids), image_weight=config.image_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite WAM loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        window_losses.append(float(loss.detach()))
        evaluate = update % config.eval_every == 0 or update == config.updates
        if evaluate:
            model.eval()
            offline = offline_metrics(model, val, config.image_weight)
            validation = closed_loop(model, train.env_config, "val", device, config.horizon)
            val_sr = validation["success_rate"]
            # Keep image prediction honest even when a planner exploits imperfections.
            accuracy = offline.get("foreground_cell_accuracy", 1.0)
            consecutive = consecutive+1 if val_sr == 1 and accuracy >= 0.995 else 0
            record = {"update": update, "mean_loss": float(np.mean(window_losses)), **losses,
                      "val_sr": val_sr, "offline_validation": offline}
            history.append(record)
            window_losses = []
            print(f"{config.mode} {update}/{config.updates}: loss={record['mean_loss']:.4f}, val SR={val_sr:.3f}", flush=True)
            key = (val_sr, -offline["loss"])
            if key >= best_key:
                best_key = key
                save_checkpoint(out/"best.pt", {"model": model.state_dict(), "model_config": asdict(config),
                    "env_config": metadata["environment"], "dataset_manifest_sha256": file_hash(dataset/"manifest.json"),
                    "update": update, "validation": record})
        # Saving every evaluation also bounds lost work after preemption to eval_every-1 updates.
        if evaluate:
            save_checkpoint(out/"latest.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "model_config": asdict(config), "update": update, "best_key": list(best_key),
                "consecutive": consecutive, "history": history, "rng": rng_state(rng)})
        if consecutive >= config.patience:
            break
    write_json(out/"metrics.json", history)
    write_json(out/"complete.json", {"updates": update,
        "session_seconds": time.perf_counter()-started, "initial_validation": initial, "runtime": runtime(device),
        "parameters": sum(p.numel() for p in model.parameters()), "selection": "validation only",
        "effective_prediction_horizon": min(config.horizon, train.env_config.depth),
        "requested_prediction_horizon": config.horizon})
    return out/"best.pt"
