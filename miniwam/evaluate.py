"""Closed-loop SR, offline forecasts, per-layout failures and inspectable previews."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import Windows, audit_dataset, save_gif
from .env import ACTION_NAMES, EnvConfig, NavigationEnv, task_banks
from .model import logits_to_rgb, pixel_classes
from .utils import file_hash, write_json


def rgb_batch(frames, config, device):
    rgb = torch.as_tensor(np.stack(frames), dtype=torch.float32, device=device).permute(0, 3, 1, 2)/255
    return F.avg_pool2d(rgb, config.tile_size)


@torch.no_grad()
def closed_loop(model, config, split, device, horizon=1, state_policy=False, random_seed=123):
    banks = task_banks(config)
    tasks = sum(banks.values(), []) if split == "all" else banks[split]
    envs = [NavigationEnv(config) for _ in tasks]
    frames = [env.reset(task) for env, task in zip(envs, tasks)]
    traces = [[] for _ in tasks]
    rng = np.random.default_rng(random_seed)
    if model is not None:
        model.eval()
    for t in range(config.depth):
        ids = [i for i, env in enumerate(envs) if not env.done]
        if not ids:
            break
        if model is None:
            actions = rng.integers(4, size=len(ids))
        elif state_policy:
            states = torch.as_tensor(np.stack([envs[i].state() for i in ids]), device=device)
            actions = model(states)[0].argmax(-1).cpu().numpy()
        else:
            pixels = rgb_batch([frames[i] for i in ids], config, device)
            actions = model.act(pixels, horizon, remaining=config.depth-t).cpu().numpy()
        for i, action in zip(ids, actions):
            before = int(abs(envs[i].agent-envs[i].goal).sum())
            frames[i], reward, done, info = envs[i].step(int(action))
            traces[i].append({"action": ACTION_NAMES[int(action)], "distance_before": before,
                              "distance_after": info["distance"], "blocked": info["blocked"]})
    successes = sum(env.success for env in envs)
    failures, by_displacement = [], {}
    for task, env, trace in zip(tasks, envs, traces):
        key = str(task.delta)
        bucket = by_displacement.setdefault(key, {"successes": 0, "episodes": 0})
        bucket["successes"] += int(env.success)
        bucket["episodes"] += 1
        if not env.success:
            reason = "boundary_collision" if any(t["blocked"] for t in trace) else "wrong_direction"
            failures.append({"task": task.key, "reason": reason, "trace": trace})
    return {"split": split, "success_rate": successes/len(tasks), "successes": successes,
        "episodes": len(tasks), "protocol": "exhaustive finite bank, argmax/MPC, reobserve after each action",
        "failure_counts": dict(Counter(f["reason"] for f in failures)), "failures": failures,
        "by_displacement": by_displacement}


@torch.no_grad()
def offline_metrics(model, windows, image_weight=1.0):
    model.eval()
    sums = Counter()
    per_step = [Counter() for _ in range(windows.horizon)]
    for ids in torch.arange(len(windows), device=windows.pixels.device).split(256):
        pixels, actions, targets, valid, expert = windows.batch(ids)
        loss, _ = model.loss(pixels, actions, targets, valid, expert, image_weight)
        sums["loss_sum"] += float(loss)*len(ids)
        sums["windows"] += len(ids)
        output = model.rollout(pixels, windows.horizon, actions)
        if output["action_logits"] is not None:
            correct = output["action_logits"].argmax(-1) == actions
            autonomous = model.rollout(pixels, windows.horizon)["actions"] == actions
            sums["expert_steps"] += int(expert.sum())
            sums["action_correct"] += int((correct & expert).sum())
            sums["open_loop_correct"] += int((autonomous & expert).sum())
            for t, bucket in enumerate(per_step):
                bucket["expert_steps"] += int(expert[:, t].sum())
                bucket["correct"] += int((correct[:, t] & expert[:, t]).sum())
        if output["image_logits"] is not None:
            predicted = output["image_logits"].argmax(2)
            truth = pixel_classes(targets)
            context = pixel_classes(pixels)[:, None]
            mask = valid[..., None, None].expand_as(truth)
            foreground = mask & ((truth != 0) | (context != 0))
            rgb = logits_to_rgb(output["image_logits"])
            permuted = logits_to_rgb(model.rollout(pixels, windows.horizon, (actions+1)%4)["image_logits"])
            mse = (rgb-targets).square().mean(2)
            persistence = (pixels[:, None]-targets).square().mean(2)
            sums["cells"] += int(mask.sum())
            sums["cell_correct"] += int(((predicted == truth) & mask).sum())
            sums["foreground"] += int(foreground.sum())
            sums["foreground_correct"] += int(((predicted == truth) & foreground).sum())
            sums["mse_sum"] += float((mse*mask).sum())
            sums["persistence_mse_sum"] += float((persistence*mask).sum())
            sums["action_effect_sum"] += float(((rgb-permuted).abs().mean(2)*mask).sum())
            for t, bucket in enumerate(per_step):
                bucket["frames"] += int(valid[:, t].sum())
                bucket["foreground"] += int(foreground[:, t].sum())
                bucket["foreground_correct"] += int(((predicted[:, t] == truth[:, t]) & foreground[:, t]).sum())
    result = {"windows": int(sums["windows"]), "loss": sums["loss_sum"]/sums["windows"],
              "requested_horizon": windows.horizon, "per_horizon": []}
    if sums["expert_steps"]:
        result.update(action_accuracy=sums["action_correct"]/sums["expert_steps"],
                      open_loop_action_accuracy=sums["open_loop_correct"]/sums["expert_steps"])
    if sums["cells"]:
        result.update(cell_accuracy=sums["cell_correct"]/sums["cells"],
            foreground_cell_accuracy=sums["foreground_correct"]/sums["foreground"],
            forecast_mse=sums["mse_sum"]/sums["cells"],
            repeat_context_mse=sums["persistence_mse_sum"]/sums["cells"],
            action_sensitivity_mae=sums["action_effect_sum"]/sums["cells"])
    for t, bucket in enumerate(per_step, 1):
        record = {"step": t, "expert_labels": int(bucket["expert_steps"]), "image_targets": int(bucket["frames"])}
        if bucket["expert_steps"]:
            record["action_accuracy"] = bucket["correct"]/bucket["expert_steps"]
        if bucket["foreground"]:
            record["foreground_cell_accuracy"] = bucket["foreground_correct"]/bucket["foreground"]
        result["per_horizon"].append(record)
    return result


@torch.no_grad()
def forecast_preview(model, windows, out):
    if model.image_head is None:
        return
    out = Path(out)
    ids = torch.arange(min(8, len(windows)), device=windows.pixels.device)
    pixels, actions, targets, valid, expert = windows.batch(ids)
    logits = model.rollout(pixels, windows.horizon, actions)["image_logits"]
    prediction = logits_to_rgb(logits).cpu().numpy()
    np.savez_compressed(out/"forecast.npz", context=pixels.cpu().numpy(), truth=targets.cpu().numpy(),
                        prediction=prediction, actions=actions.cpu().numpy(), valid=valid.cpu().numpy())
    # Row 1: truth; row 2: recorded-action-conditioned forecast. Invalid padding omitted.
    for i in range(min(4, len(ids))):
        count = int(valid[i].sum())
        initial = pixels[i].cpu().numpy()
        actual = np.concatenate([initial[None], targets[i, :count].cpu().numpy()])
        predicted = np.concatenate([initial[None], prediction[i, :count]])
        frames = []
        for truth, pred in zip(actual, predicted):
            frames.append((np.concatenate([truth, pred], axis=2).transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8))
        save_gif(out/f"forecast_{i}_truth_left.gif", frames, scale=20)


def evaluate_model(checkpoint_path, dataset, split, out, device):
    from .train import load_model
    model, config, checkpoint = load_model(checkpoint_path, device)
    audit_dataset(dataset)
    if checkpoint["dataset_manifest_sha256"] != file_hash(Path(dataset)/"manifest.json"):
        raise ValueError("Checkpoint was trained on a different dataset")
    env_config = EnvConfig(**checkpoint["env_config"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    windows = Windows(dataset, split, config.horizon, device)
    offline = offline_metrics(model, windows, config.image_weight)
    result = {"mode": config.mode, "horizon": config.horizon,
        "checkpoint_sha256": file_hash(checkpoint_path), "selected_update": checkpoint["update"],
        "validation_sr": checkpoint["validation"]["val_sr"],
        "environment": asdict(env_config), "closed_loop": closed_loop(model, env_config, split, device, config.horizon),
        "offline": offline, "diagnosis": []}
    if result["closed_loop"]["success_rate"] < 1:
        if offline.get("action_accuracy", 1) < 1:
            result["diagnosis"].append("Pixel policy makes errors even on expert-labelled offline inputs; inspect failures by displacement.")
        if model.image_head is not None:
            if offline["forecast_mse"] >= offline["repeat_context_mse"]:
                result["diagnosis"].append("Image forecast does not beat copying context; dynamics are underfit.")
            if offline["action_sensitivity_mae"] < 1e-3:
                result["diagnosis"].append("Predicted images barely change with actions; check counterfactual coverage and training budget.")
            if config.mode == "video":
                result["diagnosis"].append("Inspect model/planner errors separately: planner scores learned frames using visible goal distance.")
        if not result["diagnosis"]:
            result["diagnosis"].append("Offline accuracy does not guarantee closed-loop SR; inspect failure traces and distribution shift.")
    else:
        result["diagnosis"].append("100% on this finite held-out bank, not a guarantee for larger boards/obstacles/new seeds.")
    write_json(out/"evaluation.json", result)
    forecast_preview(model, windows, out)
    return result
