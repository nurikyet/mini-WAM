"""Diagnostic upper controls. Simulator futures never enter the learned planner."""
from __future__ import annotations

from pathlib import Path

from .common import read_json, write_json
from .evaluation import visual_score
from .vae import decode, encode
from .xland import load_bank, render_ts, simulator, single_ts
from .progress import Progress, log
from .protocol import episode_limit, renderer_version
import jax
import numpy as np
import torch


@torch.no_grad()
def scorer_controls(bank_dir, vae, device, out):
    bank = load_bank(bank_dir, "val")
    config = read_json(Path(bank_dir)/"manifest.json")["config"]
    depth, renderer = episode_limit(config), renderer_version(config)
    env, params = simulator(depth)
    step = jax.jit(env.step)
    report = {"scope": "validation only; diagnostic true-transition ceiling for visual task cost",
              "uses_oracle_futures": True, "used_for_training_or_control": False, "controls": {}}
    for label in ("perfect_rgb", "vae_reconstructed_rgb"):
        successes, failures = [], []
        progress = Progress(f"Визуальный контроль {label}: эпизоды", len(bank["goal"]))
        for task in range(len(bank["goal"])):
            ts = single_ts(bank, task)
            trace = []
            for t in range(depth):
                rgb = render_ts(ts, renderer)
                candidates = [step(params, ts, action) for action in range(6)]
                frames = np.stack([render_ts(s, renderer) for s in candidates])
                if label == "perfect_rgb":
                    predicted = torch.as_tensor(frames.copy(), device=device).permute(0,3,1,2).float()/255
                else:
                    predicted = decode(vae, encode(vae, frames, device))
                current = torch.as_tensor(rgb.copy(), device=device).permute(2,0,1).float()[None]/255
                scores = visual_score(predicted, current.expand(6,-1,-1,-1))
                action = int(scores.argmax())
                ts = candidates[action]
                trace.append({"t": t, "action": action, "scores": scores.cpu().tolist()})
                if bool(ts.last()):
                    break
            success = float(ts.reward) > 0
            successes.append(success)
            if not success:
                failures.append({"task_id": task, "goal": bank["goal"][task].tolist(), "trace": trace})
            progress.update(task+1, successes=sum(successes))
        report["controls"][label] = {"episodes": len(successes), "successes": sum(successes),
                                      "sr": float(np.mean(successes)), "failures": failures}
        log("ИТОГ ВИЗУАЛЬНОГО КОНТРОЛЯ", control=label, sr=f"{np.mean(successes):.1%}")
    write_json(Path(out)/"scorer_controls.json", report)
    return report
