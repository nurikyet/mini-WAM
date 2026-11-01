"""Aligned RGB sequences and expert masks; sharded on disk by split."""
from __future__ import annotations

from pathlib import Path

from .common import completed, file_hash, finish, read_json, seed, stage, write_json
from .expert import load_actor
from .xland import ACTION_NAMES, features, load_bank, render_ts, simulator, single_ts
from .progress import Progress, log
import imageio.v3 as iio
import jax
import numpy as np
import torch


@torch.no_grad()
def collect(bank_dir, expert_path, config, out, device, resume=False):
    bank_dir, expert_path = Path(bank_dir), Path(expert_path)
    out = stage(out, {"collection": config, "bank_sha256": file_hash(bank_dir/"manifest.json"),
                      "expert_sha256": file_hash(expert_path)}, resume)
    if completed(out):
        log("ПРОПУСК: датасет уже собран и проверен", path=out)
        return out
    depth = read_json(bank_dir / "manifest.json")["config"]["depth"]
    rng = seed(config["seed"])
    actor = load_actor(expert_path, device)
    env, params = simulator(depth)
    step = jax.jit(env.step)
    manifest = {"schema": 2, "bank_manifest_sha256": file_hash(bank_dir/"manifest.json"), "depth": depth,
        "expert_sha256": file_hash(expert_path), "actions": ACTION_NAMES, "rgb_shape": [128, 128, 3],
        "alignment": "observations[t] -- actions[t] --> observations[t+1]",
        "padding": "terminal frame repeated, action=-1, valid=False, expert_mask=False",
        "expert_mask": "True iff this actual action was selected by PPO, never for forced/random actions",
        "policy_inputs": "RGB only, including public goal/pocket panel; no states/rewards/SR/optimal-action mask",
        "splits": {}}
    paths = ["manifest.json", "audit.json"]
    for split in ("train", "val", "test"):
        bank = load_bank(bank_dir, split)
        repeats = config["train_repeats"] if split == "train" else config["eval_repeats"]
        if repeats < 7:
            raise ValueError("Need >=7 repeats: six initial actions and one pure PPO trajectory")
        n = len(bank["goal"])*repeats
        directory = out / split
        directory.mkdir(exist_ok=True)
        specs = {"observations": (np.uint8, (n, depth+1, 128, 128, 3)), "actions": (np.int64, (n, depth)),
                 "rewards": (np.float32, (n, depth)), "valid": (bool, (n, depth)), "expert_mask": (bool, (n, depth)),
                 "success": (bool, (n,)), "lengths": (np.int32, (n,)), "task_ids": (np.int32, (n,)),
                 "source_rows": (np.int64, (n,)), "pure_expert": (bool, (n,))}
        # Diagnostic states are separate arrays; LatentWindows never reads them.
        specs["agent_positions"] = (np.int32, (n, depth+1, 2))
        specs["agent_directions"] = (np.int32, (n, depth+1))
        arrays = {k: np.lib.format.open_memmap(directory/f"{k}.npy", mode="w+", dtype=dtype, shape=shape)
                  for k, (dtype, shape) in specs.items()}
        for k, arr in arrays.items():
            arr[:] = -1 if k == "actions" else 0
        metadata = []
        progress = Progress(f"Сбор RGB {split}: эпизоды", n)
        for task in range(len(bank["goal"])):
            for repeat in range(repeats):
                i = task*repeats+repeat
                ts = single_ts(bank, task)
                arrays["observations"][i, 0] = render_ts(ts)
                arrays["agent_positions"][i, 0] = np.asarray(ts.state.agent.position)
                arrays["agent_directions"][i, 0] = int(ts.state.agent.direction)
                for t in range(depth):
                    batched = jax.tree.map(lambda a: a[None], ts)
                    policy_action = int(actor(torch.as_tensor(features(batched, depth), device=device))[0].argmax(-1).item())
                    is_expert = True
                    if t == 0 and repeat < 6:
                        action, is_expert = repeat, False
                    elif repeat != 6 and rng.random() < config["epsilon"]:
                        action, is_expert = int(rng.integers(6)), False
                    else:
                        action = policy_action
                    ts = step(params, ts, action)
                    rgb = render_ts(ts)
                    arrays["observations"][i, t+1] = rgb
                    arrays["agent_positions"][i, t+1] = np.asarray(ts.state.agent.position)
                    arrays["agent_directions"][i, t+1] = int(ts.state.agent.direction)
                    arrays["actions"][i, t], arrays["rewards"][i, t] = action, float(ts.reward)
                    arrays["valid"][i, t], arrays["expert_mask"][i, t] = True, is_expert
                    if bool(ts.last()):
                        arrays["observations"][i, t+2:] = rgb
                        arrays["agent_positions"][i, t+2:] = np.asarray(ts.state.agent.position)
                        arrays["agent_directions"][i, t+2:] = int(ts.state.agent.direction)
                        break
                success = float(ts.reward) > 0
                arrays["success"][i], arrays["lengths"][i], arrays["task_ids"][i] = success, t+1, task
                arrays["source_rows"][i], arrays["pure_expert"][i] = bank["source_row"][task], repeat == 6
                metadata.append({"episode_id": i, "task_id": task, "source_row": int(bank["source_row"][task]),
                                 "group_id": int(bank["group_id"][task]), "success": success, "length": t+1,
                                 "pure_expert": repeat == 6})
                if repeat == 6 and task < 4:
                    frames = np.asarray(arrays["observations"][i, :t+2])
                    iio.imwrite(directory/f"episode_{task}.mp4", frames, fps=2, macro_block_size=1)
                progress.update(i+1)
        for arr in arrays.values():
            arr.flush()
        (directory / "episodes.jsonl").write_text("".join(__import__("json").dumps(x)+"\n" for x in metadata))
        files = {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()}
        expert_success = arrays["success"][arrays["pure_expert"]]
        manifest["splits"][split] = {"episodes": n, "transitions": int(arrays["valid"].sum()),
            "behavior_sr": float(arrays["success"].mean()), "pure_expert_sr": float(expert_success.mean()),
            "bytes": sum(p.stat().st_size for p in directory.iterdir()), "files_sha256": files}
        paths.extend(f"{split}/{name}" for name in files)
        log("ДАТАСЕТ СОБРАН", split=split, episodes=n,
            pure_expert_sr=f"{expert_success.mean():.1%}", behavior_sr=f"{arrays['success'].mean():.1%}")
        del arrays
    write_json(out / "manifest.json", manifest)
    write_json(out / "audit.json", audit(out, bank_dir))
    finish(out, paths)
    return out


def load_arrays(directory, split):
    return {p.stem: np.load(p, mmap_mode="r", allow_pickle=False) for p in (Path(directory)/split).glob("*.npy")}


def audit(directory, bank_dir):
    manifest = read_json(Path(directory)/"manifest.json")
    depth = manifest["depth"]
    env, params = simulator(depth)
    step = jax.jit(env.step)
    result = {"transition_replay": True, "all_six_initial_actions": True, "splits": {}}
    for split in ("train", "val", "test"):
        for name, expected in manifest["splits"][split]["files_sha256"].items():
            if file_hash(Path(directory)/split/name) != expected:
                raise ValueError(f"Dataset checksum mismatch: {split}/{name}")
        arrays = load_arrays(directory, split)
        bank = load_bank(bank_dir, split)
        a, valid, expert = (arrays[k] for k in ("actions", "valid", "expert_mask"))
        if (np.any(a[~valid] != -1) or np.any(expert & ~valid) or np.any((a[valid] < 0) | (a[valid] > 5))
                or not np.array_equal(valid.sum(1), arrays["lengths"])):
            raise ValueError("Invalid action/mask/padding alignment")
        coverage = set()
        progress = Progress(f"Replay-аудит RGB {split}: эпизоды", len(a))
        # Replay every frame; no approximate sample or shape-only audit.
        for i in range(len(a)):
            task = int(arrays["task_ids"][i])
            ts = single_ts(bank, task)
            np.testing.assert_array_equal(render_ts(ts), arrays["observations"][i, 0])
            coverage.add((task, int(a[i, 0])))
            length = int(arrays["lengths"][i])
            if not valid[i, :length].all() or valid[i, length:].any():
                raise ValueError("Valid mask must be a contiguous prefix")
            for t in range(length):
                if bool(ts.last()):
                    raise ValueError("Transition after terminal")
                ts = step(params, ts, int(a[i, t]))
                np.testing.assert_array_equal(render_ts(ts), arrays["observations"][i, t+1])
                np.testing.assert_array_equal(np.asarray(ts.state.agent.position), arrays["agent_positions"][i, t+1])
                if int(ts.state.agent.direction) != arrays["agent_directions"][i, t+1]:
                    raise ValueError("Diagnostic direction mismatch")
                if not np.isclose(float(ts.reward), arrays["rewards"][i, t]):
                    raise ValueError("Reward mismatch")
            if not bool(ts.last()) or (float(ts.reward) > 0) != arrays["success"][i]:
                raise ValueError("SR/termination mismatch")
            for t in range(length+1, depth+1):
                np.testing.assert_array_equal(arrays["observations"][i, t], arrays["observations"][i, length])
            progress.update(i+1)
        if len(coverage) != 6*len(bank["goal"]):
            raise ValueError("Missing counterfactual action coverage")
        result["splits"][split] = {"episodes": len(a), "replayed_transitions": int(valid.sum())}
    return result
