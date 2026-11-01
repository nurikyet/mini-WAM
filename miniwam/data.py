"""PPO RGB rollouts, negative transitions, masks and a checksummed data contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .env import ACTION_NAMES, EnvConfig, NavigationEnv, PALETTE, bank_manifest, task_banks
from .ppo import load_expert, state_evaluation
from .utils import file_hash, prepare_run, write_json


@dataclass(frozen=True)
class CollectionConfig:
    train_repeats: int = 32
    eval_repeats: int = 8
    exploration: float = 0.25
    seed: int = 43

    def __post_init__(self):
        if min(self.train_repeats, self.eval_repeats) < 5:
            raise ValueError("At least five repeats: four initial actions plus a pure PPO episode")
        if not 0 <= self.exploration <= 1:
            raise ValueError("exploration must be in [0,1]")


@torch.no_grad()
def collect(expert_path, config: CollectionConfig, out, device, resume=False):
    model, env_config = load_expert(expert_path, device)
    out = prepare_run(out, {"environment": asdict(env_config), "collection": asdict(config),
                            "expert_sha256": file_hash(expert_path)}, resume)
    if (out/"manifest.json").exists():
        audit_dataset(out)
        return out
    rng = np.random.default_rng(config.seed)
    banks = task_banks(env_config)
    manifest = {"schema_version": 1, **bank_manifest(env_config), "actions": list(ACTION_NAMES),
        "palette": PALETTE.tolist(), "expert_sha256": file_hash(expert_path),
        "collection": asdict(config), "alignment": "observations[t] --actions[t]--> observations[t+1]",
        "padding": "repeat terminal RGB, action=-1, valid=False; never a training target",
        "action_labels": "expert_mask=True only for actions actually chosen by PPO",
        "exploration": "first four repeats enumerate the four initial actions; later epsilon mixture",
        "splits": {}}
    for split, tasks in banks.items():
        repeats = config.train_repeats if split == "train" else config.eval_repeats
        count, depth, pixels = len(tasks)*repeats, env_config.depth, env_config.size*env_config.tile_size
        observations = np.zeros((count, depth+1, pixels, pixels, 3), dtype=np.uint8)
        actions = np.full((count, depth), -1, dtype=np.int64)
        rewards = np.zeros((count, depth), dtype=np.float32)
        valid = np.zeros((count, depth), dtype=bool)
        expert_mask = np.zeros_like(valid)
        successes = np.zeros(count, dtype=bool)
        lengths = np.zeros(count, dtype=np.int64)
        task_ids = np.zeros(count, dtype=np.int64)
        env = NavigationEnv(env_config)
        examples = []
        for task_id, task in enumerate(tasks):
            for repeat in range(repeats):
                i = task_id*repeats+repeat
                observations[i, 0] = env.reset(task)
                frames = [observations[i, 0]]
                for t in range(depth):
                    logits, _ = model(torch.as_tensor(env.state()[None], device=device))
                    policy_action = int(logits.argmax(-1).item())
                    is_expert = True
                    if t == 0 and repeat < 4:
                        action, is_expert = repeat, False
                    elif repeat != 4 and rng.random() < config.exploration:
                        action, is_expert = int(rng.integers(4)), False
                    else:
                        action = policy_action
                    obs, reward, done, _ = env.step(action)
                    observations[i, t+1] = obs
                    actions[i, t], rewards[i, t], valid[i, t], expert_mask[i, t] = action, reward, True, is_expert
                    frames.append(obs)
                    if done:
                        observations[i, t+2:] = obs
                        break
                lengths[i], successes[i], task_ids[i] = env.steps, env.success, task_id
                if repeat == 4 and len(examples) < 4:
                    examples.append(frames)
        path = out/f"{split}.npz"
        # Binary arrays only; never pickle Python objects into a dataset.
        np.savez_compressed(path, observations=observations, actions=actions, rewards=rewards,
                            valid=valid, expert_mask=expert_mask, success=successes,
                            lengths=lengths, task_ids=task_ids)
        manifest["splits"][split] = {"episodes": count, "transitions": int(valid.sum()),
            "unique_initial_layouts": len(tasks), "behavior_sr": float(successes.mean()),
            "expert_closed_loop_sr": state_evaluation(model, env_config, tasks, device),
            "expert_label_fraction": float(expert_mask[valid].mean()), "sha256": file_hash(path)}
        for j, frames in enumerate(examples):
            save_gif(out/f"{split}_{j}.gif", frames)
        print(f"Dataset {split}: {count} episodes, behavior SR={successes.mean():.3f}", flush=True)
    write_json(out/"manifest.json", manifest)
    write_json(out/"audit.json", audit_dataset(out))
    return out


def save_gif(path, frames, scale=5):
    frames = [Image.fromarray(np.asarray(f, dtype=np.uint8)).resize(
        (f.shape[1]*scale, f.shape[0]*scale), Image.Resampling.NEAREST) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=500, loop=0)


def audit_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    config = EnvConfig(**manifest["environment"])
    expected = bank_manifest(config)
    if manifest["tasks"] != expected["tasks"] or manifest["sha256"] != expected["sha256"]:
        raise ValueError("Task-bank manifest mismatch")
    seen_goals, seen_observations = set(), set()
    result = {"sha256_verified": True, "state_overlap_between_splits": 0, "splits": {}}
    for split, tasks in task_banks(config).items():
        goals = {(t.gy, t.gx) for t in tasks}
        if seen_goals & goals:
            raise ValueError("Goal leakage between splits")
        seen_goals |= goals
        path = directory/f"{split}.npz"
        if file_hash(path) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            # NPZ is lazy: materialize each array once, rather than decompressing
            # the complete RGB shard again for every replayed frame.
            data = {name: archive[name] for name in archive.files}
            actions, valid, expert = data["actions"], data["valid"], data["expert_mask"]
            if (np.any(actions[~valid] != -1) or np.any(expert & ~valid)
                    or np.any((actions[valid] < 0) | (actions[valid] > 3))
                    or not np.array_equal(valid.sum(1), data["lengths"])):
                raise ValueError("Invalid actions/padding/lengths")
            observed = set()
            first_action_coverage = set()
            # Replay every transition; verifies t/t+1 alignment and metadata, not only shapes.
            env = NavigationEnv(config)
            for i, task_id in enumerate(data["task_ids"]):
                env.reset(tasks[int(task_id)])
                if not np.array_equal(env.render(), data["observations"][i, 0]):
                    raise ValueError("Initial RGB mismatch")
                first_action_coverage.add((int(task_id), int(actions[i, 0])))
                for t in range(int(data["lengths"][i])):
                    observed.add(data["observations"][i, t].tobytes())
                    obs, reward, done, _ = env.step(int(actions[i, t]))
                    if (not np.array_equal(obs, data["observations"][i, t+1])
                            or not np.isclose(reward, data["rewards"][i, t])):
                        raise ValueError("RGB/action/reward transition mismatch")
                if not env.done or env.success != data["success"][i]:
                    raise ValueError("Terminal success mismatch")
            if seen_observations & observed:
                raise ValueError("Identical RGB input found in different splits")
            seen_observations |= observed
            if len(first_action_coverage) != 4*len(tasks):
                raise ValueError("Missing counterfactual first-action coverage")
            result["splits"][split] = {"episodes": len(actions), "unique_input_frames": len(observed),
                "all_initial_actions_covered": True, "alignment_replay_verified": True}
    return result


class Windows:
    """Only RGB/actions/masks enter training. No rewards, success labels or states."""
    def __init__(self, directory, split, horizon, device):
        directory = Path(directory)
        metadata = json.loads((directory/"manifest.json").read_text())
        self.env_config = EnvConfig(**metadata["environment"])
        self.horizon = horizon
        if horizon < 1:
            raise ValueError("horizon must be positive")
        with np.load(directory/f"{split}.npz", allow_pickle=False) as data:
            rgb = torch.from_numpy(data["observations"].copy()).permute(0, 1, 4, 2, 3).float()/255
            b, t, c, h, w = rgb.shape
            tiles = torch.nn.functional.avg_pool2d(rgb.flatten(0, 1), self.env_config.tile_size)
            self.pixels = tiles.reshape(b, t, c, self.env_config.size, self.env_config.size).to(device)
            self.actions = torch.from_numpy(data["actions"].copy()).to(device)
            self.valid = torch.from_numpy(data["valid"].copy()).to(device)
            self.expert = torch.from_numpy(data["expert_mask"].copy()).to(device)
        self.indices = self.valid.nonzero()

    def __len__(self):
        return len(self.indices)

    def batch(self, ids):
        episode, start = self.indices[ids].unbind(-1)
        steps = start[:, None]+torch.arange(self.horizon, device=episode.device)
        depth = self.actions.shape[1]
        safe = steps.clamp(max=depth-1)
        valid = (steps < depth) & self.valid[episode[:, None], safe]
        expert = valid & self.expert[episode[:, None], safe]
        actions = self.actions[episode[:, None], safe].clamp(min=0)
        targets = self.pixels[episode[:, None], (steps+1).clamp(max=depth)]
        return self.pixels[episode, start], actions, targets, valid, expert
