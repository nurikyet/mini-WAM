"""Finite navigation task: reach the red square in exactly one or two moves.

Four absolute movement actions, no obstacles or hidden rules. Task depth is the
initial Manhattan distance AND the episode budget. The complete finite bank is
partitioned by goal position before any training. Thus even intermediate states
of two-action trajectories cannot be shared between the three splits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


ACTION_NAMES = ("up", "right", "down", "left")
DELTAS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int64)
# Lossless, fixed palette: background, agent, goal, success. No learned VAE.
PALETTE = np.array([[16, 16, 24], [20, 90, 255], [255, 20, 20], [20, 255, 40]], dtype=np.uint8)


@dataclass(frozen=True)
class EnvConfig:
    size: int = 5
    depth: int = 1
    tile_size: int = 6
    split_seed: int = 2026

    def __post_init__(self):
        if self.depth not in (1, 2):
            raise ValueError("depth must be 1 or 2 (one/two actual movement actions)")
        if self.size < 4 or self.tile_size < 1:
            raise ValueError("size >= 4 and tile_size >= 1 required")


@dataclass(frozen=True, order=True)
class Task:
    ay: int
    ax: int
    gy: int
    gx: int

    @property
    def key(self):
        return f"{self.ay},{self.ax}->{self.gy},{self.gx}"

    @property
    def delta(self):
        return self.gy - self.ay, self.gx - self.ax


def task_banks(config: EnvConfig) -> dict[str, list[Task]]:
    rng = np.random.default_rng(config.split_seed)
    goals = rng.permutation(config.size**2)
    n_holdout = max(1, config.size**2 // 5)
    goal_splits = {}
    for name, cells in zip(("train", "val", "test"),
                           (goals[2*n_holdout:], goals[:n_holdout], goals[n_holdout:2*n_holdout])):
        goal_splits.update({int(cell): name for cell in cells})
    banks = {name: [] for name in ("train", "val", "test")}
    for ay in range(config.size):
        for ax in range(config.size):
            for gy in range(config.size):
                for gx in range(config.size):
                    task = Task(ay, ax, gy, gx)
                    if sum(map(abs, task.delta)) == config.depth:
                        banks[goal_splits[gy*config.size+gx]].append(task)
    return banks


def bank_manifest(config):
    banks = task_banks(config)
    keys = {name: [t.key for t in tasks] for name, tasks in banks.items()}
    return {"environment": asdict(config), "tasks": keys,
            "sha256": hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest(),
            "split_unit": "goal position (including all initial/intermediate states for that goal)",
            "generalization": "held-out goal positions on the same board; no obstacles"}


class NavigationEnv:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.done = True

    def reset(self, task: Task):
        if sum(map(abs, task.delta)) != self.config.depth:
            raise ValueError("Task distance must equal depth")
        if any(x < 0 or x >= self.config.size for x in asdict(task).values()):
            raise ValueError("Task outside board")
        self.task = task
        self.agent = np.array([task.ay, task.ax], dtype=np.int64)
        self.goal = np.array([task.gy, task.gx], dtype=np.int64)
        self.steps = 0
        self.done = self.success = False
        return self.render()

    def state(self):
        # Privileged PPO input; never supplied to a pixel policy or dataset loader.
        delta = (self.goal - self.agent) / self.config.size
        return np.array([*delta, (self.config.depth - self.steps) / self.config.depth], dtype=np.float32)

    def render(self):
        cells = np.zeros((self.config.size, self.config.size), dtype=np.uint8)
        cells[tuple(self.goal)] = 2
        cells[tuple(self.agent)] = 3 if self.success else 1
        return PALETTE[cells].repeat(self.config.tile_size, 0).repeat(self.config.tile_size, 1)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Call reset after a terminal transition")
        if not 0 <= int(action) < len(DELTAS):
            raise ValueError("Action must be 0..3")
        old_distance = int(abs(self.goal - self.agent).sum())
        next_agent = np.clip(self.agent + DELTAS[int(action)], 0, self.config.size - 1)
        blocked = bool(np.array_equal(next_agent, self.agent))
        self.agent = next_agent
        self.steps += 1
        distance = int(abs(self.goal - self.agent).sum())
        self.success = distance == 0
        self.done = self.success or self.steps == self.config.depth
        # Potential progress shaping plus terminal reward. SR is always goal reach.
        reward = float(old_distance - distance) * 0.1 + float(self.success)
        info = {"success": self.success, "distance": distance, "blocked": blocked,
                "failure": None if self.success else "budget_exhausted" if self.done else None}
        return self.render(), reward, self.done, info


class VectorStates:
    """Fast CPU state-only simulator for PPO, with explicit reset after done."""
    def __init__(self, config, tasks, count, rng):
        self.config, self.tasks, self.count, self.rng = config, np.array(
            [[t.ay, t.ax, t.gy, t.gx] for t in tasks]), count, rng
        self.agent = np.empty((count, 2), dtype=np.int64)
        self.goal = np.empty_like(self.agent)
        self.steps = np.zeros(count, dtype=np.int64)
        self.reset(np.ones(count, dtype=bool))

    def reset(self, mask):
        tasks = self.tasks[self.rng.integers(0, len(self.tasks), int(mask.sum()))]
        self.agent[mask], self.goal[mask], self.steps[mask] = tasks[:, :2], tasks[:, 2:], 0

    def observation(self):
        return np.concatenate([(self.goal-self.agent)/self.config.size,
                               ((self.config.depth-self.steps)/self.config.depth)[:, None]], 1).astype(np.float32)

    def step(self, actions):
        old_dist = abs(self.goal-self.agent).sum(1)
        self.agent = np.clip(self.agent + DELTAS[actions], 0, self.config.size-1)
        self.steps += 1
        dist = abs(self.goal-self.agent).sum(1)
        success = dist == 0
        done = success | (self.steps == self.config.depth)
        reward = (old_dist-dist)*0.1 + success
        self.reset(done)
        return self.observation(), reward.astype(np.float32), done
