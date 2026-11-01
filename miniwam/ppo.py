"""Clipped PPO + GAE on privileged vector states, without an oracle/action mask."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .env import EnvConfig, NavigationEnv, VectorStates, task_banks
from .utils import prepare_run, restore_rng, rng_state, runtime, save_checkpoint, write_json


@dataclass(frozen=True)
class PPOConfig:
    updates: int = 150
    num_envs: int = 128
    rollout_steps: int = 8
    epochs: int = 4
    minibatch_size: int = 256
    lr: float = 0.001
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    eval_every: int = 5
    patience: int = 3
    seed: int = 42

    def __post_init__(self):
        if min(self.updates, self.num_envs, self.rollout_steps, self.epochs,
               self.minibatch_size, self.eval_every, self.patience) < 1:
            raise ValueError("PPO counts must be positive")
        if self.lr <= 0 or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid PPO learning rate/discount")


class StatePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor, self.critic = nn.Linear(64, 4), nn.Linear(64, 1)

    def forward(self, state):
        h = self.body(state)
        return self.actor(h), self.critic(h).squeeze(-1)


def generalized_advantages(rewards, values, dones, bootstrap, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        next_value = bootstrap if t == len(rewards)-1 else values[t+1]
        alive = 1-dones[t]
        delta = rewards[t]+gamma*next_value*alive-values[t]
        gae = delta+gamma*gae_lambda*alive*gae
        advantages[t] = gae
    return advantages


@torch.no_grad()
def state_evaluation(model, env_config, tasks, device):
    env = NavigationEnv(env_config)
    successes = []
    for task in tasks:
        env.reset(task)
        while not env.done:
            logits, _ = model(torch.as_tensor(env.state()[None], device=device))
            env.step(int(logits.argmax(-1).item()))
        successes.append(env.success)
    return float(np.mean(successes))


def load_expert(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = StatePolicy().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, EnvConfig(**checkpoint["env_config"])


def train_expert(env_config: EnvConfig, config: PPOConfig, out, device, resume=False):
    out = prepare_run(out, {"environment": asdict(env_config), "ppo": asdict(config)}, resume)
    if (out/"complete.json").exists():
        return out/"best.pt"
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    banks = task_banks(env_config)
    env = VectorStates(env_config, banks["train"], config.num_envs, rng)
    model = StatePolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    start_update, best_sr, consecutive, history = 0, -1.0, 0, []
    if (out/"latest.pt").exists():
        checkpoint = torch.load(out/"latest.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update, best_sr, consecutive = checkpoint["update"], checkpoint["best_sr"], checkpoint["consecutive"]
        history = checkpoint["history"]
        for key in ("agent", "goal", "steps"):
            setattr(env, key, checkpoint["vector_env"][key].numpy())
        restore_rng(checkpoint["rng"], rng)
    started = time.perf_counter()
    update = start_update
    pending = range(start_update+1, config.updates+1) if consecutive < config.patience else ()
    for update in pending:
        observations, actions, old_logprobs, values, rewards, dones = [], [], [], [], [], []
        for _ in range(config.rollout_steps):
            obs = torch.as_tensor(env.observation(), device=device)
            with torch.no_grad():
                logits, value = model(obs)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
            _, reward, done = env.step(action.cpu().numpy())
            observations.append(obs)
            actions.append(action)
            old_logprobs.append(distribution.log_prob(action))
            values.append(value)
            rewards.append(torch.as_tensor(reward, device=device))
            dones.append(torch.as_tensor(done, dtype=torch.float32, device=device))
        observations, actions, old_logprobs, values, rewards, dones = map(torch.stack,
            (observations, actions, old_logprobs, values, rewards, dones))
        with torch.no_grad():
            _, bootstrap = model(torch.as_tensor(env.observation(), device=device))
            advantages = generalized_advantages(rewards, values, dones, bootstrap, config.gamma, config.gae_lambda)
            returns = advantages + values
        b_obs, b_actions = observations.flatten(0, 1), actions.flatten()
        b_logprobs, b_returns = old_logprobs.flatten(), returns.flatten()
        b_adv = advantages.flatten()
        b_adv = (b_adv-b_adv.mean())/(b_adv.std(unbiased=False)+1e-8)
        losses = []
        for _ in range(config.epochs):
            permutation = torch.randperm(len(b_actions), device=device)
            for ids in permutation.split(config.minibatch_size):
                logits, value = model(b_obs[ids])
                distribution = Categorical(logits=logits)
                ratio = (distribution.log_prob(b_actions[ids])-b_logprobs[ids]).exp()
                policy_loss = -torch.minimum(ratio*b_adv[ids], ratio.clamp(1-config.clip, 1+config.clip)*b_adv[ids]).mean()
                value_loss = (value-b_returns[ids]).square().mean()
                loss = policy_loss + 0.5*value_loss-config.entropy_coef*distribution.entropy().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                losses.append(float(loss.detach()))
        if update % config.eval_every == 0 or update == config.updates:
            train_sr = state_evaluation(model, env_config, banks["train"], device)
            val_sr = state_evaluation(model, env_config, banks["val"], device)
            consecutive = consecutive+1 if train_sr == val_sr == 1.0 else 0
            record = {"update": update, "transitions": update*config.num_envs*config.rollout_steps,
                      "loss": float(np.mean(losses)), "train_sr": train_sr, "val_sr": val_sr}
            history.append(record)
            print(f"PPO {update}/{config.updates}: train SR={train_sr:.3f}, val SR={val_sr:.3f}", flush=True)
            # Latest perfect train/val model wins ties; test is never used for selection.
            score = val_sr + train_sr*1e-3
            if score >= best_sr:
                best_sr = score
                save_checkpoint(out/"best.pt", {"model": model.state_dict(), "env_config": asdict(env_config),
                                                "update": update, "validation": record})
        save_checkpoint(out/"latest.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "env_config": asdict(env_config), "update": update, "best_sr": best_sr,
            "consecutive": consecutive, "history": history, "rng": rng_state(rng),
            "vector_env": {key: torch.from_numpy(getattr(env, key).copy()) for key in ("agent", "goal", "steps")}})
        if consecutive >= config.patience:
            break
    write_json(out/"metrics.json", history)
    write_json(out/"complete.json", {"updates": update,
        "session_seconds": time.perf_counter()-started, "runtime": runtime(device), "selection": "train and validation only"})
    return out/"best.pt"
