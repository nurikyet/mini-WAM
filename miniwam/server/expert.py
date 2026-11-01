"""State PPO for the real XLand curriculum. No planner or supervised oracle."""
from __future__ import annotations

from pathlib import Path

from .common import (completed, file_hash, finish, read_json, restore_rng, rng_state,
                     seed, stage, write_json)
from .xland import ACTION_NAMES, VectorEnv, features, load_bank, timesteps
from miniwam.ppo import generalized_advantages
from miniwam.utils import save_checkpoint
from .progress import Progress, log
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class StateActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(13, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, 6)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


@torch.no_grad()
def evaluate(actor, bank, depth, device, temperature=0.0, eval_seed=2027):
    rng = np.random.default_rng(eval_seed)
    vector = VectorEnv(bank, depth, len(bank["goal"]), rng)
    vector.ts = timesteps(bank)
    alive = np.ones(vector.count, bool)
    successes = np.zeros(vector.count, bool)
    traces = [[] for _ in alive]
    # No auto-reset: one independent finite task per vector lane.
    for t in range(depth):
        obs = torch.as_tensor(features(vector.ts, depth), device=device)
        logits, _ = actor(obs)
        if temperature:
            # CPU generator prevents evaluation changing the training CUDA RNG.
            u = torch.from_numpy(rng.random((len(alive), 1))).to(device)
            probs = (logits/temperature).softmax(-1)
            action = (probs.cumsum(-1) < u).sum(-1).clamp(max=5).cpu().numpy()
        else:
            action = logits.argmax(-1).cpu().numpy()
        vector.ts = vector.step_fn(vector.params, vector.ts, jnp.asarray(action))
        success = np.asarray(vector.ts.reward) > 0
        successes |= alive & success
        for i in np.flatnonzero(alive):
            traces[i].append({"t": t, "action": int(action[i]), "name": ACTION_NAMES[action[i]], "success": bool(success[i])})
        alive &= ~np.asarray(vector.ts.last())
        if not alive.any():
            break
    return {"episodes": len(alive), "successes": int(successes.sum()), "sr": float(successes.mean()),
            "temperature": temperature, "failures": [{"task_id": i, "goal": bank["goal"][i].tolist(),
            "source_row": int(bank["source_row"][i]), "trace": traces[i]} for i in np.flatnonzero(~successes)]}


def load_actor(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    actor = StateActor().to(device)
    actor.load_state_dict(saved["model"])
    return actor.eval()


def train_expert(bank_dir, config, out, device, resume=False):
    bank_dir = Path(bank_dir)
    out = stage(out, {"ppo": config, "bank_sha256": file_hash(bank_dir/"manifest.json")}, resume)
    if completed(out):
        log("ПРОПУСК: PPO уже обучен", checkpoint=out/"best.pt")
        return out / "best.pt"
    rng = seed(config["seed"])
    depth = read_json(bank_dir / "manifest.json")["config"]["depth"]
    train, val = load_bank(bank_dir, "train"), load_bank(bank_dir, "val")
    env = VectorEnv(train, depth, config["num_envs"], rng)
    actor = StateActor().to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=config["lr"], eps=1e-5)
    start, best, consecutive, history = 0, (-1., -1.), 0, []
    if (out / "latest.pt").exists():
        saved = torch.load(out / "latest.pt", map_location="cpu", weights_only=True)
        actor.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best, consecutive, history = saved["update"], tuple(saved["best"]), saved["consecutive"], saved["history"]
        leaves, tree = jax.tree.flatten(env.ts)
        env.ts = jax.tree.unflatten(tree, [jnp.asarray(a.numpy(), dtype=b.dtype) for a, b in zip(saved["env_leaves"], leaves)])
        restore_rng(saved["rng"], rng)
        log("PPO ВОССТАНОВЛЕН", update=start, checkpoint=out/"latest.pt")
    progress = Progress("PPO: обновления", config["updates"], initial=start)
    update = start
    for update in range(start+1, config["updates"]+1) if consecutive < 3 else ():
        actor.train()
        records = []
        for _ in range(config["rollout_steps"]):
            obs = torch.as_tensor(features(env.ts, depth), device=device)
            with torch.no_grad():
                logits, value = actor(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
            _, rewards, dones = env.step(action.cpu().numpy())
            records.append((obs, action, logprob, value, torch.as_tensor(rewards, device=device),
                            torch.as_tensor(dones, dtype=torch.float32, device=device)))
        obs, actions, logprobs, values, rewards, dones = [torch.stack(x) for x in zip(*records)]
        with torch.no_grad():
            bootstrap = actor(torch.as_tensor(features(env.ts, depth), device=device))[1]
            advantages = generalized_advantages(rewards, values, dones, bootstrap, 0.99, 0.95)
            returns = advantages+values
        x, a, old_lp, adv, ret = [v.flatten(0, 1) for v in (obs, actions, logprobs, advantages, returns)]
        adv = (adv-adv.mean()) / adv.std().clamp(min=1e-8)
        losses = []
        for _ in range(config["epochs"]):
            order = torch.randperm(len(x), device=device)
            for ids in order.split(config["minibatch_size"]):
                logits, value = actor(x[ids])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(a[ids])-old_lp[ids]).exp()
                policy_loss = -torch.minimum(ratio*adv[ids], ratio.clamp(0.8, 1.2)*adv[ids]).mean()
                loss = policy_loss+0.5*(value-ret[ids]).square().mean()-config["entropy_coef"]*dist.entropy().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                optimizer.step()
                losses.append(float(loss.detach()))
        progress.update(update, loss=float(np.mean(losses)), env_steps=update*config["num_envs"]*config["rollout_steps"])
        if update % config["eval_every"] == 0 or update == config["updates"]:
            actor.eval()
            train_eval = evaluate(actor, train, depth, device)
            val_eval = evaluate(actor, val, depth, device)
            key = (val_eval["sr"], train_eval["sr"])
            consecutive = consecutive+1 if key == (1., 1.) else 0
            record = {"update": update, "loss": float(np.mean(losses)), "train_sr": train_eval["sr"], "val_sr": val_eval["sr"]}
            history.append(record)
            log("PPO VALIDATION", update=update, loss=record["loss"],
                train_sr=f"{train_eval['sr']:.1%}", val_sr=f"{val_eval['sr']:.1%}", stable_checks=f"{consecutive}/3")
            if key >= best:
                best = key
                save_checkpoint(out / "best.pt", {"model": actor.state_dict(), "update": update, "selection": record})
                log("СОХРАНЕН ЛУЧШИЙ PPO", checkpoint=out/"best.pt", update=update)
            leaves = [np.asarray(x).copy() for x in jax.tree.leaves(env.ts)]
            save_checkpoint(out / "latest.pt", {"model": actor.state_dict(), "optimizer": optimizer.state_dict(),
                "update": update, "best": list(best), "consecutive": consecutive, "history": history,
                "env_leaves": [torch.from_numpy(x.astype(np.int64) if np.issubdtype(x.dtype, np.integer) else x) for x in leaves],
                "rng": rng_state(rng)})
            write_json(out / "metrics.json", history)
            log("СОХРАНЕН CHECKPOINT PPO", checkpoint=out/"latest.pt", update=update)
            if consecutive >= 3:
                log("PPO: РАННЯЯ ОСТАНОВКА", reason="SR 100% на train/val стабилен три проверки")
                break
    actor = load_actor(out / "best.pt", device)
    evaluations = {s: evaluate(actor, load_bank(bank_dir, s), depth, device) for s in ("train", "val", "test")}
    evaluations["test_sampled_T1"] = evaluate(actor, load_bank(bank_dir, "test"), depth, device, temperature=1.)
    log("ИТОГ PPO", **{f"{s}_sr": f"{evaluations[s]['sr']:.1%}" for s in ("train", "val", "test")})
    write_json(out / "evaluation.json", evaluations)
    write_json(out / "gate.json", {"passed": all(evaluations[s]["sr"] == 1. for s in ("train", "val")),
        "criterion": "100% greedy SR on every train and validation bank task; test not used for PPO selection"})
    finish(out, ["best.pt", "latest.pt", "evaluation.json", "gate.json", "metrics.json"], updates=update)
    return out / "best.pt"
