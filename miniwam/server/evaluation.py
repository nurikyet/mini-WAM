"""Closed-loop SR and visual MPC. Planning never steps the real environment."""
from __future__ import annotations

import itertools
from pathlib import Path

from .common import read_json
from .dataset import load_arrays
from .model import autocast
from .vae import decode, encode
from .progress import Progress
from .xland import (ACTION_NAMES, BOARD, BOARD_X, TILE, GOAL_TYPES, load_bank,
                    render_tile, render_ts, simulator, single_ts)
import imageio.v3 as iio
import jax
import numpy as np
import torch
from torch.nn import functional as F


def board_tiles(rgb):
    # RGB [B,3,128,128] -> tiles [B,81,3,12,12].
    board = rgb[:, :, :108, BOARD_X:BOARD_X+108]
    return board.unfold(2, TILE, TILE).unfold(3, TILE, TILE).permute(0, 2, 3, 1, 4, 5).reshape(-1, 81, 3, TILE, TILE)


def location(tiles, template):
    # The board is discrete. Averaging candidate positions can put an object
    # between two similar grey shapes, where no object actually exists.
    mse = (tiles-template[:, None]).square().mean((-3, -2, -1))
    yy, xx = torch.meshgrid(torch.arange(BOARD, device=tiles.device), torch.arange(BOARD, device=tiles.device), indexing="ij")
    coordinates = torch.stack([yy.flatten(), xx.flatten()], -1).float()
    return coordinates[mse.argmin(-1)], mse.min(-1).values


def agent_pose(rgb):
    tiles = board_tiles(rgb)
    templates = torch.from_numpy(np.stack([render_tile((1, 7), d, False, TILE) for d in range(4)]).copy()).to(rgb.device)
    templates = templates.permute(0, 3, 1, 2).float()/255
    mse = (tiles[:, :, None]-templates[None, None]).square().mean((-3, -2, -1))
    best = mse.flatten(1).argmin(-1)
    yy, xx = torch.meshgrid(torch.arange(BOARD, device=rgb.device), torch.arange(BOARD, device=rgb.device), indexing="ij")
    xy = torch.stack([yy.flatten(), xx.flatten()], -1).float()[best//4]
    directions = torch.tensor([[-1., 0.], [0., 1.], [1., 0.], [0., -1.]], device=rgb.device)
    heading = directions[best%4]
    return xy, heading


def pocket_matches_goal(predicted, goal_template):
    # Goal satisfaction is object identity, not a reward for any vaguely similar
    # colored patch. Nearest rendered symbol rejects picking a wrong grey/dark
    # object that has a small absolute pixel MSE to the requested object.
    symbols = [(0,0)]+[(tile,color) for tile in (3,4,5,7,11,12) for color in (1,2,3,4,5,6,8,9,10,11)]
    templates = torch.from_numpy(np.stack([render_tile(s,None,False,12) for s in symbols]).copy()).to(predicted.device)
    templates = templates.permute(0,3,1,2).float()/255
    goal_id = (goal_template[:,None]-templates[None]).square().mean((-3,-2,-1)).argmin(-1)
    pocket = predicted[:,:,112:124,96:108]
    pocket_id = (pocket[:,None]-templates[None]).square().mean((-3,-2,-1)).argmin(-1)
    return (pocket_id == goal_id).float()


def visual_score(predicted, current):
    """Public goal predicate + pose potential, all read from rendered pixels.

    Potential distinguishes useful turns for n=1 MPC on multistep tasks. This is
    a specified task scorer, not a learned policy or privileged success head.
    """
    tiles, now_tiles = board_tiles(predicted), board_tiles(current)
    indicators = torch.stack([current[:, :, 114:122, 2+i*8:8+i*8].mean((1, 2, 3)) for i in range(3)], -1)
    kind = indicators.argmax(-1)
    template_a = current[:, :, 112:124, 32:44]
    template_b = current[:, :, 112:124, 52:64]
    original_a, _ = location(now_tiles, template_a)
    original_b, _ = location(now_tiles, template_b)
    predicted_a, _ = location(tiles, template_a)
    predicted_b, _ = location(tiles, template_b)
    pos, heading = agent_pose(predicted)
    target = torch.where((kind == 2)[:, None], original_b, original_a)
    delta = target-pos
    desired_distance = torch.where(kind == 2, 2., 1.)
    distance_cost = (delta.abs().sum(-1)-desired_distance).abs()
    angle_cost = 1-F.cosine_similarity(heading, delta, dim=-1, eps=1e-6)
    potential = -distance_cost-0.3*angle_cost
    held = pocket_matches_goal(predicted, template_a)
    adjacent_error = ((predicted_a-predicted_b).abs().sum(-1)-1).abs()
    hold_score = 4*held+(1-held)*potential
    near_score = -((pos-original_a).abs().sum(-1)-1).abs()-0.3*angle_cost
    drop_score = 4*(1-held)*(1-adjacent_error)+held*potential
    return torch.where(kind == 0, hold_score, torch.where(kind == 1, near_score, drop_score))


@torch.no_grad()
def plan(model, vae, rgb, remaining, device, ddim_steps, planning_seed=777, chunk_size=12):
    length = min(model.horizon, remaining)
    candidates = torch.tensor(list(itertools.product(range(6), repeat=length)), device=device)
    actions = F.pad(candidates, (0, model.horizon-length))
    context = encode(vae, np.asarray(rgb)[None], device)
    current = torch.as_tensor(np.array(rgb), device=device).permute(2, 0, 1).float()[None]/255
    gen = torch.Generator(device=device).manual_seed(planning_seed)
    common_noise = torch.randn((1, model.horizon, 4, 16, 16), generator=gen, device=device)
    scores = []
    for batch in actions.split(chunk_size):
        count = len(batch)
        valid = torch.arange(model.horizon, device=device)[None].expand(count, -1) < length
        with autocast(device):
            latent = model.sample(context.expand(count, -1, -1, -1), batch, ddim_steps,
                                  common_noise.expand(count, -1, -1, -1, -1), valid)
        # Decode in bounded chunks to avoid a VAE memory spike at 6^3 plans.
        decoded = decode(vae, latent[:, :length])
        score = torch.stack([visual_score(decoded[:, t], current.expand(count, -1, -1, -1)) for t in range(length)], 1)
        weights = 0.9**torch.arange(length, device=device)
        scores.append((score*weights).sum(1))
    scores = torch.cat(scores)
    best = int(scores.argmax())
    return int(actions[best, 0]), {"best_plan": candidates[best].tolist(), "score": float(scores[best]), "plans": len(actions)}


@torch.no_grad()
def closed_loop(model, vae, bank_dir, split, device, ddim_steps=20, temperature=0., eval_seed=2027, out=None):
    bank = load_bank(bank_dir, split)
    depth = read_json(Path(bank_dir)/"manifest.json")["config"]["depth"]
    env, params = simulator(depth)
    step = jax.jit(env.step)
    rng = np.random.default_rng(eval_seed)
    successes, traces = [], []
    by_type = {str(t): {"successes": 0, "episodes": 0} for t in GOAL_TYPES}
    progress = Progress(f"Closed-loop {model.mode} {split} T={temperature}: эпизоды", len(bank["goal"]))
    for task in range(len(bank["goal"])):
        ts = single_ts(bank, task)
        frames, trace = [render_ts(ts)], []
        for t in range(depth):
            rgb = frames[-1]
            if model.mode == "video":
                action, diagnostics = plan(model, vae, rgb, depth-t, device, ddim_steps, eval_seed+task*depth+t)
            else:
                z = encode(vae, rgb[None], device)
                with autocast(device):
                    logits = model.action_logits(z)[0, 0].float()
                probabilities = (logits/max(temperature, 1e-8)).softmax(-1).cpu().numpy()
                action = int(rng.choice(6, p=probabilities/probabilities.sum())) if temperature else int(logits.argmax())
                diagnostics = {"logits": logits.cpu().tolist()}
            # The only real simulator step for this decision, after selecting the action.
            ts = step(params, ts, action)
            frames.append(render_ts(ts))
            trace.append({"t": t, "action": action, "name": ACTION_NAMES[action], "reward": float(ts.reward), **diagnostics})
            if bool(ts.last()):
                break
        success = float(ts.reward) > 0
        successes.append(success)
        typ = by_type[str(int(bank["goal"][task, 0]))]
        typ["episodes"] += 1
        typ["successes"] += success
        if not success:
            traces.append({"task_id": task, "goal": bank["goal"][task].tolist(),
                           "source_row": int(bank["source_row"][task]), "trace": trace})
        if out is not None and (task < 3 or (not success and len(traces) <= 6)):
            Path(out).mkdir(parents=True, exist_ok=True)
            iio.imwrite(Path(out)/f"task_{task}_{'success' if success else 'failure'}.mp4", np.stack(frames), fps=2, macro_block_size=1)
        progress.update(task+1, successes=sum(successes), sr=f"{np.mean(successes):.1%}")
    n, k = len(successes), sum(successes)
    # Wilson interval, plus exact finite-bank count. No population certainty claim.
    z = 1.959963984540054
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return {"split": split, "episodes": n, "successes": int(k), "sr": float(k/n),
            "wilson95": [float(center-half), float(center+half)], "temperature": temperature,
            "by_goal_type": by_type, "failures": traces,
            "protocol": "first predicted action, real RGB update, replan; finite controlled bank"}


@torch.no_grad()
def offline(model, windows, device, examples=64, ddim_steps=20, vae=None, dataset=None, split="val", out=None):
    ids = np.linspace(0, len(windows)-1, min(examples, len(windows)), dtype=int)
    batch = windows.batch(ids, device)
    current, actions, targets, valid, expert = batch
    gen = torch.Generator(device=device).manual_seed(9127)
    with autocast(device):
        loss, result = model.loss(*batch, generator=gen)
    result = {**result, "loss": float(loss), "windows": len(ids), "valid_targets": int(valid.sum()),
              "valid_by_offset": valid.sum(0).cpu().tolist()}
    if model.action_head is not None:
        with autocast(device):
            logits = model.action_logits(current)
        correct = (logits.argmax(-1) == actions) & expert
        result["expert_action_accuracy"] = float(correct.sum()/expert.sum().clamp(min=1))
    if model.mode != "policy" and vae is not None:
        # Bound VAE batches independently of validation sample count.
        count = min(8, len(current))
        current, actions, targets, valid = (a[:count] for a in (current, actions, targets, valid))
        noise = torch.randn(targets.shape, device=device, generator=gen)
        with autocast(device):
            pred = model.sample(current, actions, ddim_steps, noise, valid)
            wrong = model.sample(current, (actions+1)%6, ddim_steps, noise, valid)
        mse = (pred-targets).square().mean((-3, -2, -1))
        wrong_mse = (wrong-targets).square().mean((-3, -2, -1))
        persistence = (current[:, None]-targets).square().mean((-3, -2, -1))
        denom = valid.sum().clamp(min=1)
        result.update(latent_mse=float((mse*valid).sum()/denom),
            persistence_latent_mse=float((persistence*valid).sum()/denom),
            altered_action_latent_mse=float((wrong_mse*valid).sum()/denom),
            action_prediction_delta=float(((pred-wrong).square().mean((-3,-2,-1))*valid).sum()/denom))
        rgb_pred, rgb_truth = decode(vae, pred), decode(vae, targets)
        pred_pos, _ = agent_pose(rgb_pred.flatten(0, 1))
        truth_pos, _ = agent_pose(rgb_truth.flatten(0, 1))
        distance = (pred_pos-truth_pos).norm(dim=-1).reshape_as(valid)
        result["ade_cells_reconstructed_truth"] = float((distance*valid).sum()/denom)
        last = valid.long().sum(1)-1
        result["fde_cells_reconstructed_truth"] = float(distance[torch.arange(count, device=device), last].mean())
        result["pose_metric_note"] = "template pose on generated and VAE-reconstructed frames; inspect VAE reconstruction diagnostics"
        if dataset is not None:
            arrays = load_arrays(dataset, split)
            ep, start = windows.indices[ids[:count]].T
            offsets = start[:, None]+np.arange(model.horizon)+1
            offsets = offsets.clip(max=arrays["actions"].shape[1])
            true = torch.as_tensor(np.array(arrays["observations"][ep[:, None], offsets]), device=device).permute(0,1,4,2,3).float()/255
            if "agent_positions" in arrays:
                true_positions = torch.as_tensor(np.array(arrays["agent_positions"][ep[:, None], offsets]), device=device).float()
                position_error = (pred_pos.reshape(count, model.horizon, 2)-true_positions).norm(dim=-1)
                result["ade_cells"] = float((position_error*valid).sum()/denom)
                result["fde_cells"] = float(position_error[torch.arange(count, device=device), last].mean())
                result["position_error_by_offset"] = [float((position_error[:,t]*valid[:,t]).sum()/valid[:,t].sum().clamp(min=1)) for t in range(model.horizon)]
                result["pose_metric_note"] = "template estimate on generated frames versus exact simulator coordinates; check VAE localization error"
            error = (rgb_pred-true).square().mean(2)
            context_rgb = torch.as_tensor(np.array(arrays["observations"][ep, start]), device=device).permute(0,3,1,2).float()/255
            changed = (true-context_rgb[:, None]).abs().amax(2) > 0.02
            mask = changed & valid[..., None, None]
            result["changed_pixel_mse"] = float((error*mask).sum()/mask.sum().clamp(min=1))
            result["rgb_mse"] = float((error.mean((-2,-1))*valid).sum()/denom)
            result["psnr"] = float(-10*np.log10(max(result["rgb_mse"], 1e-12)))
            if out is not None:
                Path(out).mkdir(parents=True, exist_ok=True)
                for i in range(min(4, count)):
                    frames = torch.cat([true[i], rgb_pred[i]], -1).permute(0,2,3,1).cpu().numpy()
                    iio.imwrite(Path(out)/f"forecast_{i}_truth_left.mp4", (frames.clip(0,1)*255).astype(np.uint8), fps=2, macro_block_size=1)
    return result


@torch.no_grad()
def counterfactual_metrics(model, vae, windows, device, ddim_steps=20, tasks=3):
    """Each prediction compared with its OWN real alternative-action future.

    All six first actions are present for every task by collection contract.
    Common sampling noise removes a source of spurious action sensitivity.
    """
    if model.mode == "policy":
        return {}
    # First six episodes of each task are forced actions 0..5. Locate these
    # through the collection's task IDs rather than relying on repeat count.
    arrays = load_arrays(windows.dataset, windows.split)
    results = []
    for task in np.unique(arrays["task_ids"])[:tasks]:
        episodes = np.flatnonzero(arrays["task_ids"] == task)[:6]
        ids = np.array([np.flatnonzero((windows.indices[:,0] == e)&(windows.indices[:,1] == 0))[0] for e in episodes])
        current, actions, target, valid, expert = windows.batch(ids, device)
        if not torch.equal(actions[:,0],torch.arange(6,device=device)):
            raise ValueError("Counterfactual first-action contract violated")
        gen = torch.Generator(device=device).manual_seed(815+int(task))
        noise = torch.randn((1,model.horizon,4,16,16),generator=gen,device=device).expand(6,-1,-1,-1,-1)
        with autocast(device):
            pred = model.sample(current, actions, ddim_steps, noise, valid)
        rgb = decode(vae,pred[:,0])
        truth = torch.as_tensor(np.array(arrays["observations"][episodes,1]),device=device).permute(0,3,1,2).float()/255
        correct = (rgb-truth).square().mean()
        wrong_pairing = (rgb-truth.roll(1,0)).square().mean()
        results.append({"task_id":int(task), "rgb_mse_own_alternative":float(correct),
                        "rgb_mse_wrong_pairing":float(wrong_pairing),
                        "prediction_variance_across_actions":float(rgb.var(0).mean())})
    return {"tasks":results,"sampling":"same diffusion noise for all six alternatives",
            "own_alternative_rgb_mse":float(np.mean([r["rgb_mse_own_alternative"] for r in results])),
            "wrong_pairing_rgb_mse":float(np.mean([r["rgb_mse_wrong_pairing"] for r in results]))}
