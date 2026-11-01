"""Upstream NanoWM S/4 with an action-query decoder and v diffusion objective.

This is an explicit WAM extension, not the summer-school source (not supplied).
Action inference always sees only the current frame; future targets and recorded
actions enter a separate diffusion pass. Both passes update the same backbone.
"""
from __future__ import annotations

from contextlib import nullcontext

from . import common  # adds the pinned upstream module to sys.path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from nanowm import NanoWM


def cosine_schedule(steps=1000):
    t = torch.arange(steps+1, dtype=torch.float64)/steps
    cumulative = torch.cos((t+0.008)/1.008*np.pi/2).square()
    cumulative = cumulative/cumulative[0]
    betas = (1-cumulative[1:]/cumulative[:-1]).clamp(max=0.999)
    return (1-betas).cumprod(0).float()


class NanoWAM(nn.Module):
    def __init__(self, mode, horizon, width=384, layers=12, heads=6, patch_size=4, action_width=256):
        super().__init__()
        if mode not in ("policy", "video", "joint"):
            raise ValueError(mode)
        if layers % 2 or (width//heads) % 4 or width % heads or 16 % patch_size:
            raise ValueError("NanoWM needs even depth, head width divisible by four, and patch size dividing 16")
        self.mode, self.horizon = mode, horizon
        self.backbone = NanoWM(input_size=16, patch_size=patch_size, in_channels=4, hidden_size=width,
            depth=layers, num_heads=heads, num_frames=horizon+1, use_action=True,
            action_dim=6, action_injection_type="additive", extras=1, causal=False)
        self._features = None
        self.backbone.final_layer.register_forward_pre_hook(self._capture)
        if mode != "video":
            self.project = nn.Linear(width, action_width)
            self.queries = nn.Parameter(torch.randn(1, horizon, action_width)*0.02)
            layer = nn.TransformerDecoderLayer(action_width, 4, 4*action_width, dropout=0.1, batch_first=True, norm_first=True)
            self.action_decoder = nn.TransformerDecoder(layer, 2)
            self.action_head = nn.Linear(action_width, 6)
        else:
            self.action_head = None
        self.register_buffer("alpha_bar", cosine_schedule())

    def _capture(self, module, inputs):
        self._features = inputs[0]

    def action_logits(self, current):
        if self.action_head is None:
            raise ValueError("Video-only has no learned action head; use visual MPC")
        b = len(current)
        frames = torch.zeros((b, self.horizon+1, *current.shape[1:]), device=current.device, dtype=current.dtype)
        frames[:, 0] = current
        times = torch.full((b, self.horizon+1), 999, device=current.device, dtype=torch.long)
        times[:, 0] = 0
        # No recorded action or future frame is accepted by this interface.
        self.backbone(frames, times, action=None)
        hidden = self._features.reshape(b, self.horizon+1, self.backbone.num_patches, -1)[:, 0]
        self._features = None
        memory = self.project(hidden)
        queries = self.queries.expand(b, -1, -1)
        return self.action_head(self.action_decoder(queries, memory))

    def velocity(self, current, future, actions, times):
        b = len(current)
        frames = torch.cat([current[:, None], future], 1)
        frame_times = torch.cat([torch.zeros((b, 1), dtype=torch.long, device=current.device), times], 1)
        onehot = F.one_hot(actions, 6).to(current.dtype)
        # NanoWM internally shifts action[t] to frame[t+1]. Do NOT shift twice.
        action_input = torch.cat([onehot, torch.zeros_like(onehot[:, :1])], 1)
        output = self.backbone(frames, frame_times, action=action_input)[:, 1:]
        self._features = None
        return output

    def loss(self, current, actions, targets, valid, expert, image_weight=1., generator=None, motion_weight=0.):
        total = torch.zeros((), device=current.device)
        metrics = {}
        if self.action_head is not None:
            logits = self.action_logits(current)
            ce = F.cross_entropy(logits.flatten(0, 1), actions.flatten(), reduction="none").reshape_as(valid)
            # An open-loop expert plan from this context is supervised only
            # until the first exploratory/forced action. Later PPO actions in
            # a perturbed state are not labels for this unperturbed plan.
            plan_expert = (expert & valid).long().cumprod(1).bool()
            action_loss = (ce*plan_expert).sum()/plan_expert.sum().clamp(min=1)
            total = total+action_loss
            metrics["action_loss"] = float(action_loss.detach())
        if self.mode != "policy":
            times = torch.randint(0, 1000, valid.shape, device=current.device, generator=generator)
            alpha = self.alpha_bar[times].sqrt()[..., None, None, None]
            sigma = (1-self.alpha_bar[times]).sqrt()[..., None, None, None]
            noise = torch.randn(targets.shape, device=current.device, dtype=targets.dtype, generator=generator)
            noisy = alpha*targets+sigma*noise
            # Invalid slots must not carry terminal targets through temporal attention.
            noisy = torch.where(valid[..., None, None, None], noisy, torch.zeros_like(noisy))
            times = torch.where(valid, times, torch.full_like(times, 999))
            masked_actions = torch.where(valid, actions, torch.zeros_like(actions))
            velocity_target = alpha*noise-sigma*targets
            predicted = self.velocity(current, noisy, masked_actions, times)
            error = (predicted.float()-velocity_target).square()
            if motion_weight:
                # Foreground changes are sparse. Use only training images to
                # weight their latent regions; no positions/goals/SR labels.
                change = (targets-current[:,None]).detach().square().mean(2,keepdim=True)
                weights = 1+motion_weight*change/(change.mean((-2,-1),keepdim=True)+1e-6)
                mse = (error*weights).sum((-3,-2,-1))/(weights.sum((-3,-2,-1))*error.shape[2])
            else:
                mse = error.mean((-3, -2, -1))
            image_loss = (mse*valid).sum()/valid.sum().clamp(min=1)
            total = total+image_weight*image_loss
            metrics["image_loss"] = float(image_loss.detach())
        return total, metrics

    @torch.no_grad()
    def sample(self, current, actions, ddim_steps=20, noise=None, valid=None):
        if self.mode == "policy":
            raise ValueError("Action-only does not train image prediction")
        if actions.shape != (len(current), self.horizon):
            raise ValueError("Candidate plan must have the trained horizon")
        future = torch.randn((len(current), self.horizon, *current.shape[1:]), device=current.device) if noise is None else noise.clone()
        if valid is None:
            valid = torch.ones(actions.shape, dtype=torch.bool, device=current.device)
        active = valid[..., None, None, None]
        future = torch.where(active, future, torch.zeros_like(future))
        actions = torch.where(valid, actions, torch.zeros_like(actions))
        schedule = torch.linspace(999, 0, ddim_steps, device=current.device).long()
        for i, t in enumerate(schedule):
            times = torch.where(valid, t.expand(len(current), self.horizon), torch.full_like(actions, 999))
            alpha, sigma = self.alpha_bar[t].sqrt(), (1-self.alpha_bar[t]).sqrt()
            v = self.velocity(current, future, actions, times).float()
            x0 = alpha*future-sigma*v
            eps = sigma*future+alpha*v
            if i+1 == len(schedule):
                future = x0
            else:
                next_alpha = self.alpha_bar[schedule[i+1]].sqrt()
                next_sigma = (1-self.alpha_bar[schedule[i+1]]).sqrt()
                future = next_alpha*x0+next_sigma*eps
            future = torch.where(active, future, torch.zeros_like(future))
        return future


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
