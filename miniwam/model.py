"""Small recurrent convolutional WAM; direct palette pixels, no pretrained weights.

Spatial convolutions provide translation equivariance. A shared latent transition
is unrolled n times. Policy/joint decode actions; video/joint decode future images
conditioned on actions. Joint training updates the same encoder and transition.
This deliberately small baseline is not the paper's diffusion NanoWM.
"""
from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.nn import functional as F

from .env import PALETTE


MODES = ("policy", "video", "joint")


def palette_tensor(device):
    return torch.as_tensor(PALETTE.copy(), dtype=torch.float32, device=device)/255


def pixel_classes(rgb):
    palette = palette_tensor(rgb.device)
    # Supports [B,T,3,H,W] as well as [B,3,H,W].
    distances = (rgb.movedim(-3, -1).unsqueeze(-2)-palette).square().sum(-1)
    return distances.argmin(-1)


def logits_to_rgb(logits):
    return torch.einsum("...khw,kc->...chw", logits.softmax(-3), palette_tensor(logits.device))


class PixelWAM(nn.Module):
    def __init__(self, mode, width=32):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode, self.width = mode, width
        self.encoder = nn.Sequential(nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.transition = nn.Sequential(nn.Conv2d(width+4, width, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.action_head = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten(),
                                         nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 4)) if mode != "video" else None
        self.image_head = nn.Conv2d(width, 4, 1) if mode != "policy" else None

    def rollout(self, pixels, horizon, actions=None):
        if actions is None and self.action_head is None:
            raise ValueError("Video model requires candidate/recorded actions")
        h = self.encoder(pixels)
        action_logits, image_logits, decoded_actions = [], [], []
        for t in range(horizon):
            if self.action_head is not None:
                logits = self.action_head(h)
                action_logits.append(logits)
            action = actions[:, t] if actions is not None else logits.argmax(-1)
            decoded_actions.append(action)
            planes = F.one_hot(action, 4).float()[:, :, None, None].expand(-1, -1, *h.shape[-2:])
            h = (h+self.transition(torch.cat([h, planes], 1)))*0.7071067811865476
            if self.image_head is not None:
                image_logits.append(self.image_head(h))
        return {"actions": torch.stack(decoded_actions, 1),
                "action_logits": torch.stack(action_logits, 1) if action_logits else None,
                "image_logits": torch.stack(image_logits, 1) if image_logits else None}

    def loss(self, pixels, actions, targets, valid, expert_mask, image_weight=1.0):
        output = self.rollout(pixels, actions.shape[1], actions)
        # Disconnected zero retains correct dtype/device without privileged labels.
        total = torch.zeros((), device=pixels.device)
        metrics = {}
        if output["action_logits"] is not None:
            ce = F.cross_entropy(output["action_logits"].flatten(0, 1), actions.flatten(), reduction="none").reshape_as(valid)
            action_loss = (ce*expert_mask).sum()/expert_mask.sum().clamp(min=1)
            total = total+action_loss
            metrics["action_loss"] = float(action_loss.detach())
        if output["image_logits"] is not None:
            classes = pixel_classes(targets)
            ce = F.cross_entropy(output["image_logits"].flatten(0, 1), classes.flatten(0, 1), reduction="none").reshape_as(classes)
            # Equalize sparse objects against background; avoid a copy/background shortcut.
            weights = (1+4*(classes != 0).float())*valid[..., None, None]
            image_loss = (ce*weights).sum()/weights.sum().clamp(min=1)
            total = total+image_weight*image_loss
            metrics["image_loss"] = float(image_loss.detach())
        return total, metrics

    @torch.no_grad()
    def act(self, pixels, horizon=1, remaining=1):
        if self.mode != "video":
            return self.rollout(pixels, 1)["actions"][:, 0]
        # Image-only mode: enumerate ALL plans (at most 4^2 for this environment),
        # score predicted RGB/palette images against the red goal visible now.
        # No learned policy head, reward/success labels or real env.step in planning.
        length = min(horizon, remaining)
        candidates = torch.tensor(list(itertools.product(range(4), repeat=length)), device=pixels.device)
        batch, plans = pixels.shape[0], len(candidates)
        tiled = pixels[:, None].expand(-1, plans, -1, -1, -1).flatten(0, 1)
        actions = candidates[None].expand(batch, -1, -1).flatten(0, 1)
        probabilities = self.rollout(tiled, length, actions)["image_logits"].softmax(2)
        goal_mask = (pixel_classes(tiled) == 2).float()
        rows, cols = torch.meshgrid(torch.arange(pixels.shape[-2], device=pixels.device),
                                    torch.arange(pixels.shape[-1], device=pixels.device), indexing="ij")
        goal_y = (goal_mask*rows).sum((-2, -1))
        goal_x = (goal_mask*cols).sum((-2, -1))
        distances = (rows-goal_y[:, None, None]).abs()+(cols-goal_x[:, None, None]).abs()
        agent = probabilities[:, :, 1]
        distance = (agent*distances[:, None]).sum((-2, -1))/agent.sum((-2, -1)).clamp(min=1e-6)
        success = (probabilities[:, :, 3]*goal_mask[:, None]).sum((-2, -1))
        score = -distance*(1-success)+2*success
        discount = torch.pow(0.9, torch.arange(length, device=pixels.device))
        score = (score*discount).sum(1).reshape(batch, plans)
        return candidates[score.argmax(1), 0]
