"""Public goal images for the observable one-step curriculum.

Only the current RGB is accepted. No simulator state, action labels, rewards,
PPO policy, or candidate real futures enter this goal specification. It defines
what success should look like; NanoWM still predicts which action reaches it.
"""
from __future__ import annotations

from .xland import BOARD, BOARD_X, TILE, render_tile
import numpy as np
import torch
from torch.nn import functional as F


def goal_images(current):
    current = np.asarray(current, dtype=np.uint8)
    if current.shape != (128,128,3):
        raise ValueError("Expected the 128x128 observable-goal curriculum RGB")
    board = current[:108,BOARD_X:BOARD_X+108].reshape(BOARD,TILE,BOARD,TILE,3).transpose(0,2,1,3,4)
    indicators = [current[114:122,2+i*8:8+i*8].mean() for i in range(3)]
    kind = int(np.argmax(indicators))
    a, b = current[112:124,32:44], current[112:124,52:64]
    # The renderer supersamples in float; the public canvas stores uint8.
    floor = np.asarray(render_tile((1,7),None,False,TILE),dtype=np.uint8)
    empty = np.asarray(render_tile((0,0),None,False,TILE),dtype=np.uint8)
    templates = np.stack([render_tile((1,7),d,False,TILE) for d in range(4)]).astype(np.uint8)
    pose_error = (board[:,:,None].astype(np.float32)-templates[None,None]).astype(np.float32)**2
    cell_direction = int(pose_error.mean((-3,-2,-1)).argmin())
    cell, direction = divmod(cell_direction,4)
    ay, ax = divmod(cell,BOARD)
    def locate(template):
        error = (board.astype(np.float32)-template.astype(np.float32))**2
        return np.unravel_index(error.mean((-3,-2,-1)).argmin(),(BOARD,BOARD))
    def patch(image,y,x,tile):
        image[y*TILE:(y+1)*TILE,BOARD_X+x*TILE:BOARD_X+(x+1)*TILE] = tile
    if kind == 0:  # Hold A: A disappears from the board and appears in the pocket.
        y,x = locate(a)
        image = current.copy()
        patch(image,y,x,floor)
        image[112:124,96:108] = a
        return np.stack([image])
    gy,gx = locate(a if kind == 1 else b)
    images = []
    # Any free neighbour satisfies the public adjacency goal. None of these
    # alternatives is tied to an action ID or to a candidate simulator rollout.
    for dy,dx in ((-1,0),(0,1),(1,0),(0,-1)):
        y,x = gy+dy,gx+dx
        if not (0 <= y < BOARD and 0 <= x < BOARD):
            continue
        if not np.array_equal(board[y,x],floor):
            continue
        image = current.copy()
        if kind == 1:
            patch(image,ay,ax,floor)
            patch(image,y,x,templates[direction])
        else:
            patch(image,y,x,a)
            image[112:124,96:108] = empty
        images.append(image)
    if not images:
        raise ValueError("No visible free neighbour for the one-step goal")
    return np.stack(images)


def goal_weights(current, images, device):
    changed = np.max(np.abs(images.astype(np.int16)-current.astype(np.int16)),axis=-1) > 0
    mask = torch.as_tensor(changed.copy(),device=device).float()[:,None]
    mask = F.max_pool2d(F.avg_pool2d(mask,8),3,stride=1,padding=1)
    return 1+4*mask


def goal_scores(predictions, encoded_goals, weights):
    """Candidate latent images -> scores; no action IDs or success labels."""
    error = (predictions[:,None].float()-encoded_goals[None].float()).square()
    error = (error*weights[None]).sum((-3,-2,-1))/(weights.sum((-3,-2,-1))[None]*predictions.shape[1])
    return -error.min(-1).values
