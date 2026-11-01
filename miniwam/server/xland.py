"""Controlled single-goal navigation and legacy near-terminal XLand fixtures.

Both use genuine XLand transitions and verified trivial-1m goals. These are
observable controlled initial states, not native hidden-goal benchmark resets.
"""
from __future__ import annotations

import bz2
import itertools
from pathlib import Path
import pickle
import urllib.request

from .common import (BENCHMARK_FILE, BENCHMARK_SHA256, HF_REVISION, completed,
                     file_hash, finish, read_json, save_npz, stage, write_json)
from .progress import Progress, log
from .protocol import episode_limit, renderer_version
import jax
import jax.numpy as jnp
import numpy as np
import xminigrid
from xminigrid.core.constants import DIRECTIONS, Tiles, Colors
from xminigrid.rendering.rgb_render import render_tile
from xminigrid.types import AgentState, EnvCarry, State, StepType, TimeStep

ACTION_NAMES = ("forward", "clockwise", "counterclockwise", "pick_up", "put_down", "toggle")
GOAL_TYPES = (1, 3, 4)
BOARD = 9
TILE = 12
BOARD_X = 10


def fetch_benchmark(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / BENCHMARK_FILE
    if not path.exists():
        url = f"https://huggingface.co/datasets/Howuhh/xland_minigrid/resolve/{HF_REVISION}/{BENCHMARK_FILE}"
        temporary = path.with_suffix(".download")
        log("СКАЧИВАНИЕ BENCHMARK", name=BENCHMARK_FILE, path=path)
        download = None
        def downloaded(block, size, total):
            nonlocal download
            if total > 0:
                if download is None:
                    download = Progress("Загрузка trivial-1m: байты", total)
                download.update(min(block*size, total))
        urllib.request.urlretrieve(url, temporary, reporthook=downloaded)
        if file_hash(temporary) != BENCHMARK_SHA256:
            temporary.unlink()
            raise ValueError("trivial-1m download checksum mismatch")
        temporary.replace(path)
    log("ПРОВЕРКА SHA256 BENCHMARK", path=path)
    if file_hash(path) != BENCHMARK_SHA256:
        raise ValueError("trivial-1m file checksum mismatch")
    # Historical pickle is loaded only after authentication by pinned SHA256.
    with jax.default_device(jax.devices("cpu")[0]), bz2.open(path, "rb") as f:
        data = pickle.load(f)
    log("BENCHMARK ПРОЧИТАН", rows=len(data["goals"]))
    return {k: np.asarray(data[k]) for k in ("goals", "rules", "init_tiles", "num_rules")}


def canonical_keys(goals, rules):
    goals = goals.copy()
    # TileNear(a,b) == TileNear(b,a); prevent semantic duplicates across splits.
    mask = (goals[:, 0] == 4) & ((goals[:, 1]*256 + goals[:, 2]) > (goals[:, 3]*256 + goals[:, 4]))
    goals[mask, 1:3], goals[mask, 3:5] = goals[mask, 3:5].copy(), goals[mask, 1:3].copy()
    return np.concatenate([goals, rules.reshape(len(goals), -1)], 1)


def make_banks(config, data_dir, out, resume=False):
    out = stage(out, {"bank": config, "benchmark_sha256": BENCHMARK_SHA256}, resume)
    if completed(out):
        log("ПРОПУСК: банки уже подготовлены и проверены", path=out)
        return out
    data = fetch_benchmark(data_dir)
    pickable = np.array([3, 4, 5, 7, 11, 12])
    goals = data["goals"]
    # Historical trivial_1m_v2 stores num_rules as [N,1], not [N]. Flatten
    # per row before combining masks; implicit broadcasting would allocate N*N.
    no_rules = data["num_rules"].reshape(len(goals), -1).sum(1) == 0
    eligible = np.isin(goals[:, 0], GOAL_TYPES) & no_rules
    eligible &= np.isin(goals[:, 1], pickable)
    eligible &= (goals[:, 0] != 4) | (np.isin(goals[:, 3], pickable) & np.any(goals[:, 1:3] != goals[:, 3:5], 1))
    rows = np.flatnonzero(eligible)
    keys = canonical_keys(goals[rows], data["rules"][rows])
    unique, first, inverse, counts = np.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
    representatives = rows[first]
    rng = np.random.default_rng(config["split_seed"])
    split_groups = {s: [] for s in ("train", "val", "test")}
    for goal_type in GOAL_TYPES:
        ids = rng.permutation(np.flatnonzero(unique[:, 0] == goal_type))
        holdout = max(1, len(ids)//10)
        if len(ids) < 5:
            raise ValueError(f"Insufficient goal groups for type {goal_type}")
        for split, part in zip(split_groups, (ids[2*holdout:], ids[:holdout], ids[holdout:2*holdout])):
            split_groups[split].extend(part.tolist())
    depth = episode_limit(config)
    navigation = config.get("family") == "single_goal"
    renderer = renderer_version(config)
    if not navigation and depth not in (1, 2, 3):
        raise ValueError("Legacy curriculum depth must be 1, 2, or 3")
    manifest = {"benchmark": "trivial-1m", "filename": BENCHMARK_FILE, "hf_revision": HF_REVISION,
                "benchmark_sha256": BENCHMARK_SHA256, "config": config,
                "protocol": ("one semantic objective, multiple primitive transitions; clear axial approach; "
                             "not native benchmark resets" if navigation else
                             "legacy near-terminal curriculum; not native benchmark resets"),
                "renderer": renderer, "episode_max_steps": depth,
                "split": "canonical (goal,rules) groups; TileNear arguments sorted; 80/10/10 by goal type",
                "source_rows": len(goals), "eligible_rows": len(rows), "unique_eligible_groups": len(unique),
                "duplicate_fraction_in_eligible_rows": float(1-len(unique)/len(rows)),
                "goal_group_overlap": 0, "actions": ACTION_NAMES, "splits": {}}
    frame_sets = set()
    for split_index, (split, groups) in enumerate(split_groups.items()):
        # Keep evaluation scenes fixed when changing the number of train tasks.
        if navigation:
            rng = np.random.default_rng(np.random.SeedSequence([config["split_seed"], split_index, 901]))
        n = config["tasks"][split]
        if n < 3:
            raise ValueError("At least three tasks per split")
        records, fingerprints = [], set()
        progress = Progress(f"Банк XLand {split}: раскладки", n)
        by_type = {g: [i for i in groups if unique[i, 0] == g] for g in GOAL_TYPES}
        attempts = 0
        while len(records) < n:
            attempts += 1
            if attempts > 100*n:
                raise ValueError("Could not construct enough unique curriculum states")
            typ = GOAL_TYPES[len(records) % len(GOAL_TYPES)]
            group = int(rng.choice(by_type[typ]))
            row = int(representatives[group])
            # Historical goals are int32; engine pocket/grid tiles are uint8.
            goal, rules = goals[row].astype(np.uint8), data["rules"][row]
            # Keep the relevant object(s) and the historical distractor tiles.
            direction = int(rng.integers(4))
            offset = np.asarray(DIRECTIONS[direction])
            if navigation:
                # Balance distances within each goal type, rather than relying
                # on chance to include long examples in small evaluation banks.
                moves = int(config["travel_moves"][(len(records)//len(GOAL_TYPES))%len(config["travel_moves"])])
                distance = moves+(2 if typ == 4 else 1)
                positions = [np.array((y,x)) for y in range(1, BOARD-1) for x in range(1, BOARD-1)
                             if np.all((np.array((y,x))+offset*distance >= 1) &
                                       (np.array((y,x))+offset*distance < BOARD-1))]
                pos = positions[int(rng.integers(len(positions)))]
                target = pos+offset*distance
                facing = int(rng.integers(4))
            else:
                pos = rng.integers(3, 6, size=2)
                target = pos + offset*(1 if typ == 1 else 2)
                facing = (direction + (0 if depth == 1 else int(rng.choice([1, 3])) if depth == 2 else 2)) % 4
            grid = np.zeros((BOARD, BOARD, 2), dtype=np.uint8)
            grid[:] = (Tiles.FLOOR, Colors.BLACK)
            grid[[0, -1], :] = (Tiles.WALL, Colors.GREY)
            grid[:, [0, -1]] = (Tiles.WALL, Colors.GREY)
            pocket = np.array((Tiles.EMPTY, Colors.EMPTY), dtype=np.uint8)
            grid[tuple(target)] = goal[1:3] if typ != 4 else goal[3:5]
            if typ == 4:
                pocket = goal[1:3].copy()
            reserved = {tuple(pos), tuple(target), tuple(pos+offset)}
            if navigation:
                reserved |= {tuple(pos+offset*i) for i in range(distance+1)}
            free = [(y, x) for y in range(1, BOARD-1) for x in range(1, BOARD-1) if (y, x) not in reserved]
            rng.shuffle(free)
            for tile in data["init_tiles"][row]:
                if tile[0] == 0 or np.array_equal(tile, goal[1:3]) or (typ == 4 and np.array_equal(tile, goal[3:5])):
                    continue
                grid[free.pop()] = tile
            record = {"grid": grid, "position": pos.astype(np.int32), "direction": np.int32(facing),
                      "pocket": pocket, "goal": goal, "rules": rules, "source_row": np.int64(row), "group_id": np.int64(group)}
            if navigation:
                turns = min((direction-facing)%4, (facing-direction)%4)
                record.update(travel_moves=np.int32(moves), approach_direction=np.int32(direction),
                              minimum_transitions=np.int32(moves+turns+(typ != 3)))
            fingerprint = render_record(record, renderer).tobytes()
            if fingerprint in fingerprints:
                continue
            if fingerprint in frame_sets:
                raise ValueError("Identical initial RGB across splits")
            fingerprints.add(fingerprint)
            records.append(record)
            progress.update(len(records))
        frame_sets |= fingerprints
        save_npz(out / f"{split}.npz", **{k: np.stack([r[k] for r in records]) for k in records[0]})
        manifest["splits"][split] = {"tasks": n, "goal_groups": len(groups), "group_ids": groups,
                                    "sha256": file_hash(out / f"{split}.npz")}
    write_json(out / "manifest.json", manifest)
    # Native simulator audits construction witnesses (legacy: exhaustive search).
    audit = audit_banks(out)
    write_json(out / "audit.json", audit)
    finish(out, ["manifest.json", "audit.json", "train.npz", "val.npz", "test.npz"])
    return out


def load_bank(directory, split):
    manifest = read_json(Path(directory) / "manifest.json")
    path = Path(directory) / f"{split}.npz"
    if file_hash(path) != manifest["splits"][split]["sha256"]:
        raise ValueError(f"Bank checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def simulator(depth):
    env, params = xminigrid.make("XLand-MiniGrid-R1-9x9")
    return env, params.replace(max_steps=depth)


def timesteps(bank, ids=None):
    if ids is None:
        ids = np.arange(len(bank["goal"]))
    n = len(ids)
    return TimeStep(state=State(key=jnp.zeros((n, 2), dtype=jnp.uint32), step_num=jnp.zeros(n, dtype=jnp.int32),
        grid=jnp.asarray(bank["grid"][ids]), agent=AgentState(position=jnp.asarray(bank["position"][ids]),
        direction=jnp.asarray(bank["direction"][ids]), pocket=jnp.asarray(bank["pocket"][ids])),
        goal_encoding=jnp.asarray(bank["goal"][ids]), rule_encoding=jnp.asarray(bank["rules"][ids]), carry=EnvCarry()),
        step_type=jnp.zeros(n, dtype=jnp.uint8), reward=jnp.zeros(n), discount=jnp.ones(n),
        observation=jnp.zeros((n, 5, 5, 2), dtype=jnp.uint8))


def single_ts(bank, index):
    return jax.tree.map(lambda a: a[0], timesteps(bank, np.array([index])))


def render_record(record, renderer="legacy_v1"):
    return render_state(record["grid"], record["position"], int(record["direction"]), record["pocket"], record["goal"], renderer)


def render_ts(ts, renderer="legacy_v1"):
    s = ts.state
    return render_state(np.asarray(s.grid), np.asarray(s.agent.position), int(s.agent.direction),
                        np.asarray(s.agent.pocket), np.asarray(s.goal_encoding), renderer)


def display_tile(tile, direction=None, renderer="clean_v2"):
    """EMPTY is a blank slot, not the upstream red debug exclamation mark."""
    if renderer not in ("clean_v2", "legacy_v1"):
        raise ValueError(f"Unknown renderer: {renderer}")
    if renderer == "clean_v2" and int(tile[0]) == int(Tiles.EMPTY):
        return np.full((TILE, TILE, 3), 24, np.uint8)
    return render_tile(tuple(map(int, tile)), direction, False, TILE)


def render_state(grid, position, direction, pocket, goal, renderer="legacy_v1"):
    canvas = np.full((128, 128, 3), 24, dtype=np.uint8)
    for y in range(BOARD):
        for x in range(BOARD):
            tile = display_tile(grid[y, x], direction if np.array_equal(position, (y, x)) else None, renderer)
            canvas[y*TILE:(y+1)*TILE, BOARD_X+x*TILE:BOARD_X+(x+1)*TILE] = tile
    # Explicitly observable instruction: type, tile A, tile B, held object.
    # No success/reward/optimal action indicator is drawn.
    for i, typ in enumerate(GOAL_TYPES):
        canvas[114:122, 2+i*8:8+i*8] = 240 if int(goal[0]) == typ else 48
    for x, tile in ((32, goal[1:3]), (52, goal[3:5] if int(goal[0]) == 4 else (0, 0)), (96, pocket)):
        canvas[112:124, x:x+12] = display_tile(tile, renderer=renderer)
    return canvas


def features(ts, depth):
    """Privileged state features, used by PPO only; no oracle/action mask."""
    s = ts.state
    grid = np.asarray(s.grid)
    goals = np.asarray(s.goal_encoding)
    pos = np.asarray(s.agent.position)
    features = [np.eye(4, dtype=np.float32)[np.asarray(s.agent.direction)],
                (goals[:, :1] == np.array(GOAL_TYPES)).astype(np.float32)]
    yy, xx = np.indices(grid.shape[1:3])
    for sl in (slice(1, 3), slice(3, 5)):
        match = (grid == goals[:, None, None, sl]).all(-1)
        denom = match.sum((1, 2)).clip(1)
        target = np.stack([(match*yy).sum((1, 2))/denom, (match*xx).sum((1, 2))/denom], 1)
        features.append(((target-pos)/BOARD).astype(np.float32))
    features.append((np.asarray(s.agent.pocket) == goals[:, 1:3]).all(-1, keepdims=True).astype(np.float32))
    features.append((1-np.asarray(s.step_num)[:, None]/depth).astype(np.float32))
    return np.concatenate(features, 1)


class VectorEnv:
    def __init__(self, bank, depth, count, rng):
        self.bank, self.depth, self.count, self.rng = bank, depth, count, rng
        self.env, self.params = simulator(depth)
        self.step_fn = jax.jit(jax.vmap(self.env.step, in_axes=(None, 0, 0)))
        self.ts = timesteps(bank, rng.integers(len(bank["goal"]), size=count))

    def step(self, actions):
        self.ts = self.step_fn(self.params, self.ts, jnp.asarray(actions))
        rewards = np.asarray(self.ts.reward).copy()
        done = np.asarray(self.ts.last()).copy()
        if done.any():
            new = timesteps(self.bank, self.rng.integers(len(self.bank["goal"]), size=self.count))
            self.ts = jax.tree.map(lambda a, b: jnp.where(jnp.asarray(done).reshape((-1,)+(1,)*(a.ndim-1)), b, a), self.ts, new)
        return features(self.ts, self.depth), rewards, done


def audit_banks(directory):
    manifest = read_json(Path(directory) / "manifest.json")
    if manifest["config"].get("family") == "single_goal":
        return audit_navigation(directory)
    depth = episode_limit(manifest["config"])
    env, params = simulator(depth)
    @jax.jit
    def evaluate_plans(ts, plans):
        def one(plan):
            def step(carry, action):
                state, success = carry
                new = env.step(params, state, action)
                # Freeze after terminal, including success before later actions.
                new = jax.tree.map(lambda a, b: jnp.where(state.last(), a, b), state, new)
                return (new, success | (new.reward > 0)), new.reward > 0
            return jax.lax.scan(step, (ts, jnp.asarray(False)), plan)[1]
        return jax.vmap(one)(plans)
    result = {"exact_minimum_depth": depth, "split_goal_overlap": 0, "splits": {}}
    seen = set()
    for split in ("train", "val", "test"):
        bank = load_bank(directory, split)
        progress = Progress(f"Аудит достижимости {split}: раскладки", len(bank["goal"]))
        groups = set(map(int, bank["group_id"]))
        if seen & groups:
            raise ValueError("Goal group overlap")
        seen |= groups
        shortest = []
        plans = jnp.asarray(list(itertools.product(range(6), repeat=depth)))
        for i in range(len(bank["goal"])):
            success = np.asarray(evaluate_plans(single_ts(bank, i), plans))
            possible = success.any(0)
            if not possible[-1] or possible[:-1].any():
                raise ValueError(f"Incorrect minimum depth for {split}/{i}: {possible}")
            shortest.append(depth)
            progress.update(i+1)
        result["splits"][split] = {"tasks": len(shortest), "solvable_fraction": 1.0,
                                    "exact_depth_verified": True, "unique_goal_groups": len(groups)}
    return result


def reference_plan(record):
    """Construction witness for auditing/previews only; never a PPO label."""
    rotation = (int(record["approach_direction"])-int(record["direction"]))%4
    turns = {0: [], 1: [1], 2: [1,1], 3: [2]}[rotation]
    interaction = {1: [3], 3: [], 4: [4]}[int(record["goal"][0])]
    return turns+[0]*int(record["travel_moves"])+interaction


def audit_navigation(directory):
    manifest = read_json(Path(directory)/"manifest.json")
    budget = episode_limit(manifest["config"])
    env, params = simulator(budget)
    step = jax.jit(env.step)
    report = {"episode_max_steps": budget, "semantic_objectives": 1,
              "method": "native replay of construction witness; clear axial corridor checked; no exhaustive 6^budget search",
              "split_goal_overlap": 0, "splits": {}}
    seen = set()
    for split in ("train", "val", "test"):
        bank = load_bank(directory, split)
        groups = set(map(int, bank["group_id"]))
        if seen & groups:
            raise ValueError("Goal group overlap")
        seen |= groups
        progress = Progress(f"Аудит single-goal {split}: раскладки", len(bank["goal"]))
        for i in range(len(bank["goal"])):
            record = {k: v[i] for k,v in bank.items()}
            moves = int(record["travel_moves"])
            delta = np.asarray(DIRECTIONS[int(record["approach_direction"])])
            distance = moves+(2 if int(record["goal"][0]) == 4 else 1)
            for t in range(1, distance):
                if record["grid"][tuple(record["position"]+delta*t)][0] != Tiles.FLOOR:
                    raise ValueError("Approach corridor obstructed")
            plan = reference_plan(record)
            if len(plan) != int(record["minimum_transitions"]) or len(plan) > budget:
                raise ValueError("Witness outside episode budget")
            ts = single_ts(bank, i)
            origin = np.asarray(ts.state.agent.position).copy()
            for t, action in enumerate(plan):
                if bool(ts.last()):
                    raise ValueError("Unexpected early termination in witness")
                ts = step(params, ts, action)
                if (float(ts.reward) > 0) != (t == len(plan)-1):
                    raise ValueError("Goal reached at wrong witness transition")
            if np.abs(np.asarray(ts.state.agent.position)-origin).sum() != moves:
                raise ValueError("Travel distance does not match construction")
            progress.update(i+1)
        report["splits"][split] = {"tasks": len(bank["goal"]), "solvable_fraction": 1.,
            "unique_goal_groups": len(groups), "travel_moves": sorted(set(map(int,bank["travel_moves"]))),
            "witness_transition_range": [int(bank["minimum_transitions"].min()), int(bank["minimum_transitions"].max())]}
    return report
