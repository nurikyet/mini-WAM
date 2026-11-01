"""Behavioral checks for episode boundaries, split leakage, objectives and resume."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from miniwam.data import CollectionConfig, Windows, audit_dataset, collect
from miniwam.env import DELTAS, EnvConfig, NavigationEnv, PALETTE, Task, VectorStates, task_banks
from miniwam.evaluate import closed_loop, rgb_batch
from miniwam.model import PixelWAM, pixel_classes
from miniwam.ppo import PPOConfig, StatePolicy, generalized_advantages, train_expert
from miniwam.train import ModelConfig, train_model
from miniwam.utils import file_hash, prepare_run, save_checkpoint, seed_everything, write_json
from scripts.prepare_git_export import export


CPU = torch.device("cpu")


class EnvironmentTests(unittest.TestCase):
    def test_one_action_bank_and_all_actions(self):
        config = EnvConfig()
        tasks = sum(task_banks(config).values(), [])
        self.assertEqual(len(tasks), 80)
        for task in tasks:
            wins = 0
            for action in range(4):
                env = NavigationEnv(config)
                env.reset(task)
                _, _, done, _ = env.step(action)
                self.assertTrue(done)
                wins += env.success
                with self.assertRaises(RuntimeError):
                    env.step(0)
            self.assertEqual(wins, 1)

    def test_two_actions_and_split_states_disjoint(self):
        config = EnvConfig(depth=2)
        banks = task_banks(config)
        self.assertEqual(sum(map(len, banks.values())), 124)
        previous = set()
        for tasks in banks.values():
            frames = set()
            for task in tasks:
                wins = 0
                for plan in itertools.product(range(4), repeat=2):
                    env = NavigationEnv(config)
                    frames.add(env.reset(task).tobytes())
                    for action in plan:
                        frames.add(env.step(action)[0].tobytes())
                    wins += env.success
                self.assertGreater(wins, 0)
            self.assertFalse(previous & frames)
            previous |= frames

    def test_vector_and_rgb_environment_agree(self):
        config = EnvConfig(depth=2)
        tasks = task_banks(config)["train"][:8]
        vector = VectorStates(config, tasks, len(tasks), np.random.default_rng(1))
        vector.agent = np.array([[t.ay, t.ax] for t in tasks])
        vector.goal = np.array([[t.gy, t.gx] for t in tasks])
        singles = [NavigationEnv(config) for _ in tasks]
        for env, task in zip(singles, tasks):
            env.reset(task)
        for t in range(2):
            np.testing.assert_array_equal(vector.observation(), np.stack([env.state() for env in singles]))
            actions = np.arange(8)%4
            _, rewards, dones = vector.step(actions)
            for i, env in enumerate(singles):
                _, reward, done, _ = env.step(int(actions[i]))
                self.assertAlmostEqual(float(rewards[i]), reward, places=6)
                self.assertEqual(bool(dones[i]), done)

    def test_terminal_gae_does_not_bootstrap_next_episode(self):
        rewards = torch.tensor([[1.0], [100.0]])
        values = torch.tensor([[0.2], [50.0]])
        actual = generalized_advantages(rewards, values, torch.ones_like(values), torch.tensor([999.0]), 0.99, 0.95)
        torch.testing.assert_close(actual, torch.tensor([[0.8], [50.0]]))

    def test_palette_pixels_roundtrip(self):
        env = NavigationEnv(EnvConfig())
        rgb = env.reset(task_banks(env.config)["train"][0])
        small = rgb_batch([rgb], env.config, CPU)
        cells = pixel_classes(small)[0].numpy()
        reconstructed = PALETTE[cells].repeat(env.config.tile_size, 0).repeat(env.config.tile_size, 1)
        np.testing.assert_array_equal(rgb, reconstructed)


class ModelTests(unittest.TestCase):
    def test_objectives_and_padding(self):
        seed_everything(1)
        pixels = torch.rand(3, 3, 5, 5)
        actions = torch.zeros(3, 3, dtype=torch.long)
        targets = torch.rand(3, 3, 3, 5, 5)
        valid = torch.tensor([[True, False, False]]*3)
        expert = valid.clone()
        for mode in ("policy", "video", "joint"):
            model = PixelWAM(mode, 8)
            base, _ = model.loss(pixels, actions, targets, valid, expert)
            altered = targets.clone()
            altered[:, 1:] = 1-altered[:, 1:]
            padded_actions = actions.clone()
            padded_actions[:, 1:] = 3
            after, _ = model.loss(pixels, padded_actions, altered, valid, expert)
            torch.testing.assert_close(base, after)
            if mode == "policy":
                other, _ = model.loss(pixels, actions, 1-targets, valid, expert)
                torch.testing.assert_close(base, other)
                unlabelled, _ = model.loss(pixels, (actions+1)%4, targets, valid, torch.zeros_like(expert))
                self.assertEqual(float(unlabelled), 0.0)
                self.assertIsNone(model.image_head)
            if mode == "video":
                other, _ = model.loss(pixels, actions, targets, valid, ~expert)
                torch.testing.assert_close(base, other)
                self.assertIsNone(model.action_head)
            base.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()))

    def test_image_planner_with_perfect_visual_predictions(self):
        class PerfectVisualDynamics(PixelWAM):
            def __init__(self):
                super().__init__("video", 4)

            def rollout(self, pixels, horizon, actions=None):
                cells = pixel_classes(pixels)
                images = []
                for t in range(horizon):
                    next_cells = cells.clone()
                    for i in range(len(cells)):
                        if (cells[i] == 3).any():
                            continue
                        ay, ax = (cells[i] == 1).nonzero()[0].tolist()
                        gy, gx = (cells[i] == 2).nonzero()[0].tolist()
                        y, x = np.clip(np.array([ay, ax])+DELTAS[int(actions[i, t])], 0, cells.shape[-1]-1)
                        next_cells[i, ay, ax] = 0
                        next_cells[i, y, x] = 3 if (y, x) == (gy, gx) else 1
                    cells = next_cells
                    images.append(torch.nn.functional.one_hot(cells, 4).permute(0, 3, 1, 2).float()*40-20)
                return {"image_logits": torch.stack(images, 1)}

        model = PerfectVisualDynamics()
        for depth in (1, 2):
            result = closed_loop(model, EnvConfig(depth=depth), "all", CPU, horizon=3)
            self.assertEqual(result["success_rate"], 1.0)


class DataAndResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(42)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        checkpoint = cls.root/"expert.pt"
        save_checkpoint(checkpoint, {"model": StatePolicy().state_dict(), "env_config": {
            "size": 5, "depth": 1, "tile_size": 6, "split_seed": 2026}})
        cls.dataset = collect(checkpoint, CollectionConfig(5, 5, 0.25), cls.root/"data", CPU)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def assert_tree_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_dataset_audit_and_window_masks(self):
        result = audit_dataset(self.dataset)
        self.assertEqual(result["state_overlap_between_splits"], 0)
        for split in ("train", "val", "test"):
            windows = Windows(self.dataset, split, 3, CPU)
            pixels, actions, targets, valid, expert = windows.batch(torch.arange(len(windows)))
            self.assertTrue(valid[:, 0].all())
            self.assertFalse(valid[:, 1:].any())
            self.assertGreater(int(expert.sum()), 0)

    def test_transition_replay_rejects_corruption_even_with_new_checksum(self):
        path = self.dataset/"test.npz"
        original, manifest_bytes = path.read_bytes(), (self.dataset/"manifest.json").read_bytes()
        try:
            with np.load(path) as data:
                arrays = {key: data[key].copy() for key in data.files}
            changed = np.flatnonzero(np.any(arrays["observations"][:, 0] != arrays["observations"][:, 1], axis=(1, 2, 3)))[0]
            arrays["observations"][changed, 1] = arrays["observations"][changed, 0]
            np.savez_compressed(path, **arrays)
            metadata = json.loads(manifest_bytes)
            metadata["splits"]["test"]["sha256"] = file_hash(path)
            write_json(self.dataset/"manifest.json", metadata)
            with self.assertRaisesRegex(ValueError, "RGB/action/reward"):
                audit_dataset(self.dataset)
        finally:
            path.write_bytes(original)
            (self.dataset/"manifest.json").write_bytes(manifest_bytes)

    def check_resume(self, training_function, args, name):
        direct, interrupted = self.root/(name+"_direct"), self.root/(name+"_interrupted")
        training_function(*args, direct, CPU)

        def save_then_interrupt(path, data):
            save_checkpoint(path, data)
            if Path(path).name == "latest.pt" and data["update"] == 2:
                raise KeyboardInterrupt("simulated server preemption after atomic checkpoint")

        module = "miniwam.ppo" if name == "ppo" else "miniwam.train"
        with patch(module+".save_checkpoint", save_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                training_function(*args, interrupted, CPU)
        training_function(*args, interrupted, CPU, resume=True)
        a = torch.load(direct/"latest.pt", weights_only=True)
        b = torch.load(interrupted/"latest.pt", weights_only=True)
        for key in ("model", "optimizer", "rng", "history"):
            self.assert_tree_equal(a[key], b[key])

    def test_wam_resume_matches_uninterrupted_update(self):
        config = ModelConfig(mode="joint", width=8, horizon=3, updates=4, batch_size=8, eval_every=2, patience=999)
        self.check_resume(train_model, (self.dataset, config), "wam")

    def test_ppo_resume_matches_uninterrupted_update(self):
        config = PPOConfig(updates=4, num_envs=8, rollout_steps=2, epochs=1, minibatch_size=8, eval_every=1, patience=999)
        self.check_resume(train_expert, (EnvConfig(), config), "ppo")

    def test_existing_run_requires_same_config(self):
        path = self.root/"config_guard"
        prepare_run(path, {"budget": 1}, False)
        with self.assertRaises(FileExistsError):
            prepare_run(path, {"budget": 1}, False)
        with self.assertRaises(ValueError):
            prepare_run(path, {"budget": 2}, True)


class ExportTests(unittest.TestCase):
    def test_export_vendors_sources_and_preserves_original_histories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"source"
            (source/"upstream"/".git").mkdir(parents=True)
            (source/"upstream"/".git"/"HEAD").write_text("original history")
            (source/"upstream"/"code.py").write_text("print('source')")
            (source/"weights.pt").write_bytes(b"weights")
            destination = root/"export"
            export(source, destination)
            self.assertTrue((destination/"upstream"/"code.py").is_file())
            self.assertFalse((destination/"upstream"/".git").exists())
            self.assertFalse((destination/"weights.pt").exists())
            self.assertTrue((source/"upstream"/".git"/"HEAD").is_file())


if __name__ == "__main__":
    unittest.main()
