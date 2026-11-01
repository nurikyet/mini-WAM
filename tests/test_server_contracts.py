"""Contract verification only: no PPO runs, optimizer steps, or local experiments."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from miniwam.server.common import require_server, stage
from miniwam.server.cli import validate_config
from miniwam.server.model import NanoWAM, cosine_schedule
from miniwam.server.evaluation import pocket_matches_goal, visual_score
from miniwam.server.xland import (audit_banks, canonical_keys, features, load_bank,
    make_banks, render_ts, simulator, single_ts, timesteps)
from miniwam.server.vae import LatentWindows
from miniwam.server.dataset import collect, audit as audit_dataset
from miniwam.server.evaluation import counterfactual_metrics
from miniwam.server.common import write_json
import jax
from xminigrid.core.rules import EmptyRule

torch.set_num_threads(1)
CPU = torch.device("cpu")


def task_record(kind, depth, orientation=0, tile_a=(3,1), tile_b=(4,3), position=(4,4)):
    deltas = np.array([[-1,0],[0,1],[1,0],[0,-1]])
    pos = np.array(position, np.int32)
    target = pos+deltas[orientation]*(1 if kind == 1 else 2)
    grid = np.zeros((9,9,2),np.uint8)
    grid[:] = (1,7)
    grid[[0,-1],:] = (2,6)
    grid[:,[0,-1]] = (2,6)
    goal = np.array([kind,*tile_a,*tile_b] if kind == 4 else [kind,*tile_a,0,0],np.uint8)
    grid[tuple(target)] = goal[3:5] if kind == 4 else goal[1:3]
    return {"grid":grid, "position":pos, "direction":np.int32((orientation+depth-1)%4),
            "pocket":goal[1:3].copy() if kind == 4 else np.array([0,0],np.uint8),
            "goal":goal,"rules":np.asarray(EmptyRule().encode())[None],"source_row":np.int64(0),"group_id":np.int64(0)}


class XLandContracts(unittest.TestCase):
    def test_dark_objects_do_not_cause_premature_drop(self):
        for depth in (2,3):
            env,params=simulator(depth);step=jax.jit(env.step)
            for tile_a,tile_b in (((11,6),(11,5)),((7,4),(11,1))):
                for orientation in range(4):
                    record=task_record(4,depth,orientation,tile_a,tile_b,position=(5,4))
                    record["grid"][1,3]=(5,1)  # red pyramid distractor
                    record["grid"][1,4]=(12,6)  # grey star resembles the grey hexagon
                    ts=single_ts({k:v[None] for k,v in record.items()},0)
                    for t in range(depth):
                        current=torch.from_numpy(render_ts(ts)).permute(2,0,1).float()[None]/255
                        candidates=[step(params,ts,a) for a in range(6)]
                        frames=torch.from_numpy(np.stack([render_ts(s) for s in candidates])).permute(0,3,1,2).float()/255
                        scores=visual_score(frames,current.expand(6,-1,-1,-1))
                        ts=candidates[int(scores.argmax())]
                        if bool(ts.last()):break
                    self.assertGreater(float(ts.reward),0.,(depth,tile_a,tile_b,orientation,scores.tolist()))
        # Similar grey objects must not count as successful pickups.
        from miniwam.server.xland import render_tile
        rgb=torch.zeros(1,3,128,128)
        rgb[:,:,112:124,96:108]=torch.from_numpy(render_tile((12,6),None,False,12)).permute(2,0,1).float()/255
        target=torch.from_numpy(render_tile((11,6),None,False,12)).permute(2,0,1).float()[None]/255
        self.assertEqual(float(pocket_matches_goal(rgb,target)),0.)

    def test_native_actions_and_visual_scorer_can_solve_all_curriculum_depths(self):
        # Oracle transitions are used ONLY in this diagnostic to test the visual
        # cost. Production MPC substitutes NanoWM predictions, never env.step.
        for depth in (1,2,3):
            env, params = simulator(depth)
            step = jax.jit(env.step)
            for kind in (1,3,4):
                for orientation in range(4):
                    record = task_record(kind, depth, orientation)
                    bank = {k:v[None] for k,v in record.items()}
                    ts = single_ts(bank,0)
                    self.assertEqual(features(timesteps(bank),depth).shape,(1,13))
                    for t in range(depth):
                        current = torch.from_numpy(render_ts(ts)).permute(2,0,1).float()[None]/255
                        candidates = [step(params,ts,a) for a in range(6)]
                        frames = torch.from_numpy(np.stack([render_ts(s) for s in candidates])).permute(0,3,1,2).float()/255
                        scores = visual_score(frames,current.expand(6,-1,-1,-1))
                        action = int(scores.argmax())
                        ts = candidates[action]
                        if bool(ts.last()):
                            break
                    self.assertGreater(float(ts.reward),0.,(depth,kind,orientation,scores.tolist()))

    def test_canonical_tile_near_semantics(self):
        goals = np.array([[4,3,1,4,2],[4,4,2,3,1],[1,3,1,0,0]],np.uint8)
        rules = np.zeros((3,1,7),np.uint8)
        keys = canonical_keys(goals,rules)
        np.testing.assert_array_equal(keys[0],keys[1])
        self.assertFalse(np.array_equal(keys[0],keys[2]))

    def test_bank_builder_exact_depth_and_group_splits(self):
        goals = np.array([[k,t,c,4,3] if k==4 else [k,t,c,0,0]
                          for k in (1,3,4) for t in (3,5,7) for c in (1,2,4,5)],np.int32)
        n = len(goals)
        data = {"goals":goals,"rules":np.broadcast_to(np.asarray(EmptyRule().encode())[None],(n,1,7)).copy(),
                "init_tiles":np.zeros((n,3,2),np.uint8),"num_rules":np.zeros((n,1),np.uint8)}
        with tempfile.TemporaryDirectory() as tmp, patch("miniwam.server.xland.fetch_benchmark",return_value=data):
            for depth in (1,2,3):
                out = make_banks({"depth":depth,"split_seed":2026,"tasks":{"train":6,"val":3,"test":3}},
                                 tmp,Path(tmp)/f"d{depth}")
                self.assertTrue((out/"complete.json").exists())
                self.assertEqual(audit_banks(out)["exact_minimum_depth"],depth)
                splits = [set(load_bank(out,s)["group_id"]) for s in ("train","val","test")]
                self.assertFalse(splits[0]&splits[1] or splits[0]&splits[2] or splits[1]&splits[2])


class NanoWMContracts(unittest.TestCase):
    def make_model(self,mode):
        return NanoWAM(mode,3,width=64,layers=2,heads=4,action_width=32).eval()

    def batch(self):
        torch.manual_seed(7)
        current = torch.randn(2,4,16,16)
        actions = torch.tensor([[1,3,4],[2,4,3]])
        target = torch.randn(2,3,4,16,16)
        valid = torch.tensor([[True,False,False],[True,True,False]])
        return current,actions,target,valid,valid.clone()

    def test_objectives_and_invalid_future_independence(self):
        for mode in ("policy","video","joint"):
            model = self.make_model(mode)
            current,actions,target,valid,expert = self.batch()
            loss,_ = model.loss(current,actions,target,valid,expert,generator=torch.Generator().manual_seed(91))
            changed_target = target.clone();changed_target[~valid] = 10000
            changed_actions = actions.clone();changed_actions[~valid] = 5
            other,_ = model.loss(current,changed_actions,changed_target,valid,expert,generator=torch.Generator().manual_seed(91))
            torch.testing.assert_close(loss,other)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.parameters()))
            if mode == "video":
                self.assertIsNone(model.action_head)
            if mode == "policy":
                irrelevant,_ = model.loss(current,actions,target+100,valid,expert)
                torch.testing.assert_close(loss,irrelevant)

    def test_action_interface_has_no_future_or_recorded_action_input(self):
        model = self.make_model("joint")
        import inspect
        self.assertEqual(list(inspect.signature(model.action_logits).parameters),["current"])
        current,*_ = self.batch()
        self.assertEqual(model.action_logits(current).shape,(2,3,6))

    def test_upstream_action_alignment_is_not_double_shifted(self):
        model = self.make_model("video")
        current,actions,target,valid,expert = self.batch()
        with patch.object(model.backbone,"forward",return_value=torch.zeros(2,4,4,16,16)) as forward:
            model.velocity(current,target,actions,torch.zeros(2,3,dtype=torch.long))
            supplied = forward.call_args.kwargs["action"]
            torch.testing.assert_close(supplied[:,:3],torch.nn.functional.one_hot(actions,6).float())
            self.assertEqual(float(supplied[:,3].abs().sum()),0.)

    def test_sampler_clamps_invalid_slots(self):
        model = self.make_model("video")
        current,actions,target,valid,expert = self.batch()
        pred = model.sample(current,actions,ddim_steps=3,noise=target,valid=valid)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertEqual(float(pred[~valid].abs().sum()),0.)

    def test_cosine_schedule(self):
        a = cosine_schedule()
        self.assertEqual(len(a),1000)
        self.assertTrue(((a>0)&(a<1)).all())
        self.assertTrue((a[1:]<a[:-1]).all())


class DataAndRunContracts(unittest.TestCase):
    def test_collection_and_exact_replay_without_training(self):
        class ConstantActor(torch.nn.Module):
            def forward(self,x):
                return torch.zeros(len(x),6,device=x.device),torch.zeros(len(x),device=x.device)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);banks=root/"banks";banks.mkdir()
            (root/"expert.pt").write_bytes(b"fixture; actor loader is mocked")
            manifest={"config":{"depth":2},"splits":{}}
            from miniwam.server.common import file_hash, save_npz
            for split in ("train","val","test"):
                records=[task_record(k,2) for k in (1,3,4)]
                save_npz(banks/f"{split}.npz",**{k:np.stack([r[k] for r in records]) for k in records[0]})
                manifest["splits"][split]={"sha256":file_hash(banks/f"{split}.npz")}
            write_json(banks/"manifest.json",manifest)
            with patch("miniwam.server.dataset.load_actor",return_value=ConstantActor()), patch("miniwam.server.dataset.iio.imwrite"):
                out=collect(banks,root/"expert.pt",{"seed":7,"train_repeats":7,"eval_repeats":7,"epsilon":0.25},root/"data",CPU)
            report=audit_dataset(out,banks)
            self.assertTrue(report["transition_replay"])
            self.assertEqual(report["splits"]["train"]["episodes"],21)
            saved=np.load(out/"train/observations.npy",mmap_mode="r")
            self.assertEqual(saved.shape,(21,3,128,128,3))

    def test_counterfactual_predictions_compare_to_their_own_futures(self):
        class PerfectAlternative:
            mode="video"
            horizon=1
            def sample(self,current,actions,ddim_steps,noise,valid):
                z=torch.zeros(6,1,4,16,16)
                z[:,0,0,0,0]=actions[:,0]
                return z
        def fake_decode(vae,z):
            value=z[:,0,0,0]*25/255
            return value[:,None,None,None].expand(-1,3,128,128)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"data/train").mkdir(parents=True);(root/"cache").mkdir()
            arrays={"actions":np.arange(6)[:,None],"valid":np.ones((6,1),bool),"expert_mask":np.zeros((6,1),bool),
                    "task_ids":np.zeros(6,np.int32),"observations":np.zeros((6,2,128,128,3),np.uint8)}
            for i in range(6):arrays["observations"][i,1]=i*25
            for name,a in arrays.items():np.save(root/f"data/train/{name}.npy",a)
            np.save(root/"cache/train.npy",np.zeros((6,2,4,16,16),np.float16))
            windows=LatentWindows(root/"data",root/"cache","train",1)
            with patch("miniwam.server.evaluation.decode",side_effect=fake_decode):
                result=counterfactual_metrics(PerfectAlternative(),None,windows,CPU,3,1)
            self.assertAlmostEqual(result["own_alternative_rgb_mse"],0.)
            self.assertGreater(result["wrong_pairing_rgb_mse"],0.)

    def test_window_masks_do_not_cross_episode_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root/"dataset/train").mkdir(parents=True);(root/"cache").mkdir()
            z = np.arange(2*4*4*16*16,dtype=np.float32).reshape(2,4,4,16,16)
            np.save(root/"cache/train.npy",z)
            for name,array in {"actions":np.array([[1,2,3],[4,-1,-1]]),
                               "valid":np.array([[1,1,1],[1,0,0]],bool),
                               "expert_mask":np.array([[1,1,1],[0,0,0]],bool)}.items():
                np.save(root/f"dataset/train/{name}.npy",array)
            w = LatentWindows(root/"dataset",root/"cache","train",3)
            current,actions,target,valid,expert = w.batch([2,3],CPU)
            self.assertEqual(valid.tolist(),[[True,False,False],[True,False,False]])
            self.assertFalse(expert[1].any())
            torch.testing.assert_close(target[0,0],torch.from_numpy(z[0,3]))
            with self.assertRaises(ValueError):
                LatentWindows(root/"dataset",root/"cache","train",4)

    def test_resume_refuses_changed_source_and_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"run"
            with patch("miniwam.server.common.source_digest",return_value="first"):
                stage(out,{"a":1},False)
                stage(out,{"a":1},True)
                with self.assertRaises(ValueError): stage(out,{"a":2},True)
            with patch("miniwam.server.common.source_digest",return_value="changed"):
                with self.assertRaises(ValueError): stage(out,{"a":1},True)

    def test_server_training_refuses_local_machine(self):
        with patch("platform.system",return_value="Darwin"):
            with self.assertRaisesRegex(RuntimeError,"server-only"):
                require_server()

    def test_configs_have_real_prediction_horizons(self):
        for path in (Path(__file__).resolve().parents[1]/"configs/server").glob("*.json"):
            c=json.loads(path.read_text());validate_config(c)
            from miniwam.server.protocol import episode_limit
            c["horizons"]=[episode_limit(c["bank"])+1]
            with self.assertRaises(ValueError): validate_config(c)


if __name__ == "__main__":
    unittest.main()
