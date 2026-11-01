"""Native transitions, rendering and bounded planning; no optimizer training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np
import torch

from miniwam.server.cli import validate_config
from miniwam.server.evaluation import plan, visual_score
from miniwam.server.protocol import check_window_coverage, episode_limit, storage_estimate
from miniwam.server.xland import (audit_banks, load_bank, make_banks, reference_plan,
    render_record, render_ts, simulator, single_ts)
from miniwam.server.vae import LatentWindows
from miniwam.server.model import NanoWAM
from miniwam.server.dataset import collect, audit as audit_dataset
from miniwam.server.common import file_hash, save_npz, write_json
from xminigrid.core.rules import EmptyRule

torch.set_num_threads(1)
CPU = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[1]


def navigation_record(kind, moves=3, direction=0, facing=None):
    deltas = np.array([[-1,0],[0,1],[1,0],[0,-1]])
    pos = np.array([(7,4),(4,1),(1,4),(4,7)][direction],np.int32)
    goal = np.array([kind,3,1,4,3] if kind == 4 else [kind,3,1,0,0],np.uint8)
    grid = np.empty((9,9,2),np.uint8); grid[:]=(1,7)
    grid[[0,-1],:] = (2,6); grid[:,[0,-1]] = (2,6)
    target = pos+deltas[direction]*(moves+(2 if kind == 4 else 1))
    grid[tuple(target)] = goal[3:5] if kind == 4 else goal[1:3]
    facing = direction if facing is None else facing
    turns = min((direction-facing)%4,(facing-direction)%4)
    return {"grid":grid,"position":pos,"direction":np.int32(facing),
        "pocket":goal[1:3].copy() if kind == 4 else np.array([0,0],np.uint8),
        "goal":goal,"rules":np.asarray(EmptyRule().encode())[None],
        "source_row":np.int64(0),"group_id":np.int64(0),"travel_moves":np.int32(moves),
        "approach_direction":np.int32(direction),"minimum_transitions":np.int32(moves+turns+(kind != 3))}


class SingleGoalTests(unittest.TestCase):
    def test_three_cells_are_three_transitions_before_one_goal_is_completed(self):
        env,params=simulator(16);step=jax.jit(env.step)
        for kind in (1,3,4):
            for direction in range(4):
                record=navigation_record(kind,direction=direction)
                ts=single_ts({k:v[None] for k,v in record.items()},0)
                origin=np.asarray(ts.state.agent.position).copy()
                for t,action in enumerate(reference_plan(record)):
                    ts=step(params,ts,action)
                    self.assertEqual(float(ts.reward)>0,t==int(record["minimum_transitions"])-1)
                    self.assertEqual(bool(ts.last()),t==int(record["minimum_transitions"])-1)
                self.assertEqual(int(np.abs(np.asarray(ts.state.agent.position)-origin).sum()),3)

    def test_empty_slots_are_blank_and_raw_visual_control_handles_navigation(self):
        env,params=simulator(16);step=jax.jit(env.step)
        for kind in (1,3,4):
            for direction in range(4):
                record=navigation_record(kind,direction=direction,facing=(direction+2)%4)
                rgb=render_record(record,"clean_v2")
                if kind != 4:
                    self.assertTrue(np.all(rgb[112:124,52:64]==24))
                    self.assertTrue(np.all(rgb[112:124,96:108]==24))
                    self.assertFalse(np.array_equal(rgb,render_record(record,"legacy_v1")))
                ts=single_ts({k:v[None] for k,v in record.items()},0)
                for t in range(16):
                    current=torch.from_numpy(render_ts(ts,"clean_v2")).permute(2,0,1).float()[None]/255
                    candidates=[step(params,ts,a) for a in range(6)]
                    frames=torch.from_numpy(np.stack([render_ts(s,"clean_v2") for s in candidates])).permute(0,3,1,2).float()/255
                    scores=visual_score(frames,current.expand(6,-1,-1,-1))
                    ts=candidates[int(scores.argmax())]
                    if bool(ts.last()):break
                self.assertGreater(float(ts.reward),0,(kind,direction,scores.tolist()))

    def test_bank_witnesses_and_eval_scenes_do_not_change_with_train_count(self):
        goals=np.array([[k,t,c,4,3] if k==4 else [k,t,c,0,0]
            for k in (1,3,4) for t in (3,5,7) for c in (1,2,4,5)],np.int32)
        n=len(goals)
        data={"goals":goals,"rules":np.broadcast_to(np.asarray(EmptyRule().encode())[None],(n,1,7)).copy(),
              "init_tiles":np.zeros((n,3,2),np.uint8),"num_rules":np.zeros((n,1),np.uint8)}
        with tempfile.TemporaryDirectory() as tmp,patch("miniwam.server.xland.fetch_benchmark",return_value=data):
            config={"family":"single_goal","max_steps":16,"travel_moves":[1,2,3,4],"split_seed":2026,
                    "tasks":{"train":12,"val":6,"test":6}}
            a=make_banks(config,tmp,Path(tmp)/"a")
            config={**config,"tasks":{**config["tasks"],"train":24}}
            b=make_banks(config,tmp,Path(tmp)/"b")
            self.assertEqual(audit_banks(a)["semantic_objectives"],1)
            for split in ("val","test"):
                aa,bb=load_bank(a,split),load_bank(b,split)
                for key in aa:np.testing.assert_array_equal(aa[key],bb[key])

    def test_horizon_five_predicts_five_real_frames_with_terminal_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"data/train").mkdir(parents=True);(root/"cache").mkdir()
            directory=root/"data/train"
            valid=np.array([[1]*7+[0],[1]*3+[0]*5],bool)
            np.save(directory/"valid.npy",valid);np.save(directory/"expert_mask.npy",valid)
            np.save(directory/"actions.npy",np.where(valid,0,-1))
            np.save(directory/"task_ids.npy",np.array([0,1]))
            z=np.broadcast_to(np.arange(9)[None,:,None,None,None],(2,9,4,16,16)).astype(np.float16).copy()
            np.save(root/"cache/train.npy",z)
            windows=LatentWindows(root/"data",root/"cache","train",5)
            current,actions,future,mask,expert=windows.batch([0,7],CPU)
            self.assertEqual(future.shape,(2,5,4,16,16))
            self.assertEqual(future[0,:,0,0,0].tolist(),[1,2,3,4,5])
            self.assertEqual(mask.tolist(),[[True]*5,[True]*3+[False]*2])
            restricted=LatentWindows(root/"data",root/"cache","train",5,train_task_limit=1)
            self.assertEqual(len(np.unique(restricted.indices[:,0])),1)
            with self.assertRaises(ValueError):LatentWindows(root/"data",root/"cache","val",5,train_task_limit=1)

    def test_long_horizon_planning_is_bounded_and_reproducible(self):
        class Predictor:
            horizon=5
            def __init__(self):self.count=0
            def sample(self,current,actions,steps,noise,valid):
                self.count+=len(actions)
                z=torch.zeros(len(actions),5,4,16,16)
                z[:,:,0,0,0]=actions
                return z
        rgb=np.zeros((128,128,3),np.uint8)
        def decode_stub(vae,z):
            out=torch.zeros(len(z),z.shape[1],3,128,128)
            out[:,:,0,0,0]=z[:,:,0,0,0]
            return out
        def scorer(pred,current):return -(pred[:,0,0,0]-3).abs()
        with patch("miniwam.server.evaluation.encode",return_value=torch.zeros(1,4,16,16)), \
             patch("miniwam.server.evaluation.decode",side_effect=decode_stub), \
             patch("miniwam.server.evaluation.visual_score",side_effect=scorer):
            a=Predictor(); first,trace=plan(a,None,rgb,16,CPU,4,population=24,rounds=2,elites=4)
            b=Predictor(); second,other=plan(b,None,rgb,16,CPU,4,population=24,rounds=2,elites=4)
            self.assertEqual(first,3);self.assertEqual(second,first);self.assertEqual(trace,other)
            self.assertEqual(a.count,48);self.assertEqual(trace["plans"],48)
            self.assertEqual(len(trace["best_plan"]),5)

    def test_corrected_config_has_separate_budget_and_horizons(self):
        c=json.loads((ROOT/"configs/server/single_goal.json").read_text());validate_config(c)
        self.assertEqual(episode_limit(c["bank"]),16);self.assertEqual(c["horizons"],[1,3,5])
        self.assertLess(storage_estimate(c)["data_gib"],20)

    def test_padding_cannot_pass_for_a_real_long_horizon(self):
        info={"splits":{"train":{"diversity":{
            "complete_windows_by_horizon":{"1":10,"5":0},
            "expert_complete_windows_by_horizon":{"1":5,"5":0}}}}}
        check_window_coverage(info,[1])
        with self.assertRaisesRegex(ValueError,"no real"):
            check_window_coverage(info,[1,5])

    def test_motion_weighted_objective_ignores_invalid_targets(self):
        torch.manual_seed(1)
        model=NanoWAM("video",3,width=64,layers=2,heads=4,action_width=32).eval()
        current=torch.randn(2,4,16,16)
        target=torch.randn(2,3,4,16,16)
        valid=torch.tensor([[True,True,False],[True,False,False]])
        actions=torch.zeros(2,3,dtype=torch.long)
        one,_=model.loss(current,actions,target,valid,valid,generator=torch.Generator().manual_seed(42),motion_weight=4.)
        other=target.clone();other[~valid]=1000
        two,_=model.loss(current,actions,other,valid,valid,generator=torch.Generator().manual_seed(42),motion_weight=4.)
        torch.testing.assert_close(one,two)
        self.assertTrue(torch.isfinite(one))

    def test_actual_nanowm_handles_five_future_transitions_in_all_modes(self):
        current=torch.randn(1,4,16,16);future=torch.randn(1,5,4,16,16)
        actions=torch.tensor([[0,0,0,3,1]])
        valid=torch.tensor([[True,True,True,True,False]])
        for mode in ("policy","video","joint"):
            model=NanoWAM(mode,5,width=64,layers=2,heads=4,action_width=32).eval()
            loss,_=model.loss(current,actions,future,valid,valid,motion_weight=4.)
            self.assertTrue(torch.isfinite(loss));loss.backward()
            if mode != "video":self.assertEqual(model.action_logits(current).shape,(1,5,6))
            if mode != "policy":
                predicted=model.sample(current,actions,ddim_steps=2,noise=future,valid=valid)
                self.assertEqual(predicted.shape,future.shape)
                self.assertEqual(float(predicted[:,4].abs().sum()),0.)

    def test_action_plan_labels_end_at_the_first_exploratory_action(self):
        model=NanoWAM("policy",3,width=64,layers=2,heads=4,action_width=32).eval()
        current=torch.randn(1,4,16,16);target=torch.zeros(1,3,4,16,16)
        valid=torch.ones(1,3,dtype=torch.bool);expert=torch.tensor([[True,False,True]])
        actions=torch.tensor([[0,1,2]])
        one,_=model.loss(current,actions,target,valid,expert)
        actions[0,2]=5
        two,_=model.loss(current,actions,target,valid,expert)
        torch.testing.assert_close(one,two)

    def test_multistep_collection_replays_clean_rgb_and_audits_window_lengths(self):
        class ForwardActor(torch.nn.Module):
            def forward(self,x):
                return torch.zeros(len(x),6,device=x.device),torch.zeros(len(x),device=x.device)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);banks=root/"banks";banks.mkdir()
            (root/"expert.pt").write_bytes(b"fixture; PPO loader mocked, no training")
            config={"family":"single_goal","max_steps":16,"renderer":"clean_v2"}
            manifest={"config":config,"splits":{}}
            for split in ("train","val","test"):
                records=[navigation_record(k) for k in (1,3,4)]
                save_npz(banks/f"{split}.npz",**{k:np.stack([r[k] for r in records]) for k in records[0]})
                manifest["splits"][split]={"sha256":file_hash(banks/f"{split}.npz")}
            write_json(banks/"manifest.json",manifest)
            with patch("miniwam.server.dataset.load_actor",return_value=ForwardActor()),patch("miniwam.server.dataset.iio.imwrite"):
                out=collect(banks,root/"expert.pt",{"seed":1,"train_repeats":7,"eval_repeats":7,"epsilon":0.25},root/"dataset",CPU)
            audit=audit_dataset(out,banks)
            self.assertTrue(audit["transition_replay"])
            data=json.loads((out/"manifest.json").read_text())
            self.assertEqual(data["episode_max_steps"],16)
            self.assertEqual(data["renderer"],"clean_v2")
            self.assertGreater(data["splits"]["train"]["diversity"]["moving_transitions"],0)
            self.assertGreater(data["splits"]["train"]["diversity"]["complete_windows_by_horizon"]["5"],0)
            frames=np.load(out/"train/observations.npy",mmap_mode="r")
            self.assertEqual(frames.shape,(21,17,128,128,3))
            self.assertTrue(np.all(frames[0,0,112:124,96:108]==24))


if __name__=="__main__":unittest.main()
