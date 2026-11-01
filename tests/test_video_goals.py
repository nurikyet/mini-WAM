import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np
import torch
from torch.nn import functional as F

from miniwam.server.goal_images import goal_images, goal_scores, goal_weights
from miniwam.server.xland import render_ts, simulator, single_ts
from miniwam.server.common import file_hash, save_npz, write_json
from miniwam.server.diagnostics import diagnose_video
from test_server_contracts import task_record


class PublicGoalImageTests(unittest.TestCase):
    def test_rgb_goal_specification_contains_the_actual_success_image(self):
        env,params=simulator(1);step=jax.jit(env.step)
        for kind in (1,3,4):
            for orientation in range(4):
                for tile_a,tile_b in (((3,1),(4,3)),((11,6),(11,5)),((7,4),(12,1))):
                    record=task_record(kind,1,orientation,tile_a,tile_b)
                    ts=single_ts({k:v[None] for k,v in record.items()},0)
                    current=render_ts(ts);original=current.copy()
                    goals=goal_images(current)
                    np.testing.assert_array_equal(current,original)
                    candidates=[step(params,ts,a) for a in range(6)]
                    futures=np.stack([render_ts(s) for s in candidates])
                    successful=np.array([float(s.reward)>0 for s in candidates])
                    self.assertTrue(any(np.array_equal(g,t) for g in goals for t in futures[successful]),(kind,orientation,tile_a))
                    encode=lambda rgb:F.avg_pool2d(torch.from_numpy(rgb.copy()).permute(0,3,1,2).float()/255,8)
                    scores=goal_scores(encode(futures),encode(goals),goal_weights(current,goals,torch.device('cpu')))
                    self.assertTrue(successful[int(scores.argmax())],(kind,orientation,scores.tolist()))
        self.assertEqual(list(inspect.signature(goal_images).parameters),["current"])

    def test_score_depends_on_predictions_not_candidate_action_order(self):
        torch.manual_seed(3)
        predictions=torch.randn(6,4,16,16);goals=torch.randn(3,4,16,16);weights=torch.ones(3,1,16,16)
        order=torch.tensor([3,1,5,0,4,2])
        torch.testing.assert_close(goal_scores(predictions[order],goals,weights),goal_scores(predictions,goals,weights)[order])

    def test_invalid_rgb_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'128x128'):
            goal_images(np.zeros((64,64,3),np.uint8))

    def test_diagnostic_reads_old_weights_and_writes_only_a_new_report(self):
        record=task_record(3,1)
        bank={k:v[None] for k,v in record.items()}
        ts=single_ts(bank,0);env,params=simulator(1)
        frames=np.stack([render_ts(env.step(params,ts,a)) for a in range(6)])
        def fake_encode(vae,rgb,device):
            x=torch.as_tensor(np.array(rgb)).permute(0,3,1,2).float()/255
            z=F.avg_pool2d(x,8)
            return torch.cat([z,torch.zeros(len(z),1,16,16)],1)
        def fake_decode(vae,z):
            return F.interpolate(z[:,:3],size=(128,128),mode='nearest')
        encoded=fake_encode(None,frames,torch.device('cpu'))
        class FrozenFixture:
            def sample(self,current,actions,steps,noise):
                return encoded[actions[:,0]][:,None]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'original';banks=run/'banks';banks.mkdir(parents=True)
            save_npz(banks/'val.npz',**bank)
            write_json(banks/'manifest.json',{'config':{'depth':1},'splits':{'val':{'sha256':file_hash(banks/'val.npz')}}})
            model=run/'models/video_n1_seed3407';model.mkdir(parents=True)
            (model/'best.pt').write_bytes(b'original frozen weights')
            write_json(model/'complete.json',{'files_sha256':{'best.pt':file_hash(model/'best.pt')}})
            before={p:file_hash(p) for p in run.rglob('*') if p.is_file()}
            with patch('miniwam.server.diagnostics.load_model',return_value=(FrozenFixture(),{'step':35000})), \
                 patch('miniwam.server.diagnostics.load_vae',return_value=None), \
                 patch('miniwam.server.diagnostics.encode',side_effect=fake_encode), \
                 patch('miniwam.server.diagnostics.decode',side_effect=fake_decode):
                out=diagnose_video(run,root/'new-report',torch.device('cpu'),ddim_steps=3)
            after={p:file_hash(p) for p in run.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            result=json.loads((out/'diagnosis.json').read_text())
            self.assertFalse(result['training'])
            self.assertEqual(result['split'],'val')
            self.assertEqual(result['aggregate']['model_goal_cost']['sr'],1.)
            self.assertEqual(len(result['records'][0]['actions']),6)
            self.assertEqual(result['moving_counterfactuals'],1)
            self.assertEqual(len(list(out.glob('*.gif'))),6)
            self.assertTrue((out/'complete.json').exists())


if __name__ == "__main__":unittest.main()
