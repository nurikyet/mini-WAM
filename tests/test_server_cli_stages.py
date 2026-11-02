"""Check worker CLI data isolation and scientific config, with training mocked."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from miniwam.server.cli import main
from miniwam.server.common import stage, write_json

ROOT = Path(__file__).resolve().parents[1]


class WorkerCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config=json.loads((ROOT/"configs/server/single_goal.json").read_text())
        self.run=stage(self.root/"shared",{"pipeline":self.config},False)
        write_json(self.run/"resolved_config.json",self.config)
        write_json(self.run/"expert/gate.json",{"passed":True})
        for name in ("banks","dataset","latents"):
            write_json(self.run/name/"complete.json",{"files_sha256":{}})

    def invoke(self,*extra):
        args=["train-model","--run",str(self.run),"--out",str(self.root/"model"),
              "--mode","joint","--horizon","5","--seed","3407","--resume",*extra]
        return main(args)

    def test_full_worker_preserves_model_config_and_ablation_changes_only_training_limit(self):
        with patch("miniwam.server.cli.require_server",return_value=torch.device("cpu")), \
             patch("pathlib.Path.home",return_value=self.root), \
             patch("miniwam.server.training.train") as train:
            self.assertEqual(self.invoke(),0)
            self.assertEqual(train.call_args.args[3],self.config["model"])
            shared_inputs=train.call_args.args[:3]
            self.assertEqual(self.invoke("--train-tasks","128"),0)
            self.assertEqual(train.call_args.args[:3],shared_inputs)
            self.assertEqual(train.call_args.args[3],{**self.config["model"],"train_task_limit":128})

    def test_worker_refuses_incomplete_shared_data(self):
        (self.run/"dataset/complete.json").unlink()
        with patch("miniwam.server.cli.require_server",return_value=torch.device("cpu")), \
             patch("pathlib.Path.home",return_value=self.root), \
             patch("miniwam.server.training.train") as train:
            with self.assertRaisesRegex(RuntimeError,"not complete"):self.invoke()
            train.assert_not_called()

    def test_prepare_entrypoint_does_not_run_the_wam_pipeline(self):
        with patch("miniwam.server.cli.require_server",return_value=torch.device("cpu")), \
             patch("pathlib.Path.home",return_value=self.root), \
             patch("miniwam.server.cli.prepare") as prepare, \
             patch("miniwam.server.cli.pipeline") as pipeline:
            result=main(["prepare","--config",str(ROOT/"configs/server/single_goal.json"),
                         "--out",str(self.root/"prepared"),"--resume"])
        self.assertEqual(result,0);prepare.assert_called_once();pipeline.assert_not_called()


if __name__=="__main__":unittest.main()
