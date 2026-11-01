import json
from pathlib import Path
import re
import tempfile
import unittest
from urllib.parse import unquote

from scripts.view_results import build_gallery


class GalleryTests(unittest.TestCase):
    def test_gallery_references_real_files_and_keeps_weights_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"downloaded run";root.mkdir()
            summary={"models":[{"mode":"video","horizon":1,"sr":{"train":0.8,"val":0.4,"test":0.3}}]}
            (root/"summary.json").write_text(json.dumps(summary))
            model=root/"models/video_n1_seed3407";(model/"val").mkdir(parents=True)
            weight=model/"best.pt";weight.write_bytes(b"unchanged checkpoint")
            (model/"val/evaluation.json").write_text(json.dumps({"closed_loop":{"successes":2,"episodes":5,"sr":0.4}}))
            (model/"val/task_0_failure.mp4").write_bytes(b"video fixture")
            (model/"val/forecast_0_truth_left.mp4").write_bytes(b"video fixture")
            (root/"latents").mkdir();(root/"latents/val_reconstruction_0.png").write_bytes(b"image fixture")
            gallery=build_gallery(root,Path(tmp)/"report with spaces/index.html")
            content=gallery.read_text()
            self.assertIn("40.0%",content)
            self.assertIn("только один будущий кадр",content)
            for link in re.findall(r'(?:src|href)="([^"]+)"',content):
                self.assertTrue((gallery.parent/unquote(link)).is_file(),link)
            self.assertEqual(weight.read_bytes(),b"unchanged checkpoint")

    def test_missing_experiment_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError,"summary.json"):
                build_gallery(tmp)


if __name__ == "__main__":unittest.main()
