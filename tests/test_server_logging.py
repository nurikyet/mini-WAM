"""Logging and launcher checks without CUDA, training, or network access."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from miniwam.server.progress import Progress, duration, log, logged_stage


class ProgressLoggingTests(unittest.TestCase):
    def test_progress_has_timestamp_counts_metrics_and_resume_eta(self):
        output = io.StringIO()
        with redirect_stdout(output), patch("miniwam.server.progress.time.monotonic", side_effect=[100.,110.]):
            progress = Progress("NanoWM joint", 100, initial=50)
            progress.update(60, force=True, loss=0.25, val_sr="100%")
        value = output.getvalue()
        self.assertRegex(value, r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\]")
        self.assertIn("done=60 total=100 percent=60.0%", value)
        self.assertIn("elapsed=00:00:10 eta=00:00:40", value)
        self.assertIn("loss=0.25 val_sr=100%", value)

    def test_progress_limits_output_and_always_reports_completion(self):
        output = io.StringIO()
        with redirect_stdout(output), patch("miniwam.server.progress.time.monotonic", side_effect=[0.,1.,2.,31.,32.]):
            progress = Progress("frames", 10)
            for step in (1,2,3,10):
                progress.update(step)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines),4)  # start, first, timed update, completion
        self.assertNotIn("done=2 total",output.getvalue())
        self.assertIn("done=10 total=10 percent=100.0%",lines[-1])
        self.assertEqual(duration(3661),"01:01:01")

    def test_stage_failure_logs_error_and_preserves_original_exception(self):
        output = io.StringIO()
        with redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError,"test failure"):
                with logged_stage("validation"):
                    raise RuntimeError("test failure")
        value = output.getvalue()
        self.assertIn("НАЧАЛО",value)
        self.assertIn("ОШИБКА СТАДИИ",value)
        self.assertIn("RuntimeError: test failure",value)
        self.assertNotIn("ГОТОВО",value)

    def test_slow_stage_emits_heartbeat_and_stops_it_on_exit(self):
        heartbeat = threading.Event()
        real_log = log
        def observe(message, **kwargs):
            real_log(message, **kwargs)
            if message == "ОЖИДАНИЕ ЗАВЕРШЕНИЯ СТАДИИ":
                heartbeat.set()
        with redirect_stdout(io.StringIO()), patch.dict(os.environ,{"MINIWAM_LOG_INTERVAL":"0.01"}), \
                patch("miniwam.server.progress.log",side_effect=observe):
            with logged_stage("long audit"):
                self.assertTrue(heartbeat.wait(timeout=1))
        self.assertFalse(any(t.name == "miniwam-log-heartbeat" for t in threading.enumerate()))


class ServerLauncherLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root/"scripts"
        scripts.mkdir()
        source = Path(__file__).resolve().parents[1]/"scripts/train_server.sh"
        shutil.copyfile(source,scripts/"train_server.sh")
        binaries = self.root/"bin"
        binaries.mkdir()
        self.python = binaries/"fake-python"
        self.python.write_text("""#!/bin/sh
printf 'FAKE STDOUT: %s\\n' "$*"
printf 'FAKE STDERR: %s\\n' "$*" >&2
case "$*" in
  *doctor*) exit "${FAKE_DOCTOR_CODE:-0}" ;;
  *pipeline*)
    while [ -n "${FAKE_RELEASE_PATH:-}" ] && [ ! -f "$FAKE_RELEASE_PATH" ]; do sleep 0.02; done
    exit "${FAKE_PIPELINE_CODE:-0}" ;;
esac
""")
        self.python.chmod(0o755)
        # The host may be macOS; only the Linux runner's lock command is stubbed.
        flock = binaries/"flock"
        flock.write_text('#!/bin/sh\nexit "${FAKE_LOCK_CODE:-0}"\n')
        flock.chmod(0o755)
        self.env = {**os.environ,"PATH":str(binaries)+os.pathsep+os.environ["PATH"],
                    "PYTHON":str(self.python),"LOG_DIR":str(self.root/"log files"),
                    "OUT_ROOT":str(self.root/"output files"),"MINIWAM_TMPDIR":str(self.root/"tmp")}

    def run_script(self,target="all",**env):
        return subprocess.run(["bash",str(self.root/"scripts/train_server.sh"),target],
                              cwd=self.root,env={**self.env,**env},capture_output=True,text=True,timeout=10)

    def latest(self,target="all"):
        return self.root/"log files"/f"{target}-latest.log"

    def test_success_writes_both_streams_and_separate_logs_on_each_launch(self):
        result = self.run_script()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        latest = self.latest()
        self.assertTrue(latest.is_symlink())
        contents = latest.read_text()
        self.assertIn("FAKE STDOUT",contents)
        self.assertIn("FAKE STDERR",contents)
        self.assertIn("single_goal.json",contents)
        self.assertNotIn("one_step.json",contents)
        self.assertIn("ГОТОВО: выполнение завершено; код=0",contents)
        first = latest.resolve()
        result = self.run_script()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertNotEqual(latest.resolve(),first)
        self.assertTrue(first.exists())

    def test_sr_gate_exit_two_is_preserved_and_next_stage_does_not_run(self):
        result = self.run_script(FAKE_PIPELINE_CODE="2")
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        contents = self.latest().read_text()
        self.assertIn("SR НИЖЕ 100%",contents)
        self.assertNotIn("two_step.json",contents)
        self.assertNotIn("ГОТОВО: выполнение завершено",contents)

    def test_log_lines_are_visible_while_the_experiment_is_running(self):
        release = self.root/"release"
        process = subprocess.Popen(["bash",str(self.root/"scripts/train_server.sh"),"one-step"],
            cwd=self.root,env={**self.env,"FAKE_RELEASE_PATH":str(release)},
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            latest = self.latest("one-step")
            deadline = time.monotonic()+5
            contents = ""
            while time.monotonic() < deadline:
                if latest.exists():
                    contents = latest.read_text()
                    if "FAKE STDERR: -m miniwam.server pipeline" in contents:
                        break
                threading.Event().wait(0.02)
            self.assertIn("FAKE STDERR: -m miniwam.server pipeline",contents)
            self.assertIsNone(process.poll())
            self.assertNotIn("ГОТОВО: выполнение завершено",contents)
        finally:
            release.touch()
            stdout,stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode,0,stdout+stderr)

    def test_doctor_failure_is_logged_and_training_does_not_start(self):
        result = self.run_script(FAKE_DOCTOR_CODE="7")
        self.assertEqual(result.returncode,7,result.stdout+result.stderr)
        contents = self.latest().read_text()
        self.assertIn("ОШИБКА: запуск завершился с кодом 7",contents)
        self.assertNotIn("pipeline",contents)

    def test_diagnostic_command_uses_old_run_without_starting_a_training_pipeline(self):
        result=self.run_script("diagnose-video")
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        contents=self.latest("diagnose-video").read_text()
        self.assertIn("miniwam.server diagnose-video",contents)
        self.assertIn("one_step_video_diagnostic",contents)
        self.assertNotIn("miniwam.server pipeline",contents)

    def test_lock_rejection_does_not_replace_active_log_link(self):
        result = self.run_script()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        previous = self.latest().resolve()
        result = self.run_script(FAKE_LOCK_CODE="1")
        self.assertEqual(result.returncode,1)
        self.assertEqual(self.latest().resolve(),previous)
        self.assertIn("другой процесс",result.stdout)


if __name__ == "__main__":
    unittest.main()
