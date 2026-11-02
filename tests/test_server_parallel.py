"""Exercise real short subprocesses and cancellation without CUDA or training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from miniwam.server.parallel import (Job, WorkerFailed, build_jobs, run_jobs,
                                    run_parallel, worker_environment)
from miniwam.server.worker import configure_cpu

ROOT = Path(__file__).resolve().parents[1]


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="miniwam parallel ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = self.root/"events.jsonl"
        self.fixture = self.root/"fake worker.py"
        self.fixture.write_text('''import json,os,sys,time
from pathlib import Path
events=Path(sys.argv[1]);name=sys.argv[2];barrier=int(sys.argv[3])
def event(kind):
    record={"name":name,"event":kind,"pid":os.getpid(),"time":time.monotonic(),
            "fraction":os.environ.get("MINIWAM_CUDA_MEMORY_FRACTION"),
            "affinity":os.environ.get("MINIWAM_CPU_AFFINITY")}
    fd=os.open(events,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
    try:os.write(fd,(json.dumps(record)+"\\n").encode())
    finally:os.close(fd)
event("start")
print("stdout "+name,flush=True);print("stderr "+name,file=sys.stderr,flush=True)
if name == "fail":sys.exit(7)
if name == "long":time.sleep(30)
deadline=time.monotonic()+5
while barrier and sum(json.loads(line)["event"]=="start" for line in events.read_text().splitlines())<barrier:
    if time.monotonic()>deadline:raise RuntimeError("pool did not admit the promised workers")
    time.sleep(0.01)
time.sleep(0.1)
event("end")
''')

    def job(self,name,barrier=0):
        return Job(name,(sys.executable,str(self.fixture),str(self.events),name,str(barrier)))

    def test_real_process_pool_runs_nine_jobs_with_at_most_three_concurrently(self):
        jobs=[self.job(f"model{i}",3) for i in range(9)]
        state=run_jobs(jobs,self.root,3,poll_seconds=0.01)
        events=[json.loads(line) for line in self.events.read_text().splitlines()]
        active,maximum=0,0
        for event in events:
            active += 1 if event["event"]=="start" else -1
            maximum=max(maximum,active)
            self.assertLessEqual(active,3)
            self.assertAlmostEqual(float(event["fraction"]),0.3)
            self.assertTrue(event["affinity"])
        self.assertEqual(active,0);self.assertEqual(maximum,3)
        self.assertEqual(state["state"],"completed")
        self.assertTrue(all(record["state"]=="completed" for record in state["jobs"].values()))
        for job in jobs:
            path=self.root/"job_logs"/f"{job.name}-latest.log"
            self.assertTrue(path.is_symlink())
            self.assertIn("stdout "+job.name,path.read_text())
            self.assertIn("stderr "+job.name,path.read_text())
        saved=json.loads((self.root/"parallel_status.json").read_text())
        self.assertEqual(saved["workers"],3)

    def test_worker_failure_stops_other_processes_and_does_not_start_more_jobs(self):
        with self.assertRaisesRegex(WorkerFailed,"exit 7"):
            run_jobs([self.job("fail"),self.job("long"),self.job("never")],self.root,2,poll_seconds=0.01)
        status=json.loads((self.root/"parallel_status.json").read_text())
        self.assertEqual(status["state"],"failed")
        self.assertEqual(status["jobs"]["fail"]["state"],"failed")
        self.assertEqual(status["jobs"]["long"]["state"],"interrupted")
        self.assertEqual(status["jobs"]["never"]["state"],"not_started")
        with self.assertRaises(ProcessLookupError):os.kill(status["jobs"]["long"]["pid"],0)

    def test_queue_cleans_up_when_the_coordinator_is_interrupted(self):
        def interrupt(_):raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run_jobs([self.job("quick"),self.job("long"),self.job("never")],self.root,2,
                     on_complete=interrupt,poll_seconds=0.01)
        status=json.loads((self.root/"parallel_status.json").read_text())
        self.assertEqual(status["state"],"interrupted")
        self.assertEqual(status["jobs"]["long"]["state"],"interrupted")
        with self.assertRaises(ProcessLookupError):os.kill(status["jobs"]["long"]["pid"],0)

    def test_shared_preparation_failure_never_starts_models(self):
        config=ROOT/"configs/server/single_goal.json"
        with patch("miniwam.server.parallel.run_command",return_value=7) as command, \
             patch("miniwam.server.parallel.run_jobs") as pool:
            self.assertEqual(run_parallel(config,self.root/"run",3),7)
        self.assertIn("prepare",command.call_args.args[0]);pool.assert_not_called()

    def test_preparation_finishes_before_pool_and_reports_preserve_sr_gate_code(self):
        out=self.root/"run";order=[]
        def command(args,env):
            if "prepare" in args:
                order.append("prepare");out.mkdir()
            elif "report" in args:
                order.append("report");(out/"summary.json").write_text('{"all_sr_100":false}')
            return 0
        def pool(jobs,directory,workers,**kwargs):
            self.assertEqual(order,["prepare"]);order.append("pool")
            self.assertEqual(len(jobs),9);self.assertEqual(workers,3)
            self.assertTrue(all("--train-tasks" not in j.command for j in jobs))
            kwargs["on_complete"](jobs[0])
        with patch("miniwam.server.parallel.run_command",side_effect=command), \
             patch("miniwam.server.parallel.run_jobs",side_effect=pool):
            result=run_parallel(ROOT/"configs/server/single_goal.json",out,3,True,True)
        self.assertEqual(result,2);self.assertEqual(order,["prepare","pool","report","report"])

    def test_jobs_have_unique_outputs_and_resource_settings_can_change_on_resume(self):
        config=json.loads((ROOT/"configs/server/single_goal.json").read_text())
        jobs=build_jobs(self.root,config,True)
        self.assertEqual(len({j.name for j in jobs}),9)
        self.assertIn("--resume",jobs[0].command)
        self.assertEqual(jobs[0].name,"video_n5_seed3407")
        for workers in (1,2,3):
            env=worker_environment({},workers,0,list(range(12)),4)
            self.assertAlmostEqual(float(env["MINIWAM_CUDA_MEMORY_FRACTION"]),0.9/workers)
            self.assertEqual(env["MINIWAM_CPU_AFFINITY"],"0,1,2,3")
        config["horizons"]=[1,1]
        with self.assertRaisesRegex(ValueError,"unique"):build_jobs(self.root,config,True)

    def test_invalid_worker_counts_are_rejected_before_gpu_preparation(self):
        for workers in (0,10):
            with patch("miniwam.server.parallel.run_command") as command:
                with self.assertRaises(ValueError):
                    run_parallel(ROOT/"configs/server/single_goal.json",self.root/"run",workers)
                command.assert_not_called()

    def test_coordinator_does_not_import_torch_or_jax(self):
        result=subprocess.run([sys.executable,"-c",
            'import sys; import miniwam.server.parallel; assert "torch" not in sys.modules; assert "jax" not in sys.modules'],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_cpu_affinity_is_set_before_loading_ml_libraries(self):
        with patch.dict(os.environ,{"MINIWAM_CPU_AFFINITY":"2,3"}), \
             patch("miniwam.server.worker.os.sched_setaffinity",create=True) as set_affinity:
            configure_cpu()
        set_affinity.assert_called_once_with(0,{2,3})


if __name__=="__main__":unittest.main()
