"""Bounded independent experiments on one CUDA GPU; coordinator uses stdlib only.

Shared preparation exits before workers start, releasing its CUDA context. Each
worker is a fresh executable with its own RNG, optimizer, log and checkpoint.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from .progress import interval, log


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]


class WorkerFailed(RuntimeError):
    pass


class SignalReceived(KeyboardInterrupt):
    def __init__(self, signum):
        self.signum = signum


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,delete=False,encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def stop_processes(processes, timeout=10.):
    processes = list(processes)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic()+timeout
    for process in processes:
        try:
            process.wait(timeout=max(0.,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_command(command, env):
    process = subprocess.Popen(command,env=env,start_new_session=True)
    try:
        return process.wait()
    except BaseException:
        stop_processes([process])
        raise


def build_jobs(out, config, resume):
    out = Path(out)
    jobs = []
    # Start the longest horizons first; free slots immediately take the next job.
    for horizon in sorted(config["horizons"],reverse=True):
        for mode in ("video", "joint", "policy"):
            for seed in config["seeds"]:
                name = f"{mode}_n{horizon}_seed{seed}"
                command = [sys.executable,"-m","miniwam.server.worker","train-model",
                           "--run",str(out),"--out",str(out/"models"/name),
                           "--mode",mode,"--horizon",str(horizon),"--seed",str(seed)]
                if resume:
                    command.append("--resume")
                jobs.append(Job(name,tuple(command)))
    if not jobs or len({job.name for job in jobs}) != len(jobs):
        raise ValueError("Model names must be nonempty and unique; check horizons/seeds")
    return jobs


def cpu_settings(workers, env):
    available = sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else list(range(os.cpu_count() or 1))
    count = int(env.get("CPUS_PER_JOB",min(4,max(1,len(available)//workers))))
    if count < 1:
        raise ValueError("CPUS_PER_JOB must be positive")
    return available,min(count,len(available))


def worker_environment(base, workers, slot, available, cpus_per_job):
    env = dict(base)
    start = slot*cpus_per_job % len(available)
    affinity = available[start:start+cpus_per_job]
    threads = min(len(affinity),int(env.get("OMP_NUM_THREADS","2")))
    if threads < 1:
        raise ValueError("OMP_NUM_THREADS must be positive")
    env.update(MINIWAM_CPU_AFFINITY=",".join(map(str,affinity)),
               MINIWAM_CUDA_MEMORY_FRACTION=str(0.9/workers),
               PYTHONUNBUFFERED="1",JAX_PLATFORMS="cpu",XLA_PYTHON_CLIENT_PREALLOCATE="false",
               OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads),NUMEXPR_NUM_THREADS=str(threads))
    return env


def run_jobs(jobs, out, workers, *, env=None, on_complete=None, poll_seconds=0.25):
    """Real process pool, also testable with short CPU-only fixture commands."""
    if not 1 <= workers <= len(jobs):
        raise ValueError(f"workers must be in [1,{len(jobs)}]")
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("Job names must be unique")
    out = Path(out)
    log_dir = out/"job_logs"
    log_dir.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ if env is None else env)
    available, cpus_per_job = cpu_settings(workers,env)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    records = {job.name:{"state":"queued"} for job in jobs}
    state = {"started_at":timestamp(),"state":"running","coordinator_pid":os.getpid(),"workers":workers,
             "gpu_memory_fraction_per_worker":0.9/workers,"jobs":records}
    status_path = out/"parallel_status.json"
    pending, active = deque(jobs), {}
    next_heartbeat = time.monotonic()+interval()
    log("ПАРАЛЛЕЛЬНАЯ ОЧЕРЕДЬ", jobs=len(jobs),workers=workers,
        gpu_memory_fraction_per_worker=0.9/workers,cpus_per_job=cpus_per_job)
    write_status(status_path,state)
    try:
        while pending or active:
            for slot in range(workers):
                if slot in active or not pending:
                    continue
                job = pending.popleft()
                log_path = log_dir/f"{job.name}-{stamp}.log"
                worker_env = worker_environment(env,workers,slot,available,cpus_per_job)
                handle = log_path.open("w",encoding="utf-8")
                try:
                    process = subprocess.Popen(job.command,env=worker_env,stdout=handle,
                                               stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException:
                    handle.close()
                    records[job.name] = {"state":"failed","log":str(log_path),"reason":"could not start process"}
                    raise
                active[slot] = (job,process,handle)
                records[job.name] = {"state":"running","pid":process.pid,"slot":slot,
                    "started_at":timestamp(),"log":str(log_path),
                    "cpu_affinity":worker_env["MINIWAM_CPU_AFFINITY"]}
                latest = log_dir/f"{job.name}-latest.log"
                latest.unlink(missing_ok=True)
                latest.symlink_to(log_path.name)
                log("МОДЕЛЬ ЗАПУЩЕНА", model=job.name,pid=process.pid,slot=slot,log=log_path)
                write_status(status_path,state)
            for slot,(job,process,handle) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[slot]
                records[job.name].update(state="completed" if code == 0 else "failed",
                                         exit_code=code,finished_at=timestamp())
                write_status(status_path,state)
                log("МОДЕЛЬ ЗАВЕРШЕНА",level="INFO" if code == 0 else "ERROR",
                    model=job.name,exit_code=code,log=records[job.name]["log"])
                if code:
                    raise WorkerFailed(f"{job.name}: exit {code}; see {records[job.name]['log']}. "
                                       "For CUDA OOM reduce JOBS and resume the same output.")
                if on_complete:
                    on_complete(job)
            if time.monotonic() >= next_heartbeat:
                log("СОСТОЯНИЕ ОЧЕРЕДИ",running=",".join(item[0].name for item in active.values()),
                    completed=sum(r["state"] == "completed" for r in records.values()),queued=len(pending))
                next_heartbeat = time.monotonic()+interval()
            if active:
                time.sleep(poll_seconds)
        state.update(state="completed",finished_at=timestamp())
        write_status(status_path,state)
        return state
    except BaseException as error:
        stop_processes(item[1] for item in active.values())
        for job,process,handle in active.values():
            handle.close()
            records[job.name].update(state="interrupted",exit_code=process.returncode,finished_at=timestamp())
        for job in pending:
            records[job.name]["state"] = "not_started"
        state.update(state="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",finished_at=timestamp())
        write_status(status_path,state)
        raise


def run_parallel(config_path, out, workers=3, resume=False, require_perfect=False):
    out = Path(out).expanduser().resolve()
    config = json.loads(Path(config_path).read_text())
    jobs = build_jobs(out,config,resume)
    if not 1 <= workers <= len(jobs):
        raise ValueError(f"workers must be in [1,{len(jobs)}]")
    cpu_settings(workers,os.environ)  # validate resource settings before preparation
    command = [sys.executable,"-m","miniwam.server","prepare","--config",str(config_path),"--out",str(out)]
    if resume:
        command.append("--resume")
    log("ОБЩАЯ ПОДГОТОВКА",out=out,note="После подготовки процесс завершится и освободит GPU")
    code = run_command(command,os.environ.copy())
    if code:
        return code

    def report(_job=None):
        result = run_command([sys.executable,"-m","miniwam.server","report","--out",str(out)],os.environ.copy())
        if result:
            raise RuntimeError(f"Report failed with code {result}")

    try:
        run_jobs(jobs,out,workers,on_complete=report)
    except WorkerFailed:
        report()
        raise
    report()
    summary = json.loads((out/"summary.json").read_text())
    return 2 if require_perfect and not summary["all_sr_100"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--require-perfect",action="store_true")
    args = parser.parse_args(argv)
    def interrupt(signum,_frame):
        raise SignalReceived(signum)
    previous = {s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        return run_parallel(args.config,args.out,args.workers,args.resume,args.require_perfect)
    except SignalReceived as error:
        log("ПАРАЛЛЕЛЬНЫЙ ЗАПУСК ПРЕРВАН",level="ERROR",signal=error.signum)
        return 128+error.signum
    except (ValueError,RuntimeError,OSError) as error:
        log("ОШИБКА ПАРАЛЛЕЛЬНОГО ЗАПУСКА",level="ERROR",error=str(error))
        return 1
    finally:
        for signum,handler in previous.items():
            signal.signal(signum,handler)


if __name__ == "__main__":
    raise SystemExit(main())
