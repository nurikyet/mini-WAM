"""Console progress captured by train_server.sh; no ML dependencies or RNG use."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import math
import os
import threading
import time

_output_lock = threading.Lock()


def log(message, *, level="INFO", **fields):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    values = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            value = f"{value:.6g}"
        values.append(f"{key}={value}")
    suffix = " | " + " ".join(values) if values else ""
    with _output_lock:
        print(f"[{timestamp}] [{level}] {message}{suffix}", flush=True)


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def interval():
    value = float(os.environ.get("MINIWAM_LOG_INTERVAL", "30"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("MINIWAM_LOG_INTERVAL must be a positive number of seconds")
    return value


@contextmanager
def logged_stage(name, **fields):
    started = time.monotonic()
    stopped = threading.Event()
    period = interval()
    log("НАЧАЛО", stage=name, **fields)

    def heartbeat():
        while not stopped.wait(period):
            log("ОЖИДАНИЕ ЗАВЕРШЕНИЯ СТАДИИ", stage=name, elapsed=duration(time.monotonic()-started))

    worker = threading.Thread(target=heartbeat, daemon=True, name="miniwam-log-heartbeat")
    worker.start()
    try:
        yield
    except BaseException as error:
        stopped.set()
        worker.join(timeout=1)
        log("ОШИБКА СТАДИИ", level="ERROR", stage=name,
            elapsed=duration(time.monotonic()-started), error=f"{type(error).__name__}: {error}")
        raise
    else:
        stopped.set()
        worker.join(timeout=1)
        log("ГОТОВО", stage=name, elapsed=duration(time.monotonic()-started))


class Progress:
    def __init__(self, name, total, initial=0):
        self.name, self.total, self.initial = name, total, initial
        self.started = time.monotonic()
        self.last_log = self.started
        self.period = interval()
        log("ПРОГРЕСС", stage=name, done=initial, total=total,
            percent=f"{100*initial/max(total,1):.1f}%")

    def update(self, current, *, force=False, **fields):
        now = time.monotonic()
        if not (force or current == self.initial+1 or current == self.total or now-self.last_log >= self.period):
            return
        self.last_log = now
        elapsed = now-self.started
        completed = current-self.initial
        remaining = elapsed*(self.total-current)/completed if completed > 0 else None
        log("ПРОГРЕСС", stage=self.name, done=current, total=self.total,
            percent=f"{100*current/max(self.total,1):.1f}%", elapsed=duration(elapsed),
            eta=duration(remaining) if remaining is not None else "unknown", **fields)
