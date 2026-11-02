"""Set CPU affinity BEFORE importing Torch/JAX, then enter the server CLI."""
from __future__ import annotations

import os


def configure_cpu():
    value = os.environ.get("MINIWAM_CPU_AFFINITY")
    if value and hasattr(os,"sched_setaffinity"):
        cpus = {int(cpu) for cpu in value.split(",")}
        if not cpus or min(cpus) < 0:
            raise ValueError("Invalid MINIWAM_CPU_AFFINITY")
        os.sched_setaffinity(0,cpus)


def main():
    configure_cpu()
    from .cli import main as server_main
    return server_main()


if __name__ == "__main__":
    raise SystemExit(main())
