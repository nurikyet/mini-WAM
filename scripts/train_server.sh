#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Override PYTHON/DEVICE/OUT_ROOT for a different environment, CPU or storage disk.
miniwam_python="${PYTHON:-.venv-wam/bin/python}"
miniwam_device="${DEVICE:-cuda}"
miniwam_out_root="${OUT_ROOT:-runs}"
"$miniwam_python" -m miniwam --device "$miniwam_device" doctor
"$miniwam_python" -m miniwam --device "$miniwam_device" pipeline \
  --config configs/one_step.json --out "$miniwam_out_root/one_step" --resume --require-perfect
# The next command runs only after every one-step test SR reaches 100%.
"$miniwam_python" -m miniwam --device "$miniwam_device" pipeline \
  --config configs/two_step.json --out "$miniwam_out_root/two_step" --resume --require-perfect
