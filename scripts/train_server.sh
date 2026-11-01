#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

miniwam_python="${PYTHON:-.venv-server/bin/python}"
miniwam_out_root="${OUT_ROOT:-runs/server}"
miniwam_target="${1:-one-step}"
export TMPDIR="$HOME/tmp"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$miniwam_out_root"
command -v flock >/dev/null || { echo 'flock is required to avoid competing GPU runs.' >&2; exit 1; }
exec 9>"$miniwam_out_root/.run.lock"
flock -n 9 || { echo 'Another server experiment holds this output root.' >&2; exit 1; }
"$miniwam_python" -m miniwam.server doctor
run_stage() {
  "$miniwam_python" -m miniwam.server pipeline --config "configs/server/$1.json" \
    --out "$miniwam_out_root/$1" --resume --require-perfect
}
case "$miniwam_target" in
  smoke)
    "$miniwam_python" -m miniwam.server pipeline --config configs/server/smoke.json \
      --out "$miniwam_out_root/smoke" --resume ;;
  one-step) run_stage one_step ;;
  two-step) run_stage two_step ;;
  three-step) run_stage three_step ;;
  all)
    run_stage one_step
    run_stage two_step
    run_stage three_step ;;
  *) echo 'Usage: bash scripts/train_server.sh [smoke|one-step|two-step|three-step|all]' >&2; exit 1 ;;
esac
