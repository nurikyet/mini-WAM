#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this script on gangway (Linux/H100), after git clone.' >&2
  exit 1
fi
case "$(pwd -P)/" in
  "$(cd "$HOME" && pwd -P)/"*) ;;
  *) echo 'Clone this repository inside your persistent server HOME.' >&2; exit 1 ;;
esac
mkdir -p "$HOME/tmp" logs
export TMPDIR="$HOME/tmp"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export HF_HUB_DISABLE_XET=1
miniwam_python="${SERVER_PYTHON:-python3.11}"
"$miniwam_python" -c 'import sys; assert sys.version_info[:2] == (3,11), sys.version'
uv venv --python "$miniwam_python" --allow-existing .venv-server
# Preserve the server's configured package mirror and proxy settings.
uv pip install --python .venv-server/bin/python -r requirements-server.txt
uv pip install --python .venv-server/bin/python --no-deps -e .
uv pip freeze --python .venv-server/bin/python > logs/server-packages.txt
.venv-server/bin/python -m miniwam.server doctor --assets
echo 'Server setup complete. Next: bash scripts/train_server.sh smoke'
