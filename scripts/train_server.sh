#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

miniwam_python="${PYTHON:-.venv-server/bin/python}"
miniwam_out_root="${OUT_ROOT:-runs/server}"
miniwam_target="${1:-one-step}"
miniwam_log_dir="${LOG_DIR:-logs}"
case "$miniwam_target" in
  smoke|one-step|two-step|three-step|all) ;;
  *) echo 'Usage: bash scripts/train_server.sh [smoke|one-step|two-step|three-step|all]' >&2; exit 1 ;;
esac
mkdir -p "$miniwam_log_dir"
miniwam_log_dir="$(cd "$miniwam_log_dir" && pwd -P)"
miniwam_log_file="$miniwam_log_dir/$miniwam_target-$(date -u +%Y%m%d-%H%M%S)-$$.log"
miniwam_latest_log="$miniwam_log_dir/$miniwam_target-latest.log"
export TMPDIR="${MINIWAM_TMPDIR:-$HOME/tmp}"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

finish_run() {
  miniwam_exit_code="$1"
  case "$miniwam_exit_code" in
    0) log "ГОТОВО: выполнение завершено; код=0" ;;
    2) log "SR НИЖЕ 100%: серия остановлена; код=2. Смотрите summary.md и diagnosis.json в $miniwam_out_root" ;;
    *) log "ОШИБКА: запуск завершился с кодом $miniwam_exit_code. Traceback и сообщения выше в этом логе." ;;
  esac
  log "Длительность=$((SECONDS-miniwam_started)) сек.; лог=$miniwam_log_file"
}

run_stage() {
  log "НАЧАЛО ЭКСПЕРИМЕНТА: $1; результаты=$miniwam_out_root/$1"
  "$miniwam_python" -m miniwam.server pipeline --config "configs/server/$1.json" \
    --out "$miniwam_out_root/$1" --resume --require-perfect
  log "ГОТОВО: $1; отчет=$miniwam_out_root/$1/summary.md"
}

main() {
  miniwam_started=$SECONDS
  trap 'finish_run "$?"' EXIT
  log "ЗАПУСК: $miniwam_target; PID=${BASHPID:-$$}; python=$miniwam_python; время UTC"
  log "Полный лог stdout/stderr: $miniwam_log_file"
  mkdir -p "$TMPDIR" "$miniwam_out_root"
  command -v flock >/dev/null || { log 'ОШИБКА: flock is required to avoid competing GPU runs.'; exit 1; }
  exec 9>"$miniwam_out_root/.run.lock"
  flock -n 9 || { log 'ОШИБКА: другой процесс уже использует этот output root.'; exit 1; }
  # A rejected concurrent launch must not replace the running job's log link.
  ln -sfn "$(basename "$miniwam_log_file")" "$miniwam_latest_log"
  log "Следить: tail -F \"$miniwam_latest_log\""
  log 'НАЧАЛО: проверка сервера / CUDA / NanoWM (doctor)'
  "$miniwam_python" -m miniwam.server doctor
  log 'ГОТОВО: doctor'
  case "$miniwam_target" in
    smoke)
      log 'НАЧАЛО: smoke — короткая проверка; SR 100% для WAM не требуется'
      "$miniwam_python" -m miniwam.server pipeline --config configs/server/smoke.json \
        --out "$miniwam_out_root/smoke" --resume ;;
    one-step) run_stage one_step ;;
    two-step) run_stage two_step ;;
    three-step) run_stage three_step ;;
    all)
      run_stage one_step
      run_stage two_step
      run_stage three_step ;;
  esac
}

# tee mirrors both streams immediately. Preserve the experiment's exit status,
# including SR gate code 2, rather than accidentally returning tee's zero.
set +e
(set -euo pipefail; main) 2>&1 | tee -a "$miniwam_log_file"
miniwam_pipe_status=("${PIPESTATUS[@]}")
set -e
if [[ "${miniwam_pipe_status[0]}" -ne 0 ]]; then
  exit "${miniwam_pipe_status[0]}"
fi
exit "${miniwam_pipe_status[1]}"
