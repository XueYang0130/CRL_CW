#!/bin/zsh

set -eu

ROOT_DIR="/Users/xueyang/crl_cw"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
LOG_PATH="$ROOT_DIR/outputs/experiment_queue_autorun.log"

mkdir -p "$ROOT_DIR/outputs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] queue watcher started" >> "$LOG_PATH"

run_status() {
  local summary_path="$1"
  if [[ ! -f "$summary_path" ]]; then
    echo "missing"
    return
  fi
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
path = Path("$summary_path")
data = json.loads(path.read_text())
if "status" in data:
    print(data["status"])
else:
    print("completed")
PY
}

launch_when_ready() {
  local prerequisite_summary="$1"
  local config_path="$2"
  local run_dir="$3"
  local summary_path="$4"
  local label="$5"

  while true; do
    local prerequisite_status
    prerequisite_status=$(run_status "$prerequisite_summary")
    if [[ "$prerequisite_status" == "failed" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] prerequisite failed for $label" >> "$LOG_PATH"
      exit 1
    fi
    if [[ "$prerequisite_status" == "completed" ]]; then
      break
    fi
    sleep 60
  done

  while true; do
    local current_status
    current_status=$(run_status "$summary_path")
    if [[ "$current_status" == "completed" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label already completed" >> "$LOG_PATH"
      return
    fi
    if [[ "$current_status" == "failed" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label failed previously" >> "$LOG_PATH"
      exit 1
    fi
    if [[ -d "$run_dir" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label directory exists; waiting for completion" >> "$LOG_PATH"
      sleep 60
      continue
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching $label" >> "$LOG_PATH"
    (
      cd "$ROOT_DIR"
      "$PYTHON_BIN" scripts/run.py --config "$config_path"
    ) >> "$LOG_PATH" 2>&1
  done
}

BASELINE_SUMMARY="$ROOT_DIR/outputs/single_task_baselines/single_task_baseline_cw5_500k_seed0/summary.json"

FINE_TUNING_DIR="$ROOT_DIR/outputs/fine_tuning/fine_tuning_cw5_500k_seed0"
FINE_TUNING_SUMMARY="$FINE_TUNING_DIR/summary.json"
FINE_TUNING_CONFIG="$ROOT_DIR/configs/fine_tuning_cw5_500k.yaml"

TASK_CONDITIONED_DIR="$ROOT_DIR/outputs/task_conditioned/task_conditioned_cw5_500k_seed0"
TASK_CONDITIONED_SUMMARY="$TASK_CONDITIONED_DIR/summary.json"
TASK_CONDITIONED_CONFIG="$ROOT_DIR/configs/task_conditioned_cw5_500k.yaml"

PACKNET_DIR="$ROOT_DIR/outputs/packnet/packnet_cw5_500k_seed0"
PACKNET_SUMMARY="$PACKNET_DIR/summary.json"
PACKNET_CONFIG="$ROOT_DIR/configs/packnet_cw5_500k.yaml"

WSRL_CURRENT_DIR="$ROOT_DIR/outputs/wsrl_continual/wsrl_continual_cw5_500k_current_reset_seed0"
WSRL_CURRENT_SUMMARY="$WSRL_CURRENT_DIR/summary.json"
WSRL_CURRENT_CONFIG="$ROOT_DIR/configs/wsrl_continual_cw5_500k_current_reset.yaml"

WSRL_BEST_RETURN_DIR="$ROOT_DIR/outputs/wsrl_continual/wsrl_continual_cw5_500k_best_return_reset_seed0"
WSRL_BEST_RETURN_SUMMARY="$WSRL_BEST_RETURN_DIR/summary.json"
WSRL_BEST_RETURN_CONFIG="$ROOT_DIR/configs/wsrl_continual_cw5_500k_best_return_reset.yaml"

launch_when_ready "$BASELINE_SUMMARY" "$FINE_TUNING_CONFIG" "$FINE_TUNING_DIR" "$FINE_TUNING_SUMMARY" "fine_tuning"
launch_when_ready "$FINE_TUNING_SUMMARY" "$TASK_CONDITIONED_CONFIG" "$TASK_CONDITIONED_DIR" "$TASK_CONDITIONED_SUMMARY" "task_conditioned"
launch_when_ready "$TASK_CONDITIONED_SUMMARY" "$PACKNET_CONFIG" "$PACKNET_DIR" "$PACKNET_SUMMARY" "packnet"
launch_when_ready "$PACKNET_SUMMARY" "$WSRL_CURRENT_CONFIG" "$WSRL_CURRENT_DIR" "$WSRL_CURRENT_SUMMARY" "wsrl_continual current/reset"
launch_when_ready "$WSRL_CURRENT_SUMMARY" "$WSRL_BEST_RETURN_CONFIG" "$WSRL_BEST_RETURN_DIR" "$WSRL_BEST_RETURN_SUMMARY" "wsrl_continual best_return/reset"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] experiment queue completed" >> "$LOG_PATH"
