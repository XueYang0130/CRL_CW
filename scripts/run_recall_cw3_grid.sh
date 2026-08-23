#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

sequences=(cw3_0 cw3_1 cw3_2 cw3_3 cw3_4 cw3_5 cw3_6 cw3_7)
seeds=(1 2 3 4 5)
common=(
  --mode continual
  --method recall
  --env-version v3
  --reward-function-version cw10_v1
  --steps-per-task 500000
  --eval-every 20000
  --stoch-eval-episodes 5
  --det-eval-episodes 0
  --best-return-eval-episodes 10
  --batch-size 128
  --episodic-memory-per-task 10000
  --episodic-batch-size 128
  --actor-cloning-coefficient 10.0
  --gradient-clip-norm 0.1
  --reset-buffer-on-task-change
  --reset-optimizer-on-task-change
  --gradient-diagnostics
  --gradient-diagnostics-interval 500
  --gradient-diagnostics-source-batch-size 128
  --baseline-curves outputs/single_task_baselines/cw10_v3_v1_500k_seeds1_2/aggregate/baseline_curves.json
  --device cpu
  --output-dir outputs/cw10_continual
)

for sequence in "${sequences[@]}"; do
  for seed in "${seeds[@]}"; do
    run_name="recall_${sequence}_v3_v1_500k_seed${seed}"
    run_dir="outputs/cw10_continual/${run_name}"
    if [[ -f "${run_dir}/summary.json" ]]; then
      echo "[skip] completed ${run_name}"
      continue
    fi
    if [[ -d "${run_dir}" ]]; then
      echo "[error] incomplete directory exists: ${run_dir}" >&2
      exit 1
    fi
    echo "[start] ${run_name}"
    caffeinate -dimsu env PYTHONUNBUFFERED=1 PYTHONPATH=src \
      .venv/bin/python -u scripts/run.py \
      "${common[@]}" \
      --task-sequence "${sequence}" \
      --seed "${seed}" \
      --run-name "${run_name}"
  done
done
