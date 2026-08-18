#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/xueyang/crl_cw"
cd "$ROOT"

SEED4_SPARSE="outputs/stickpull_causal_ablation/stickpull_success_sparse_bc_interval4_adaptive_pcgrad_500k_seed4/summary.json"
SEED4_WEAK="outputs/stickpull_causal_ablation/stickpull_success_continuous_weak_quarter_adaptive_pcgrad_500k_seed4/summary.json"

while [[ ! -f "$SEED4_SPARSE" || ! -f "$SEED4_WEAK" ]]; do
  sleep 60
done

mkdir -p logs/bc_cadence

run_condition() {
  local seed="$1"
  local manifest="$2"
  local interval="$3"
  local target_ratio="$4"
  local conflict_ratio="$5"
  local condition="$6"
  local run_name="stickpull_success_${condition}_adaptive_pcgrad_500k_seed${seed}"

  env PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
    scripts/run_stickpull_causal_ablation.py \
    --run-dir "outputs/cw10_continual/success_replay_best_adaptive_pcgrad_cw10_v3_v1_500k_seed${seed}" \
    --memory-manifest "$manifest" \
    --new-task-index 4 \
    --memory-mode success \
    --integration adaptive_pcgrad \
    --bc-update-interval "$interval" \
    --bc-adaptive-target-ratio "$target_ratio" \
    --bc-adaptive-conflict-ratio "$conflict_ratio" \
    --steps 500000 \
    --eval-every 20000 \
    --stoch-eval-episodes 5 \
    --retention-eval-episodes 25 \
    --batch-size 128 \
    --seed "$seed" \
    --device cpu \
    --run-name "$run_name" \
    > "logs/bc_cadence/${run_name}.log" 2>&1
}

run_condition 0 \
  outputs/memory_libraries/stickpull_matched_srbestpcgrad_seed0_v2/manifest.json \
  4 0.2 0.05 sparse_bc_interval4 &
pid_seed0_sparse=$!

run_condition 0 \
  outputs/memory_libraries/stickpull_matched_srbestpcgrad_seed0_v2/manifest.json \
  1 0.05 0.0125 continuous_weak_quarter &
pid_seed0_weak=$!

run_condition 1 \
  outputs/memory_libraries/stickpull_matched_srbestpcgrad_seed1/manifest.json \
  4 0.2 0.05 sparse_bc_interval4 &
pid_seed1_sparse=$!

run_condition 1 \
  outputs/memory_libraries/stickpull_matched_srbestpcgrad_seed1/manifest.json \
  1 0.05 0.0125 continuous_weak_quarter &
pid_seed1_weak=$!

status=0
for pid in \
  "$pid_seed0_sparse" \
  "$pid_seed0_weak" \
  "$pid_seed1_sparse" \
  "$pid_seed1_weak"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
