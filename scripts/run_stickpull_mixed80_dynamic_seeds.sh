#!/usr/bin/env bash
set -euo pipefail

cd /Users/xueyang/crl_cw

build_manifest_if_needed() {
  local seed="$1"
  local memory_name="stickpull_matched_srbestpcgrad_seed${seed}_dynamic"
  local manifest="outputs/memory_libraries/${memory_name}/manifest.json"
  if [[ -f "${manifest}" ]]; then
    return
  fi
  PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
    scripts/build_stickpull_matched_memory.py \
    --run-dir "outputs/cw10_continual/success_replay_best_adaptive_pcgrad_cw10_v3_v1_500k_seed${seed}" \
    --max-source-task-index 3 \
    --pool-episodes-per-task 100 \
    --random-episode-ratio 0.2 \
    --memory-states-per-task 10000 \
    --seed "${seed}" \
    --device cpu \
    --run-name "${memory_name}"
}

run_probe() {
  local seed="$1"
  local manifest="$2"
  local run_name="stickpull_mixed80_dynamic_adaptive_pcgrad_500k_seed${seed}"
  if [[ -f "outputs/stickpull_causal_ablation/${run_name}/summary.json" ]]; then
    return
  fi
  PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
    scripts/run_stickpull_causal_ablation.py \
    --run-dir "outputs/cw10_continual/success_replay_best_adaptive_pcgrad_cw10_v3_v1_500k_seed${seed}" \
    --memory-manifest "${manifest}" \
    --new-task-index 4 \
    --memory-mode mixed80_dynamic \
    --integration adaptive_pcgrad \
    --steps 500000 \
    --eval-every 20000 \
    --stoch-eval-episodes 5 \
    --retention-eval-episodes 25 \
    --batch-size 128 \
    --seed "${seed}" \
    --device cpu \
    --run-name "${run_name}"
}

run_probe 0 outputs/memory_libraries/stickpull_matched_srbestpcgrad_seed0_v2/manifest.json
run_probe 1 outputs/memory_libraries/stickpull_matched_srbestpcgrad_seed1/manifest.json

build_manifest_if_needed 3
run_probe 3 outputs/memory_libraries/stickpull_matched_srbestpcgrad_seed3_dynamic/manifest.json

run_probe 4 outputs/memory_libraries/stickpull_matched_srbestpcgrad_seed4/manifest.json

build_manifest_if_needed 5
run_probe 5 outputs/memory_libraries/stickpull_matched_srbestpcgrad_seed5_dynamic/manifest.json
