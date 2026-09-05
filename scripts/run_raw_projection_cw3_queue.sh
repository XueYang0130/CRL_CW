#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="outputs/cw10_continual"
BASELINE_CURVES="outputs/single_task_baselines/cw10_v3_v1_500k_seeds1_2/aggregate/baseline_curves.json"

if [[ ! -f "${BASELINE_CURVES}" ]]; then
    echo "Missing baseline curves: ${BASELINE_CURVES}" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c \
    "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"; then
    echo "CUDA is not available to ${PYTHON_BIN}." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH=src

for seed in 1 2 3 4 5; do
    for index in 0 1 2 3 4 5 6 7; do
        sequence="cw3_${index}"
        run_name="success_replay_best_raw_pcgrad_projection_${sequence}_v3_v1_500k_seed${seed}"
        run_dir="${OUTPUT_DIR}/${run_name}"

        if [[ -f "${run_dir}/summary.json" ]]; then
            echo "Skipping completed: ${run_name}"
            continue
        fi

        if [[ -e "${run_dir}" ]]; then
            echo "Incomplete run directory exists: ${run_dir}" >&2
            exit 1
        fi
        echo "===== Starting ${run_name}: $(date) ====="
        if ! "${PYTHON_BIN}" -u scripts/run.py \
            --mode continual \
            --method success_replay_best_pcgrad \
            --task-sequence "${sequence}" \
            --env-version v3 \
            --reward-function-version cw10_v1 \
            --steps-per-task 500000 \
            --eval-every 20000 \
            --stoch-eval-episodes 5 \
            --det-eval-episodes 0 \
            --episodic-memory-per-task 10000 \
            --bc-max-norm-ratio 1000000 \
            --baseline-curves "${BASELINE_CURVES}" \
            --seed "${seed}" \
            --device cuda \
            --output-dir "${OUTPUT_DIR}" \
            --run-name "${run_name}"; then
            echo "Run failed: ${run_name}" >&2
            exit 1
        fi
        echo "===== Finished ${run_name}: $(date) ====="
    done
done
