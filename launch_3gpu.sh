#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${LPAN_DATA_ROOT:-data}"
OUTPUT_ROOT="${LPAN_BASELINE_OUTPUT_ROOT:-runs/v3_formal_baselines}"
PLAN_FILE="${LPAN_BASELINE_PLAN:-runs/baseline_matrix_plan.json}"

mkdir -p "${OUTPUT_ROOT}"

CUDA_VISIBLE_DEVICES=0 python main.py baseline-matrix --action run-worker \
  --seed 123 --device cuda --plan-file "${PLAN_FILE}" \
  --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
  > "${OUTPUT_ROOT}/worker_seed123.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main.py baseline-matrix --action run-worker \
  --seed 456 --device cuda --plan-file "${PLAN_FILE}" \
  --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
  > "${OUTPUT_ROOT}/worker_seed456.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 python main.py baseline-matrix --action run-worker \
  --seed 789 --device cuda --plan-file "${PLAN_FILE}" \
  --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
  > "${OUTPUT_ROOT}/worker_seed789.log" 2>&1 &

wait
