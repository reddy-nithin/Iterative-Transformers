#!/usr/bin/env bash
# T23: per-loop gradient norms of the shared block, 4 models (no training).
# Usage (repo root, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T23_per_loop_grad_norms/run.sh
# Env: PYTHON. SMOKE=1: 2 batches and 512 test images, results in results/smoke.
set -euo pipefail

TASK_DIR="tasks/T23_per_loop_grad_norms"
PYTHON="${PYTHON:-.venv/bin/python}"
SMOKE="${SMOKE:-0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root (no $TASK_DIR here)" >&2
    exit 2
fi

out="$TASK_DIR/results"
extra=()
if [[ "$SMOKE" == "1" ]]; then
    out="$TASK_DIR/results/smoke"
    extra=(--batches 2 --test-limit 512)
fi
mkdir -p "$out"

PYTHONPATH=. "$PYTHON" "$TASK_DIR/grad_norms.py" --out-dir "$out" "${extra[@]}" 2>&1 | tee "$out/grad_norms.log"
PYTHONPATH=. "$PYTHON" "$TASK_DIR/plot.py" --results "$out" 2>&1 | tee "$out/plot.log"
