#!/usr/bin/env bash
# T26: CLS starting vector (learned / zeros / noise / learned + noise / swapped) on T11-d and T13-c (no training).
# Usage (repo root, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T26_cls_start_vector/run.sh
# Env: PYTHON. SMOKE=1: 256 test images, results in results/smoke.
set -euo pipefail

TASK_DIR="tasks/T26_cls_start_vector"
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
    extra=(--limit 256)
fi
mkdir -p "$out"

PYTHONPATH=. "$PYTHON" "$TASK_DIR/cls_start.py" --out-dir "$out" "${extra[@]}" 2>&1 | tee "$out/cls_start.log"
PYTHONPATH=. "$PYTHON" "$TASK_DIR/plot.py" --results "$out" 2>&1 | tee "$out/plot.log"
