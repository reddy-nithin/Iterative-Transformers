#!/usr/bin/env bash
# T22: block similarity matrices + contiguous max-cut for the pretrained model and the baseline (no training).
# Usage (repo root, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T22_block_phase_maxcut/run.sh
# Env: PYTHON. SMOKE=1: 200 test images, results in results/smoke.
set -euo pipefail

TASK_DIR="tasks/T22_block_phase_maxcut"
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
    extra=(--limit 200)
fi
mkdir -p "$out"

PYTHONPATH=. "$PYTHON" "$TASK_DIR/block_phases.py" --out-dir "$out" "${extra[@]}" 2>&1 | tee "$out/block_phases.log"
PYTHONPATH=. "$PYTHON" "$TASK_DIR/plot.py" --results "$out" 2>&1 | tee "$out/plot.log"
