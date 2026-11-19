#!/usr/bin/env bash
# T34: frozen-update test and angular speed on T11-d, T13-c, T31-a (no training), then the plots.
# Usage (repo root, tmux, one GPU):  CUDA_VISIBLE_DEVICES=N bash tasks/T34_frozen_update_test/run.sh
# Env: PYTHON, OUT_ROOT. SMOKE=1: first 500 test images, results in results/smoke (no sanity checks).
set -euo pipefail

TASK_DIR="tasks/T34_frozen_update_test"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
SMOKE="${SMOKE:-0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root (no $TASK_DIR here)" >&2
    exit 2
fi

limit=()
out="$OUT_ROOT"
if [[ "$SMOKE" == "1" ]]; then
    limit=(--limit 500)
    out="$OUT_ROOT/smoke"
fi
mkdir -p "$out"

echo "[run] T34 frozen update on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
PYTHONPATH=. "$PYTHON" "$TASK_DIR/frozen_update.py" "${limit[@]}" --out "$out/frozen_update.json" \
    2>&1 | tee "$out/frozen_update.log"
"$PYTHON" "$TASK_DIR/plot.py" --results "$out" 2>&1 | tee "$out/plot.log"
echo "[run] T34 done ($(date '+%F %T'))"
