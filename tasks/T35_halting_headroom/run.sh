#!/usr/bin/env bash
# T35: halting headroom on T31-a (no training): store per-loop outputs, oracle, settling, AUROC, then the plots.
# Usage (repo root, tmux, one GPU):  CUDA_VISIBLE_DEVICES=N bash tasks/T35_halting_headroom/run.sh
# Env: PYTHON, OUT_ROOT, CKPT_DIR. SMOKE=1: first 500 test images, results in results/smoke, outputs file under
# $CKPT_DIR/smoke (no sanity check).
set -euo pipefail

TASK_DIR="tasks/T35_halting_headroom"
OUTPUTS_NAME="T35_T31-a_per_loop_outputs.pt"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
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
outputs="$CKPT_DIR/$OUTPUTS_NAME"
if [[ "$SMOKE" == "1" ]]; then
    limit=(--limit 500)
    out="$OUT_ROOT/smoke"
    outputs="$CKPT_DIR/smoke/$OUTPUTS_NAME"
fi
mkdir -p "$out"

echo "[run] T35 halting headroom on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
PYTHONPATH=. "$PYTHON" "$TASK_DIR/halting.py" "${limit[@]}" --outputs "$outputs" \
    --out "$out/halting_headroom.json" 2>&1 | tee "$out/halting.log"
"$PYTHON" "$TASK_DIR/plot.py" --results "$out" 2>&1 | tee "$out/plot.log"
echo "[run] T35 done ($(date '+%F %T'))"
