#!/usr/bin/env bash
# T36: T11-d layout (B0, B1, B2 looped) with a fixed loop count: T36-a = 2+1 (no loop), T36-b = 2+4.
# Recipe identical to T11-d (lr 5e-4, 30 epochs, seed 0, otherwise src.train defaults).
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T36_truncation_controls/run.sh T36-a|T36-b
# SMOKE=1: 1 epoch on --subset 512, results in $OUT_ROOT/smoke, no checkpoint. Env: PYTHON, OUT_ROOT, CKPT_DIR.
set -euo pipefail

TASK_DIR="tasks/T36_truncation_controls"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
RUN="${1:-}"

case "$RUN" in
    T36-a) LOOPS=1; CKPT="T36-a_2p1_noloop_30ep.pt" ;;
    T36-b) LOOPS=4; CKPT="T36-b_2p4_30ep.pt" ;;
    *)     echo "name the run: T36-a or T36-b" >&2; exit 2 ;;
esac
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root" >&2
    exit 2
fi

MODEL_ARGS=(--model unroll --unique-blocks 3 --body-loops "$LOOPS" --lr 5e-4 --seed 0)

if [[ "$SMOKE" == "1" ]]; then
    out="$OUT_ROOT/smoke"
    mkdir -p "$out"
    "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --epochs 1 --subset 512 \
        --out-dir "$out" --tag "_$RUN" 2>&1 | tee "$out/$RUN.log"
    exit 0
fi

mkdir -p "$OUT_ROOT" "$CKPT_DIR"
echo "[T36] $RUN ($LOOPS loops) on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
"$PYTHON" -m src.train "${MODEL_ARGS[@]}" --epochs 30 \
    --out-dir "$OUT_ROOT" --tag "_$RUN" --save-checkpoint "$CKPT_DIR/$CKPT" \
    2>&1 | tee "$OUT_ROOT/$RUN.log"
echo "[T36] $RUN done ($(date '+%F %T'))"
