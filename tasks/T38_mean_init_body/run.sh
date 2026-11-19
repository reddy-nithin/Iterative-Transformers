#!/usr/bin/env bash
# T38: T11-d layout (2+10) with the looped block started as the mean of pretrained blocks 2..11 (0-based).
# Recipe identical to T11-d (lr 5e-4, 30 epochs, seed 0, otherwise src.train defaults).
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T38_mean_init_body/run.sh T38-a
# SMOKE=1: 1 epoch on --subset 512, results in $OUT_ROOT/smoke, no checkpoint. Env: PYTHON, OUT_ROOT, CKPT_DIR.
set -euo pipefail

TASK_DIR="tasks/T38_mean_init_body"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
RUN="${1:-}"

case "$RUN" in
    T38-a) CKPT="T38-a_2p10_meaninit_30ep.pt" ;;
    *)     echo "name the run: T38-a" >&2; exit 2 ;;
esac
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root" >&2
    exit 2
fi

MODEL_ARGS=(--model unroll --unique-blocks 3 --body-init-range 2 11 --lr 5e-4 --seed 0)

if [[ "$SMOKE" == "1" ]]; then
    out="$OUT_ROOT/smoke"
    mkdir -p "$out"
    "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --epochs 1 --subset 512 \
        --out-dir "$out" --tag "_$RUN" 2>&1 | tee "$out/$RUN.log"
    exit 0
fi

mkdir -p "$OUT_ROOT" "$CKPT_DIR"
echo "[T38] $RUN (looped block = mean of blocks 2..11) on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
"$PYTHON" -m src.train "${MODEL_ARGS[@]}" --epochs 30 \
    --out-dir "$OUT_ROOT" --tag "_$RUN" --save-checkpoint "$CKPT_DIR/$CKPT" \
    2>&1 | tee "$OUT_ROOT/$RUN.log"
echo "[T38] $RUN done ($(date '+%F %T'))"
