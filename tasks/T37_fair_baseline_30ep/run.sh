#!/usr/bin/env bash
# T37: plain 12-block pretrained ViT-Tiny on the T11-d recipe (30 ep, lr 5e-4, rest = src.train defaults:
# AdamW wd 0.05, 1 warmup epoch + cosine, label smoothing 0.1, batch 128, AMP, seed 0).
# Usage (repo root, tmux, one GPU):  CUDA_VISIBLE_DEVICES=N bash tasks/T37_fair_baseline_30ep/run.sh
# SMOKE=1: 1 epoch on --subset 512, results in results/smoke, no checkpoint.
set -euo pipefail

TASK_DIR="tasks/T37_fair_baseline_30ep"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
RUN="T37-a"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root" >&2
    exit 2
fi

if [[ "$SMOKE" == "1" ]]; then
    out="$OUT_ROOT/smoke"
    mkdir -p "$out"
    "$PYTHON" -m src.train --model baseline --lr 5e-4 --epochs 1 --seed 0 --subset 512 \
        --out-dir "$out" --tag "_$RUN" 2>&1 | tee "$out/$RUN.log"
    exit 0
fi

mkdir -p "$OUT_ROOT" "$CKPT_DIR"
echo "[T37] $RUN on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
"$PYTHON" -m src.train --model baseline --lr 5e-4 --epochs 30 --seed 0 \
    --out-dir "$OUT_ROOT" --tag "_$RUN" --save-checkpoint "$CKPT_DIR/T37-a_baseline_12blk_30ep.pt" \
    2>&1 | tee "$OUT_ROOT/$RUN.log"
echo "[T37] done ($(date '+%F %T'))"
