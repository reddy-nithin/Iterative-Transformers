#!/usr/bin/env bash
# T28: frozen-patch CLS-only loop on the T11-d layout. Train, then per-loop readout and class separation.
# Usage (repo root, tmux, one GPU):  CUDA_VISIBLE_DEVICES=N bash tasks/T28_frozen_patch_cls_loop/run.sh
# Env: PYTHON, CKPT_DIR, OUT_ROOT. SMOKE=1: 1 epoch on 300 images, results in results/smoke.
set -euo pipefail

TASK_DIR="tasks/T28_frozen_patch_cls_loop"
RUN="T28-a"
CHECKPOINT_NAME="T28-a_cls_only_loop.pt"
MODEL_ARGS=(--model unroll --unique-blocks 3 --cls-only-loop)
CLS_SVD="tasks/T5_cls_svd_per_block/cls_svd.py"
EXIT_ANALYSIS="tasks/T13_per_loop_readout_loss/exit_analysis.py"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
LR=5e-4
EPOCHS=30

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root (no $TASK_DIR here)" >&2
    exit 2
fi

is_complete() {  # true if the run's JSON exists and has as many epochs as it asked for
    "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(len(d['history']) != d['args']['epochs'])" "$1"
}

if [[ "$SMOKE" == "1" ]]; then
    out="$OUT_ROOT/smoke"
    checkpoint="$CKPT_DIR/smoke/$CHECKPOINT_NAME"
    mkdir -p "$out" "$CKPT_DIR/smoke"
    "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs 1 --subset 300 \
        --out-dir "$out" --tag "_$RUN" --save-checkpoint "$checkpoint" 2>&1 | tee "$out/$RUN.log"
    PYTHONPATH=. "$PYTHON" "$EXIT_ANALYSIS" "$checkpoint" --limit 500 --out "$out/exit_analysis.json" \
        2>&1 | tee "$out/exit_analysis.log"
    exit 0
fi

mkdir -p "$OUT_ROOT" "$CKPT_DIR"
checkpoint="$CKPT_DIR/$CHECKPOINT_NAME"
json="$OUT_ROOT/unroll_$RUN.json"
if [[ -f "$json" ]] && is_complete "$json"; then
    echo "[run] $RUN already trained, skipping training"
else
    echo "[run] $RUN on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
    "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs "$EPOCHS" \
        --out-dir "$OUT_ROOT" --tag "_$RUN" --save-checkpoint "$checkpoint" 2>&1 | tee "$OUT_ROOT/$RUN.log"
fi
echo "[run] $RUN per-loop readout ($(date '+%F %T'))"
PYTHONPATH=. "$PYTHON" "$EXIT_ANALYSIS" "$checkpoint" --out "$OUT_ROOT/exit_analysis.json" \
    2>&1 | tee "$OUT_ROOT/exit_analysis.log"
echo "[run] $RUN class separation, train + test ($(date '+%F %T'))"
PYTHONPATH=. "$PYTHON" "$CLS_SVD" --checkpoint "$checkpoint" --model-tag "${RUN}_train" \
    --out-dir "$OUT_ROOT" 2>&1 | tee "$OUT_ROOT/cls_svd_${RUN}_train.log"
PYTHONPATH=. "$PYTHON" "$CLS_SVD" --checkpoint "$checkpoint" --split test --per-class 100 --control-size 100 \
    --model-tag "${RUN}_test" --out-dir "$OUT_ROOT" 2>&1 | tee "$OUT_ROOT/cls_svd_${RUN}_test.log"
echo "[run] done ($(date '+%F %T'))"
