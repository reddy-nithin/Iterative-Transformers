#!/usr/bin/env bash
# T30: per-loop center loss on T13-c (T11-d layout + readout every loop). Train, then per-loop readout,
# class separation, and T25's class-cloud analysis (T25's train-set centres) on T30-a.
# Usage (repo root, tmux, one GPU):  CUDA_VISIBLE_DEVICES=N bash tasks/T30_per_loop_center_loss/run.sh
# Env: PYTHON, CKPT_DIR, OUT_ROOT. SMOKE=1: 1 epoch on 300 images, results in results/smoke.
set -euo pipefail

TASK_DIR="tasks/T30_per_loop_center_loss"
RUN="T30-a"
CHECKPOINT_NAME="T30-a_center_loss.pt"
MODEL_ARGS=(--model unroll --unique-blocks 3 --readout-loss every_loop --center-loss-weight 0.1)
CLS_SVD="tasks/T5_cls_svd_per_block/cls_svd.py"
EXIT_ANALYSIS="tasks/T13_per_loop_readout_loss/exit_analysis.py"
CLASS_CLOUD="tasks/T25_class_cloud_exit/class_cloud.py"
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
    PYTHONPATH=. "$PYTHON" "$CLASS_CLOUD" --checkpoint "$checkpoint" --tag "$RUN" --limit 500 --train-limit 1000 \
        --out-dir "$out" 2>&1 | tee "$out/class_cloud.log"
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
echo "[run] $RUN class-cloud exit (T25 code, T25 train-set centres) ($(date '+%F %T'))"
PYTHONPATH=. "$PYTHON" "$CLASS_CLOUD" --checkpoint "$checkpoint" --tag "$RUN" --out-dir "$OUT_ROOT" \
    2>&1 | tee "$OUT_ROOT/class_cloud_${RUN}.log"
echo "[run] done ($(date '+%F %T'))"
