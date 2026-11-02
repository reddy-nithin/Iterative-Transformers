#!/usr/bin/env bash
# T21-a: T11-c (head b0, body b1 x10, tail b11) with the tail replaced by pretrained block 2 (--tail-idx 2), nothing
# else changed. Pretrained init, shared recipe (lr 5e-4, 30 epochs, otherwise src.train defaults). After training, in
# the same job: class separation per block call (T5 cls_svd.py, T11 settings, train + test) and T12's per-loop
# diagnostics incl. (d), the readout through the tail at every loop.
# A finished training run is skipped (the analysis is redone), so rerun after a crash.
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T21_contiguous_tail_b2/run.sh
# Env: PYTHON, CKPT_DIR, OUT_ROOT. SMOKE=1: 300 images, 1 epoch, results in $OUT_ROOT/smoke, checkpoint in
# $CKPT_DIR/smoke (never in the repo), T12 diagnostics on 200 images.
set -euo pipefail

TASK_DIR="tasks/T21_contiguous_tail_b2"
CLS_SVD="tasks/T5_cls_svd_per_block/cls_svd.py"
LOOP_DIAG="tasks/T12_per_loop_cls_diagnostics/loop_diagnostics.py"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
RUN="T21-a"
MODEL_ARGS=(--model hbt --hbt-pretrained --head-idx 0 --body-idx 1 --tail-idx 2)
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
    checkpoint="$CKPT_DIR/smoke/${RUN}_hbt_tail_b2.pt"
    mkdir -p "$out" "$CKPT_DIR/smoke"
    "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs 1 --subset 300 \
        --out-dir "$out" --tag "_$RUN" --save-checkpoint "$checkpoint" 2>&1 | tee "$out/$RUN.log"
    PYTHONPATH=. "$PYTHON" "$LOOP_DIAG" "$checkpoint" --collapse-images 200 --limit 200 --out-dir "$out/t12_diag" \
        2>&1 | tee "$out/t12_diag.log"
    exit 0
fi

mkdir -p "$OUT_ROOT" "$CKPT_DIR"
checkpoint="$CKPT_DIR/${RUN}_hbt_tail_b2.pt"
json="$OUT_ROOT/hbt_$RUN.json"
if [[ -f "$json" ]] && is_complete "$json"; then
    echo "[run] $RUN already trained, skipping training"
else
    echo "[run] $RUN on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
    "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs "$EPOCHS" \
        --out-dir "$OUT_ROOT" --tag "_$RUN" --save-checkpoint "$checkpoint" 2>&1 | tee "$OUT_ROOT/$RUN.log"
fi
echo "[run] $RUN class separation, train + test ($(date '+%F %T'))"
PYTHONPATH=. "$PYTHON" "$CLS_SVD" --checkpoint "$checkpoint" --model-tag "${RUN}_train" \
    --out-dir "$OUT_ROOT" 2>&1 | tee "$OUT_ROOT/cls_svd_${RUN}_train.log"
PYTHONPATH=. "$PYTHON" "$CLS_SVD" --checkpoint "$checkpoint" --split test --per-class 100 --control-size 100 \
    --model-tag "${RUN}_test" --out-dir "$OUT_ROOT" 2>&1 | tee "$OUT_ROOT/cls_svd_${RUN}_test.log"
echo "[run] $RUN per-loop diagnostics + readout through the tail ($(date '+%F %T'))"
PYTHONPATH=. "$PYTHON" "$LOOP_DIAG" "$checkpoint" --out-dir "$OUT_ROOT/t12_diag" 2>&1 | tee "$OUT_ROOT/t12_diag.log"
echo "[run] done ($(date '+%F %T'))"
