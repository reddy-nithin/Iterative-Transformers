#!/usr/bin/env bash
# T14-a: T13-d (unrolling n=3: b0, b1, b2 x10, no tail; readout every loop + loss margin) plus the coverage mask on
# b2's CLS attention (+1 parameter, eta). Pretrained init as in T11 / T13, shared recipe (lr 5e-4, 30 epochs, otherwise
# src.train defaults). After training, in the same job: class separation per block call (T5 cls_svd.py, T13 settings,
# train + test), the T13 exit analysis and the T12 metrics (this folder's diagnostics.py).
# A finished training run is skipped (the analysis is redone), so rerun after a crash.
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T14_addon_from_diagnostics/run.sh
# Env: PYTHON, CKPT_DIR, OUT_ROOT. SMOKE=1: 300 images, 1 epoch, results in $OUT_ROOT/smoke, checkpoint in
# $CKPT_DIR/smoke (never in the repo), short analysis on 500 test images.
set -euo pipefail

TASK_DIR="tasks/T14_addon_from_diagnostics"
CLS_SVD="tasks/T5_cls_svd_per_block/cls_svd.py"
EXIT_ANALYSIS="tasks/T13_per_loop_readout_loss/exit_analysis.py"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
RUN="T14-a"
MODEL_ARGS=(--model unroll --unique-blocks 3 --coverage-mask --readout-loss loss_margin)
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
    mkdir -p "$out"
    mkdir -p "$CKPT_DIR/smoke"
    checkpoint="$CKPT_DIR/smoke/${RUN}_smoke.pt"
    echo "[run] smoke $RUN on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
    "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs 1 --subset 300 \
        --out-dir "$out" --tag "_$RUN" --save-checkpoint "$checkpoint" 2>&1 | tee "$out/$RUN.log"
    PYTHONPATH=. "$PYTHON" "$EXIT_ANALYSIS" "$checkpoint" --limit 500 --out "$out/exit_analysis.json" 2>&1 \
        | tee "$out/exit_analysis.log"
    PYTHONPATH=. "$PYTHON" "$TASK_DIR/diagnostics.py" "$checkpoint" --limit 500 --collapse-images 200 \
        --out-dir "$out/t12_diag" 2>&1 | tee "$out/diagnostics.log"
    echo "[run] smoke done ($(date '+%F %T'))"
    exit 0
fi

mkdir -p "$OUT_ROOT" "$CKPT_DIR"
checkpoint="$CKPT_DIR/T14-a_coverage_mask.pt"
json="$OUT_ROOT/unroll_$RUN.json"
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
echo "[run] $RUN exit analysis ($(date '+%F %T'))"
PYTHONPATH=. "$PYTHON" "$EXIT_ANALYSIS" "$checkpoint" --out "$OUT_ROOT/exit_analysis.json" 2>&1 \
    | tee "$OUT_ROOT/exit_analysis.log"
echo "[run] $RUN T12 metrics ($(date '+%F %T'))"
PYTHONPATH=. "$PYTHON" "$TASK_DIR/diagnostics.py" "$checkpoint" 2>&1 | tee "$OUT_ROOT/diagnostics.log"
echo "[run] done ($(date '+%F %T'))"
