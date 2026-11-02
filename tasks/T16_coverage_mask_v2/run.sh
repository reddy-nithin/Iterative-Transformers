#!/usr/bin/env bash
# T16: T13-d (unrolling n=3: b0, b1, b2 x10, no tail; readout every loop + loss margin) plus coverage mask v2 on b2's
# CLS attention with fixed eta (T16-a eta 0.5, T16-b eta 1.0; no new parameters). Pretrained init, shared recipe
# (lr 5e-4, 30 epochs, otherwise src.train defaults). After each run, in the same job: exit curves (T13
# exit_analysis.py), T12 metrics with the mask replayed (T14 diagnostics.py, rule v2) and class separation per block
# call (T5 cls_svd.py, T11 settings, train + test). Finished training runs are skipped, so rerun after a crash.
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T16_coverage_mask_v2/run.sh RUN ...
# Runs: T16-a T16-b. Env: PYTHON, CKPT_DIR, OUT_ROOT.
# SMOKE=1: 300 images, 1 epoch, results in $OUT_ROOT/smoke, checkpoint in $CKPT_DIR/smoke, short analysis.
set -euo pipefail

TASK_DIR="tasks/T16_coverage_mask_v2"
CLS_SVD="tasks/T5_cls_svd_per_block/cls_svd.py"
EXIT_ANALYSIS="tasks/T13_per_loop_readout_loss/exit_analysis.py"
MASK_DIAG="tasks/T14_addon_from_diagnostics/diagnostics.py"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
LR=5e-4
EPOCHS=30

run_eta() {  # run -> fixed eta
    case "$1" in
        T16-a) echo 0.5 ;;
        T16-b) echo 1.0 ;;
        *)     return 1 ;;
    esac
}

checkpoint_name() {  # run -> checkpoint file name (names fixed by the task spec)
    case "$1" in
        T16-a) echo "T16-a_covmask_v2_eta05.pt" ;;
        T16-b) echo "T16-b_covmask_v2_eta1.pt" ;;
        *)     return 1 ;;
    esac
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root (no $TASK_DIR here)" >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then
    echo "name the run(s): T16-a T16-b" >&2
    exit 2
fi
for run in "$@"; do  # fail on a typo before any GPU time is spent
    if ! run_eta "$run" > /dev/null; then
        echo "unknown run '$run' (use T16-a T16-b)" >&2
        exit 2
    fi
done

is_complete() {  # true if the run's JSON exists and has as many epochs as it asked for
    "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(len(d['history']) != d['args']['epochs'])" "$1"
}

for run in "$@"; do
    MODEL_ARGS=(--model unroll --unique-blocks 3 --readout-loss loss_margin --coverage-mask-v2-eta "$(run_eta "$run")")
    if [[ "$SMOKE" == "1" ]]; then
        out="$OUT_ROOT/smoke"
        checkpoint="$CKPT_DIR/smoke/$(checkpoint_name "$run")"
        mkdir -p "$out" "$CKPT_DIR/smoke"
        "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs 1 --subset 300 \
            --out-dir "$out" --tag "_$run" --save-checkpoint "$checkpoint" 2>&1 | tee "$out/$run.log"
        PYTHONPATH=. "$PYTHON" "$MASK_DIAG" "$checkpoint" --collapse-images 200 --limit 200 \
            --out-dir "$out/t12_diag" 2>&1 | tee "$out/t12_diag_$run.log"
        continue
    fi
    mkdir -p "$OUT_ROOT" "$CKPT_DIR"
    checkpoint="$CKPT_DIR/$(checkpoint_name "$run")"
    json="$OUT_ROOT/unroll_$run.json"
    if [[ -f "$json" ]] && is_complete "$json"; then
        echo "[run] $run already trained, skipping training"
    else
        echo "[run] $run on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
        "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs "$EPOCHS" \
            --out-dir "$OUT_ROOT" --tag "_$run" --save-checkpoint "$checkpoint" 2>&1 | tee "$OUT_ROOT/$run.log"
    fi
    echo "[run] $run exit curves, T12 metrics with the mask, class separation ($(date '+%F %T'))"
    PYTHONPATH=. "$PYTHON" "$EXIT_ANALYSIS" "$checkpoint" --out "$OUT_ROOT/exit_analysis.json" \
        2>&1 | tee "$OUT_ROOT/exit_analysis_$run.log"
    PYTHONPATH=. "$PYTHON" "$MASK_DIAG" "$checkpoint" --out-dir "$OUT_ROOT/t12_diag" 2>&1 | tee "$OUT_ROOT/t12_diag_$run.log"
    PYTHONPATH=. "$PYTHON" "$CLS_SVD" --checkpoint "$checkpoint" --model-tag "${run}_train" \
        --out-dir "$OUT_ROOT" 2>&1 | tee "$OUT_ROOT/cls_svd_${run}_train.log"
    PYTHONPATH=. "$PYTHON" "$CLS_SVD" --checkpoint "$checkpoint" --split test --per-class 100 --control-size 100 \
        --model-tag "${run}_test" --out-dir "$OUT_ROOT" 2>&1 | tee "$OUT_ROOT/cls_svd_${run}_test.log"
done
echo "[run] done ($(date '+%F %T'))"
