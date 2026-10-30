#!/usr/bin/env bash
# T13: readout at every body loop (and the loss-margin version) on the T11-c layout (head b0, body b1 x10, tail b11)
# and the T11-d layout (unrolling n=3: b0, b1, b2 x10, no tail). Pretrained init as in T11, shared recipe (lr 5e-4,
# 30 epochs, otherwise src.train defaults); each run differs from T11-c / T11-d only in --readout-loss. After each run,
# class separation of the CLS token after every block call (T5's cls_svd.py, same settings as T11), train and test.
# Finished training runs are skipped (their CLS analysis is redone), so rerun after a crash.
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T13_per_loop_readout_loss/sweep.sh RUN ...
# Runs: T13-a T13-b T13-c T13-d. Env: PYTHON, CKPT_DIR, OUT_ROOT.
# SMOKE=1: 300 images, 1 epoch, results in $OUT_ROOT/smoke, no checkpoint and no CLS analysis.
set -euo pipefail

TASK_DIR="tasks/T13_per_loop_readout_loss"
CLS_SVD="tasks/T5_cls_svd_per_block/cls_svd.py"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
LR=5e-4
EPOCHS=30

run_args() {  # model flags for one run: the T11-c or T11-d model, plus the T13 loss
    local t11c="--model hbt --hbt-pretrained --head-idx 0 --body-idx 1 --tail-idx 11"
    local t11d="--model unroll --unique-blocks 3"
    case "$1" in
        T13-a) echo "$t11c --readout-loss every_loop" ;;
        T13-b) echo "$t11c --readout-loss loss_margin" ;;
        T13-c) echo "$t11d --readout-loss every_loop" ;;
        T13-d) echo "$t11d --readout-loss loss_margin" ;;
        *)     return 1 ;;
    esac
}

checkpoint_name() {  # run -> checkpoint file name (names fixed by the task spec)
    case "$1" in
        T13-a) echo "T13-a_readout_every_loop.pt" ;;
        T13-b) echo "T13-b_loss_margin.pt" ;;
        T13-c) echo "T13-c_unroll_readout_every_loop.pt" ;;
        T13-d) echo "T13-d_unroll_loss_margin.pt" ;;
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
    echo "name the run(s): T13-a T13-b T13-c T13-d" >&2
    exit 2
fi
for run in "$@"; do  # fail on a typo before any GPU time is spent
    if ! run_args "$run" > /dev/null; then
        echo "unknown run '$run' (use T13-a T13-b T13-c T13-d)" >&2
        exit 2
    fi
done

is_complete() {  # true if the run's JSON exists and has as many epochs as it asked for
    "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(len(d['history']) != d['args']['epochs'])" "$1"
}

for run in "$@"; do
    read -r -a MODEL_ARGS <<< "$(run_args "$run")"
    if [[ "$SMOKE" == "1" ]]; then
        out="$OUT_ROOT/smoke"
        mkdir -p "$out"
        echo "[sweep] smoke $run on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
        "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs 1 --subset 300 \
            --out-dir "$out" --tag "_$run" 2>&1 | tee "$out/$run.log"
        continue
    fi
    mkdir -p "$OUT_ROOT" "$CKPT_DIR"
    checkpoint="$CKPT_DIR/$(checkpoint_name "$run")"
    json="$OUT_ROOT/${MODEL_ARGS[1]}_$run.json"
    if [[ -f "$json" ]] && is_complete "$json"; then
        echo "[sweep] $run already trained, skipping training"
    else
        echo "[sweep] $run on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
        "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs "$EPOCHS" \
            --out-dir "$OUT_ROOT" --tag "_$run" --save-checkpoint "$checkpoint" 2>&1 | tee "$OUT_ROOT/$run.log"
    fi
    echo "[sweep] $run class separation, train + test ($(date '+%F %T'))"
    PYTHONPATH=. "$PYTHON" "$CLS_SVD" --checkpoint "$checkpoint" --model-tag "${run}_train" \
        --out-dir "$OUT_ROOT" 2>&1 | tee "$OUT_ROOT/cls_svd_${run}_train.log"
    PYTHONPATH=. "$PYTHON" "$CLS_SVD" --checkpoint "$checkpoint" --split test --per-class 100 --control-size 100 \
        --model-tag "${run}_test" --out-dir "$OUT_ROOT" 2>&1 | tee "$OUT_ROOT/cls_svd_${run}_test.log"
done
echo "[sweep] done ($(date '+%F %T'))"
