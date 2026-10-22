#!/usr/bin/env bash
# T9: head + repeated body + tail, trained from scratch, lr 5e-4, 30 epochs. After each run, class separation of the
# CLS token after every loop (T5's cls_svd.py, same 4-class settings as the iter b0 numbers), train and test.
# Finished training runs are skipped (their CLS analysis is redone), so rerun after a crash.
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T9_head_body_tail/sweep.sh [run ...]
# Runs: hbt hbt_inject head_only loop_only (default: all). Env: EPOCHS, OUT_ROOT, EXTRA_ARGS, PYTHON, CKPT_DIR,
# CLS_ARGS (extra flags for cls_svd.py, e.g. "--per-class 20 --control-size 0" for a smoke test).
set -euo pipefail

TASK_DIR="tasks/T9_head_body_tail"
CLS_SVD="tasks/T5_cls_svd_per_block/cls_svd.py"
PYTHON="${PYTHON:-.venv/bin/python}"
EPOCHS="${EPOCHS:-30}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CLS_ARGS="${CLS_ARGS:-}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
LR=5e-4

run_args() {  # model flags for one run name
    case "$1" in
        hbt)        echo "--head-blocks 1 --tail-blocks 1" ;;            # meeting design: head + body x10 + tail
        hbt_inject) echo "--head-blocks 1 --tail-blocks 1 --inject" ;;   # (b) body(x + head output) every loop
        head_only)  echo "--head-blocks 1 --tail-blocks 0" ;;            # (a) head + body x11
        loop_only)  echo "--head-blocks 0 --tail-blocks 0" ;;            # one random block x12
        *)          return 1 ;;
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

RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then RUNS=(hbt hbt_inject head_only loop_only); fi
for run in "${RUNS[@]}"; do  # fail on a typo before any GPU time is spent
    if ! run_args "$run" > /dev/null; then
        echo "unknown run '$run' (use hbt hbt_inject head_only loop_only)" >&2
        exit 2
    fi
done
mkdir -p "$OUT_ROOT" "$CKPT_DIR"

is_complete() {  # true if the run's JSON exists and has as many epochs as it asked for
    "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(len(d['history']) != d['args']['epochs'])" "$1"
}

for run in "${RUNS[@]}"; do
    read -r -a MODEL_ARGS <<< "$(run_args "$run")"
    tag="_${run}"
    json="$OUT_ROOT/hbt${tag}.json"
    checkpoint="$CKPT_DIR/t9_${run}.pt"
    if [[ -f "$json" ]] && is_complete "$json"; then
        echo "[sweep] $run already trained, skipping training"
    else
        echo "[sweep] $run on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
        # shellcheck disable=SC2086  # EXTRA_ARGS is meant to split into separate flags
        "$PYTHON" -m src.train --model hbt "${MODEL_ARGS[@]}" --lr "$LR" --epochs "$EPOCHS" \
            --out-dir "$OUT_ROOT" --tag "$tag" --save-checkpoint "$checkpoint" $EXTRA_ARGS \
            2>&1 | tee "$OUT_ROOT/hbt${tag}.log"
    fi
    echo "[sweep] $run class separation, train + test ($(date '+%F %T'))"
    # shellcheck disable=SC2086
    PYTHONPATH=. "$PYTHON" "$CLS_SVD" --checkpoint "$checkpoint" --model-tag "${run}_train" \
        --out-dir "$OUT_ROOT" $CLS_ARGS 2>&1 | tee "$OUT_ROOT/cls_svd_${run}_train.log"
    # shellcheck disable=SC2086
    PYTHONPATH=. "$PYTHON" "$CLS_SVD" --checkpoint "$checkpoint" --split test --per-class 100 --control-size 100 \
        --model-tag "${run}_test" --out-dir "$OUT_ROOT" $CLS_ARGS 2>&1 | tee "$OUT_ROOT/cls_svd_${run}_test.log"
done
echo "[sweep] done ($(date '+%F %T'))"
