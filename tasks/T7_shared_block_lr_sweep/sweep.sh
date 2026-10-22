#!/usr/bin/env bash
# T7: lr ceiling of the single shared block (iter b0, block 0 x12), 30 epochs, lrs in increasing order.
# Stop rule: after each run, if its final test acc is below the previous lr's, or any epoch's loss is NaN, the higher
# lrs are skipped. The previous point for the first lr is T4's lr 5e-4 run (58.06). Finished runs are skipped, so
# rerun after a crash.
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T7_shared_block_lr_sweep/sweep.sh [lr ...]
# Default lrs: 1e-3 2e-3 5e-3. Env: EPOCHS, OUT_ROOT, EXTRA_ARGS, PYTHON, CKPT_DIR.
set -euo pipefail

TASK_DIR="tasks/T7_shared_block_lr_sweep"
PYTHON="${PYTHON:-.venv/bin/python}"
EPOCHS="${EPOCHS:-30}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
PREVIOUS_JSON="tasks/T4_longer_runs_lr5e-4/results/iterative_iter_b0.json"  # lr 5e-4, 30 ep

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root (no $TASK_DIR here)" >&2
    exit 2
fi

LRS=("$@")
if [[ ${#LRS[@]} -eq 0 ]]; then LRS=(1e-3 2e-3 5e-3); fi
mkdir -p "$OUT_ROOT" "$CKPT_DIR"

is_complete() {  # true if the run's JSON exists and has as many epochs as it asked for
    "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(len(d['history']) != d['args']['epochs'])" "$1"
}
final_acc() {  # final test acc of a run JSON
    "$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['final_test_acc'])" "$1"
}
has_nan() {  # true if any epoch's train or test loss is NaN
    "$PYTHON" -c "import json,math,sys; h=json.load(open(sys.argv[1]))['history']; sys.exit(0 if any(math.isnan(e['train_loss']) or math.isnan(e['test_loss']) for e in h) else 1)" "$1"
}

previous_acc="$(final_acc "$PREVIOUS_JSON")"
previous_lr="5e-4"
for lr in "${LRS[@]}"; do
    tag="_lr${lr}"
    json="$OUT_ROOT/iterative${tag}.json"
    if [[ -f "$json" ]] && is_complete "$json"; then
        echo "[sweep] lr $lr already complete, skipping"
    else
        echo "[sweep] lr $lr on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
        # shellcheck disable=SC2086  # EXTRA_ARGS is meant to split into separate flags
        "$PYTHON" -m src.train --model iterative --block-idx 0 --lr "$lr" --epochs "$EPOCHS" \
            --out-dir "$OUT_ROOT" --tag "$tag" --save-checkpoint "$CKPT_DIR/iter_b0${tag}.pt" $EXTRA_ARGS \
            2>&1 | tee "$OUT_ROOT/iterative${tag}.log"
    fi
    acc="$(final_acc "$json")"
    if has_nan "$json"; then
        echo "[sweep] STOP: lr $lr has a NaN loss (final acc $acc); higher lrs skipped"
        break
    fi
    if "$PYTHON" -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$acc" "$previous_acc"; then
        echo "[sweep] STOP: lr $lr final acc $acc < lr $previous_lr final acc $previous_acc; higher lrs skipped"
        break
    fi
    previous_acc="$acc"; previous_lr="$lr"
done
echo "[sweep] done ($(date '+%F %T')); best so far: lr $previous_lr at $previous_acc%"
