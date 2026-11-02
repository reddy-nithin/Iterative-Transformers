#!/usr/bin/env bash
# T18: progressive unrolling (blocks 0..n-2 kept, block n-1 repeated, 12 applications in total) at 30 epochs for
# n = 2, 4, 5, 6. Existing unrolling code unchanged, only epochs 10 -> 30; lr 5e-4, seed 0, pretrained init, otherwise
# src.train defaults. Finished training runs are skipped, so rerun after a crash.
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T18_unroll_curve_30ep/sweep.sh RUN ...
# Runs: T18-n2 T18-n4 T18-n5 T18-n6. Env: PYTHON, CKPT_DIR, OUT_ROOT.
# SMOKE=1: 300 images, 1 epoch, results in $OUT_ROOT/smoke, no checkpoint.
set -euo pipefail

TASK_DIR="tasks/T18_unroll_curve_30ep"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
LR=5e-4
EPOCHS=30
SEED=0

unique_blocks() {  # run -> n distinct blocks
    case "$1" in
        T18-n2) echo 2 ;;
        T18-n4) echo 4 ;;
        T18-n5) echo 5 ;;
        T18-n6) echo 6 ;;
        *)      return 1 ;;
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
    echo "name the run(s): T18-n2 T18-n4 T18-n5 T18-n6" >&2
    exit 2
fi
for run in "$@"; do  # fail on a typo before any GPU time is spent
    if ! unique_blocks "$run" > /dev/null; then
        echo "unknown run '$run' (use T18-n2 T18-n4 T18-n5 T18-n6)" >&2
        exit 2
    fi
done

is_complete() {  # true if the run's JSON exists and has as many epochs as it asked for
    "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(len(d['history']) != d['args']['epochs'])" "$1"
}

for run in "$@"; do
    n="$(unique_blocks "$run")"
    if [[ "$SMOKE" == "1" ]]; then
        out="$OUT_ROOT/smoke"
        mkdir -p "$out"
        echo "[sweep] smoke $run (n=$n) on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
        "$PYTHON" -m src.train --model unroll --unique-blocks "$n" --seed "$SEED" --lr "$LR" --epochs 1 --subset 300 \
            --out-dir "$out" --tag "_$run" 2>&1 | tee "$out/$run.log"
        continue
    fi
    mkdir -p "$OUT_ROOT" "$CKPT_DIR"
    checkpoint="$CKPT_DIR/${run}_unroll_30ep.pt"
    json="$OUT_ROOT/unroll_$run.json"
    if [[ -f "$json" ]] && is_complete "$json"; then
        echo "[sweep] $run already trained, skipping"
        continue
    fi
    echo "[sweep] $run (n=$n) on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
    "$PYTHON" -m src.train --model unroll --unique-blocks "$n" --seed "$SEED" --lr "$LR" --epochs "$EPOCHS" \
        --out-dir "$OUT_ROOT" --tag "_$run" --save-checkpoint "$checkpoint" 2>&1 | tee "$OUT_ROOT/$run.log"
done
echo "[sweep] done ($(date '+%F %T'))"
