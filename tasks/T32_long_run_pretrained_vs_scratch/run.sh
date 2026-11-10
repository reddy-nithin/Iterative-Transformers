#!/usr/bin/env bash
# T32: the 2+10 model (unrolling n=3: blocks 0, 1 distinct, block 2 x10, no tail) for 100 epochs, from pretrained
# weights (T32-a) and from random weights (T32-b). Same recipe for both (lr 5e-4, 100 epochs, otherwise src.train
# defaults: AdamW wd 0.05, 1 warmup epoch + cosine, label smoothing 0.1, batch 128, AMP, seed 0); only the init differs.
# Usage (repo root, tmux, one GPU per run):  CUDA_VISIBLE_DEVICES=N bash tasks/T32_long_run_pretrained_vs_scratch/run.sh RUN
# Runs: T32-a T32-b. Env: PYTHON, CKPT_DIR, OUT_ROOT.
# SMOKE=1: 300 images, 1 epoch, results in $OUT_ROOT/smoke, no checkpoint.
# A finished run (JSON with all 100 epochs) is skipped. There is no resume: a crashed run restarts from epoch 1.
set -euo pipefail

TASK_DIR="tasks/T32_long_run_pretrained_vs_scratch"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
LR=5e-4
EPOCHS=100
SEED=0

run_args() {  # model flags for one run; both give B0, B1, B2 x10 with 1,539,940 params
    case "$1" in
        T32-a) echo "--model unroll --unique-blocks 3" ;;               # pretrained init, exactly T11-d's flags
        T32-b) echo "--model hbt --head-blocks 2 --tail-blocks 0" ;;    # same layout, timm default random init
        *)     return 1 ;;
    esac
}

checkpoint_name() {  # run -> checkpoint file name (names fixed by the task spec)
    case "$1" in
        T32-a) echo "T32-a_2p10_pretrained_100ep.pt" ;;
        T32-b) echo "T32-b_2p10_scratch_100ep.pt" ;;
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
    echo "name the run(s): T32-a T32-b" >&2
    exit 2
fi
for run in "$@"; do  # fail on a typo before any GPU time is spent
    if ! run_args "$run" > /dev/null; then
        echo "unknown run '$run' (use T32-a T32-b)" >&2
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
        echo "[T32] smoke $run on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
        "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs 1 --seed "$SEED" --subset 300 \
            --out-dir "$out" --tag "_$run" 2>&1 | tee "$out/$run.log"
        continue
    fi
    mkdir -p "$OUT_ROOT" "$CKPT_DIR"
    checkpoint="$CKPT_DIR/$(checkpoint_name "$run")"
    json="$OUT_ROOT/${MODEL_ARGS[1]}_$run.json"
    if [[ -f "$json" ]] && is_complete "$json"; then
        echo "[T32] $run already trained, skipping"
        continue
    fi
    echo "[T32] $run on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
    "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs "$EPOCHS" --seed "$SEED" \
        --out-dir "$OUT_ROOT" --tag "_$run" --save-checkpoint "$checkpoint" 2>&1 | tee "$OUT_ROOT/$run.log"
done
echo "[T32] done ($(date '+%F %T'))"
