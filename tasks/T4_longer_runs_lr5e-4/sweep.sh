#!/usr/bin/env bash
# T4: 30-epoch runs at lr 5e-4 with full per-epoch curves. Finished runs are skipped, so rerun after a crash.
# Usage (repo root):  CUDA_VISIBLE_DEVICES=N bash tasks/T4_longer_runs_lr5e-4/sweep.sh [run ...]
# Runs: iter_b0 iter_b6 iter_avg lora_r2 lora_r8 scratch (default: all). Env: EPOCHS, OUT_ROOT, EXTRA_ARGS, PYTHON.
set -euo pipefail

TASK_DIR="tasks/T4_longer_runs_lr5e-4"
PYTHON="${PYTHON:-.venv/bin/python}"
EPOCHS="${EPOCHS:-30}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
LR=5e-4
LORA_LR=5e-3   # ~10x the shared block's lr

run_args() {  # model flags for one run name
    case "$1" in
        iter_b0)  echo "--model iterative --block-idx 0" ;;
        iter_b6)  echo "--model iterative --block-idx 6" ;;
        iter_avg) echo "--model iterative --init avg" ;;
        lora_r2)  echo "--model lora --init block --block-idx 0 --rank 2 --lora-lr $LORA_LR" ;;
        lora_r8)  echo "--model lora --init block --block-idx 0 --rank 8 --lora-lr $LORA_LR" ;;
        scratch)  echo "--model scratch" ;;
        *)        return 1 ;;
    esac
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first (check nvidia-smi)" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root (no $TASK_DIR here)" >&2
    exit 2
fi

RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then RUNS=(iter_b0 iter_b6 iter_avg lora_r2 lora_r8 scratch); fi
for run in "${RUNS[@]}"; do  # fail on a typo before any GPU time is spent
    if ! run_args "$run" > /dev/null; then
        echo "unknown run '$run' (use iter_b0 iter_b6 iter_avg lora_r2 lora_r8 scratch)" >&2
        exit 2
    fi
done
mkdir -p "$OUT_ROOT"

is_complete() {  # true if the run's JSON exists and has as many epochs as it asked for
    "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(len(d['history']) != d['args']['epochs'])" "$1"
}

for run in "${RUNS[@]}"; do
    read -r -a MODEL_ARGS <<< "$(run_args "$run")"
    model="${MODEL_ARGS[1]}"
    tag="_${run}"
    json="$OUT_ROOT/${model}${tag}.json"
    if [[ -f "$json" ]] && is_complete "$json"; then
        echo "[sweep] $run already complete, skipping"
        continue
    fi
    echo "[sweep] $run on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
    "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --lr "$LR" --epochs "$EPOCHS" \
        --out-dir "$OUT_ROOT" --tag "$tag" $EXTRA_ARGS 2>&1 | tee "$OUT_ROOT/${model}${tag}.log"
done
echo "[sweep] done ($(date '+%F %T'))"
