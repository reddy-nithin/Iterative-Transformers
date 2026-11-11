#!/usr/bin/env bash
# T25: class-cloud exit on T13-c and T11-d (no training), then the plots.
# Usage (repo root, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T25_class_cloud_exit/run.sh
# Env: PYTHON, CKPT_DIR. SMOKE=1: 256 test and 3,000 train images, results in results/smoke.
set -euo pipefail

TASK_DIR="tasks/T25_class_cloud_exit"
PYTHON="${PYTHON:-.venv/bin/python}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
declare -A CHECKPOINTS=(
    [T13-c]="$CKPT_DIR/T13-c_unroll_readout_every_loop.pt"
    [T11-d]="$CKPT_DIR/T11-d_unroll_n3_30ep.pt"
)

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root (no $TASK_DIR here)" >&2
    exit 2
fi

out="$TASK_DIR/results"
extra=()
if [[ "$SMOKE" == "1" ]]; then
    out="$TASK_DIR/results/smoke"
    extra=(--limit 256 --train-limit 3000)
fi
mkdir -p "$out"

for tag in T13-c T11-d; do
    PYTHONPATH=. "$PYTHON" "$TASK_DIR/class_cloud.py" --checkpoint "${CHECKPOINTS[$tag]}" --tag "$tag" \
        --out-dir "$out" "${extra[@]}" 2>&1 | tee "$out/class_cloud_$tag.log"
done
PYTHONPATH=. "$PYTHON" "$TASK_DIR/plot.py" --results "$out" 2>&1 | tee "$out/plot.log"
