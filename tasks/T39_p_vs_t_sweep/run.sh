#!/usr/bin/env bash
# T39: loop-count sweep at fixed parameters. h+r = h distinct pretrained blocks, then block h looped r times
# (--unique-blocks h+1 --body-loops r). 3W runs: 2+2, 2+8, 2+16. 4W runs: 3+1, 3+2, 3+4, 3+8.
# Recipe identical to T11-d (lr 5e-4, 30 epochs, seed 0, otherwise src.train defaults).
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T39_p_vs_t_sweep/run.sh T39-a ... T39-g
# SMOKE=1: 1 epoch on --subset 512, results in $OUT_ROOT/smoke, no checkpoint. Env: PYTHON, OUT_ROOT, CKPT_DIR.
set -euo pipefail

TASK_DIR="tasks/T39_p_vs_t_sweep"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
RUN="${1:-}"

case "$RUN" in
    T39-a) UNIQUE=3; LOOPS=2;  CKPT="T39-a_2p2_30ep.pt" ;;
    T39-b) UNIQUE=3; LOOPS=8;  CKPT="T39-b_2p8_30ep.pt" ;;
    T39-c) UNIQUE=3; LOOPS=16; CKPT="T39-c_2p16_30ep.pt" ;;
    T39-d) UNIQUE=4; LOOPS=1;  CKPT="T39-d_3p1_30ep.pt" ;;
    T39-e) UNIQUE=4; LOOPS=2;  CKPT="T39-e_3p2_30ep.pt" ;;
    T39-f) UNIQUE=4; LOOPS=4;  CKPT="T39-f_3p4_30ep.pt" ;;
    T39-g) UNIQUE=4; LOOPS=8;  CKPT="T39-g_3p8_30ep.pt" ;;
    *)     echo "name the run: T39-a ... T39-g" >&2; exit 2 ;;
esac
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root" >&2
    exit 2
fi

MODEL_ARGS=(--model unroll --unique-blocks "$UNIQUE" --body-loops "$LOOPS" --lr 5e-4 --seed 0)

if [[ "$SMOKE" == "1" ]]; then
    out="$OUT_ROOT/smoke"
    mkdir -p "$out"
    "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --epochs 1 --subset 512 \
        --out-dir "$out" --tag "_$RUN" 2>&1 | tee "$out/$RUN.log"
    exit 0
fi

mkdir -p "$OUT_ROOT" "$CKPT_DIR"
echo "[T39] $RUN ($((UNIQUE - 1))+$LOOPS) on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
"$PYTHON" -m src.train "${MODEL_ARGS[@]}" --epochs 30 \
    --out-dir "$OUT_ROOT" --tag "_$RUN" --save-checkpoint "$CKPT_DIR/$CKPT" \
    2>&1 | tee "$OUT_ROOT/$RUN.log"
echo "[T39] $RUN done ($(date '+%F %T'))"
