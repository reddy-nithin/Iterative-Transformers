#!/usr/bin/env bash
# T11: pretrained head + body x10 + tail with different body blocks, plus progressive unrolling n=3 for 30 epochs,
# all on the shared recipe (lr 5e-4, 30 epochs, otherwise src.train defaults). After each run, class separation of the
# CLS token after every loop (T5's cls_svd.py, same 4-class settings as T9), train and test.
# Finished training runs are skipped (their CLS analysis is redone), so rerun after a crash.
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T11_pretrained_hbt_body_sweep/sweep.sh RUN ...
# Runs: T11-a T11-b T11-c T11-d. Env: PYTHON, CKPT_DIR, OUT_ROOT.
# SMOKE=1: 300 images, 1 epoch, results in $OUT_ROOT/smoke, no checkpoint and no CLS analysis.
set -euo pipefail

TASK_DIR="tasks/T11_pretrained_hbt_body_sweep"
CLS_SVD="tasks/T5_cls_svd_per_block/cls_svd.py"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
LR=5e-4
EPOCHS=30

run_args() {  # model flags for one run
    case "$1" in
        T11-a) echo "--model hbt --hbt-pretrained --head-idx 0 --body-idx 0 --tail-idx 11" ;;
        T11-b) echo "--model hbt --hbt-pretrained --head-idx 0 --body-idx 6 --tail-idx 11" ;;
        T11-c) echo "--model hbt --hbt-pretrained --head-idx 0 --body-idx 1 --tail-idx 11" ;;
        T11-d) echo "--model unroll --unique-blocks 3" ;;  # existing unrolling code, only epochs 10 -> 30
        *)     return 1 ;;
    esac
}

checkpoint_name() {  # run -> checkpoint file name (names fixed by the task spec)
    case "$1" in
        T11-a) echo "T11-a_hbt_body_b0.pt" ;;
        T11-b) echo "T11-b_hbt_body_b6.pt" ;;
        T11-c) echo "T11-c_hbt_body_b1.pt" ;;
        T11-d) echo "T11-d_unroll_n3_30ep.pt" ;;
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
    echo "name the run(s): T11-a T11-b T11-c T11-d" >&2
    exit 2
fi
for run in "$@"; do  # fail on a typo before any GPU time is spent
    if ! run_args "$run" > /dev/null; then
        echo "unknown run '$run' (use T11-a T11-b T11-c T11-d)" >&2
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
