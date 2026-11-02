#!/usr/bin/env bash
# T15: seeds 1 and 2 for T11-c (head b0, body b1 x10, tail b11), T11-d (unrolling n=3: b0, b1, b2 x10) and T13-d
# (T11-d + readout every loop + loss margin). Each run is the existing config with only --seed changed. Pretrained init,
# shared recipe (lr 5e-4, 30 epochs, otherwise src.train defaults). After T15-e/f, T13's exit analysis gives the
# fixed-depth curve (top-1 at loop 1..10) on the full test set.
# Finished training runs are skipped (their analysis is redone), so rerun after a crash.
# Usage (repo root, tmux, GPU pinned):  CUDA_VISIBLE_DEVICES=N bash tasks/T15_seed_repeats/sweep.sh RUN ...
# Runs: T15-a .. T15-f. Env: PYTHON, CKPT_DIR, OUT_ROOT.
# SMOKE=1: 300 images, 1 epoch, results in $OUT_ROOT/smoke, checkpoint in $CKPT_DIR/smoke (never in the repo),
# exit analysis on 500 test images.
set -euo pipefail

TASK_DIR="tasks/T15_seed_repeats"
EXIT_ANALYSIS="tasks/T13_per_loop_readout_loss/exit_analysis.py"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"
CKPT_DIR="${CKPT_DIR:-/storage/INSLAB/Data/nrs3qy/checkpoints/itertf}"
SMOKE="${SMOKE:-0}"
LR=5e-4
EPOCHS=30

run_args() {  # model flags for one run: the T11-c, T11-d or T13-d config, unchanged
    local t11c="--model hbt --hbt-pretrained --head-idx 0 --body-idx 1 --tail-idx 11"
    local t11d="--model unroll --unique-blocks 3"
    case "$1" in
        T15-a|T15-b) echo "$t11c" ;;
        T15-c|T15-d) echo "$t11d" ;;
        T15-e|T15-f) echo "$t11d --readout-loss loss_margin" ;;
        *)           return 1 ;;
    esac
}

run_seed() {  # run -> seed (seed 0 is the existing T11 / T13 run)
    case "$1" in
        T15-a|T15-c|T15-e) echo 1 ;;
        T15-b|T15-d|T15-f) echo 2 ;;
        *)                 return 1 ;;
    esac
}

checkpoint_name() {  # run -> checkpoint file name (names fixed by the task spec)
    case "$1" in
        T15-a) echo "T15-a_t11c_seed1.pt" ;;
        T15-b) echo "T15-b_t11c_seed2.pt" ;;
        T15-c) echo "T15-c_t11d_seed1.pt" ;;
        T15-d) echo "T15-d_t11d_seed2.pt" ;;
        T15-e) echo "T15-e_t13d_seed1.pt" ;;
        T15-f) echo "T15-f_t13d_seed2.pt" ;;
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
    echo "name the run(s): T15-a T15-b T15-c T15-d T15-e T15-f" >&2
    exit 2
fi
for run in "$@"; do  # fail on a typo before any GPU time is spent
    if ! run_args "$run" > /dev/null; then
        echo "unknown run '$run' (use T15-a .. T15-f)" >&2
        exit 2
    fi
done

is_complete() {  # true if the run's JSON exists and has as many epochs as it asked for
    "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(len(d['history']) != d['args']['epochs'])" "$1"
}

for run in "$@"; do
    read -r -a MODEL_ARGS <<< "$(run_args "$run")"
    seed="$(run_seed "$run")"
    if [[ "$SMOKE" == "1" ]]; then
        out="$OUT_ROOT/smoke"
        checkpoint="$CKPT_DIR/smoke/$(checkpoint_name "$run")"
        mkdir -p "$out" "$CKPT_DIR/smoke"
        echo "[sweep] smoke $run (seed $seed) on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
        "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --seed "$seed" --lr "$LR" --epochs 1 --subset 300 \
            --out-dir "$out" --tag "_$run" --save-checkpoint "$checkpoint" 2>&1 | tee "$out/$run.log"
        if [[ "$run" == T15-e || "$run" == T15-f ]]; then
            PYTHONPATH=. "$PYTHON" "$EXIT_ANALYSIS" "$checkpoint" --limit 500 --out "$out/exit_analysis.json" \
                2>&1 | tee "$out/exit_analysis_$run.log"
        fi
        continue
    fi
    mkdir -p "$OUT_ROOT" "$CKPT_DIR"
    checkpoint="$CKPT_DIR/$(checkpoint_name "$run")"
    json="$OUT_ROOT/${MODEL_ARGS[1]}_$run.json"
    if [[ -f "$json" ]] && is_complete "$json"; then
        echo "[sweep] $run already trained, skipping training"
    else
        echo "[sweep] $run (seed $seed) on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
        "$PYTHON" -m src.train "${MODEL_ARGS[@]}" --seed "$seed" --lr "$LR" --epochs "$EPOCHS" \
            --out-dir "$OUT_ROOT" --tag "_$run" --save-checkpoint "$checkpoint" 2>&1 | tee "$OUT_ROOT/$run.log"
    fi
    if [[ "$run" == T15-e || "$run" == T15-f ]]; then
        echo "[sweep] $run fixed-depth curve and exit rules ($(date '+%F %T'))"
        PYTHONPATH=. "$PYTHON" "$EXIT_ANALYSIS" "$checkpoint" --out "$OUT_ROOT/exit_analysis_$run.json" \
            2>&1 | tee "$OUT_ROOT/exit_analysis_$run.log"
    fi
done
echo "[sweep] done ($(date '+%F %T'))"
