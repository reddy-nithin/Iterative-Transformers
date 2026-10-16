#!/usr/bin/env bash
# T3 progressive-unrolling sweep: one pass over n = number of distinct blocks.
#
# Usage (from the repo root, inside tmux, GPU pinned):
#   CUDA_VISIBLE_DEVICES=0 bash tasks/T3_progressive_unrolling/sweep.sh probe          # n = 1..12
#   CUDA_VISIBLE_DEVICES=1 bash tasks/T3_progressive_unrolling/sweep.sh ft 1 2 3       # only these n
# Passes:
#   probe  body lr 1e-6, head lr 1e-3  (blocks stay near pretrained)
#   ft     everything at lr 1e-4       (the T1/T2 recipe)
# Optional env: EXTRA_ARGS (e.g. "--subset 512 --epochs 1" for a smoke test), OUT_ROOT (default: this folder's results/).
# A run whose JSON already has all its epochs is skipped, so re-running the script resumes a broken sweep.
set -euo pipefail

TASK_DIR="tasks/T3_progressive_unrolling"
PYTHON="${PYTHON:-.venv/bin/python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUT_ROOT="${OUT_ROOT:-$TASK_DIR/results}"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <probe|ft> [n ...]" >&2
    exit 2
fi
PASS="$1"; shift
case "$PASS" in
    probe) LR_ARGS=(--lr 1e-6 --head-lr 1e-3) ;;
    ft)    LR_ARGS=(--lr 1e-4) ;;
    *)     echo "unknown pass '$PASS' (use probe or ft)" >&2; exit 2 ;;
esac
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "set CUDA_VISIBLE_DEVICES to one idle GPU first (check nvidia-smi)" >&2
    exit 2
fi
if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root (no $TASK_DIR here)" >&2
    exit 2
fi

NS=("$@")
if [[ ${#NS[@]} -eq 0 ]]; then NS=($(seq 1 12)); fi
OUT_DIR="$OUT_ROOT/$PASS"
mkdir -p "$OUT_DIR"

is_complete() {  # true if the run's JSON exists and has as many epochs as it asked for
    "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(len(d['history']) != d['args']['epochs'])" "$1"
}

for n in "${NS[@]}"; do
    tag="_u${n}"
    json="$OUT_DIR/unroll${tag}.json"
    if [[ -f "$json" ]] && is_complete "$json"; then
        echo "[sweep] $PASS n=$n already complete, skipping"
        continue
    fi
    echo "[sweep] $PASS n=$n on GPU $CUDA_VISIBLE_DEVICES ($(date '+%F %T'))"
    # shellcheck disable=SC2086  # EXTRA_ARGS is meant to split into separate flags
    "$PYTHON" -m src.train --model unroll --unique-blocks "$n" "${LR_ARGS[@]}" \
        --out-dir "$OUT_DIR" --tag "$tag" $EXTRA_ARGS 2>&1 | tee "$OUT_DIR/unroll${tag}.log"
done
echo "[sweep] $PASS done ($(date '+%F %T'))"
