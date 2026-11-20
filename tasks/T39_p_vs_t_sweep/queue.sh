#!/usr/bin/env bash
# T39 queue: runs the 7 T39 trainings on GPUs 0 and 1 ONLY (GPUs 2 and 3 stay free for T40), at most one run per GPU.
# Each run gets its own tmux session (itertf-<run>-<date>). A GPU takes the next run when its previous run's session
# has ended AND the GPU shows under FREE_MIB MiB in use (so a labmate's job is never overrun).
# Start it in its own tmux session from the repo root:
#     tmux new-session -d -s itertf-T39-queue "bash tasks/T39_p_vs_t_sweep/queue.sh"
# Log: tasks/T39_p_vs_t_sweep/results/queue.log. Each run's exit code: results/<run>.exit.
set -euo pipefail

TASK_DIR="tasks/T39_p_vs_t_sweep"
RESULTS="$TASK_DIR/results"
LOG="$RESULTS/queue.log"
GPUS=(0 1)
ORDER=(T39-c T39-g T39-b T39-f T39-e T39-a T39-d)  # longest first (most block passes)
FREE_MIB="${FREE_MIB:-1024}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DATE_TAG="$(date '+%Y%m%d')"

declare -A LANE_SESSION=()  # GPU -> tmux session of the run it holds

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

gpu_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"
}

lane_is_free() {
    local gpu="$1" session="${LANE_SESSION[$1]:-}"
    if [[ -n "$session" ]] && tmux has-session -t "$session" 2>/dev/null; then
        return 1
    fi
    [[ "$(gpu_used_mib "$gpu")" -lt "$FREE_MIB" ]]
}

start_run() {
    local run="$1" gpu="$2" session="itertf-$1-$DATE_TAG"
    tmux new-session -d -s "$session" \
        "CUDA_VISIBLE_DEVICES=$gpu bash $TASK_DIR/run.sh $run; echo \$? > $RESULTS/$run.exit"
    LANE_SESSION[$gpu]="$session"
    log "started $run on GPU $gpu (tmux $session)"
}

report_finished() {
    local gpu session run
    for gpu in "${GPUS[@]}"; do
        session="${LANE_SESSION[$gpu]:-}"
        [[ -z "$session" ]] && continue
        tmux has-session -t "$session" 2>/dev/null && continue
        run="${session#itertf-}"; run="${run%-$DATE_TAG}"
        log "$run finished on GPU $gpu, exit code $(cat "$RESULTS/$run.exit" 2>/dev/null || echo missing)"
        LANE_SESSION[$gpu]=""
    done
}

if [[ ! -d "$TASK_DIR" ]]; then
    echo "run this from the repo root" >&2
    exit 2
fi
mkdir -p "$RESULTS"
log "queue start: ${ORDER[*]} on GPUs ${GPUS[*]}"

next=0
while (( next < ${#ORDER[@]} )); do
    report_finished
    for gpu in "${GPUS[@]}"; do
        (( next < ${#ORDER[@]} )) || break
        if lane_is_free "$gpu"; then
            start_run "${ORDER[$next]}" "$gpu"
            next=$((next + 1))
            sleep 30  # let the run claim its GPU memory before the next check
        fi
    done
    sleep "$POLL_SECONDS"
done

log "all runs started; waiting for the last ones"
while [[ -n "${LANE_SESSION[0]:-}${LANE_SESSION[1]:-}" ]]; do
    report_finished
    sleep "$POLL_SECONDS"
done
log "queue done"
