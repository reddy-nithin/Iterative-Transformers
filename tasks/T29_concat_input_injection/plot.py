"""T29 results: concat + adapter injection (T29-a) vs T11-d, with T9's plain-add run (T9-c) for the norms.

Tables: params / GFLOPs / epoch time / top-1, per-loop readout, mean ‖x_t‖ per loop, separation per loop, adapter split.
Plots: results/acc_per_loop.png, results/state_norm_per_loop.png, results/separation_per_loop.png.
Inputs: results/{unroll_T29-a.json, exit_analysis.json, cls_svd_T29-a_*.json, state_norms.json}, T11-d's files (T11,
T13) and T9-c's separation (tasks/T9_head_body_tail/results/cls_svd_hbt_inject_*.json).

Run from the repo root:  PYTHONPATH=. .venv/bin/python tasks/T29_concat_input_injection/plot.py
"""
import json
import logging
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
TASKS = TASK_DIR.parent
sys.path.insert(0, str(TASKS / "T28_frozen_patch_cls_loop"))

from compare import (T11D, RunFiles, load_json, markdown_row, per_loop_readout, plot_curves,  # noqa: E402
                     separation_per_loop, train_summary)

T29A = RunFiles("T29-a concat + adapter", RESULTS / "unroll_T29-a.json", RESULTS / "exit_analysis.json",
                "T29-a_concat_inject", RESULTS / "cls_svd_T29-a", "tab:blue")
T9C = RunFiles("T9-c plain add (scratch init)", TASKS / "T9_head_body_tail/results/hbt_hbt_inject.json",
               RESULTS / "unused", "", TASKS / "T9_head_body_tail/results/cls_svd_hbt_inject", "tab:gray")
T9_HEAD_CALLS = 1  # T9 layout: head 1, body x10, tail 1
NORM_KEYS = {T29A.label: "T29-a", T11D.label: "T11-d", T9C.label: "T9-c"}

log = logging.getLogger("t29_plot")


def separation(run: RunFiles, split: str) -> list[float]:
    """Separation after loops 1..10 (T9-c has one head block, the others two)."""
    head_calls = T9_HEAD_CALLS if run is T9C else 2
    return separation_per_loop(run, split, head_calls=head_calls)


def main() -> None:
    """Tables and plots."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    norms = load_json(RESULTS / "state_norms.json")
    main_runs, all_runs = (T11D, T29A), (T11D, T29A, T9C)
    summaries = {run.label: train_summary(run) for run in main_runs}
    readout = {run.label: per_loop_readout(run) for run in main_runs}
    sep = {(run.label, split): separation(run, split) for run in all_runs for split in ("test", "train")}
    log.info(markdown_row(["model", "params", "GFLOPs / image", "throughput img/s", "median epoch s", "final top-1",
                           "best top-1"]))
    for run in main_runs:
        s = summaries[run.label]
        log.info(markdown_row([run.label, f"{s['params']:,}", s["gflops"], s["throughput_img_s"],
                               s["median_epoch_s"], s["final_top1"], s["best_top1"]]))
    for run in main_runs:
        log.info(markdown_row([f"{run.label} readout", *readout[run.label]]))
    for run in all_runs:
        log.info(markdown_row([f"{run.label} mean ‖x_t‖, t = 0..10", *norms[NORM_KEYS[run.label]]["mean_token_norm"]]))
        log.info(markdown_row([f"{run.label} sep. test", *sep[(run.label, "test")]]))
        log.info(markdown_row([f"{run.label} sep. train", *sep[(run.label, "train")]]))
    log.info("adapter: %s", norms["adapter"])
    plot_curves({run.label: (readout[run.label], run.colour) for run in main_runs}, "test top-1 (%)",
                "T29 · readout after every loop", RESULTS / "acc_per_loop.png")
    plot_curves({run.label: (norms[NORM_KEYS[run.label]]["mean_token_norm"][1:], run.colour) for run in all_runs},
                "mean token norm ‖x_t‖", "T29 · state norm after every loop (test)", RESULTS / "state_norm_per_loop.png")
    plot_curves({run.label: (sep[(run.label, "test")], run.colour) for run in all_runs},
                "within / between class spread (lower = tighter)", "T29 · class separation after every loop (test)",
                RESULTS / "separation_per_loop.png")
    (RESULTS / "summary.json").write_text(json.dumps(
        {"train": summaries, "readout": readout, "state_norms": norms,
         "separation": {f"{label} {split}": v for (label, split), v in sep.items()}}, indent=1))


if __name__ == "__main__":
    main()
