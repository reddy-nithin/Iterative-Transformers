"""T30 results: per-loop center loss (T30-a) vs T13-c.

Tables: params / epoch time / top-1, per-loop readout, separation per loop (test, train), the center term per epoch,
and T25's analysis on both models: (a) NCC vs readout per loop, (b) best gain over fixed depth per exit rule.
Plots: results/acc_per_loop.png, results/separation_per_loop.png, results/class_cloud_best_gain.png.
Inputs: results/{unroll_T30-a.json, exit_analysis.json, cls_svd_T30-a_*.json, class_cloud_T30-a.json}; T13-c's files
in T13 and T25 (tasks/T25_class_cloud_exit/results/class_cloud_T13-c.json).

Run from the repo root:  PYTHONPATH=. .venv/bin/python tasks/T30_per_loop_center_loss/plot.py
"""
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
TASKS = TASK_DIR.parent
sys.path.insert(0, str(TASKS / "T28_frozen_patch_cls_loop"))

from compare import (T13C, RunFiles, load_json, markdown_row, per_loop_readout, plot_curves,  # noqa: E402
                     separation_per_loop, train_summary)

T30A = RunFiles("T30-a center loss", RESULTS / "unroll_T30-a.json", RESULTS / "exit_analysis.json",
                "T30-a_center_loss", RESULTS / "cls_svd_T30-a", "tab:blue")
CLASS_CLOUD = {T13C.label: TASKS / "T25_class_cloud_exit/results/class_cloud_T13-c.json",
               T30A.label: RESULTS / "class_cloud_T30-a.json"}
T13_BEST_RULE_GAIN = 0.79  # T13's best rule on T13-c (T25 README baseline)

log = logging.getLogger("t30_plot")


def plot_best_gain(clouds: dict[str, dict], out_path: Path) -> None:
    """Best gain over fixed depth for each class-cloud rule, both models (bars)."""
    rules = list(next(iter(clouds.values()))["best_gain"])
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    width = 0.8 / len(clouds)
    for index, (label, cloud) in enumerate(clouds.items()):
        gains = [cloud["best_gain"][rule]["gain_vs_fixed"] for rule in rules]
        ax.bar([r + index * width for r in range(len(rules))], gains, width, label=label,
               color=T30A.colour if label == T30A.label else T13C.colour)
    ax.axhline(T13_BEST_RULE_GAIN, color="0.4", ls="--", lw=1, label=f"T13 best rule on T13-c (+{T13_BEST_RULE_GAIN})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([r + 0.4 - width / 2 for r in range(len(rules))], rules, rotation=20, fontsize=8)
    ax.set(ylabel="best top-1 gain over fixed depth (pt)", title="T30 · class-cloud exit (T25 rules, train-set centres)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Tables and plots."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    runs = (T13C, T30A)
    summaries = {run.label: train_summary(run) for run in runs}
    readout = {run.label: per_loop_readout(run) for run in runs}
    sep = {(run.label, split): separation_per_loop(run, split) for run in runs for split in ("test", "train")}
    clouds = {label: load_json(path) for label, path in CLASS_CLOUD.items()}
    history = load_json(T30A.train_json)["history"]
    log.info(markdown_row(["model", "params", "median epoch s", "final top-1", "best top-1"]))
    for run in runs:
        s = summaries[run.label]
        log.info(markdown_row([run.label, f"{s['params']:,}", s["median_epoch_s"], s["final_top1"], s["best_top1"]]))
    for run in runs:
        log.info(markdown_row([f"{run.label} readout", *readout[run.label]]))
        log.info(markdown_row([f"{run.label} sep. test", *sep[(run.label, "test")]]))
        log.info(markdown_row([f"{run.label} sep. train", *sep[(run.label, "train")]]))
    log.info("center term per epoch (alpha): " + " ".join(
        f"{h['epoch']}:{h['train_center_loss']:.3f}({h['center_alpha']:g})" for h in history))
    for label, cloud in clouds.items():
        for row in cloud["ncc"]:
            log.info(markdown_row([label, row["loop"], row["ncc_per_loop"], row["ncc_shared"], row["readout"],
                                   row["mismatch"]]))
        for rule, point in cloud["best_gain"].items():
            log.info(markdown_row([label, rule, point]))
    plot_curves({run.label: (readout[run.label], run.colour) for run in runs}, "test top-1 (%)",
                "T30 · readout after every loop", RESULTS / "acc_per_loop.png")
    plot_curves({f"{run.label} {split}": (sep[(run.label, split)], run.colour) for run in runs for split in ("test",)},
                "within / between class spread (lower = tighter)", "T30 · class separation after every loop (test)",
                RESULTS / "separation_per_loop.png")
    plot_best_gain(clouds, RESULTS / "class_cloud_best_gain.png")
    (RESULTS / "summary.json").write_text(json.dumps(
        {"train": summaries, "readout": readout,
         "separation": {f"{label} {split}": v for (label, split), v in sep.items()},
         "center_per_epoch": [h["train_center_loss"] for h in history],
         "class_cloud_best_gain": {label: cloud["best_gain"] for label, cloud in clouds.items()}}, indent=1))


if __name__ == "__main__":
    main()
