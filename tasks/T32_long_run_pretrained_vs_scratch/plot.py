"""T32 plots and summary numbers: 2+10 for 100 epochs, pretrained (T32-a) vs scratch (T32-b).

1. results/acc_vs_epoch_100ep.png: test top-1 vs epoch, both runs, T11-d's 74.11 as a dotted reference line.
2. results/loss_vs_epoch_100ep.png: train and test loss vs epoch, both runs.
3. results/gap_vs_epoch.png: g(e) = T32-a top-1 minus T32-b top-1, with a line at 1 pt.
4. results/lr_vs_epoch.png: learning rate per epoch (the cosine stretched over 100 epochs).

Also prints the numbers for the README results table.

Inputs: results/unroll_T32-a.json and results/hbt_T32-b.json (per-epoch history from src/train.py).

Run from the repo root:  .venv/bin/python tasks/T32_long_run_pretrained_vs_scratch/plot.py
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
T11D_TOP1 = 74.11
GAP_THRESHOLD = 1.0
REPORT_EPOCHS = (10, 30, 50, 75, 100)

log = logging.getLogger("t32_plot")


@dataclass(frozen=True)
class Run:
    """One training run: its history file and how it is drawn."""
    label: str
    path: Path
    colour: str


RUNS = (
    Run("T32-a pretrained init", RESULTS / "unroll_T32-a.json", "tab:blue"),
    Run("T32-b random init (scratch)", RESULTS / "hbt_T32-b.json", "tab:orange"),
)


def load_history(path: Path) -> list[dict]:
    """Per-epoch history of one run; raises if the file is missing or incomplete."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run run.sh first")
    history = json.loads(path.read_text())["history"]
    if len(history) != 100:
        raise ValueError(f"{path} has {len(history)} epochs, expected 100")
    return history


def series(history: list[dict], key: str) -> list[float]:
    """One per-epoch quantity as a list (index 0 = epoch 1)."""
    return [record[key] for record in history]


def finish(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Axis labels, title, grid, legend."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_accuracy(histories: dict[str, list[dict]], out_path: Path) -> None:
    """Test top-1 vs epoch for both runs, with the T11-d reference line."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    epochs = range(1, 101)
    for run in RUNS:
        ax.plot(epochs, series(histories[run.label], "test_acc"), color=run.colour, label=run.label)
    ax.axhline(T11D_TOP1, color="0.4", lw=1, ls=":", label="T11-d, 30-ep run (different lr schedule)")
    finish(ax, "epoch", "test top-1 (%)", "T32 · 2+10, 100 epochs: test top-1")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss(histories: dict[str, list[dict]], out_path: Path) -> None:
    """Train loss (dashed, includes label smoothing) and test loss (solid) for both runs."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    epochs = range(1, 101)
    for run in RUNS:
        history = histories[run.label]
        ax.plot(epochs, series(history, "test_loss"), color=run.colour, label=f"{run.label}, test")
        ax.plot(epochs, series(history, "train_loss"), color=run.colour, ls="--",
                label=f"{run.label}, train (label smoothing 0.1)")
    finish(ax, "epoch", "cross-entropy loss", "T32 · train and test loss")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def gap_per_epoch(histories: dict[str, list[dict]]) -> list[float]:
    """g(e) = T32-a top-1 minus T32-b top-1, per epoch."""
    pretrained = series(histories[RUNS[0].label], "test_acc")
    scratch = series(histories[RUNS[1].label], "test_acc")
    return [a - b for a, b in zip(pretrained, scratch)]


def plot_gap(gap: list[float], out_path: Path) -> None:
    """g(e) vs epoch with the 1-pt catch-up line."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(range(1, 101), gap, color="tab:purple", label="g(e) = T32-a − T32-b")
    ax.axhline(GAP_THRESHOLD, color="0.4", lw=1, ls="--", label="1 pt (caught up)")
    ax.set_ylim(bottom=0)
    finish(ax, "epoch", "gap (top-1 points)", "T32 · pretrained minus scratch")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lr(history: list[dict], out_path: Path) -> None:
    """Learning rate at each epoch's last step (identical for both runs)."""
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    lr = series(history, "lr")
    ax.plot(range(1, 101), lr, color="0.2", label="lr (end of epoch)")
    ax.axvline(30, color="tab:red", lw=1, ls=":", label=f"epoch 30: lr = {lr[29] / lr[0]:.0%} of max")
    finish(ax, "epoch", "learning rate", "T32 · 1 warmup epoch + cosine over 100 epochs")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_summary(histories: dict[str, list[dict]], gap: list[float]) -> None:
    """Numbers for the README results table."""
    for run in RUNS:
        acc = series(histories[run.label], "test_acc")
        best = max(acc)
        picks = ", ".join(f"ep{e} {acc[e - 1]:.2f}" for e in REPORT_EPOCHS)
        print(f"{run.label}: {picks}; best {best:.2f} at ep{acc.index(best) + 1}; "
              f"last-10 gain {acc[99] - acc[89]:+.2f}")
    below = [epoch for epoch, value in enumerate(gap, start=1) if value < GAP_THRESHOLD]
    print("gap at", ", ".join(f"ep{e} {gap[e - 1]:.2f}" for e in REPORT_EPOCHS))
    print(f"min gap {min(gap):.2f} at ep{gap.index(min(gap)) + 1}; max gap {max(gap):.2f} at ep{gap.index(max(gap)) + 1}")
    print("first epoch with g < 1 pt:", below[0] if below else "never")


def main() -> None:
    """Load both histories, draw the four plots, print the summary."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    histories = {run.label: load_history(run.path) for run in RUNS}
    gap = gap_per_epoch(histories)
    plot_accuracy(histories, RESULTS / "acc_vs_epoch_100ep.png")
    plot_loss(histories, RESULTS / "loss_vs_epoch_100ep.png")
    plot_gap(gap, RESULTS / "gap_vs_epoch.png")
    plot_lr(histories[RUNS[0].label], RESULTS / "lr_vs_epoch.png")
    log.info("wrote 4 plots to %s", RESULTS)
    print_summary(histories, gap)


if __name__ == "__main__":
    main()
