"""T38 loss curves: T38-a (2+10, looped block starts as the mean of blocks 2..11) vs T11-d (2+10, starts as block 2).

Saves results/loss_curves.png and prints the numbers used in the README table.

Run from the repo root:  .venv/bin/python tasks/T38_mean_init_body/plot.py
"""
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
T11D_JSON = TASK_DIR.parent / "T11_pretrained_hbt_body_sweep" / "results" / "unroll_T11-d.json"
T11D_3SEED_MEAN = 73.79
RUNS = {"T38-a mean start": (RESULTS / "unroll_T38-a.json", "#d62728")}

log = logging.getLogger("t38_plot")


def load_run(path: Path) -> dict:
    """Load one src.train results JSON; raise if it is missing or has no 30-epoch history."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    data = json.loads(path.read_text())
    if len(data.get("history", [])) != 30:
        raise ValueError(f"{path}: expected 30 epochs of history, got {len(data.get('history', []))}")
    return data


def summarize(name: str, data: dict) -> None:
    """Print the README table numbers for one run."""
    hist = data["history"]
    best_acc, best_ep = max((r["test_acc"], r["epoch"]) for r in hist)
    log.info("%s: params=%s GFLOPs=%.3f ep1=%.2f ep10=%.2f ep30=%.2f best=%.2f (ep %d) diff_vs_73.79=%+.2f",
             name, f"{data['params']:,}", data["gflops"], hist[0]["test_acc"], hist[9]["test_acc"],
             hist[-1]["test_acc"], best_acc, best_ep, hist[-1]["test_acc"] - T11D_3SEED_MEAN)


def plot_run(ax_loss, ax_acc, data: dict, label: str, color: str) -> None:
    """Draw one run's train/test loss and test accuracy."""
    hist = data["history"]
    epochs = [r["epoch"] for r in hist]
    ax_loss.plot(epochs, [r["train_loss"] for r in hist], color=color, ls="--", label=f"{label} train")
    ax_loss.plot(epochs, [r["test_loss"] for r in hist], color=color, label=f"{label} test")
    ax_acc.plot(epochs, [r["test_acc"] for r in hist], color=color, label=label)


def main() -> None:
    """Load T38-a and T11-d histories and save the two-panel figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))
    reference = load_run(T11D_JSON)
    summarize("T11-d 2+10 (seed 0)", reference)
    plot_run(ax_loss, ax_acc, reference, "T11-d 2+10, block-2 start", "#1f77b4")
    for label, (path, color) in RUNS.items():
        data = load_run(path)
        summarize(label, data)
        plot_run(ax_loss, ax_acc, data, label, color)
    ax_acc.axhline(T11D_3SEED_MEAN, ls=":", color="black", lw=1, label="T11-d 3-seed mean (73.79)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (cross-entropy, train with label smoothing 0.1)")
    ax_loss.legend(fontsize=7)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%) on CIFAR-100 test")
    ax_acc.legend(fontsize=8)
    fig.suptitle("T38: 2+10, looped block started as the mean of blocks 2..11 vs block 2 (T11-d); 30 ep, seed 0")
    fig.tight_layout()
    out = RESULTS / "loss_curves.png"
    fig.savefig(out, dpi=150)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
