"""T7 plot + table: iter b0 (block 0 x12, 0.65M) test accuracy vs epoch, one line per lr.

New runs (lr 1e-3, 2e-3, 5e-3, 30 epochs) come from this folder; lr 5e-4 (30 epochs) from T4 and lr 1e-4 (10 epochs
only) from T1. NaN epochs are left as gaps and flagged in the table.

Run from the repo root:  python tasks/T7_shared_block_lr_sweep/plot.py
"""
import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
EARLIER_RUNS = {  # lr label -> run JSON from an earlier task
    "1e-4 (10 ep)": TASK_DIR.parent / "T1_baseline_vs_shared_block" / "results" / "iterative.json",
    "5e-4": TASK_DIR.parent / "T4_longer_runs_lr5e-4" / "results" / "iterative_iter_b0.json",
}
NEW_LRS = ("1e-3", "2e-3", "5e-3")
FULL_EPOCHS = 30
COLORS = {"1e-4 (10 ep)": "tab:gray", "5e-4": "tab:green", "1e-3": "tab:blue", "2e-3": "tab:purple",
          "5e-3": "tab:red"}  # fixed per lr so reruns never repaint

log = logging.getLogger(__name__)


def load_runs(results: Path) -> dict[str, dict]:
    """lr label -> run JSON, for every run that exists (earlier tasks first, then this task's)."""
    paths = dict(EARLIER_RUNS) | {lr: results / f"iterative_lr{lr}.json" for lr in NEW_LRS}
    runs = {}
    for label, path in paths.items():
        if path.exists():
            runs[label] = json.loads(path.read_text())
        else:
            log.warning("no run for lr %s at %s", label, path)
    return runs


def nan_epochs(history: list[dict]) -> list[int]:
    """Epochs whose train or test loss is NaN (runs older than T4 logged no test loss)."""
    return [e["epoch"] for e in history if math.isnan(e["train_loss"]) or math.isnan(e.get("test_loss", 0.0))]


def build_table(runs: dict[str, dict]) -> pd.DataFrame:
    """One row per lr: epochs run, final and best test acc, final train acc, NaN epochs."""
    rows = []
    for label, run in runs.items():
        history = run["history"]
        rows.append({"lr": label, "epochs": len(history), "final acc (%)": round(run["final_test_acc"], 2),
                     "best acc (%)": round(run["best_test_acc"], 2),
                     "final train acc (%)": round(history[-1]["train_acc"], 2) if "train_acc" in history[-1] else None,
                     "NaN epochs": ",".join(map(str, nan_epochs(history))) or "none"})
    return pd.DataFrame(rows)


def plot(runs: dict[str, dict], out_path: Path) -> None:
    """Test acc per epoch, one line per lr."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, run in runs.items():
        history = run["history"]
        bad = set(nan_epochs(history))
        acc = [float("nan") if e["epoch"] in bad else e["test_acc"] for e in history]
        ax.plot([e["epoch"] for e in history], acc, marker="o", ms=3, lw=2, color=COLORS[label], label=f"lr {label}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("CIFAR-100 test acc (%)")
    ax.set_title("T7 iter b0 (block 0 x12, 0.65M params), lr sweep, seed 0", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> None:
    """Build the table and plot from whatever runs have finished so far."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    results = parser.parse_args().results_dir
    runs = load_runs(results)
    if not any(lr in runs for lr in NEW_LRS):
        raise FileNotFoundError(f"no T7 runs in {results}")
    table = build_table(runs)
    full_length = table[table["epochs"] == FULL_EPOCHS]
    if full_length.empty:
        verdict = f"no finished {FULL_EPOCHS}-epoch run yet"
    else:
        best = full_length.loc[full_length["final acc (%)"].idxmax()]
        verdict = f"best lr ({FULL_EPOCHS} epochs): {best['lr']} at {best['final acc (%)']}%"
    text = table.to_string(index=False) + f"\n\n{verdict}\n"
    print(text)
    (results / "summary.md").write_text(text)
    plot(runs, results / "acc_vs_epoch_by_lr.png")
    log.info("wrote summary.md and acc_vs_epoch_by_lr.png in %s", results)


if __name__ == "__main__":
    main()
