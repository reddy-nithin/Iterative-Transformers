"""T4 plots + table: per-run loss/accuracy curves (train and test), an all-runs test-acc overview, iter b0 vs
scratch side by side, and a plateau verdict per run.

Run from the repo root:  python tasks/T4_longer_runs_lr5e-4/plot.py [--results-dir DIR]
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
TASKS = TASK_DIR.parent
# run name -> (JSON stem, legend label, colour). Colours are fixed per run so reruns never repaint.
RUNS = {
    "iter_b0": ("iterative_iter_b0", "iter b0", "tab:blue"),
    "iter_b6": ("iterative_iter_b6", "iter b6", "tab:orange"),
    "iter_avg": ("iterative_iter_avg", "iter avg", "tab:green"),
    "lora_r2": ("lora_lora_r2", "LoRA r2 (b0 base)", "tab:red"),
    "lora_r8": ("lora_lora_r8", "LoRA r8 (b0 base)", "tab:purple"),
    "scratch": ("scratch_scratch", "scratch (5.5M)", "tab:gray"),
}
# Old 10-epoch results for reference: run name -> [(JSON path, label)]. LoRA refs used the avg base, not block 0.
OLD_REFS = {
    "iter_b0": [(TASKS / "T1_baseline_vs_shared_block/results/iterative.json", "old 10 ep, lr 1e-4"),
                (TASKS / "T2_shared_block_with_lora/results/iterative_lr5e-4.json", "old 10 ep, lr 5e-4")],
    "iter_b6": [(TASKS / "T1_baseline_vs_shared_block/results/iterative_b6.json", "old 10 ep, lr 1e-4")],
    "iter_avg": [(TASKS / "T2_shared_block_with_lora/results/iterative_avg.json", "old 10 ep, lr 1e-4")],
    "lora_r2": [(TASKS / "T2_shared_block_with_lora/results/lora_r2.json", "old 10 ep, lr 1e-4, avg base")],
    "lora_r8": [(TASKS / "T2_shared_block_with_lora/results/lora_r8.json", "old 10 ep, lr 1e-4, avg base")],
    "scratch": [(TASKS / "T2_shared_block_with_lora/results/scratch.json", "old 10 ep, lr 1e-4")],
}
PLATEAU_WINDOW = 5      # epochs
PLATEAU_GAIN = 0.5      # points of test acc over the window

log = logging.getLogger(__name__)


def load_runs(results: Path) -> dict[str, dict]:
    """Every T4 run JSON found in `results`, keyed by run name."""
    runs = {}
    for name, (stem, _, _) in RUNS.items():
        path = results / f"{stem}.json"
        if path.exists():
            runs[name] = json.loads(path.read_text())
        else:
            log.warning("no results yet for %s (%s)", name, path)
    return runs


def gain_over(history: list[dict], start_epoch: int, end_epoch: int) -> float | None:
    """Test-acc change from `start_epoch` to `end_epoch` (1-based), or None if the run is too short."""
    if start_epoch < 1 or end_epoch > len(history):
        return None
    return history[end_epoch - 1]["test_acc"] - history[start_epoch - 1]["test_acc"]


def verdict(history: list[dict]) -> str:
    """'plateaued' if test acc rose less than PLATEAU_GAIN points over the last PLATEAU_WINDOW epochs."""
    n = len(history)
    gain = gain_over(history, n - PLATEAU_WINDOW, n)
    if gain is None:
        return "too short"
    return "plateaued" if gain < PLATEAU_GAIN else "still climbing"


def summarise(runs: dict[str, dict]) -> pd.DataFrame:
    """One row per run: params, final/best test acc, late gains, train-test gap, verdict."""
    rows = []
    for name, run in runs.items():
        h = run["history"]
        n = len(h)
        last = h[-1]
        late = gain_over(h, n - PLATEAU_WINDOW, n)
        mid = gain_over(h, n - 2 * PLATEAU_WINDOW, n - PLATEAU_WINDOW)
        rows.append({
            "run": name, "params (M)": round(run["params"] / 1e6, 3), "epochs": n,
            "final acc (%)": round(last["test_acc"], 2), "best acc (%)": round(run["best_test_acc"], 2),
            f"gain last {PLATEAU_WINDOW} ep": None if late is None else round(late, 2),
            f"gain {PLATEAU_WINDOW} ep before": None if mid is None else round(mid, 2),
            "train acc (%)": round(last.get("train_acc", float("nan")), 2),
            "train-test gap": round(last.get("train_acc", float("nan")) - last["test_acc"], 2),
            "verdict": verdict(h),
        })
    return pd.DataFrame(rows)


def plot_curves(runs: dict[str, dict], out_path: Path, title: str) -> None:
    """Loss (left) and accuracy (right) per epoch. Solid = test, dashed = train; colour = run."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, run in runs.items():
        _, label, color = RUNS[name]
        h = run["history"]
        epochs = [e["epoch"] for e in h]
        for key, ax, style, split in (("train_loss", ax_loss, "--", "train"), ("test_loss", ax_loss, "-", "test"),
                                      ("train_acc", ax_acc, "--", "train"), ("test_acc", ax_acc, "-", "test")):
            if key in h[0]:
                ax.plot(epochs, [e[key] for e in h], ls=style, lw=1.8, color=color, label=f"{label} {split}")
        for (ref_path, ref_label), marker in zip(OLD_REFS[name] if len(runs) == 1 else [], ("x", "+")):
            if not ref_path.exists():
                log.warning("reference run not found: %s", ref_path)
                continue
            old = json.loads(ref_path.read_text())["final_test_acc"]
            ax_acc.scatter([10], [old], marker=marker, s=70, color="black", zorder=3, label=f"{ref_label}: {old:.1f}%")
    ax_loss.set_ylabel("loss (label smoothing 0.1)")
    ax_acc.set_ylabel("CIFAR-100 acc (%)")
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_test_acc_all(runs: dict[str, dict], out_path: Path) -> None:
    """Test acc per epoch for every run, final value labelled at the line end."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, run in runs.items():
        _, label, color = RUNS[name]
        h = run["history"]
        ax.plot([e["epoch"] for e in h], [e["test_acc"] for e in h], lw=2, color=color, label=label)
        ax.annotate(f"{h[-1]['test_acc']:.1f}", (h[-1]["epoch"], h[-1]["test_acc"]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=color)
    ax.set_xlabel("epoch"); ax.set_ylabel("CIFAR-100 test acc (%)")
    ax.set_title("T4: all runs at lr 5e-4 (LoRA factors 5e-3), cosine schedule, seed 0", fontsize=10)
    ax.set_xlim(right=ax.get_xlim()[1] + 1.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main() -> None:
    """Build the table and all plots from whatever runs have finished so far."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS, help="folder holding the T4 run JSONs")
    results = parser.parse_args().results_dir
    runs = load_runs(results)
    if not runs:
        raise FileNotFoundError(f"no T4 runs under {results}")
    table = summarise(runs).to_string(index=False)
    print(table)
    (results / "summary.md").write_text(table + "\n")
    for name, run in runs.items():
        plot_curves({name: run}, results / f"curves_{name}.png",
                    f"T4 {RUNS[name][1]}: lr 5e-4, {len(run['history'])} epochs, seed 0")
    plot_test_acc_all(runs, results / "test_acc_all.png")
    pair = {k: runs[k] for k in ("iter_b0", "scratch") if k in runs}
    if len(pair) == 2:
        plot_curves(pair, results / "b0_vs_scratch.png", "T4: iter b0 (0.65M) vs scratch (5.5M), both lr 5e-4")
    log.info("wrote summary.md and plots in %s", results)


if __name__ == "__main__":
    main()
