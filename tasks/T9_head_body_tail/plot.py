"""T9 plots + table: head + repeated body + tail (from scratch, lr 5e-4, 30 epochs).

1. Test accuracy vs epoch for every T9 run, with the 12-block scratch model and iter b0 (T4, same lr and epochs)
   as references.
2. Class separation of the CLS token after every loop (tr within / tr between, 4 classes, from cls_svd.py), train and
   test, with the fine-tuned 12-block ViT and iter b0 (T5) as reference lines.

Run from the repo root:  python tasks/T9_head_body_tail/plot.py [--results-dir DIR]
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
T4_RESULTS = TASK_DIR.parent / "T4_longer_runs_lr5e-4" / "results"
T5_RESULTS = TASK_DIR.parent / "T5_cls_svd_per_block" / "results"
# run -> (legend label, colour); fixed so reruns never repaint
RUNS = {
    "hbt": ("head + body x10 + tail", "tab:blue"),
    "hbt_inject": ("head + body x10 + tail, input injection", "tab:purple"),
    "head_only": ("head + body x11", "tab:orange"),
    "loop_only": ("body x12 (one random block)", "tab:red"),
}
ACC_REFERENCES = {  # label -> (T4 run JSON, colour)
    "scratch, 12 distinct blocks (5.5M)": (T4_RESULTS / "scratch_scratch.json", "0.45"),
    "iter b0, pretrained block 0 x12": (T4_RESULTS / "iterative_iter_b0.json", "tab:green"),
}
SEPARATION_REFERENCES = {  # label -> (T5 cls_svd tag without split, colour)
    "fine-tuned 12-block ViT": ("ft_baseline", "0.45"),
    "iter b0 (pretrained init)": ("iter_b0", "tab:green"),
}
SEPARATION_SET = "all classes"

log = logging.getLogger(__name__)


def load_json(path: Path) -> dict | None:
    """A JSON file, or None (with a warning) if it is missing."""
    if not path.exists():
        log.warning("missing %s", path)
        return None
    return json.loads(path.read_text())


def separation_curve(cls_svd: dict) -> list[float | None]:
    """Separation after loop 1..12 (drops layer 0, where all CLS vectors are equal, and the final-norm layer)."""
    by_layer = cls_svd["separation"][SEPARATION_SET]
    last_loop = cls_svd["final_norm_layer"] - 1
    return [by_layer[str(layer)] for layer in range(1, last_loop + 1)]


def minimum(curve: list[float | None]) -> tuple[float, int]:
    """(smallest value, loop where it happens, 1-based)."""
    values = [(v, loop) for loop, v in enumerate(curve, 1) if v is not None]
    return min(values)


def build_table(results: Path) -> pd.DataFrame:
    """One row per finished T9 run: params, final / best acc, separation minimum (train, test) and where."""
    rows = []
    for run, (label, _) in RUNS.items():
        train = load_json(results / f"hbt_{run}.json")
        if train is None:
            continue
        row = {"run": run, "layout": label, "params (M)": round(train["params"] / 1e6, 3),
               "epochs": len(train["history"]), "final acc (%)": round(train["final_test_acc"], 2),
               "best acc (%)": round(train["best_test_acc"], 2)}
        for split in ("train", "test"):
            cls_svd = load_json(results / f"cls_svd_{run}_{split}.json")
            if cls_svd is not None:
                value, loop = minimum(separation_curve(cls_svd))
                row[f"min separation {split} (loop)"] = f"{value:.2f} ({loop})"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_accuracy(results: Path, out_path: Path) -> None:
    """Test acc per epoch: T9 runs (solid) and the two T4 references (dashed)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for run, (label, colour) in RUNS.items():
        train = load_json(results / f"hbt_{run}.json")
        if train is not None:
            history = train["history"]
            ax.plot([e["epoch"] for e in history], [e["test_acc"] for e in history], lw=2, marker="o", ms=3,
                    color=colour, label=f"{label} ({train['params'] / 1e6:.2f}M)")
    for label, (path, colour) in ACC_REFERENCES.items():
        reference = load_json(path)
        if reference is not None:
            history = reference["history"]
            ax.plot([e["epoch"] for e in history], [e["test_acc"] for e in history], lw=1.5, ls="--",
                    color=colour, label=f"{label}: {reference['final_test_acc']:.1f}%")
    ax.set_xlabel("epoch")
    ax.set_ylabel("CIFAR-100 test acc (%)")
    ax.set_title("T9 head / body / tail from scratch (lr 5e-4, 30 epochs, seed 0)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_separation(results: Path, out_path: Path) -> None:
    """Separation vs loop, train and test panels; below 1 = classes tighter than the distances between them."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, split in zip(axes, ("train", "test")):
        for run, (label, colour) in RUNS.items():
            cls_svd = load_json(results / f"cls_svd_{run}_{split}.json")
            if cls_svd is not None:
                curve = separation_curve(cls_svd)
                ax.plot(range(1, len(curve) + 1), curve, lw=2, marker="o", ms=3, color=colour, label=label)
        for label, (tag, colour) in SEPARATION_REFERENCES.items():
            cls_svd = load_json(T5_RESULTS / f"cls_svd_{tag}_{split}.json")
            if cls_svd is not None:
                curve = separation_curve(cls_svd)
                ax.plot(range(1, len(curve) + 1), curve, lw=1.5, ls="--", color=colour, label=label)
        ax.axhline(1.0, color="black", lw=0.8, ls=":")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("loop / block")
        ax.set_title(f"{split} images (4 classes)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("separation: tr(within) / tr(between)")
    axes[1].legend(fontsize=7)
    fig.suptitle("T9 class tightness of the CLS token after each loop", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Build the table and plots from whatever runs have finished so far."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    results = parser.parse_args().results_dir
    table = build_table(results)
    if table.empty:
        raise FileNotFoundError(f"no T9 runs in {results}")
    text = table.to_string(index=False)
    print(text)
    (results / "summary.md").write_text(text + "\n")
    plot_accuracy(results, results / "acc_vs_epoch.png")
    plot_separation(results, results / "separation_vs_loop.png")
    log.info("wrote summary.md, acc_vs_epoch.png and separation_vs_loop.png in %s", results)


if __name__ == "__main__":
    main()
