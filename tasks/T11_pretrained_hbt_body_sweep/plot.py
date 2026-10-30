"""T11 plots + table: pretrained head + body x10 + tail with body block 0 / 6 / 1, and unrolling n=3 at 30 epochs.

1. Train and test loss vs epoch for every run (results/loss_curves.png).
2. Test accuracy vs epoch (results/acc_vs_epoch.png).
3. Class separation of the CLS token after every block call (tr within / tr between, 4 classes, from cls_svd.py),
   train and test, with the fine-tuned 12-block ViT, iter b0 and the from-scratch T9 hbt as references
   (results/separation_vs_loop.png).

Run from the repo root:  python tasks/T11_pretrained_hbt_body_sweep/plot.py [--results-dir DIR]
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
T5_RESULTS = TASK_DIR.parent / "T5_cls_svd_per_block" / "results"
T9_RESULTS = TASK_DIR.parent / "T9_head_body_tail" / "results"
# run -> (training JSON name, legend label, colour); fixed so reruns never repaint
RUNS = {
    "T11-a": ("hbt_T11-a.json", "T11-a: head b0, body b0 x10, tail b11", "tab:blue"),
    "T11-b": ("hbt_T11-b.json", "T11-b: head b0, body b6 x10, tail b11", "tab:orange"),
    "T11-c": ("hbt_T11-c.json", "T11-c: head b0, body b1 x10, tail b11", "tab:purple"),
    "T11-d": ("unroll_T11-d.json", "T11-d: unrolling n=3 (b0, b1, b2 x10), no tail", "tab:red"),
}
# reference rows for the table: label -> (params in M, final test top-1), numbers from earlier tasks
REFERENCE_ROWS = {
    "baseline, 12 distinct blocks, fine-tuned (T5-a)": (5.544, 86.77),
    "iter b0, block 0 x12, 30 ep (T4/T5-b)": (0.650, 58.06),
    "per-loop LoRA r8 on b0 (T4)": (0.945, 61.33),
    "head + body x10 + tail from scratch (T9-a)": (1.540, 44.87),
    "unrolling n=3, lr 5e-4, 10 ep (T3)": (1.540, 67.12),
}
SEPARATION_REFERENCES = {  # label -> (cls_svd JSON without the split suffix, colour)
    "fine-tuned 12-block ViT (T5-a)": (T5_RESULTS / "cls_svd_ft_baseline", "0.45"),
    "iter b0 (T5-b)": (T5_RESULTS / "cls_svd_iter_b0", "tab:green"),
    "hbt from scratch (T9-a)": (T9_RESULTS / "cls_svd_hbt", "tab:brown"),
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
    """Separation after block call 1..12 (drops layer 0 and the final-norm layer)."""
    by_layer = cls_svd["separation"][SEPARATION_SET]
    last_call = cls_svd["final_norm_layer"] - 1
    return [by_layer[str(layer)] for layer in range(1, last_call + 1)]


def run_row(results: Path, run: str) -> dict | None:
    """Table row for one finished T11 run: params, final / best top-1, separation after the last block call (test)
    and after the last body loop (test; call 11 for H-B-T, call 12 for T11-d, which has no tail)."""
    json_name, label, _ = RUNS[run]
    train = load_json(results / json_name)
    if train is None:
        return None
    row = {"run": run, "layout": label, "params (M)": round(train["params"] / 1e6, 3),
           "epochs": len(train["history"]), "final top-1": round(train["final_test_acc"], 2),
           "best top-1": round(train["best_test_acc"], 2)}
    cls_svd = load_json(results / f"cls_svd_{run}_test.json")
    if cls_svd is not None:
        curve = separation_curve(cls_svd)
        last_body_call = len(curve) if run == "T11-d" else len(curve) - 1
        row["sep last block (test)"] = round(curve[-1], 2)
        row["sep last body loop (test)"] = round(curve[last_body_call - 1], 2)
    return row


def build_table(results: Path) -> pd.DataFrame:
    """The finished T11 runs plus the reference rows."""
    rows = [row for run in RUNS if (row := run_row(results, run)) is not None]
    rows += [{"run": "ref", "layout": label, "params (M)": params, "final top-1": acc}
             for label, (params, acc) in REFERENCE_ROWS.items()]
    return pd.DataFrame(rows)


def plot_history(results: Path, out_path: Path, keys: tuple[str, ...], ylabel: str, title: str) -> None:
    """One line per run and history key vs epoch (first key solid, second dashed)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for run, (json_name, label, colour) in RUNS.items():
        train = load_json(results / json_name)
        if train is None:
            continue
        epochs = [e["epoch"] for e in train["history"]]
        for key, style in zip(keys, ("-", "--")):
            suffix = f" ({key.split('_')[0]})" if len(keys) > 1 else ""
            ax.plot(epochs, [e[key] for e in train["history"]], lw=2 if style == "-" else 1.5, ls=style,
                    marker="o" if style == "-" else None, ms=3, color=colour, label=f"{label}{suffix}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_separation(results: Path, out_path: Path) -> None:
    """Separation vs block call, train and test panels; below 1 = classes tighter than the distances between them."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, split in zip(axes, ("train", "test")):
        for run, (_, label, colour) in RUNS.items():
            cls_svd = load_json(results / f"cls_svd_{run}_{split}.json")
            if cls_svd is not None:
                curve = separation_curve(cls_svd)
                ax.plot(range(1, len(curve) + 1), curve, lw=2, marker="o", ms=3, color=colour, label=label)
        for label, (stem, colour) in SEPARATION_REFERENCES.items():
            cls_svd = load_json(Path(f"{stem}_{split}.json"))
            if cls_svd is not None:
                curve = separation_curve(cls_svd)
                ax.plot(range(1, len(curve) + 1), curve, lw=1.5, ls="--", color=colour, label=label)
        ax.axhline(1.0, color="black", lw=0.8, ls=":")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("block call (H-B-T: 1 = head, 2-11 = body loops, 12 = tail)")
        ax.set_title(f"{split} images (4 classes)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("separation: tr(within) / tr(between)")
    axes[1].legend(fontsize=7)
    fig.suptitle("T11 class tightness of the CLS token after each block call", fontsize=10)
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
    text = table.to_string(index=False)
    print(text)
    (results / "summary.md").write_text(text + "\n")
    recipe = "pretrained init, lr 5e-4, 30 epochs, seed 0"
    plot_history(results, results / "loss_curves.png", ("train_loss", "test_loss"), "loss (label smoothing 0.1)",
                 f"T11 train (solid) and test (dashed) loss ({recipe})")
    plot_history(results, results / "acc_vs_epoch.png", ("test_acc",), "CIFAR-100 test top-1 (%)",
                 f"T11 test accuracy per epoch ({recipe})")
    plot_separation(results, results / "separation_vs_loop.png")
    log.info("wrote summary.md, loss_curves.png, acc_vs_epoch.png and separation_vs_loop.png in %s", results)


if __name__ == "__main__":
    main()
