"""T3 plots + table: final test accuracy vs number of distinct blocks n (one line per pass), and
per-epoch learning curves for the n=1..4 fine-tune lr comparison (1e-4 vs 5e-4).

Run from the repo root:  python tasks/T3_progressive_unrolling/plot.py [--results-dir DIR]
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
BASELINE_JSON = TASK_DIR.parent / "T1_baseline_vs_shared_block" / "results" / "baseline.json"
# pass key -> (results subfolder, legend label, colour). Colours are fixed per pass so reruns never repaint.
PASSES = {
    "probe": ("probe", "probe: body lr 1e-6, head lr 1e-3", "tab:blue"),
    "ft": ("ft", "fine-tune: all lr 1e-4", "tab:orange"),
    "ft_lr5e-4": ("lr5e-4/ft", "fine-tune: all lr 5e-4 (n=1..4 rerun)", "tab:green"),
}
LR_COMPARE = {"ft": ("lr 1e-4", "--"), "ft_lr5e-4": ("lr 5e-4", "-")}  # pass key -> (label, line style)

log = logging.getLogger(__name__)


def load_runs(results: Path, subdir: str) -> dict[int, dict]:
    """All run JSONs in one pass folder, keyed by n."""
    return {run["args"]["unique_blocks"]: run
            for run in (json.loads(path.read_text()) for path in sorted((results / subdir).glob("unroll_u*.json")))}


def load_pass(results: Path, subdir: str) -> pd.DataFrame:
    """One row per finished run of a pass: n, params (M), final and best test acc, epochs run."""
    rows = []
    for path in sorted((results / subdir).glob("unroll_u*.json")):
        run = json.loads(path.read_text())
        rows.append({"n": run["args"]["unique_blocks"], "params (M)": round(run["params"] / 1e6, 3),
                     "final acc (%)": round(run["final_test_acc"], 2), "best acc (%)": round(run["best_test_acc"], 2),
                     "epochs": len(run["history"])})
    if not rows:
        log.warning("no runs found in %s", results / subdir)
        return pd.DataFrame(columns=["n", "params (M)", "final acc (%)", "best acc (%)", "epochs"])
    return pd.DataFrame(rows).sort_values("n").reset_index(drop=True)


def build_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Side-by-side table: n, params, then final acc for each pass."""
    keyed = [df.set_index("n")[["final acc (%)"]].rename(columns={"final acc (%)": f"{p} acc (%)"})
             for p, df in frames.items() if not df.empty]
    params = pd.concat([df.set_index("n")["params (M)"] for df in frames.values() if not df.empty])
    params = params[~params.index.duplicated()]
    return pd.concat([params, *keyed], axis=1).sort_index().reset_index()


def plot(frames: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Final test acc vs n, one line per pass, baseline (T1) as a dashed reference."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for pass_name, df in frames.items():
        if not df.empty:
            _, label, color = PASSES[pass_name]
            ax.plot(df["n"], df["final acc (%)"], marker="o", lw=2, color=color, label=label)
    if BASELINE_JSON.exists():
        base = json.loads(BASELINE_JSON.read_text())["final_test_acc"]
        ax.axhline(base, ls="--", color="gray", lw=1)
        ax.text(1, base, f"T1 baseline {base:.1f}%", va="bottom", fontsize=8, color="gray")
    else:
        log.warning("baseline reference not found at %s", BASELINE_JSON)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("n = number of distinct blocks (blocks 0..n-2 kept, block n-1 repeated)")
    ax.set_ylabel("final CIFAR-100 test acc (%)")
    ax.set_title("T3 progressive unrolling (ViT-Tiny, 10 epochs, seed 0)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)


def plot_lr_curves(results: Path, out_path: Path) -> None:
    """Test acc per epoch for each n run at both fine-tune lrs: colour = n, line style = lr."""
    runs = {key: load_runs(results, PASSES[key][0]) for key in LR_COMPARE}
    shared_n = sorted(set.intersection(*(set(r) for r in runs.values())))
    if not shared_n:
        log.warning("no n was run at both fine-tune lrs; skipping %s", out_path.name)
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, n in enumerate(shared_n):
        color = plt.cm.tab10.colors[i]
        for key, (lr_label, style) in LR_COMPARE.items():
            history = runs[key][n]["history"]
            ax.plot([e["epoch"] for e in history], [e["test_acc"] for e in history], ls=style, lw=2,
                    marker="o", ms=4, color=color, label=f"n={n}, {lr_label}")
        final = runs["ft_lr5e-4"][n]["final_test_acc"]
        gain = final - runs["ft"][n]["final_test_acc"]
        ax.annotate(f"+{gain:.1f}", (len(runs["ft_lr5e-4"][n]["history"]), final), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color="0.25")
    ax.set_xlabel("epoch")
    ax.set_ylabel("CIFAR-100 test acc (%)")
    ax.set_title("T3 fine-tune lr check: lr 5e-4 (solid) vs 1e-4 (dashed); label = final gain", fontsize=10)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.6)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)


def main() -> None:
    """Build the table and plot from whatever runs have finished so far."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS, help="folder holding probe/, ft/ and lr5e-4/ft/")
    results = parser.parse_args().results_dir
    frames = {key: load_pass(results, subdir) for key, (subdir, _, _) in PASSES.items()}
    if all(df.empty for df in frames.values()):
        raise FileNotFoundError(f"no T3 runs under {results}/{{probe,ft}}")
    table = build_table(frames).to_string(index=False)
    print(table)
    (results / "summary.md").write_text(table + "\n")
    plot(frames, results / "acc_vs_n.png")
    plot_lr_curves(results, results / "lr_compare_curves.png")
    log.info("wrote summary.md, acc_vs_n.png and lr_compare_curves.png in %s", results)


if __name__ == "__main__":
    main()
