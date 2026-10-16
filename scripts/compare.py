"""Build results table + plots from run JSONs.

Writes summary.md, accuracy.png (per epoch), acc_vs_params.png, final_acc_bars.png into --out-dir.
Example:  python -m scripts.compare --results-dir tasks/T1_baseline_vs_shared_block/results \
              tasks/T2_shared_block_with_lora/results --out-dir tasks/T2_shared_block_with_lora/results
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SMALL = 1.5e6  # shared-block runs are below this; they get labels only in the zoom panel
FAMILIES = [  # (name prefix, legend label, colour) for the bar chart; first match wins
    ("baseline", "pretrained, 12 blocks", "tab:blue"),
    ("scratch", "random init, 12 blocks", "tab:gray"),
    ("iterative_lr", "shared block, higher lr", "tab:green"),
    ("iterative", "shared block", "tab:cyan"),
    ("lora", "shared block + per-loop LoRA", "tab:orange"),
]
OTHER_FAMILY = ("", "other", "tab:purple")


def parse_args() -> argparse.Namespace:
    """Folders to read run JSONs from, and where to write the table and plots."""
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", nargs="+", default=["results"], help="one or more folders of run JSONs")
    p.add_argument("--out-dir", default=None, help="where summary.md and plots go (default: first --results-dir)")
    return p.parse_args()


def load_runs(dirs: list[str]) -> dict[str, dict]:
    """All run JSONs from `dirs`, keyed by file stem, sorted by name."""
    paths = sorted((p for d in dirs for p in Path(d).glob("*.json")), key=lambda p: p.stem)
    if not paths:
        raise FileNotFoundError(f"no *.json run files in {dirs}")
    return {p.stem: json.loads(p.read_text()) for p in paths}


def run_colors(runs: dict[str, dict]) -> dict[str, tuple]:
    """One fixed colour per run, reused by every plot so a run looks the same everywhere."""
    return {k: plt.cm.tab20.colors[i % 20] for i, k in enumerate(runs)}


def family(name: str) -> tuple[str, str, str]:
    """(prefix, legend label, colour) of the model family a run belongs to."""
    return next((f for f in FAMILIES if name.startswith(f[0])), OTHER_FAMILY)


def write_table(runs: dict[str, dict], out_dir: Path) -> None:
    """Print the summary table and save it as summary.md."""
    rows = [{"model": k, "params (M)": round(v["params"] / 1e6, 3), "GFLOPs": round(v["gflops"], 2),
             "throughput (img/s)": round(v["throughput_img_s"]), "final acc (%)": round(v["final_test_acc"], 2),
             "best acc (%)": round(v["best_test_acc"], 2), "epochs": len(v["history"])} for k, v in runs.items()]
    table = pd.DataFrame(rows).to_string(index=False)
    print(table)
    (out_dir / "summary.md").write_text(table + "\n")


def plot_accuracy(runs: dict[str, dict], colors: dict[str, tuple], out_dir: Path) -> None:
    """Test accuracy per epoch, one line per run."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, v in runs.items():
        ax.plot([h["epoch"] for h in v["history"]], [h["test_acc"] for h in v["history"]],
                marker="o", label=k, color=colors[k])
    ax.set_xlabel("epoch"); ax.set_ylabel("CIFAR-100 test acc (%)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(); fig.savefig(out_dir / "accuracy.png", dpi=150)


def plot_acc_vs_params(runs: dict[str, dict], colors: dict[str, tuple], out_dir: Path) -> None:
    """Accuracy vs parameter count; baseline and scratch drawn as dashed reference lines.

    Left: all runs (log x). Right: zoom on the shared-block runs (< 1.5M params) so their labels don't overlap.
    """
    refs = {k: v["final_test_acc"] for k, v in runs.items() if k in ("baseline", "scratch")}
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, v in runs.items():
        x, y = v["params"] / 1e6, v["final_test_acc"]
        ax.scatter(x, y, color=colors[k])
        if v["params"] >= SMALL:
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(5, 2), fontsize=7)
        else:
            zoom.scatter(x, y, color=colors[k])
            zoom.annotate(k, (x, y), textcoords="offset points", xytext=(5, 2), fontsize=7)
    small_accs = [v["final_test_acc"] for v in runs.values() if v["params"] < SMALL]
    zoom_top = max(small_accs) + 3 if small_accs else 100
    for k, y in refs.items():
        ax.axhline(y, ls="--", color="gray", lw=1)
        if y < zoom_top:  # baseline sits far above the zoomed range
            zoom.axhline(y, ls="--", color="gray", lw=1)
            zoom.text(0.99, y, k, transform=zoom.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color="gray")
    for a in (ax, zoom):
        a.set_ylabel("final CIFAR-100 test acc (%)"); a.grid(alpha=0.3)
    ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xticks([0.5, 1, 2, 5], ["0.5", "1", "2", "5"])
    ax.set_xlabel("params (M, log scale)"); ax.set_title("all runs", fontsize=9)
    zoom.set_xlabel("params (M)"); zoom.set_title("shared-block runs (zoom)", fontsize=9)
    zoom.set_ylim(top=zoom_top)
    fig.tight_layout(); fig.savefig(out_dir / "acc_vs_params.png", dpi=150)


def plot_final_bars(runs: dict[str, dict], out_dir: Path) -> None:
    """One bar per run, sorted by accuracy and coloured by kind of model: the quickest read of what moved accuracy."""
    order = sorted(runs, key=lambda k: runs[k]["final_test_acc"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, k in enumerate(order):
        acc = runs[k]["final_test_acc"]
        ax.barh(i, acc, color=family(k)[2])
        ax.text(acc + 0.8, i, f"{acc:.1f}%  ({runs[k]['params'] / 1e6:.2f}M)", va="center", fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xlim(0, 100); ax.set_xlabel("final CIFAR-100 test acc (%)"); ax.grid(axis="x", alpha=0.3)
    used = {family(k)[1]: family(k)[2] for k in order}
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in used.values()], labels=list(used),
              fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out_dir / "final_acc_bars.png", dpi=150)


def main() -> None:
    """Load runs, write the table and all three plots."""
    args = parse_args()
    out_dir = Path(args.out_dir or args.results_dir[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results_dir)
    colors = run_colors(runs)
    write_table(runs, out_dir)
    plot_accuracy(runs, colors, out_dir)
    plot_acc_vs_params(runs, colors, out_dir)
    plot_final_bars(runs, out_dir)


if __name__ == "__main__":
    main()
