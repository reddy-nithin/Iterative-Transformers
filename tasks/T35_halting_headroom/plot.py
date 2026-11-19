"""T35 plots from results/halting_headroom.json.

1. oracle_curve.png: accuracy vs mean loops, the oracle curve and T31-a's fixed-depth curve (loops 1..10).
2. settling_hist.png: histogram of the settling loop.
3. auroc_per_loop.png: AUROC vs loop t = 2..10, one line per signal.

Run from the repo root:  .venv/bin/python tasks/T35_halting_headroom/plot.py [--results DIR]
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
SIGNAL_LABELS = {"margin": "margin (top1 - top2 logit)", "max_softmax": "max softmax", "neg_kl": "-KL(p_t || p_t-1)",
                 "neg_angular_step": "-(1 - cos(z_t, z_t-1))", "log_cls_norm": "log raw CLS norm"}

log = logging.getLogger("t35_plot")


def load(results: Path) -> dict:
    """halting.py's output; raises if missing or incomplete."""
    path = results / "halting_headroom.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run halting.py first")
    data = json.loads(path.read_text())
    if "oracle" not in data:
        raise RuntimeError(f"{path} has no analysis (sanity check failed?)")
    return data


def plot_oracle(data: dict, out_path: Path) -> None:
    """Oracle curve and fixed-depth points."""
    curve = data["oracle"]["curve"]
    fixed = [row["top1"] for row in data["per_loop"]]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot([p[0] for p in curve], [p[1] for p in curve], "-", color="tab:purple", label="oracle (knows the label)")
    ax.plot(range(1, len(fixed) + 1), fixed, "-o", ms=4, color="tab:blue", label="T31-a fixed depth")
    ax.axvline(3, color="grey", ls=":", lw=1)
    ax.set(xlabel="mean loops per image", ylabel="test top-1 (%)", xlim=(1, len(fixed)),
           title=f"Halting headroom at 3 loops: {data['oracle']['headroom_at_3']:+.2f} pt")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_settling(data: dict, out_path: Path) -> None:
    """Settling-loop histogram."""
    histogram = data["settling"]["histogram"]
    loops = [int(t) for t in histogram]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(loops, [histogram[str(t)] for t in loops], color="tab:blue")
    ax.set(xlabel="settling loop", ylabel="images", xticks=loops,
           title=f"Settling loop (mean {data['settling']['mean']:.2f}, std {data['settling']['std']:.2f})")
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_auroc(data: dict, out_path: Path) -> None:
    """AUROC vs loop, one line per signal; pooled value in the legend."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name, per_loop in data["auroc"]["per_loop"].items():
        loops = [int(t) for t in per_loop]
        pooled = data["auroc"]["pooled_2_10"][name]
        ax.plot(loops, [per_loop[str(t)] for t in loops], "-o", ms=3,
                label=f"{SIGNAL_LABELS[name]} (pooled {pooled})")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set(xlabel="loop t", ylabel="AUROC (target: loop-t prediction correct)", xticks=range(2, 11),
           title="How well each signal predicts 'already right'")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    """All three plots."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(TASK_DIR / "results"))
    results = Path(parser.parse_args().results)
    data = load(results)
    plot_oracle(data, results / "oracle_curve.png")
    plot_settling(data, results / "settling_hist.png")
    plot_auroc(data, results / "auroc_per_loop.png")
    log.info("wrote oracle_curve.png, settling_hist.png, auroc_per_loop.png to %s", results)


if __name__ == "__main__":
    main()
