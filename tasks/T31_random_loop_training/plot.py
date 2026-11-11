"""T31 plots and tables from results/depth_sweep.json.

1. results/acc_vs_loops.png: test top-1 at fixed r = 1..20, T31-a and T11-d (both trained at 10 loops on average /
   exactly); T31-a's exit-rule points (mean loops, top-1) on top.
2. Tables: top-1 per r for both models; best gain over fixed depth per exit rule on T31-a (20-loop budget).

Run from the repo root:  .venv/bin/python tasks/T31_random_loop_training/plot.py [--results DIR]
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
COLOURS = {"T31-a": "tab:blue", "T11-d": "tab:red"}
RULE_MARKERS = {"confidence": "^", "kl": "s", "patience": "D"}

log = logging.getLogger("t31_plot")


def load(results: Path) -> dict:
    """depth_sweep.py's output; raises if missing."""
    path = results / "depth_sweep.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run depth_sweep.py first")
    return json.loads(path.read_text())


def plot_depth(result: dict, out_path: Path) -> None:
    """Top-1 vs fixed number of loops, with T31-a's exit-rule points."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for tag, model in result.items():
        curve = model["acc_per_loops"]
        ax.plot(range(1, len(curve) + 1), curve, "-o", ms=3, color=COLOURS[tag], label=f"{tag} fixed r")
        for rule, points in model.get("exit_rules_20_loops", {}).get("curves", {}).items():
            ax.scatter([p["mean_loops"] for p in points], [p["acc"] for p in points], marker=RULE_MARKERS[rule],
                       s=22, facecolors="none", edgecolors=COLOURS[tag], label=f"{tag} {rule} exit")
    ax.axvline(10, color="0.5", ls=":", lw=1, label="10 loops (trained)")
    ax.set(xlabel="loops r (fixed) / mean loops (exit rule)", ylabel="test top-1 (%)",
           title="T31 · accuracy vs number of loops at test time")
    ax.set_xticks(range(1, 21))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_tables(result: dict) -> None:
    """README tables."""
    log.info("| model | " + " | ".join(f"r={r}" for r in range(1, 21)) + " |")
    for tag, model in result.items():
        log.info(f"| {tag} | " + " | ".join(f"{a:.2f}" for a in model["acc_per_loops"]) + " |")
    for tag, model in result.items():
        for rule, summary in model.get("exit_rules_20_loops", {}).get("vs_fixed", {}).items():
            best = summary["best_gain"]
            log.info(f"| {tag} | {rule} | threshold {best['threshold']} | top-1 {best['acc']} | mean loops "
                     f"{best['mean_loops']} | gain {best['gain_vs_fixed']:+.2f} | cheapest near full: "
                     f"{summary['cheapest_near_full']} | fixed loops near full {summary['fixed_loops_near_full']} |")


def main() -> None:
    """Plot and print."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(TASK_DIR / "results"))
    results = Path(parser.parse_args().results)
    result = load(results)
    plot_depth(result, results / "acc_vs_loops.png")
    print_tables(result)


if __name__ == "__main__":
    main()
