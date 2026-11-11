"""T26 plot and README tables from results/cls_start.json.

cls_start_gap_per_stage.png: g_s (relative distance of the CLS to the learned-start CLS) vs stage, one line per start
(noise starts: mean over 5 seeds with a min-max band), one panel per model.

Run from the repo root:  .venv/bin/python tasks/T26_cls_start_vector/plot.py [--results DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
NOISE_SEEDS = (0, 1, 2, 3, 4)
SINGLE_STARTS = {"zeros": "tab:red", "swap": "tab:green"}
SEEDED_STARTS = {"noise": "tab:blue", "learned+noise": "tab:purple"}


def load(results: Path) -> dict:
    """The cls_start.py output; raises if it is missing."""
    path = results / "cls_start.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run cls_start.py first")
    return json.loads(path.read_text())


def seeded(result: dict, kind: str, key: str) -> np.ndarray:
    """(seeds, values) array of `key` for the seeded starts of one kind."""
    return np.array([result["starts"][f"{kind} s{seed}"][key] for seed in NOISE_SEEDS])


def plot_gap(results: dict, out_path: Path) -> None:
    """g_s per stage for every start, one panel per model."""
    fig, axes = plt.subplots(1, len(results), figsize=(13, 4.5), constrained_layout=True, sharey=True)
    for ax, (tag, result) in zip(axes, results.items()):
        stages = result["stages"]
        x = np.arange(len(stages))
        for name, colour in SINGLE_STARTS.items():
            ax.plot(x, result["starts"][name]["gap_per_stage"], "-o", ms=3, color=colour, label=name)
        for kind, colour in SEEDED_STARTS.items():
            gaps = seeded(result, kind, "gap_per_stage")
            ax.plot(x, gaps.mean(axis=0), "-s", ms=3, color=colour, label=f"{kind} (mean of 5 seeds)")
            ax.fill_between(x, gaps.min(axis=0), gaps.max(axis=0), color=colour, alpha=0.2)
        ax.axhline(0.1, color="0.5", ls=":", lw=1, label="0.1")
        ax.set_xticks(x, stages, rotation=45, ha="right")
        ax.set(ylabel="g_s = ‖CLS(alt) − CLS(learned)‖ / ‖CLS(learned)‖", title=f"{tag}: gap to the learned start")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summary_rows(result: dict) -> list[tuple[str, float, list[float]]]:
    """(start name, final top-1, gap per stage) for the README table; seeded starts as means over seeds."""
    starts = result["starts"]
    rows = [("learned", starts["learned"]["final"], starts["learned"]["gap_per_stage"]),
            ("zeros", starts["zeros"]["final"], starts["zeros"]["gap_per_stage"])]
    for kind in SEEDED_STARTS:
        summary = result[f"{kind}_final"]
        rows.append((f"{kind} (5 seeds, spread {summary['spread']})", summary["mean"],
                     seeded(result, kind, "gap_per_stage").mean(axis=0).tolist()))
    rows.append(("swap (other model's token)", starts["swap"]["final"], starts["swap"]["gap_per_stage"]))
    return rows


def print_summary(results: dict) -> None:
    """Tables for the README."""
    for tag, result in results.items():
        reference = result["starts"]["learned"]["final"]
        print(f"\n[{tag}] sigma of learned token {result['sigma']}, {result['n_images']} images")
        print("| start | final top-1 | Δ vs learned | g after B0 | g after B1 | g loop 1 | g loop 5 | g loop 10 |")
        for name, final, gaps in summary_rows(result):
            print(f"| {name} | {final:.2f} | {final - reference:+.2f} | {gaps[0]:.3f} | {gaps[1]:.3f} | "
                  f"{gaps[2]:.3f} | {gaps[6]:.3f} | {gaps[-1]:.3f} |")
        print(f"[{tag}] per-loop top-1 learned: {result['starts']['learned']['acc_per_loop']}")
        print(f"[{tag}] per-loop top-1 zeros:   {result['starts']['zeros']['acc_per_loop']}")
        print(f"[{tag}] per-loop top-1 noise (mean): "
              f"{np.round(seeded(result, 'noise', 'acc_per_loop').mean(axis=0), 2).tolist()}")
        print(f"[{tag}] LINE: {result['line']}")


def main() -> None:
    """Plot and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(TASK_DIR / "results"))
    results_dir = Path(parser.parse_args().results)
    results = load(results_dir)
    plot_gap(results, results_dir / "cls_start_gap_per_stage.png")
    print_summary(results)


if __name__ == "__main__":
    main()
