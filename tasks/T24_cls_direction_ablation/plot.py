"""T24 plots and README tables from results/cls_direction.json.

1. singular_spectrum.png: energy share of the top 20 singular values of D, 2 models.
2. push_along_u1_per_loop.png: mean |d_t . u1| and mean ‖d_t‖ vs loop, 2 models.
3. fisher_u1_vs_random.png: histogram of random-direction Fisher ratios with u1 marked, one panel per model.

Run from the repo root:  .venv/bin/python tasks/T24_cls_direction_ablation/plot.py [--results DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
COLOURS = {"T11-d": "tab:orange", "T13-c": "tab:green"}
SETTINGS = ("no edit", "u1 loops 6-10", "u1 loops 1-10", "random loops 6-10", "u2 loops 6-10")


def load(results: Path) -> dict:
    """cls_direction.py's output; raises if missing."""
    path = results / "cls_direction.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run cls_direction.py first")
    return json.loads(path.read_text())


def plot_spectrum(result: dict, out_path: Path) -> None:
    """Energy share of the top singular values."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for tag, model in result.items():
        spectrum = model["a"]["spectrum_top20"]
        ax.plot(range(1, len(spectrum) + 1), spectrum, "-o", ms=3, color=COLOURS[tag], label=tag)
    ax.set(xlabel="singular value index", ylabel="σ_k² / Σσ²", yscale="log",
           title="T24 · spectrum of late CLS updates (loops 6-10, uncentred)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_push(result: dict, out_path: Path) -> None:
    """Mean ‖d_t‖ (dashed) and mean |d_t . u1| (solid) per loop."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for tag, model in result.items():
        loops = range(1, len(model["b"]["d_norm_per_loop"]) + 1)
        ax.plot(loops, model["b"]["d_norm_per_loop"], "--o", ms=3, color=COLOURS[tag], label=f"{tag} ‖d_t‖")
        ax.plot(loops, model["b"]["push_u1_per_loop"], "-s", ms=3, color=COLOURS[tag], label=f"{tag} |d_t · u1|")
    ax.set(xlabel="loop t", ylabel="mean over test images", title="T24 · CLS update size and its part along u1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fisher(result: dict, out_path: Path) -> None:
    """Random-direction Fisher ratios with u1 marked."""
    fig, axes = plt.subplots(1, len(result), figsize=(12, 4.5), constrained_layout=True)
    for ax, (tag, model) in zip(axes, result.items()):
        info = model["b"]
        ax.hist(info["random_ratios"], bins=30, color="0.6", label="200 random directions")
        ax.axvline(info["fisher_u1"], color=COLOURS[tag], lw=2, label=f"u1 ({info['fisher_u1']:.3g})")
        ax.axvline(info["random_median"], color="k", ls=":", label="random median")
        ax.set(xlabel="Fisher ratio of CLS_10 · direction (100 classes)", ylabel="count",
               title=f"{tag}: u1 at the {info['u1_percentile']:.0f}th percentile")
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_summary(result: dict) -> None:
    """Tables for the README."""
    print("| model | energy share σ1² | top-5 cumulative | cos(u1, mean d) | centred energy share |")
    for tag, model in result.items():
        a = model["a"]
        print(f"| {tag} | {a['energy_share']:.3f} | {' / '.join(f'{v:.3f}' for v in a['cumulative_top5'])} | "
              f"{a['cos_u1_mean']:.3f} | {a['centred_energy_share']:.3f} |")
    print("\n| model | Fisher u1 | random median | random p95 | u1 percentile |")
    for tag, model in result.items():
        b = model["b"]
        print(f"| {tag} | {b['fisher_u1']:.4g} | {b['random_median']:.4g} | {b['random_p95']:.4g} | "
              f"{b['u1_percentile']:.1f} |")
    for tag, model in result.items():
        b = model["b"]
        print(f"[{tag}] ‖d_t‖:     " + " ".join(f"{v:.3f}" for v in b["d_norm_per_loop"]))
        print(f"[{tag}] |d_t·u1|:  " + " ".join(f"{v:.3f}" for v in b["push_u1_per_loop"]))
    print("\n| setting | " + " | ".join(result) + " |")
    for setting in SETTINGS:
        cells = []
        for model in result.values():
            run = model["c"][setting]
            delta = run["final"] - model["c"]["no edit"]["final"]
            extra = f" (spread {run['spread']:.2f})" if "spread" in run else ""
            cells.append(f"{run['final']:.2f} ({delta:+.2f}){extra}")
        print(f"| {setting} | " + " | ".join(cells) + " |")
    for tag, model in result.items():
        for setting in ("no edit", "u1 loops 6-10"):
            print(f"[{tag}] per-loop {setting}: {model['c'][setting]['acc_per_loop']}")
        print(f"[{tag}] LINE: {model['line']}")


def main() -> None:
    """Plot and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(TASK_DIR / "results"))
    results_dir = Path(parser.parse_args().results)
    result = load(results_dir)
    plot_spectrum(result, results_dir / "singular_spectrum.png")
    plot_push(result, results_dir / "push_along_u1_per_loop.png")
    plot_fisher(result, results_dir / "fisher_u1_vs_random.png")
    print_summary(result)


if __name__ == "__main__":
    main()
