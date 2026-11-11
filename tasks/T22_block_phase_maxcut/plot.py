"""T22 plots and README tables from block_phases.py's outputs.

1. layer_similarity_pretrained.png / layer_similarity_baseline.png: S_tok heatmap with the best k = 3 cut (solid lines)
   and the T11-c cut (dashed).
2. layer_similarity_cls.png: S_cls for both models side by side, same lines.

Run from the repo root:  .venv/bin/python tasks/T22_block_phase_maxcut/plot.py [--results DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
TAGS = ("pretrained", "baseline")
T11C_CUT = (1, 11)


def load(results: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """(block_phases.json, {'S_tok_pretrained': matrix, ...}); raises if anything is missing."""
    path = results / "block_phases.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run block_phases.py first")
    matrices = {f"{kind}_{tag}": np.load(results / f"{kind}_{tag}.npy") for kind in ("S_tok", "S_cls") for tag in TAGS}
    return json.loads(path.read_text()), matrices


def draw_cut(ax, cut, style: str, colour: str) -> None:
    """Group boundaries as horizontal and vertical lines between blocks."""
    for point in cut:
        ax.axhline(point - 0.5, color=colour, ls=style, lw=1.5)
        ax.axvline(point - 0.5, color=colour, ls=style, lw=1.5)


def heatmap(ax, similarity: np.ndarray, best_cut: list[int], title: str):
    """One annotated similarity heatmap with the best cut (solid red) and the T11-c cut (dashed white)."""
    image = ax.imshow(similarity, cmap="viridis", vmin=min(0.0, float(similarity.min())), vmax=1)
    for l in range(similarity.shape[0]):
        for m in range(similarity.shape[1]):
            ax.text(m, l, f"{similarity[l, m]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="w" if similarity[l, m] < 0.6 else "k")
    draw_cut(ax, best_cut, "-", "red")
    draw_cut(ax, T11C_CUT, "--", "white")
    ax.set(xticks=range(12), yticks=range(12), xlabel="block m", ylabel="block l", title=title)
    return image


def plot_tok(result: dict, similarity: np.ndarray, tag: str, out_path: Path) -> None:
    """S_tok heatmap of one model."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    best = result[tag]["S_tok"]["k3_top5"][0]
    image = heatmap(ax, similarity, best["cut"],
                    f"T22 · {tag}: S_tok (mean cos over all 197 tokens)\nred = best k=3 cut, white dashed = T11-c cut")
    fig.colorbar(image, ax=ax)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cls(result: dict, matrices: dict[str, np.ndarray], out_path: Path) -> None:
    """S_cls heatmaps of both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    image = None
    for ax, tag in zip(axes, TAGS):
        best = result[tag]["S_cls"]["k3_top5"][0]
        image = heatmap(ax, matrices[f"S_cls_{tag}"], best["cut"], f"{tag}: S_cls (CLS token only)")
    fig.colorbar(image, ax=axes)
    fig.suptitle("T22 · red = best k=3 cut, white dashed = T11-c cut")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def groups_text(groups: list[list[int]]) -> str:
    """Groups as text, e.g. '{0} {1..10} {11}'."""
    return " ".join(f"{{{g[0]}..{g[-1]}}}" if len(g) > 1 else f"{{{g[0]}}}" for g in groups)


def fmt(value: float | None) -> str:
    """A score or '–' for None."""
    return "–" if value is None else f"{value:.3f}"


def print_summary(result: dict) -> None:
    """Ranking tables for the README."""
    for tag in TAGS:
        for kind in ("S_tok", "S_cls"):
            info = result[tag][kind]
            print(f"\n[{tag} {kind}]")
            print("| rank | groups (k=3) | cut score | within | across mean |")
            for row in info["k3_top5"]:
                print(f"| {row['rank']} | {groups_text(row['groups'])} | {row['cut_score']:.3f} | "
                      f"{fmt(row['within_score'])} | {row['across_mean']:.3f} |")
            for name, row in info["hand_cuts"].items():
                print(f"| {row['rank']} ({name}) | {groups_text(row['groups'])} | {row['cut_score']:.3f} | "
                      f"{fmt(row['within_score'])} | {row['across_mean']:.3f} |")
            for k in ("k2_best", "k4_best"):
                row = info[k]
                print(f"best {k[:2]}: {groups_text(row['groups'])} cut {row['cut_score']:.3f} "
                      f"within {fmt(row['within_score'])} across {row['across_mean']:.3f}")
    print(f"\nDECISION: {result['decision']}")


def main() -> None:
    """Plot and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(TASK_DIR / "results"))
    results_dir = Path(parser.parse_args().results)
    result, matrices = load(results_dir)
    for tag in TAGS:
        plot_tok(result, matrices[f"S_tok_{tag}"], tag, results_dir / f"layer_similarity_{tag}.png")
    plot_cls(result, matrices, results_dir / "layer_similarity_cls.png")
    print_summary(result)


if __name__ == "__main__":
    main()
