"""T27 plots and summary tables from the outputs of trajectories.py.

1. pca_paths_<tag>.png: PC1-PC2, PC3-PC4, PC5-PC6; one line per image (500), coloured by class, light to dark along
   the 13 stages, class mean at the last stage marked.
2. pca_mean_paths_<tag>.png: the same panels with each class's mean path only.
3. dist_to_final_<tag>.png: ||CLS_s - CLS_12|| / ||CLS_12|| for 100 images (10 per class), rows grouped by class.
4. split_steps.png: histogram of split steps over the 45 class pairs, both models.
Also prints the split-step table, the 5 latest-splitting pairs, and the distance-to-final medians per stage.

Run from the repo root:  .venv/bin/python tasks/T27_cls_trajectories/plot.py [--results DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
TAGS = ("T11-d", "T13-c")
PANELS = ((0, 1), (2, 3), (4, 5))
N_LATEST = 5


def load(results: Path, tag: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """(analysis json, CLS states (S, N, D) float32, class index per image) for one model."""
    json_path, npz_path = results / f"trajectories_{tag}.json", results / f"cls_{tag}.npz"
    for path in (json_path, npz_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run trajectories.py first")
    data = np.load(npz_path)
    return json.loads(json_path.read_text()), data["cls"].astype(np.float32), data["labels"]


def project(cls: np.ndarray, result: dict) -> np.ndarray:
    """CLS states projected on the saved PCA components: (S, N, k)."""
    return (cls - np.array(result["pca_mean"])) @ np.array(result["pca_components"]).T


def stage_shades(colour, n_stages: int) -> np.ndarray:
    """n_stages RGB colours from a light tint of `colour` (first stage) to the full colour (last stage)."""
    base = np.array(to_rgb(colour))
    mix = np.linspace(0.15, 1.0, n_stages)[:, None]
    return (1 - mix) * np.ones(3) + mix * base


def class_colours(n_classes: int) -> list:
    """One tab10 colour per class."""
    return [plt.get_cmap("tab10")(i) for i in range(n_classes)]


def draw_paths(ax, paths: np.ndarray, colour, line_alpha: float, marker_size: float) -> None:
    """Lines through the stages of each path (paths: (S, n, 2)), points shaded light to dark."""
    shades = stage_shades(colour, paths.shape[0])
    for i in range(paths.shape[1]):
        ax.plot(paths[:, i, 0], paths[:, i, 1], color=colour, alpha=line_alpha, lw=0.6)
    for stage in range(paths.shape[0]):
        ax.scatter(paths[stage, :, 0], paths[stage, :, 1], color=shades[stage], s=marker_size, lw=0)


def plot_pca(result: dict, cls: np.ndarray, labels: np.ndarray, out_path: Path, mean_only: bool) -> None:
    """PCA paths, one image per line or one class mean per line."""
    projected = project(cls, result)
    names, share = result["classes"], result["pca_variance_share"]
    colours = class_colours(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, (pc_x, pc_y) in zip(axes, PANELS):
        for c, name in enumerate(names):
            paths = projected[:, labels == c][:, :, [pc_x, pc_y]]
            mean_path = paths.mean(axis=1, keepdims=True)
            if mean_only:
                draw_paths(ax, mean_path, colours[c], line_alpha=0.9, marker_size=30)
            else:
                draw_paths(ax, paths, colours[c], line_alpha=0.12, marker_size=4)
            ax.scatter(*mean_path[-1, 0], color=colours[c], s=140, marker="*", edgecolor="k", lw=0.6, zorder=5,
                       label=name)
        start = projected[0, 0, [pc_x, pc_y]]
        ax.annotate("input (all images)", start, fontsize=7, color="0.3")
        ax.set(xlabel=f"PC{pc_x + 1} ({share[pc_x]:.1%})", ylabel=f"PC{pc_y + 1} ({share[pc_y]:.1%})")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, title="class (★ = mean at loop 10)", title_fontsize=8)
    kind = "class-mean paths" if mean_only else "one line per image"
    fig.suptitle(f"T27 · {result['tag']} raw CLS through 13 stages (input, B0, B1, loops 1-10), {kind}; "
                 "light → dark = early → late")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_distance(result: dict, cls: np.ndarray, out_path: Path) -> None:
    """Heatmap of relative distance to the final CLS, 100 images grouped by class."""
    rows = np.array(result["heatmap_rows"])
    final = cls[-1, rows]
    distance = (np.linalg.norm(cls[:, rows] - final, axis=-1) / np.linalg.norm(final, axis=-1)).T
    per_class = len(rows) // len(result["classes"])
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    image = ax.imshow(distance, aspect="auto", cmap="viridis", vmin=0, vmax=min(1.5, float(distance.max())))
    for boundary in range(per_class, len(rows), per_class):
        ax.axhline(boundary - 0.5, color="w", lw=0.6)
    ax.set_yticks([per_class * c + per_class / 2 - 0.5 for c in range(len(result["classes"]))], result["classes"])
    ax.set_xticks(range(len(result["stages"])), result["stages"], rotation=45, ha="right")
    fig.colorbar(image, ax=ax, label="‖CLS_s − CLS_final‖ / ‖CLS_final‖")
    ax.set_title(f"T27 · {result['tag']}: distance to the final CLS (10 images per class)")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def split_counts(result: dict) -> tuple[list[str], list[int]]:
    """Histogram of split steps over stages 1..12 plus 'none'."""
    stages = result["stages"][1:] + ["none"]
    steps = [row["split_step"] for row in result["split_steps"]]
    counts = [steps.count(s) for s in range(1, len(result["stages"]))] + [steps.count(None)]
    return stages, counts


def plot_split_steps(results: dict[str, dict], out_path: Path) -> None:
    """Split-step histograms, both models side by side."""
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    width = 0.4
    stages: list[str] = []
    for i, (tag, result) in enumerate(results.items()):
        stages, counts = split_counts(result)
        ax.bar(np.arange(len(stages)) + (i - 0.5) * width, counts, width, label=tag)
    ax.set_xticks(range(len(stages)), stages, rotation=45, ha="right")
    ax.set(ylabel="class pairs (of 45)", title="T27 · split step: first stage where sep(a, b) ≥ 1")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def lateness(row: dict, n_stages: int) -> tuple[int, float]:
    """Sort key: later split step first (never = latest), then smaller best separation."""
    step = row["split_step"]
    return (n_stages if step is None else step, -max(row["sep"]))


def latest_pairs(result: dict) -> list[tuple[str, str, str]]:
    """The N_LATEST pairs that split last, with the stage name where they split ('none' if never)."""
    n_stages = len(result["stages"])
    ordered = sorted(result["split_steps"], key=lambda row: lateness(row, n_stages), reverse=True)[:N_LATEST]
    return [(*row["pair"], "none" if row["split_step"] is None else result["stages"][row["split_step"]])
            for row in ordered]


def print_summary(results: dict[str, dict]) -> None:
    """Tables for the README."""
    for tag, result in results.items():
        stages, counts = split_counts(result)
        print(f"\n[{tag}] top-1 {result['top1_test']} (subset {result['subset_top1']}); "
              f"PCA shares PC1-6 {result['pca_variance_share']}")
        print(f"[{tag}] median dist to final: " + ", ".join(
            f"{name} {value:.3f}" for name, value in zip(result["stages"], result["dist_to_final_median"])))
        print(f"[{tag}] p90 dist to final:    " + ", ".join(
            f"{name} {value:.3f}" for name, value in zip(result["stages"], result["dist_to_final_p90"])))
        print(f"[{tag}] settles (median < 0.1 from then on) at stage {result['settle_stage_median_below_0.1']}")
        print(f"[{tag}] mean ||CLS||: " + ", ".join(
            f"{name} {value:.1f}" for name, value in zip(result["stages"], result["cls_norm_mean"])))
        print(f"[{tag}] sep median / max over pairs: " + ", ".join(
            f"{name} {med:.2f}/{top:.2f}" for name, med, top in
            zip(result["stages"][1:], result["sep_median_over_pairs"][1:], result["sep_max_over_pairs"][1:])))
        print(f"[{tag}] split-step histogram: " + ", ".join(f"{s}: {c}" for s, c in zip(stages, counts) if c))
        print(f"[{tag}] latest pairs: " + "; ".join(f"{a}-{b} ({s})" for a, b, s in latest_pairs(result)))


def main() -> None:
    """Load both models' outputs, draw the plots, print the tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(TASK_DIR / "results"))
    results_dir = Path(parser.parse_args().results)
    results = {}
    for tag in TAGS:
        result, cls, labels = load(results_dir, tag)
        plot_pca(result, cls, labels, results_dir / f"pca_paths_{tag}.png", mean_only=False)
        plot_pca(result, cls, labels, results_dir / f"pca_mean_paths_{tag}.png", mean_only=True)
        plot_distance(result, cls, results_dir / f"dist_to_final_{tag}.png")
        results[tag] = result
    plot_split_steps(results, results_dir / "split_steps.png")
    print_summary(results)


if __name__ == "__main__":
    main()
