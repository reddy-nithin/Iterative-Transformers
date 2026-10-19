"""T5 plots + table from results/cls_svd_<tag>.json: singular-value decay per layer (raw and centred) and
effective rank vs block for every class and the mixed-class control.

Run from the repo root:  .venv/bin/python tasks/T5_cls_svd_per_block/plot.py [--model-tag pretrained]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
CONTROL_STYLE = {"color": "grey", "linestyle": "--"}
CLASS_COLOURS = plt.get_cmap("tab10").colors
VARIANTS = ("raw", "centred")


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", default="pretrained")
    p.add_argument("--results-dir", default=str(RESULTS))
    return p.parse_args()


def group_style(index: int, name: str, control_name: str | None) -> dict:
    """Same colour per class in every figure; the control is dashed grey."""
    return dict(CONTROL_STYLE) if name == control_name else {"color": CLASS_COLOURS[index % 10], "linestyle": "-"}


def plot_decay(data: dict, variant: str, out_path: Path) -> None:
    """One panel per group: s_i / s_1 vs index i, one line per layer coloured by depth (log y-axis)."""
    groups = data["groups"]
    n_layers = len(next(iter(groups.values()))["layers"])
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.8), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (name, group) in zip(axes, groups.items()):
        for layer in range(n_layers):
            s = np.array(group["layers"][str(layer)][variant]["singular_values"])
            if s[0] <= 0:  # centred layer 0: every image has the same CLS, nothing to plot
                continue
            positive = s > 0
            ax.plot(np.arange(1, len(s) + 1)[positive], (s / s[0])[positive], color=cmap(layer / (n_layers - 1)),
                    linewidth=1)
        ax.set_yscale("log")
        ax.set_title(f"{name} (N={group['n_images']})", fontsize=10)
        ax.set_xlabel("singular value index i")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("s_i / s_1")
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_layers - 1))
    fig.colorbar(mappable, ax=axes.tolist(), label="layer (0 = before block 1, k = after block k)")
    note = " (layer 0 left out: identical CLS for every image)" if variant == "centred" else ""
    fig.suptitle(f"CLS singular-value decay per layer, {variant}{note}, {data['model_tag']} {data['model']}",
                 fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_effective_rank(data: dict, control_name: str | None, out_path: Path) -> None:
    """Effective rank vs layer, raw and centred side by side, one line per group."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, variant in zip(axes, VARIANTS):
        for index, (name, group) in enumerate(data["groups"].items()):
            layers = sorted(int(layer) for layer in group["layers"])
            ranks = [group["layers"][str(layer)][variant]["effective_rank"] for layer in layers]
            ax.plot(layers, ranks, marker="o", markersize=3, label=name, **group_style(index, name, control_name))
        ax.set_title(variant)
        ax.set_xlabel("layer (0 = before block 1, k = after block k)")
        ax.set_ylabel("effective rank (of max 192)")
        ax.set_xticks(layers)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Per-class CLS effective rank vs depth, {data['model_tag']} {data['model']}", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summary_table(data: dict, variant: str) -> str:
    """Markdown table: rows = layers, columns = groups, cells = 'effective rank / 90%-energy rank'."""
    names = list(data["groups"])
    n_layers = len(data["groups"][names[0]]["layers"])
    lines = ["| layer | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for layer in range(n_layers):
        cells = [data["groups"][name]["layers"][str(layer)][variant] for name in names]
        lines.append(f"| {layer} | " + " | ".join(f"{c['effective_rank']:.1f} / {c['energy_rank_90']}" for c in cells)
                     + " |")
    return "\n".join(lines)


def write_summary(data: dict, out_path: Path) -> None:
    """Both tables (raw, centred) with a short legend."""
    parts = [f"# CLS SVD summary ({data['model_tag']} {data['model']})", "",
             "Cell = effective rank (entropy-based) / number of singular values holding 90% of the energy. "
             "Layer 0 = CLS before block 1; layer k = after block k (before the final norm). Max rank 192.", ""]
    for variant in VARIANTS:
        parts += [f"## {variant}", "", summary_table(data, variant), ""]
    out_path.write_text("\n".join(parts))


def main() -> None:
    """Build every figure and the summary table for one model tag."""
    args = parse_args()
    results = Path(args.results_dir)
    json_path = results / f"cls_svd_{args.model_tag}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found; run cls_svd.py first")
    data = json.loads(json_path.read_text())
    control_name = next((name for name in data["groups"] if name not in data["classes"]), None)
    suffix = "" if args.model_tag == "pretrained" else f"_{args.model_tag}"
    for variant in VARIANTS:
        plot_decay(data, variant, results / f"sv_decay_{variant}{suffix}.png")
    plot_effective_rank(data, control_name, results / f"effective_rank_vs_block{suffix}.png")
    write_summary(data, results / f"summary{suffix}.md")
    print(f"wrote plots and summary to {results}", flush=True)


if __name__ == "__main__":
    main()
