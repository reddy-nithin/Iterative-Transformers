"""T5 plots + table from results/cls_svd_<tag>.json: singular-value decay per layer (raw and centred), effective
rank vs block for every class and the mixed-class control, and class separation vs block.

Run from the repo root:  .venv/bin/python tasks/T5_cls_svd_per_block/plot.py [--model-tag pretrained]
Compare models/datasets: .venv/bin/python tasks/T5_cls_svd_per_block/plot.py --compare pretrained ft_baseline_train ...
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
    p.add_argument("--compare", nargs="+", default=None, metavar="TAG",
                   help="instead: one figure comparing these tags (writes compare.png)")
    return p.parse_args()


def layer_labels(data: dict, layers: list[int]) -> list[str]:
    """Tick labels: layer numbers, with the CLS after the final LayerNorm shown as 'N'."""
    final_norm = data.get("final_norm_layer")
    return ["N" if layer == final_norm else str(layer) for layer in layers]


def class_names(data: dict) -> list[str]:
    """Group names that are classes (not the mixed control)."""
    return [name for name in data["groups"] if name in data["classes"]]


def control_name(data: dict) -> str | None:
    """Name of the mixed-class control group, if there is one."""
    return next((name for name in data["groups"] if name not in data["classes"]), None)


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
    fig.colorbar(mappable, ax=axes.tolist(), label="layer (0 = before block 1, k = after block k, last = final norm)")
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
        ax.set_xlabel("layer (0 = before block 1, k = after block k, N = after final norm)")
        ax.set_ylabel("effective rank (of max 192)")
        ax.set_xticks(layers, layer_labels(data, layers))
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Per-class CLS effective rank vs depth, {data['model_tag']} {data['model']}", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def separation_series(by_layer: dict) -> tuple[list[int], list[float]]:
    """Layers and ratios, leaving out layers where the ratio is undefined (None)."""
    pairs = [(int(layer), ratio) for layer, ratio in by_layer.items() if ratio is not None]
    return [layer for layer, _ in pairs], [ratio for _, ratio in pairs]


def plot_separation(data: dict, out_path: Path) -> None:
    """Within/between class spread vs layer, one line per class set (log y-axis)."""
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for index, (set_name, by_layer) in enumerate(data["separation"].items()):
        layers, ratios = separation_series(by_layer)
        ax.plot(layers, ratios, marker="o", markersize=3, label=set_name, color=CLASS_COLOURS[index % 10])
    all_layers = sorted(int(layer) for layer in next(iter(data["separation"].values())))
    ax.set_xticks(all_layers, layer_labels(data, all_layers))
    ax.set_yscale("log")
    ax.set_xlabel("layer (0 = before block 1, k = after block k, N = after final norm)")
    ax.set_ylabel("within / between class spread (lower = tighter classes)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.suptitle(f"Class separation vs depth, {data['model_tag']} ({data['dataset']} {data['split']})", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def mean_class_rank(data: dict, variant: str = "centred") -> tuple[list[int], np.ndarray]:
    """Layers and the mean effective rank over the class groups (control excluded)."""
    names = class_names(data)
    layers = sorted(int(layer) for layer in data["groups"][names[0]]["layers"])
    ranks = np.array([[data["groups"][n]["layers"][str(layer)][variant]["effective_rank"] for layer in layers]
                      for n in names])
    return layers, ranks.mean(axis=0)


def plot_compare(datasets: list[dict], out_path: Path) -> None:
    """Headline figure: per tag, mean class centred effective rank (control dashed) and class separation.

    Separation uses the 'matched 4' set when present (ImageNet), so every tag is measured over 4 classes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for index, data in enumerate(datasets):
        colour = CLASS_COLOURS[index % 10]
        label = f"{data['model_tag']} ({data['dataset']} {data['split']})"
        layers, ranks = mean_class_rank(data)
        axes[0].plot(layers, ranks, marker="o", markersize=3, color=colour, label=f"{label}: class mean")
        control = control_name(data)
        if control:
            control_ranks = [data["groups"][control]["layers"][str(layer)]["centred"]["effective_rank"]
                             for layer in layers]
            axes[0].plot(layers, control_ranks, linestyle="--", color=colour, alpha=0.6, label=f"{label}: control")
        set_name = "matched 4" if "matched 4" in data["separation"] else "all classes"
        sep_layers, ratios = separation_series(data["separation"][set_name])
        axes[1].plot(sep_layers, ratios, marker="o", markersize=3, color=colour, label=f"{label}, {set_name}")
    ticks = layers
    for ax in axes:
        ax.set_xticks(ticks, layer_labels(datasets[0], ticks))
        ax.set_xlabel("layer (N = after final norm)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("centred effective rank (of max 192)")
    axes[0].set_title("How many directions one class spreads over")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("within / between class spread")
    axes[1].set_title("How tight classes are vs their distances (lower = tighter)")
    fig.suptitle("CLS per class vs depth: models and datasets compared", fontsize=11)
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


def separation_table(data: dict) -> str:
    """Markdown table: rows = layers, columns = class sets, cells = within/between class spread."""
    sets = list(data["separation"])
    layers = sorted(int(layer) for layer in data["separation"][sets[0]])
    lines = ["| layer | " + " | ".join(sets) + " |", "|---" * (len(sets) + 1) + "|"]
    for layer, label in zip(layers, layer_labels(data, layers)):
        cells = [data["separation"][s][str(layer)] for s in sets]
        lines.append(f"| {label} | " + " | ".join("-" if c is None else f"{c:.2f}" for c in cells) + " |")
    return "\n".join(lines)


def write_summary(data: dict, out_path: Path) -> None:
    """Rank tables (raw, centred), separation table and top-1 per group, with a short legend."""
    parts = [f"# CLS SVD summary ({data['model_tag']}: {data['model']}, {data.get('dataset', 'cifar100')} "
             f"{data.get('split', 'train')})", "",
             "Cell = effective rank (entropy-based) / number of singular values holding 90% of the energy. "
             "Layer 0 = CLS before block 1; layer k = after block k (before the final norm); the last layer "
             "(N in the plots) = after the final norm, what the head reads. Max rank 192.", ""]
    for variant in VARIANTS:
        parts += [f"## {variant}", "", summary_table(data, variant), ""]
    if "separation" in data:
        parts += ["## separation (within / between class spread; lower = tighter classes)", "",
                  separation_table(data), ""]
    top1 = {name: group.get("top1") for name, group in data["groups"].items()}
    if any(value is not None for value in top1.values()):
        parts += ["## top-1 of the model's own head (%)", "",
                  ", ".join(f"{name} {value}" for name, value in top1.items()), ""]
    out_path.write_text("\n".join(parts))


def main() -> None:
    """Build every figure and the summary table for one model tag."""
    args = parse_args()
    results = Path(args.results_dir)
    if args.compare:
        plot_compare([load_results(results, tag) for tag in args.compare], results / "compare.png")
        print(f"wrote {results / 'compare.png'}", flush=True)
        return
    data = load_results(results, args.model_tag)
    suffix = "" if args.model_tag == "pretrained" else f"_{args.model_tag}"
    for variant in VARIANTS:
        plot_decay(data, variant, results / f"sv_decay_{variant}{suffix}.png")
    plot_effective_rank(data, control_name(data), results / f"effective_rank_vs_block{suffix}.png")
    if "separation" in data:
        plot_separation(data, results / f"separation_vs_block{suffix}.png")
    write_summary(data, results / f"summary{suffix}.md")
    print(f"wrote plots and summary to {results}", flush=True)


def load_results(results: Path, tag: str) -> dict:
    """The JSON that cls_svd.py wrote for one model tag."""
    json_path = results / f"cls_svd_{tag}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found; run cls_svd.py first")
    return json.loads(json_path.read_text())


if __name__ == "__main__":
    main()
