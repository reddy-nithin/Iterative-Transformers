"""T34 plots from results/frozen_update.json.

1. frozen_vs_k.png: top-1 vs k, one panel per checkpoint: frozen(k), loop-k readout, dashed line at true-10.
2. angular_speed.png: mean 1 - cos(final_norm(x_t), final_norm(x_{t-1})) vs loop 1..20, CLS (solid) and patches
   (dashed), one colour per checkpoint, log y-axis.
3. cls_norm.png: mean raw CLS L2 norm vs loop 1..20.

Run from the repo root:  .venv/bin/python tasks/T34_frozen_update_test/plot.py [--results DIR]
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
COLOURS = {"T11-d": "tab:red", "T13-c": "tab:green", "T31-a": "tab:blue"}

log = logging.getLogger("t34_plot")


def load(results: Path) -> dict:
    """frozen_update.py's output, only checkpoints that passed the sanity check."""
    path = results / "frozen_update.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run frozen_update.py first")
    data = json.loads(path.read_text())
    return {tag: value for tag, value in data.items() if tag in COLOURS and value.get("status") == "ok"}


def plot_frozen(models: dict, out_path: Path) -> None:
    """Frozen(k) and loop-k readout vs k, with true-10 dashed."""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), constrained_layout=True, squeeze=False)
    for ax, (tag, model) in zip(axes[0], models.items()):
        ks = sorted(int(k) for k in model["frozen"])
        ax.plot(ks, [model["frozen"][str(k)] for k in ks], "-o", color=COLOURS[tag], label="frozen(k)")
        ax.plot(ks, [model["loop_k_readout"][str(k)] for k in ks], "--s", color="grey", label="loop-k readout")
        ax.axhline(model["true_10"], color="black", ls=":", label=f"true-10 ({model['true_10']:.2f})")
        ax.set(title=tag, xlabel="k (last real loop)", ylabel="test top-1 (%)", xticks=ks)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_angular(models: dict, out_path: Path) -> None:
    """Angular step vs loop, CLS solid and patches dashed, log y."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for tag, model in models.items():
        loops = range(1, len(model["angular_step_cls_1_20"]) + 1)
        ax.plot(loops, model["angular_step_cls_1_20"], "-o", ms=3, color=COLOURS[tag], label=f"{tag} CLS")
        ax.plot(loops, model["angular_step_patches_1_20"], "--", color=COLOURS[tag], label=f"{tag} patches")
    ax.axvline(10, color="grey", ls=":", lw=1)
    ax.set(yscale="log", xlabel="loop t", ylabel="mean 1 - cos(norm(x_t), norm(x_t-1))",
           title="Angular step of the normalised state", xticks=range(1, 21))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_cls_norm(models: dict, out_path: Path) -> None:
    """Raw CLS norm vs loop."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for tag, model in models.items():
        loops = range(1, len(model["cls_norm_1_20"]) + 1)
        ax.plot(loops, model["cls_norm_1_20"], "-o", ms=3, color=COLOURS[tag], label=tag)
    ax.axvline(10, color="grey", ls=":", lw=1)
    ax.set(xlabel="loop t", ylabel="mean ||x_t[CLS]|| (before final norm)", title="Raw CLS norm",
           xticks=range(1, 21))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    """All three plots."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(TASK_DIR / "results"))
    results = Path(parser.parse_args().results)
    models = load(results)
    if not models:
        raise RuntimeError("no checkpoint passed the sanity check; nothing to plot")
    plot_frozen(models, results / "frozen_vs_k.png")
    plot_angular(models, results / "angular_speed.png")
    plot_cls_norm(models, results / "cls_norm.png")
    log.info("wrote frozen_vs_k.png, angular_speed.png, cls_norm.png to %s", results)


if __name__ == "__main__":
    main()
