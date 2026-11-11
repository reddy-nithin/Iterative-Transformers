"""T23 plots and README tables from results/grad_norms.json.

1. grad_norm_per_loop.png: whole-block ‖g_t‖ vs loop, 4 models, log y-axis.
2. grad_norm_per_group.png: 6 panels (one per parameter group), 4 lines each, log y-axis.
3. residual_norm_per_loop.png: ‖x_t‖ vs loop, 4 models.

Run from the repo root:  .venv/bin/python tasks/T23_per_loop_grad_norms/plot.py [--results DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
MODELS = {"b0-init": ("tab:blue", "--"), "b0-trained": ("tab:blue", "-"),
          "d-init": ("tab:orange", "--"), "d-trained": ("tab:orange", "-")}
LAYER_GROUPS = ("norm1", "attn.qkv", "attn.proj", "norm2", "mlp.fc1", "mlp.fc2")
SUMMARY_GROUPS = ("block", *LAYER_GROUPS)


def load(results: Path) -> dict:
    """grad_norms.py's output; raises if missing."""
    path = results / "grad_norms.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run grad_norms.py first")
    return json.loads(path.read_text())


def draw_lines(ax, result: dict, values_of, log_y: bool) -> None:
    """One line per model, x = loop 1..n."""
    for tag, (colour, style) in MODELS.items():
        values = values_of(result[tag])
        ax.plot(range(1, len(values) + 1), values, style, marker="o", ms=3, color=colour, label=tag)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("loop t")
    ax.grid(alpha=0.3)


def plot_whole(result: dict, out_path: Path) -> None:
    """Whole-block ‖g_t‖ per loop."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    draw_lines(ax, result, lambda model: model["groups"]["block"]["norm_per_loop"], log_y=True)
    ax.set(ylabel="‖g_t‖ (whole shared block, mean of 20 batches)", title="T23 · per-loop gradient norm, shared block")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_groups(result: dict, out_path: Path) -> None:
    """‖g_t‖ per loop, one panel per parameter group."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    for ax, group in zip(axes.flat, LAYER_GROUPS):
        draw_lines(ax, result, lambda model, g=group: model["groups"][g]["norm_per_loop"], log_y=True)
        ax.set(title=group, ylabel="‖g_t‖")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("T23 · per-loop gradient norm per parameter group")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residual(result: dict, out_path: Path) -> None:
    """‖x_t‖ per loop."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    draw_lines(ax, result, lambda model: model["residual_norm_per_loop"], log_y=False)
    ax.set(ylabel="‖x_t‖ (mean over tokens and images)", title="T23 · residual-state norm after each loop")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_summary(result: dict) -> None:
    """Per-loop and summary tables for the README."""
    for tag in MODELS:
        model = result[tag]
        loops = model["loops"]
        print(f"\n[{tag}] {loops} loops at block positions {model['shared_positions']}, top-1 {model['top1']}, "
              f"gate rel err {model['gate']['grad_rel_error']:.2e}, logits diff {model['gate']['logits_max_abs_diff']:.2e}")
        print("| group | " + " | ".join(f"t{t}" for t in range(1, loops + 1)) + " |")
        for group in SUMMARY_GROUPS:
            print(f"| {group} | " + " | ".join(f"{v:.3g}" for v in model["groups"][group]["norm_per_loop"]) + " |")
        print("| ‖x_t‖ | " + " | ".join(f"{v:.1f}" for v in model["residual_norm_per_loop"]) + " |")
    print("\n| model | group | spread | dominance | agreement | pairwise cos |")
    for tag in MODELS:
        for group in SUMMARY_GROUPS:
            s = result[tag]["groups"][group]
            print(f"| {tag} | {group} | {s['spread']:.2f} | {s['dominance']:.3f} | {s['agreement']:.3f} | "
                  f"{s['pairwise_cos']:.3f} |")
    print(f"\nDECISION: {result['decision']}")


def main() -> None:
    """Plot and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(TASK_DIR / "results"))
    results_dir = Path(parser.parse_args().results)
    result = load(results_dir)
    plot_whole(result, results_dir / "grad_norm_per_loop.png")
    plot_groups(result, results_dir / "grad_norm_per_group.png")
    plot_residual(result, results_dir / "residual_norm_per_loop.png")
    print_summary(result)


if __name__ == "__main__":
    main()
