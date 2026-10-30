"""T13 plots + table: readout at every loop (T13-a, T13-c) and the loss-margin version (T13-b, T13-d) on the T11-c
and T11-d layouts, with T11-c and T11-d as references. T11-c-layout runs are drawn in purple/blue, T11-d-layout runs
in red/orange; references dashed.

1. results/exit_curves.png: one panel per model, fixed-depth curve + confidence, KL and patience exit rules.
2. results/acc_per_loop.png: fixed-depth curve (readout at loop t) for all six models.
3. results/rel_update.png: T12 metric (a), rel_cls (solid) and rel_patch (dotted) per loop.
4. results/loss_curves.png: train CE of the final output and test loss vs epoch.
5. results/separation_vs_loop.png: class separation of CLS after every block call, train and test.
Also writes results/summary.md (the results table).

Inputs: exit_analysis.json (exit_analysis.py), t12_diag/diag_*.json (T12 loop_diagnostics.py run on the T13
checkpoints), the training and cls_svd JSONs of T13 (this folder) and T11 / T12 (their folders).

Run from the repo root:  python tasks/T13_per_loop_readout_loss/plot.py
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
T11_RESULTS = TASK_DIR.parent / "T11_pretrained_hbt_body_sweep" / "results"
T12_RESULTS = TASK_DIR.parent / "T12_per_loop_cls_diagnostics" / "results"
SEPARATION_SET = "all classes"
EARLY_LOOPS = 5  # "loops 1-5" in the T13-b vs T13-a prediction

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Run:
    """One model in the comparison: where its files are and how it is drawn."""
    name: str
    layout: str  # "T11-c" or "T11-d"
    loss: str
    checkpoint_stem: str
    train_json: Path
    cls_svd_prefix: Path  # + "_train.json" / "_test.json"
    diag_json: Path  # T12 metrics
    colour: str
    reference: bool = False

    @property
    def body_calls(self) -> range:
        """Block calls (1-based) that are body loops: 2..11 with a head and tail, 3..12 for unrolling n = 3."""
        return range(2, 12) if self.layout == "T11-c" else range(3, 13)

    @property
    def label(self) -> str:
        """Legend label."""
        return f"{self.name} ({self.loss})"


RUNS = [
    Run("T11-c", "T11-c", "ref: final loss only", "T11-c_hbt_body_b1", T11_RESULTS / "hbt_T11-c.json",
        T11_RESULTS / "cls_svd_T11-c", T12_RESULTS / "diag_T11-c_hbt_body_b1.json", "tab:purple", reference=True),
    Run("T13-a", "T11-c", "readout every loop", "T13-a_readout_every_loop", RESULTS / "hbt_T13-a.json",
        RESULTS / "cls_svd_T13-a", RESULTS / "t12_diag" / "diag_T13-a_readout_every_loop.json", "#1f4e9c"),
    Run("T13-b", "T11-c", "every loop + margin", "T13-b_loss_margin", RESULTS / "hbt_T13-b.json",
        RESULTS / "cls_svd_T13-b", RESULTS / "t12_diag" / "diag_T13-b_loss_margin.json", "#5aa0e0"),
    Run("T11-d", "T11-d", "ref: final loss only", "T11-d_unroll_n3_30ep", T11_RESULTS / "unroll_T11-d.json",
        T11_RESULTS / "cls_svd_T11-d", T12_RESULTS / "diag_T11-d_unroll_n3_30ep.json", "tab:red", reference=True),
    Run("T13-c", "T11-d", "readout every loop", "T13-c_unroll_readout_every_loop", RESULTS / "unroll_T13-c.json",
        RESULTS / "cls_svd_T13-c", RESULTS / "t12_diag" / "diag_T13-c_unroll_readout_every_loop.json", "#d95f02"),
    Run("T13-d", "T11-d", "every loop + margin", "T13-d_unroll_loss_margin", RESULTS / "unroll_T13-d.json",
        RESULTS / "cls_svd_T13-d", RESULTS / "t12_diag" / "diag_T13-d_unroll_loss_margin.json", "#e6ab02"),
]
RULE_STYLES = {"confidence": ("^", "tab:green"), "kl": ("s", "tab:cyan"), "patience": ("D", "0.35")}


def load_json(path: Path) -> dict | None:
    """A JSON file, or None (with a warning) if it is missing."""
    if not path.exists():
        log.warning("missing %s", path)
        return None
    return json.loads(path.read_text())


def line_style(run: Run) -> dict:
    """Matplotlib kwargs for a run's lines: own colour, dashed for references."""
    return {"color": run.colour, "ls": "--" if run.reference else "-", "lw": 1.6 if run.reference else 2.0,
            "label": run.label}


def separation_curve(cls_svd: dict) -> list[float]:
    """Separation after block call 1..12 (drops layer 0 and the final-norm layer); same as T11's plot.py."""
    by_layer = cls_svd["separation"][SEPARATION_SET]
    last_call = cls_svd["final_norm_layer"] - 1
    return [by_layer[str(layer)] for layer in range(1, last_call + 1)]


def finish(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Axis labels, title, grid, legend."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def plot_exit_curves(exits: dict, out_path: Path) -> None:
    """2 x 3 panels (row = layout): fixed-depth curve and the three exit rules on the same axes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for ax, run in zip(axes.flat, RUNS):
        result = exits.get(run.checkpoint_stem)
        if result is None:
            ax.set_title(f"{run.label}: no exit analysis yet", fontsize=10)
            continue
        fixed = result["acc_per_loop"]
        ax.plot(range(1, len(fixed) + 1), fixed, color=run.colour, lw=2, marker="o", ms=3, label="fixed depth")
        for rule, (marker, colour) in RULE_STYLES.items():
            points = sorted(result["curves"][rule], key=lambda p: p["mean_loops"])
            ax.plot([p["mean_loops"] for p in points], [p["acc"] for p in points], color=colour, marker=marker, ms=4,
                    lw=1, label=rule)
        ax.set_xlim(3.5, 10.3)
        ax.set_ylim(max(0.0, fixed[-1] - 35), fixed[-1] + 3)
        finish(ax, "loops used (mean per image for the rules)", "test top-1 (%)",
               f"{run.label}, flip 9→10 {result['flip_rate_9_10']}%")
    fig.suptitle("Early exit vs fixed depth (T11-c layout top row, T11-d layout bottom row)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_acc_per_loop(exits: dict, out_path: Path) -> None:
    """Fixed-depth top-1 at loop t for every model."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for run in RUNS:
        if (result := exits.get(run.checkpoint_stem)) is not None:
            acc = result["acc_per_loop"]
            ax.plot(range(1, len(acc) + 1), acc, marker="o", ms=3, **line_style(run))
    finish(ax, "loop t (stop here, read out)", "CIFAR-100 test top-1 (%)",
           "Readout after every loop (T11-c layout: through the tail; T11-d layout: final norm + head)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rel_update(out_path: Path) -> None:
    """T12 metric (a): rel_cls (solid) and rel_patch (dotted, thinner) per loop, one panel per layout."""
    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True, sharey=True)
    for run in RUNS:
        diag = load_json(run.diag_json)
        if diag is None:
            continue
        ax = ax_c if run.layout == "T11-c" else ax_d
        means = diag["means"]
        loops = range(1, len(means["rel_cls"]) + 1)
        ax.plot(loops, means["rel_cls"], color=run.colour, lw=2, marker="o", ms=3, label=f"{run.label} CLS")
        ax.plot(loops, means["rel_patch"], color=run.colour, lw=1, ls=":", label=f"{run.name} patches")
    for ax, layout in ((ax_c, "T11-c"), (ax_d, "T11-d")):
        finish(ax, "loop t", "||x_t - x_t-1|| / ||x_t-1|| (mean)", f"{layout} layout: CLS (solid) vs patches (dotted)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(out_path: Path) -> None:
    """Train CE of the final output (T13: train_ce_final; T11: train_loss, the same quantity) and test loss."""
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for run in RUNS:
        train = load_json(run.train_json)
        if train is None:
            continue
        history = train["history"]
        epochs = [h["epoch"] for h in history]
        ax_train.plot(epochs, [h.get("train_ce_final", h["train_loss"]) for h in history], **line_style(run))
        ax_test.plot(epochs, [h["test_loss"] for h in history], **line_style(run))
    finish(ax_train, "epoch", "train CE of the final output", "Train loss (final output only, comparable across runs)")
    finish(ax_test, "epoch", "test CE (label smoothing 0.1)", "Test loss")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_separation(out_path: Path) -> None:
    """Separation after every block call, train and test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for ax, split in zip(axes, ("train", "test")):
        for run in RUNS:
            cls_svd = load_json(Path(f"{run.cls_svd_prefix}_{split}.json"))
            if cls_svd is not None:
                curve = separation_curve(cls_svd)
                ax.plot(range(1, len(curve) + 1), curve, marker="o", ms=3, **line_style(run))
        ax.axhline(1.0, color="0.5", ls=":", lw=1)
        finish(ax, "block call (body: 2-11 for T11-c layout, 3-12 for T11-d layout)",
               "tr(within) / tr(between)", f"Class separation of CLS, 4 classes, {split}")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def best_rule(result: dict) -> tuple[str, dict] | None:
    """The exit rule whose cheapest point within 1 pt of loop 10 uses the fewest mean loops."""
    options = [(rule, summary) for rule, summary in result["vs_fixed"].items() if summary["cheapest_near_full"]]
    return min(options, key=lambda item: item[1]["cheapest_near_full"]["mean_loops"]) if options else None


def exit_cells(result: dict | None) -> list[str]:
    """Loop-5 acc, flip rate and the best exit rule (or dashes when the exit analysis is missing)."""
    if result is None:
        return ["–"] * 3
    chosen = best_rule(result)
    if chosen is None:
        rule_text = "none within 1 pt"
    else:
        rule, summary = chosen
        point = summary["cheapest_near_full"]
        rule_text = (f"{rule} {point['threshold']}: {point['mean_loops']} loops ({point['acc']:.2f}%) vs fixed "
                     f"{summary['fixed_loops_near_full']}")
    return [f"{result['acc_per_loop'][4]:.2f}", f"{result['flip_rate_9_10']:.2f}", rule_text]


def separation_cells(run: Run) -> list[str]:
    """Separation (test) at the last body loop, the lowest body value (and its call), and after the last block."""
    cls_svd = load_json(Path(f"{run.cls_svd_prefix}_test.json"))
    if cls_svd is None:
        return ["–"] * 3
    curve = separation_curve(cls_svd)
    body = [curve[call - 1] for call in run.body_calls]
    lowest = min(body)
    return [f"{body[-1]:.2f}", f"{lowest:.2f} ({run.body_calls[body.index(lowest)]})", f"{curve[-1]:.2f}"]


def table_row(run: Run, exits: dict) -> str:
    """One markdown row: accuracy, exit numbers, early rel_cls and separation."""
    train = load_json(run.train_json)
    cells = [run.name, run.layout, run.loss]
    if train:
        cells += [f"{train['params'] / 1e6:.3f}", f"{train['final_test_acc']:.2f}", f"{train['best_test_acc']:.2f}"]
    else:
        cells += ["–"] * 3
    cells += exit_cells(exits.get(run.checkpoint_stem))
    diag = load_json(run.diag_json)
    cells.append(f"{sum(diag['means']['rel_cls'][:EARLY_LOOPS]) / EARLY_LOOPS:.3f}" if diag else "–")
    cells += separation_cells(run)
    return "| " + " | ".join(cells) + " |"


def write_table(exits: dict, out_path: Path) -> str:
    """The T13 results table (markdown), also written to `out_path`."""
    header = ("| Run | Layout | Loss | Params (M) | Final top-1 | Best top-1 | Loop-5 acc | Flip 9→10 (%) | "
              "Fewest loops within 1 pt of loop 10 | Mean rel_cls loops 1-5 | Sep. last body loop (test) | "
              "Min body sep. (test, call) | Sep. last block (test) |")
    lines = [header, "|" + "---|" * 13] + [table_row(run, exits) for run in RUNS]
    table = "\n".join(lines) + "\n"
    out_path.write_text(table)
    return table


def main() -> None:
    """All five plots and the table from whatever results exist."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    exits = load_json(RESULTS / "exit_analysis.json") or {}
    plot_exit_curves(exits, RESULTS / "exit_curves.png")
    plot_acc_per_loop(exits, RESULTS / "acc_per_loop.png")
    plot_rel_update(RESULTS / "rel_update.png")
    plot_loss_curves(RESULTS / "loss_curves.png")
    plot_separation(RESULTS / "separation_vs_loop.png")
    print(write_table(exits, RESULTS / "summary.md"))


if __name__ == "__main__":
    main()
