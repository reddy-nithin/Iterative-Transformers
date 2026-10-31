"""T14 plots + table: T14-a (T13-d + coverage mask) against its base T13-d, in every plot and table.

1. results/exit_curves.png: one panel per model, fixed-depth curve + confidence, KL and patience exit rules (T13).
2. results/acc_per_loop.png: fixed-depth curve (readout at loop t).
3. results/rel_update.png: T12 metric (a), rel_cls (solid) and rel_patch (dotted) per loop.
4. results/attn_cos_entropy.png: T12 metric (b), cos(A_t, A_t+1) (with the 0.8 rule line and the 0.5 prediction line)
   and entropy of A_t; plus the mean coverage supply left after each loop for T14-a.
5. results/new_direction.png: T12 metric (c), new-direction share and cos of successive CLS updates.
6. results/separation_vs_loop.png: class separation of CLS after every block call, train and test.
7. results/loss_curves.png: train CE of the final output, test loss, and the learned eta per epoch.
Also writes results/summary.md (the results table).

T13-d's numbers are read from the T13 folder (same scripts, same images), nothing is recomputed.
Run from the repo root:  .venv/bin/python tasks/T14_addon_from_diagnostics/plot.py
"""
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
T13_RESULTS = TASK_DIR.parent / "T13_per_loop_readout_loss" / "results"


def load_t13_plot():
    """T13's plot.py, loaded by path (this file has the same name, so a plain import would be ambiguous)."""
    spec = importlib.util.spec_from_file_location("t13_plot", TASK_DIR.parent / "T13_per_loop_readout_loss" / "plot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_t13 = load_t13_plot()
RULE_STYLES, best_rule, finish, load_json, separation_curve = (
    _t13.RULE_STYLES, _t13.best_rule, _t13.finish, _t13.load_json, _t13.separation_curve)

BODY_CALLS = range(3, 13)  # unrolling n = 3: blocks 1, 2 distinct, then the body at calls 3..12
LATE_LOOP_START = 4  # T12 rule: late loops = loops >= 4
PREDICTED_COS = 0.5  # T14 prediction: metric (b) drops below this
RULE_COS = 0.8  # T12 decision-rule threshold
EARLY_LOOPS = 5
MAX_ENTROPY = 5.278  # log 196

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Run:
    """One model in the comparison: where its files are and how it is drawn."""
    name: str
    description: str
    checkpoint_stem: str
    train_json: Path
    exit_json: Path
    cls_svd_prefix: Path  # + "_train.json" / "_test.json"
    diag_json: Path
    colour: str
    reference: bool = False

    @property
    def label(self) -> str:
        """Legend label."""
        return f"{self.name} ({self.description})"

    def style(self) -> dict:
        """Matplotlib kwargs: own colour, dashed for the reference."""
        return {"color": self.colour, "ls": "--" if self.reference else "-", "lw": 1.6 if self.reference else 2.0,
                "label": self.label}


RUNS = [
    Run("T13-d", "base: every loop + margin", "T13-d_unroll_loss_margin", T13_RESULTS / "unroll_T13-d.json",
        T13_RESULTS / "exit_analysis.json", T13_RESULTS / "cls_svd_T13-d",
        T13_RESULTS / "t12_diag" / "diag_T13-d_unroll_loss_margin.json", "#e6ab02", reference=True),
    Run("T14-a", "base + coverage mask", "T14-a_coverage_mask", RESULTS / "unroll_T14-a.json",
        RESULTS / "exit_analysis.json", RESULTS / "cls_svd_T14-a",
        RESULTS / "t12_diag" / "diag_T14-a_coverage_mask.json", "tab:brown"),
]


def exit_result(run: Run) -> dict | None:
    """The run's entry in its exit-analysis JSON."""
    exits = load_json(run.exit_json) or {}
    return exits.get(run.checkpoint_stem)


def plot_exit_curves(out_path: Path) -> None:
    """One panel per model: fixed-depth curve and the three exit rules on the same axes."""
    fig, axes = plt.subplots(1, len(RUNS), figsize=(11, 4.3), constrained_layout=True, sharey=True)
    for ax, run in zip(axes, RUNS):
        result = exit_result(run)
        if result is None:
            ax.set_title(f"{run.label}: no exit analysis yet", fontsize=10)
            continue
        fixed = result["acc_per_loop"]
        ax.plot(range(1, len(fixed) + 1), fixed, color=run.colour, lw=2, marker="o", ms=3, label="fixed depth")
        for rule, (marker, colour) in RULE_STYLES.items():
            points = sorted(result["curves"][rule], key=lambda p: p["mean_loops"])
            ax.plot([p["mean_loops"] for p in points], [p["acc"] for p in points], color=colour, marker=marker, ms=4,
                    lw=1, label=rule)
        ax.set_xlim(0.7, 10.3)
        finish(ax, "loops used (mean per image for the rules)", "test top-1 (%)",
               f"{run.label}, flip 9→10 {result['flip_rate_9_10']}%")
    fig.suptitle("Early exit vs fixed depth")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_acc_per_loop(out_path: Path) -> None:
    """Fixed-depth top-1 at loop t."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for run in RUNS:
        if (result := exit_result(run)) is not None:
            acc = result["acc_per_loop"]
            ax.plot(range(1, len(acc) + 1), acc, marker="o", ms=3, **run.style())
    finish(ax, "loop t (stop here, read out)", "CIFAR-100 test top-1 (%)", "Readout after every loop (final norm + head)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def diag_lines(ax, key: str, first_loop: int = 1, **extra) -> None:
    """One line per model of the T12 per-loop mean `key`."""
    for run in RUNS:
        if (diag := load_json(run.diag_json)) is None or key not in diag["means"]:
            continue
        values = diag["means"][key]
        style = {**run.style(), **extra}
        ax.plot(range(first_loop, first_loop + len(values)), values, marker="o", ms=3, **style)


def plot_rel_update(out_path: Path) -> None:
    """T12 metric (a): rel_cls (solid) and rel_patch (dotted)."""
    fig, ax = plt.subplots(figsize=(7, 4.3), constrained_layout=True)
    for run in RUNS:
        if (diag := load_json(run.diag_json)) is None:
            continue
        means = diag["means"]
        loops = range(1, len(means["rel_cls"]) + 1)
        ax.plot(loops, means["rel_cls"], color=run.colour, lw=2, marker="o", ms=3, label=f"{run.label} CLS")
        ax.plot(loops, means["rel_patch"], color=run.colour, lw=1, ls=":", label=f"{run.name} patches")
    finish(ax, "loop t", "||x_t - x_t-1|| / ||x_t-1|| (mean)", "(a) relative update: CLS (solid) vs patches (dotted)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attention(out_path: Path) -> None:
    """T12 metric (b) and the supply left after each loop."""
    fig, (ax_cos, ax_ent, ax_sup) = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True)
    diag_lines(ax_cos, "attn_cos")
    ax_cos.axhline(RULE_COS, color="0.5", ls=":", lw=1)
    ax_cos.axhline(PREDICTED_COS, color="tab:red", ls=":", lw=1)
    finish(ax_cos, "loop t", "cos(A_t, A_t+1), per head then mean",
           "(b) same patches next loop? (grey 0.8 rule, red 0.5 prediction)")
    diag_lines(ax_ent, "attn_entropy")
    ax_ent.axhline(MAX_ENTROPY, color="0.5", ls=":", lw=1)
    finish(ax_ent, "loop t", "entropy of A_t (nats, max 5.28)", "(b) how spread is CLS attention?")
    diag_lines(ax_sup, "supply_mean")
    ax_sup.set_ylim(0, 1.05)
    finish(ax_sup, "loop t", "mean supply c over patches after loop t", "coverage supply left (T14-a only)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_new_direction(out_path: Path) -> None:
    """T12 metric (c)."""
    fig, (ax_new, ax_succ) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    diag_lines(ax_new, "new_direction_share")
    finish(ax_new, "loop t", "||o_perp|| / ||o||", "(c) share of the CLS attention update off the CLS direction")
    diag_lines(ax_succ, "cls_update_cos", first_loop=2)
    ax_succ.axhline(0.0, color="0.5", ls=":", lw=1)
    finish(ax_succ, "loop t", "cos(ΔCLS_t, ΔCLS_t-1)", "(c) are successive CLS updates orthogonal?")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_separation(out_path: Path) -> None:
    """Separation after every block call, train and test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for ax, split in zip(axes, ("train", "test")):
        for run in RUNS:
            if (cls_svd := load_json(Path(f"{run.cls_svd_prefix}_{split}.json"))) is not None:
                curve = separation_curve(cls_svd)
                ax.plot(range(1, len(curve) + 1), curve, marker="o", ms=3, **run.style())
        ax.axhline(1.0, color="0.5", ls=":", lw=1)
        finish(ax, "block call (body loops = calls 3-12)", "tr(within) / tr(between)",
               f"Class separation of CLS, 4 classes, {split}")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(out_path: Path) -> None:
    """Train CE of the final output, test loss, and eta per epoch."""
    fig, (ax_train, ax_test, ax_eta) = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True)
    for run in RUNS:
        if (train := load_json(run.train_json)) is None:
            continue
        history = train["history"]
        epochs = [h["epoch"] for h in history]
        ax_train.plot(epochs, [h.get("train_ce_final", h["train_loss"]) for h in history], **run.style())
        ax_test.plot(epochs, [h["test_loss"] for h in history], **run.style())
        if all("eta" in h for h in history):
            ax_eta.plot(epochs, [h["eta"] for h in history], marker="o", ms=3, **run.style())
    finish(ax_train, "epoch", "train CE of the final output", "Train loss (final output only)")
    finish(ax_test, "epoch", "test CE (label smoothing 0.1)", "Test loss")
    ax_eta.axhline(0.5, color="0.5", ls=":", lw=1)
    finish(ax_eta, "epoch", "eta (end of epoch)", "Learned depletion rate eta (init 0.5)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def rule_text(result: dict) -> str:
    """The cheapest exit rule within 1 pt of loop 10, vs fixed depth."""
    chosen = best_rule(result)
    if chosen is None:
        return "none within 1 pt"
    rule, summary = chosen
    point = summary["cheapest_near_full"]
    return (f"{rule} {point['threshold']}: {point['mean_loops']} loops ({point['acc']:.2f}%) vs fixed "
            f"{summary['fixed_loops_near_full']}")


def late_mean(values: list[float]) -> float:
    """Mean of a pair metric over loops >= LATE_LOOP_START (values[t-1] = loop t)."""
    late = values[LATE_LOOP_START - 1:]
    return sum(late) / len(late)


def table_row(run: Run) -> str:
    """One markdown row."""
    train, result, diag = load_json(run.train_json), exit_result(run), load_json(run.diag_json)
    cells = [run.name, run.description]
    if train:
        eta = train["history"][-1].get("eta")
        cells += [f"{train['params']:,}", "–" if eta is None else f"{eta:.4f}",
                  f"{train['final_test_acc']:.2f}", f"{train['best_test_acc']:.2f}"]
    else:
        cells += ["–"] * 4
    if result:
        cells += [f"{result['acc_per_loop'][4]:.2f}", f"{result['flip_rate_9_10']:.2f}", rule_text(result)]
    else:
        cells += ["–"] * 3
    if diag:
        means = diag["means"]
        cells += [f"{sum(means['rel_cls'][:EARLY_LOOPS]) / EARLY_LOOPS:.3f}", f"{late_mean(means['attn_cos']):.3f}",
                  f"{sum(means['attn_entropy']) / len(means['attn_entropy']):.2f}"]
    else:
        cells += ["–"] * 3
    cls_svd = load_json(Path(f"{run.cls_svd_prefix}_test.json"))
    if cls_svd:
        body = [separation_curve(cls_svd)[call - 1] for call in BODY_CALLS]
        lowest = min(body)
        cells += [f"{body[-1]:.2f}", f"{lowest:.2f} ({BODY_CALLS[body.index(lowest)]})"]
    else:
        cells += ["–"] * 2
    return "| " + " | ".join(cells) + " |"


def write_table(out_path: Path) -> str:
    """The T14 results table (markdown), also written to `out_path`."""
    header = ("| Run | Model | Params | Learned η | Final top-1 | Best top-1 | Loop-5 acc | Flip 9→10 (%) | "
              "Fewest loops within 1 pt of loop 10 | Mean rel_cls loops 1-5 | Late-loop cos(A_t, A_t+1) | "
              "Mean H(A_t) | Sep. last loop (test) | Min loop sep. (test, call) |")
    lines = [header, "|" + "---|" * 14] + [table_row(run) for run in RUNS]
    table = "\n".join(lines) + "\n"
    out_path.write_text(table)
    return table


def main() -> None:
    """All plots and the table from whatever results exist."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RESULTS.mkdir(parents=True, exist_ok=True)
    plot_exit_curves(RESULTS / "exit_curves.png")
    plot_acc_per_loop(RESULTS / "acc_per_loop.png")
    plot_rel_update(RESULTS / "rel_update.png")
    plot_attention(RESULTS / "attn_cos_entropy.png")
    plot_new_direction(RESULTS / "new_direction.png")
    plot_separation(RESULTS / "separation_vs_loop.png")
    plot_loss_curves(RESULTS / "loss_curves.png")
    print(write_table(RESULTS / "summary.md"))


if __name__ == "__main__":
    main()
