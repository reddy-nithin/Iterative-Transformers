"""T16 plots and tables: coverage mask v2 (T16-a eta 0.5, T16-b eta 1.0) vs the reference T13-d.

1. results/attn_cos_entropy.png: CLS attention cos between successive loops (left) and CLS attention entropy per loop
   (right).
2. results/supply_per_loop.png: mean supply and share of patches with supply < 0.1 after each loop.
3. results/acc_per_loop.png: fixed-depth curve (readout at loop t), full test set.
4. results/exit_curves.png: fixed depth + confidence, KL and patience exit rules, one panel per model.
Also writes results/tables.md (top-1, per-loop tables, T12 (a)-(c), separation).

Inputs: results/exit_analysis.json, results/t12_diag/diag_*.json, results/cls_svd_*_test.json and the training JSONs
of this task; T13's exit_analysis.json, t12_diag and cls_svd JSONs for T13-d.

Run from the repo root:  .venv/bin/python tasks/T16_coverage_mask_v2/plot.py
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
T13_RESULTS = TASK_DIR.parent / "T13_per_loop_readout_loss" / "results"
SEPARATION_SET = "all classes"
LATE_PAIRS = range(6, 10)  # cos pairs (6,7) .. (9,10), as in the pre-flight
RULE_STYLES = {"confidence": ("^", "tab:green"), "kl": ("s", "tab:cyan"), "patience": ("D", "0.35")}
MAX_ENTROPY = 5.28  # ln 196

log = logging.getLogger("t16_plot")


@dataclass(frozen=True)
class Run:
    """One model in the comparison: where its files are and how it is drawn."""
    label: str
    stem: str  # checkpoint stem (key in exit_analysis.json, suffix of the diag JSON)
    results: Path
    train_json: str
    cls_svd_tag: str
    colour: str
    masked: bool


RUNS = (
    Run("T13-d (ref)", "T13-d_unroll_loss_margin", T13_RESULTS, "unroll_T13-d.json", "T13-d", "tab:brown", False),
    Run("T16-a η 0.5", "T16-a_covmask_v2_eta05", RESULTS, "unroll_T16-a.json", "T16-a", "tab:blue", True),
    Run("T16-b η 1.0", "T16-b_covmask_v2_eta1", RESULTS, "unroll_T16-b.json", "T16-b", "tab:purple", True),
)


def load(path: Path) -> dict:
    """A results JSON; raises if missing (the plots need it)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run run.sh for this model first")
    return json.loads(path.read_text())


def run_data(run: Run) -> dict:
    """Everything the plots and tables need for one model."""
    exits = load(run.results / "exit_analysis.json")[run.stem]
    diag = load(run.results / "t12_diag" / f"diag_{run.stem}.json")
    separation = load(run.results / f"cls_svd_{run.cls_svd_tag}_test.json")
    history = load(run.results / run.train_json)["history"]
    by_layer = separation["separation"][SEPARATION_SET]
    calls = range(1, separation["final_norm_layer"])
    return {"exits": exits, "means": diag["means"], "final_top1": history[-1]["test_acc"],
            "separation": [round(by_layer[str(c)], 3) for c in calls]}


def finish(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Axis labels, title, grid, legend."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def plot_attention(data: dict[str, dict], out_path: Path) -> None:
    """Successive-loop CLS attention cos (left) and attention entropy (right) per loop."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for run in RUNS:
        means = data[run.label]["means"]
        left.plot(range(2, 11), means["attn_cos"], color=run.colour, marker="o", ms=3, label=run.label)
        right.plot(range(1, 11), means["attn_entropy"], color=run.colour, marker="o", ms=3, label=run.label)
    left.axhline(0.8, color="0.5", lw=0.8, ls="--", label="0.8 (prediction)")
    right.axhline(MAX_ENTROPY, color="0.5", lw=0.8, ls="--", label="uniform (5.28)")
    finish(left, "loop t", "cos(A_t-1, A_t)", "CLS → patch attention, successive loops")
    finish(right, "loop t", "entropy (nats)", "CLS → patch attention entropy")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_supply(data: dict[str, dict], out_path: Path) -> None:
    """Mean supply and share of near-empty patches after each loop (masked runs only)."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for run in (r for r in RUNS if r.masked):
        means = data[run.label]["means"]
        ax.plot(range(1, 11), means["supply_mean"], color=run.colour, marker="o", ms=3, label=f"{run.label} mean c")
        ax.plot(range(1, 11), means["supply_share_below_0.1"], color=run.colour, ls="--", marker="s", ms=3,
                label=f"{run.label} share c < 0.1")
    ax.set_ylim(-0.02, 1.02)
    finish(ax, "after loop t", "value", "Coverage supply per loop (test set)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_acc_per_loop(data: dict[str, dict], out_path: Path) -> None:
    """Fixed-depth readout accuracy per loop."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for run in RUNS:
        curve = data[run.label]["exits"]["acc_per_loop"]
        ax.plot(range(1, len(curve) + 1), curve, color=run.colour, marker="o", ms=3, label=run.label)
    finish(ax, "loop t (readout after loop t)", "test top-1 (%)", "Fixed-depth accuracy per loop")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_exit_curves(data: dict[str, dict], out_path: Path) -> None:
    """Fixed depth and the three exit rules, one panel per model."""
    fig, axes = plt.subplots(1, len(RUNS), figsize=(16, 5), constrained_layout=True)
    for ax, run in zip(axes, RUNS):
        exits = data[run.label]["exits"]
        fixed = exits["acc_per_loop"]
        ax.plot(range(1, len(fixed) + 1), fixed, color=run.colour, lw=2, marker="o", ms=3, label="fixed depth")
        for rule, (marker, colour) in RULE_STYLES.items():
            points = sorted(exits["curves"][rule], key=lambda p: p["mean_loops"])
            ax.plot([p["mean_loops"] for p in points], [p["acc"] for p in points], color=colour, marker=marker,
                    ms=4, lw=1, label=rule)
        finish(ax, "loops used (mean per image for the rules)", "test top-1 (%)", run.label)
    fig.suptitle("Exit rules vs fixed depth (T13 exit_analysis.py)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_loop_table(title: str, data: dict[str, dict], values: Callable[[dict], list[float]], columns: range,
                   digits: int, runs: tuple[Run, ...] = RUNS) -> list[str]:
    """Markdown table with one row per model, one column per loop (or block call)."""
    lines = [f"**{title}**", "", "| model | " + " | ".join(str(c) for c in columns) + " |",
             "|---" * (len(columns) + 1) + "|"]
    for run in runs:
        lines.append(f"| {run.label} | " + " | ".join(f"{v:.{digits}f}" for v in values(data[run.label])) + " |")
    return lines + [""]


def summary_table(data: dict[str, dict]) -> list[str]:
    """One row per model: top-1, peak of the per-loop curve, late attention cos, final separation."""
    lines = ["| model | final top-1 (logged) | loop-10 top-1 (fp32) | peak per-loop (loop) | late attn cos "
             "(6-7..9-10) | sep. after last call (test) |", "|---|---|---|---|---|---|"]
    for run in RUNS:
        d = data[run.label]
        curve = d["exits"]["acc_per_loop"]
        late_cos = sum(d["means"]["attn_cos"][t - 1] for t in LATE_PAIRS) / len(LATE_PAIRS)
        lines.append(f"| {run.label} | {d['final_top1']:.2f} | {curve[-1]:.2f} | {max(curve):.2f} "
                     f"({curve.index(max(curve)) + 1}) | {late_cos:.3f} | {d['separation'][-1]:.3f} |")
    return lines + [""]


def write_tables(data: dict[str, dict], out_path: Path) -> None:
    """Summary plus per-loop tables, as markdown."""
    loops, pairs = range(1, 11), range(2, 11)
    masked = tuple(r for r in RUNS if r.masked)
    n_calls = len(data[RUNS[0].label]["separation"])
    lines = ["# T16 tables (generated by plot.py)", ""] + summary_table(data)
    lines += per_loop_table("Fixed-depth top-1 per loop", data, lambda d: d["exits"]["acc_per_loop"], loops, 2)
    lines += per_loop_table("T12 (a) rel_cls = ‖ΔCLS‖ / ‖CLS‖", data, lambda d: d["means"]["rel_cls"], loops, 3)
    lines += per_loop_table("T12 (b) new-direction share", data, lambda d: d["means"]["new_direction_share"], loops, 3)
    lines += per_loop_table("T12 (c) cos(Δ_t, Δ_t-1)", data, lambda d: d["means"]["cls_update_cos"], pairs, 3)
    lines += per_loop_table("CLS attention cos(A_t-1, A_t)", data, lambda d: d["means"]["attn_cos"], pairs, 3)
    lines += per_loop_table("CLS attention entropy (nats, max 5.28)", data, lambda d: d["means"]["attn_entropy"],
                            loops, 2)
    lines += per_loop_table("Mean supply after loop t", data, lambda d: d["means"]["supply_mean"], loops, 3, masked)
    lines += per_loop_table("Share of patches with supply < 0.1 after loop t", data,
                            lambda d: d["means"]["supply_share_below_0.1"], loops, 3, masked)
    lines += per_loop_table("Class separation per block call (test, 4 classes; calls 1-2 = B0, B1; 3-12 = B2 loops)",
                            data, lambda d: d["separation"], range(1, n_calls + 1), 3)
    out_path.write_text("\n".join(lines))


def main() -> None:
    """All four plots and the tables."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = {run.label: run_data(run) for run in RUNS}
    plot_attention(data, RESULTS / "attn_cos_entropy.png")
    plot_supply(data, RESULTS / "supply_per_loop.png")
    plot_acc_per_loop(data, RESULTS / "acc_per_loop.png")
    plot_exit_curves(data, RESULTS / "exit_curves.png")
    write_tables(data, RESULTS / "tables.md")
    log.info("wrote the 4 plots and tables.md in %s", RESULTS)


if __name__ == "__main__":
    main()
