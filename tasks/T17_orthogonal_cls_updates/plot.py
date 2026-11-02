"""T17 plots and tables: orthogonal CLS-update loss (T17-a) and orthogonal + size floor (T17-b) vs the reference T13-c.

1. results/update_cos_per_loop.png: successive CLS-update cos, T12 (c), per loop.
2. results/rel_update.png: T12 (a) rel_cls = ||CLS_t - CLS_t-1|| / ||CLS_t-1|| per loop (r_min = 0.25 marked).
3. results/acc_per_loop.png: fixed-depth curve (readout at loop t), full test set.
4. results/separation_vs_loop.png: class separation of CLS after every block call, train and test.
Also writes results/tables.md (top-1, exit rules, per-loop tables).

Inputs: results/exit_analysis.json, results/t12_diag/diag_*.json, results/cls_svd_*_{train,test}.json and the training
JSONs of this task; T13's files for T13-c.

Run from the repo root:  .venv/bin/python tasks/T17_orthogonal_cls_updates/plot.py
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
LATE_LOOPS = range(6, 11)  # loops 6..10
R_MIN = 0.25  # T17-b size floor

log = logging.getLogger("t17_plot")


@dataclass(frozen=True)
class Run:
    """One model in the comparison: where its files are and how it is drawn."""
    label: str
    stem: str  # checkpoint stem (key in exit_analysis.json, suffix of the diag JSON)
    results: Path
    train_json: str
    cls_svd_tag: str
    colour: str


RUNS = (
    Run("T13-c (ref)", "T13-c_unroll_readout_every_loop", T13_RESULTS, "unroll_T13-c.json", "T13-c", "tab:orange"),
    Run("T17-a orth", "T17-a_orth_cls", RESULTS, "unroll_T17-a.json", "T17-a", "tab:blue"),
    Run("T17-b orth + large", "T17-b_orth_large_cls", RESULTS, "unroll_T17-b.json", "T17-b", "tab:purple"),
)


def load(path: Path) -> dict:
    """A results JSON; raises if missing (the plots need it)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run sweep.sh for this model first")
    return json.loads(path.read_text())


def separation_curve(path: Path) -> list[float]:
    """Separation after block call 1..12 (drops layer 0 and the final-norm layer), as in T11 / T13."""
    cls_svd = load(path)
    by_layer = cls_svd["separation"][SEPARATION_SET]
    return [round(by_layer[str(c)], 3) for c in range(1, cls_svd["final_norm_layer"])]


def run_data(run: Run) -> dict:
    """Everything the plots and tables need for one model."""
    return {"exits": load(run.results / "exit_analysis.json")[run.stem],
            "means": load(run.results / "t12_diag" / f"diag_{run.stem}.json")["means"],
            "final_top1": load(run.results / run.train_json)["history"][-1]["test_acc"],
            "sep_test": separation_curve(run.results / f"cls_svd_{run.cls_svd_tag}_test.json"),
            "sep_train": separation_curve(run.results / f"cls_svd_{run.cls_svd_tag}_train.json")}


def finish(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Axis labels, title, grid, legend."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def plot_per_loop(data: dict[str, dict], values: Callable[[dict], list[float]], first_loop: int, ylabel: str,
                  title: str, out_path: Path, reference_line: tuple[float, str] | None = None) -> None:
    """One line per model of a per-loop quantity."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for run in RUNS:
        curve = values(data[run.label])
        ax.plot(range(first_loop, first_loop + len(curve)), curve, color=run.colour, marker="o", ms=3, label=run.label)
    if reference_line is not None:
        ax.axhline(reference_line[0], color="0.5", lw=0.8, ls="--", label=reference_line[1])
    finish(ax, "loop t", ylabel, title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_separation(data: dict[str, dict], out_path: Path) -> None:
    """Separation per block call, test (left) and train (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for ax, key, split in zip(axes, ("sep_test", "sep_train"), ("test", "train")):
        for run in RUNS:
            curve = data[run.label][key]
            ax.plot(range(1, len(curve) + 1), curve, color=run.colour, marker="o", ms=3, label=run.label)
        ax.axhline(1.0, color="0.5", lw=0.8, ls="--", label="within = between")
        finish(ax, "block call (1-2 = B0, B1; 3-12 = B2 loops 1-10)", "tr(within) / tr(between)",
               f"Class separation of CLS ({split}, 4 classes)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def late_mean(values: list[float], first_loop: int) -> float:
    """Mean over loops 6..10 of a per-loop list whose entry 0 is loop `first_loop`."""
    return sum(values[t - first_loop] for t in LATE_LOOPS) / len(LATE_LOOPS)


def summary_table(data: dict[str, dict]) -> list[str]:
    """One row per model: top-1, per-loop peak, late update cos and rel_cls, best exit-rule gain, final separation."""
    lines = ["| model | final top-1 (logged) | loop-10 top-1 (fp32) | peak per-loop (loop) | late update cos (loops "
             "6-10) | late rel_cls (loops 6-10) | best exit-rule gain (rule, loops) | sep. last call test / train |",
             "|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        d = data[run.label]
        curve = d["exits"]["acc_per_loop"]
        best_rule, best = max(((rule, s["best_gain"]) for rule, s in d["exits"]["vs_fixed"].items()),
                              key=lambda item: item[1]["gain_vs_fixed"])
        lines.append(f"| {run.label} | {d['final_top1']:.2f} | {curve[-1]:.2f} | {max(curve):.2f} "
                     f"({curve.index(max(curve)) + 1}) | {late_mean(d['means']['cls_update_cos'], 2):.3f} | "
                     f"{late_mean(d['means']['rel_cls'], 1):.3f} | {best['gain_vs_fixed']:+.2f} ({best_rule}, "
                     f"{best['mean_loops']}) | {d['sep_test'][-1]:.3f} / {d['sep_train'][-1]:.3f} |")
    return lines + [""]


def per_loop_table(title: str, data: dict[str, dict], values: Callable[[dict], list[float]], columns: range,
                   digits: int) -> list[str]:
    """Markdown table with one row per model, one column per loop (or block call)."""
    lines = [f"**{title}**", "", "| model | " + " | ".join(str(c) for c in columns) + " |",
             "|---" * (len(columns) + 1) + "|"]
    for run in RUNS:
        lines.append(f"| {run.label} | " + " | ".join(f"{v:.{digits}f}" for v in values(data[run.label])) + " |")
    return lines + [""]


def write_tables(data: dict[str, dict], out_path: Path) -> None:
    """Summary plus per-loop tables, as markdown."""
    loops, pairs = range(1, 11), range(2, 11)
    calls = range(1, len(data[RUNS[0].label]["sep_test"]) + 1)
    lines = ["# T17 tables (generated by plot.py)", ""] + summary_table(data)
    lines += per_loop_table("Fixed-depth top-1 per loop", data, lambda d: d["exits"]["acc_per_loop"], loops, 2)
    lines += per_loop_table("T12 (a) rel_cls = ‖ΔCLS‖ / ‖CLS‖", data, lambda d: d["means"]["rel_cls"], loops, 3)
    lines += per_loop_table("T12 (b) new-direction share", data, lambda d: d["means"]["new_direction_share"], loops, 3)
    lines += per_loop_table("T12 (c) cos(Δ_t, Δ_t-1)", data, lambda d: d["means"]["cls_update_cos"], pairs, 3)
    lines += per_loop_table("Class separation per block call (test)", data, lambda d: d["sep_test"], calls, 3)
    lines += per_loop_table("Class separation per block call (train)", data, lambda d: d["sep_train"], calls, 3)
    out_path.write_text("\n".join(lines))


def main() -> None:
    """All four plots and the tables."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = {run.label: run_data(run) for run in RUNS}
    plot_per_loop(data, lambda d: d["means"]["cls_update_cos"], 2, "cos(Δ_t, Δ_t-1)",
                  "Successive CLS-update cos (T12 (c))", RESULTS / "update_cos_per_loop.png", (0.5, "0.5 (prediction)"))
    plot_per_loop(data, lambda d: d["means"]["rel_cls"], 1, "‖ΔCLS_t‖ / ‖CLS_t-1‖", "Relative CLS update (T12 (a))",
                  RESULTS / "rel_update.png", (R_MIN, "r_min 0.25 (T17-b floor)"))
    plot_per_loop(data, lambda d: d["exits"]["acc_per_loop"], 1, "test top-1 (%)", "Fixed-depth accuracy per loop",
                  RESULTS / "acc_per_loop.png")
    plot_separation(data, RESULTS / "separation_vs_loop.png")
    write_tables(data, RESULTS / "tables.md")
    log.info("wrote the 4 plots and tables.md in %s", RESULTS)


if __name__ == "__main__":
    main()
