"""T20 plots: zero-mean value readout for CLS in B2 (T20-a) vs the reference T11-d.

1. results/attn_cos_entropy.png: CLS attention cos between successive loops (left) and CLS attention entropy (right).
2. results/update_cos_per_loop.png: successive-update cos of the full CLS update d_t (T12 (c)) and of its attention
   part a_t and MLP part m_t (T19), per loop.
3. results/acc_per_loop.png: fixed-depth curve (readout at loop t), full test set.

Inputs: results/exit_analysis.json, results/t12_diag/diag_*.json, results/t19/decomposition.json of this task; the
T11-d files in T12, T13 and T19.

Run from the repo root:  .venv/bin/python tasks/T20_zero_mean_value_readout/plot.py
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
TASKS = TASK_DIR.parent
MAX_ENTROPY = 5.28  # ln 196

log = logging.getLogger("t20_plot")


@dataclass(frozen=True)
class Run:
    """One model: where its exit, T12 and T19 results are and how it is drawn."""
    label: str
    stem: str
    exits: Path
    diag: Path
    decomposition: Path
    colour: str


RUNS = (
    Run("T11-d (ref)", "T11-d_unroll_n3_30ep", TASKS / "T13_per_loop_readout_loss" / "results" / "exit_analysis.json",
        TASKS / "T12_per_loop_cls_diagnostics" / "results",
        TASKS / "T19_cls_update_decomposition" / "results" / "decomposition.json", "tab:red"),
    Run("T20-a zero-mean value", "T20-a_zero_mean_value", RESULTS / "exit_analysis.json", RESULTS / "t12_diag",
        RESULTS / "t19" / "decomposition.json", "tab:green"),
)


def load(path: Path) -> dict:
    """A results JSON; raises if missing (the plots need it)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run run.sh and the T12 / T19 diagnostics first")
    return json.loads(path.read_text())


def run_data(run: Run) -> dict:
    """Per-loop curves for one model."""
    return {"acc": load(run.exits)[run.stem]["acc_per_loop"],
            "t12": load(run.diag / f"diag_{run.stem}.json")["means"],
            "t19": load(run.decomposition)[run.stem]["means"]}


def finish(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Axis labels, title, grid, legend."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def plot_attention(data: dict[str, dict], out_path: Path) -> None:
    """Successive-loop CLS attention cos (left) and attention entropy (right)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for run in RUNS:
        t12 = data[run.label]["t12"]
        left.plot(range(2, 11), t12["attn_cos"], color=run.colour, marker="o", ms=3, label=run.label)
        right.plot(range(1, 11), t12["attn_entropy"], color=run.colour, marker="o", ms=3, label=run.label)
    right.axhline(MAX_ENTROPY, color="0.5", lw=0.8, ls="--", label="uniform (5.28)")
    right.axhline(3.5, color="0.5", lw=0.8, ls=":", label="3.5 (Gemini prediction)")
    finish(left, "loop t", "cos(A_t-1, A_t)", "CLS → patch attention, successive loops")
    finish(right, "loop t", "entropy (nats)", "CLS → patch attention entropy")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_update_cos(data: dict[str, dict], out_path: Path) -> None:
    """cos(part_t, part_t-1) for d (solid), a (dashed) and m (dotted), per model."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for run in RUNS:
        t19 = data[run.label]["t19"]
        ax.plot(range(2, 11), t19["cos_d"], color=run.colour, marker="o", ms=3, label=f"{run.label} d_t")
        ax.plot(range(2, 11), t19["cos_a"], color=run.colour, ls="--", marker="^", ms=3, label=f"{run.label} a_t")
        ax.plot(range(2, 11), t19["cos_m"], color=run.colour, ls=":", marker="s", ms=3, label=f"{run.label} m_t")
    ax.axhline(0.8, color="0.5", lw=0.8, ls="--", label="0.8")
    finish(ax, "loop t", "cos(part_t, part_t-1)", "Successive CLS-update cos: full, attention and MLP part")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_acc_per_loop(data: dict[str, dict], out_path: Path) -> None:
    """Fixed-depth readout accuracy per loop."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for run in RUNS:
        curve = data[run.label]["acc"]
        ax.plot(range(1, len(curve) + 1), curve, color=run.colour, marker="o", ms=3, label=run.label)
    finish(ax, "loop t (readout after loop t)", "test top-1 (%)", "Fixed-depth accuracy per loop")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """All three plots."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = {run.label: run_data(run) for run in RUNS}
    plot_attention(data, RESULTS / "attn_cos_entropy.png")
    plot_update_cos(data, RESULTS / "update_cos_per_loop.png")
    plot_acc_per_loop(data, RESULTS / "acc_per_loop.png")
    log.info("wrote the 3 plots in %s", RESULTS)


if __name__ == "__main__":
    main()
