"""T18 table and plot: top-1 vs n distinct blocks at 10 and 30 epochs (lr 5e-4). n = 1 (iter b0, T4) and n = 3 (T11-d)
already existed at 30 epochs; 10-epoch numbers from T3 (n = 1..4) and T6 (n = 5, 6). Writes results/table.md and
results/acc_vs_n_10_vs_30ep.png.

Run from the repo root:  .venv/bin/python tasks/T18_unroll_curve_30ep/results_table.py
"""
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
RESULTS = TASK_DIR / "results"
TEN_EPOCHS = {n: TASKS / f"T3_progressive_unrolling/results/lr5e-4/ft/unroll_u{n}.json" for n in (1, 2, 3, 4)}
TEN_EPOCHS.update({n: TASKS / f"T6_unroll_lr5e-4_full_curve/results/ft/unroll_u{n}.json" for n in (5, 6)})
THIRTY_EPOCHS = {1: TASKS / "T4_longer_runs_lr5e-4/results/iterative_iter_b0.json",
                 3: TASKS / "T11_pretrained_hbt_body_sweep/results/unroll_T11-d.json"}
THIRTY_EPOCHS.update({n: RESULTS / f"unroll_T18-n{n}.json" for n in (2, 4, 5, 6)})
SOURCE_30 = {1: "T4 iter b0", 3: "T11-d", 2: "T18-n2", 4: "T18-n4", 5: "T18-n5", 6: "T18-n6"}

log = logging.getLogger("t18_table")


def finished_run(path: Path, epochs: int) -> dict | None:
    """The run JSON if it finished `epochs` epochs at lr 5e-4, else None."""
    if not path.exists():
        return None
    run = json.loads(path.read_text())
    if len(run["history"]) != epochs or run["history"][-1]["epoch"] != epochs:
        return None
    if abs(run["args"]["lr"] - 5e-4) > 1e-12:
        raise ValueError(f"{path} is at lr {run['args']['lr']}, not 5e-4")
    return run


def main() -> None:
    """Table + plot."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows, points = ["| n | params | top-1 10 ep | top-1 30 ep | gain | 30-ep run |", "|---|---|---|---|---|---|"], []
    for n in range(1, 7):
        ten, thirty = finished_run(TEN_EPOCHS[n], 10), finished_run(THIRTY_EPOCHS[n], 30)
        acc10 = ten["history"][-1]["test_acc"] if ten else None
        acc30 = thirty["history"][-1]["test_acc"] if thirty else None
        params = (thirty or ten)["params"]
        gain = "pending" if acc30 is None or acc10 is None else f"{acc30 - acc10:+.2f}"
        rows.append(f"| {n} | {params / 1e6:.2f}M | {acc10:.2f} | "
                    + ("pending" if acc30 is None else f"{acc30:.2f}") + f" | {gain} | {SOURCE_30[n]} |")
        points.append((n, params, acc10, acc30))
    (RESULTS / "table.md").write_text("\n".join(rows) + "\n")
    log.info("\n".join(rows))

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot([p[0] for p in points], [p[2] for p in points], marker="o", color="tab:gray", label="10 epochs (T3 / T6)")
    done = [p for p in points if p[3] is not None]
    ax.plot([p[0] for p in done], [p[3] for p in done], marker="o", color="tab:red", label="30 epochs (T4, T11-d, T18)")
    for n, params, acc10, acc30 in points:
        top = acc30 if acc30 is not None else acc10
        ax.annotate(f"{params / 1e6:.2f}M", (n, top), textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=7)
    ax.set(xlabel="n distinct blocks (blocks 0..n−2 kept, block n−1 repeated; 12 applications)",
           ylabel="CIFAR-100 test top-1 (%)", xticks=range(1, 7))
    ax.set_title("Progressive unrolling at lr 5e-4: 10 vs 30 epochs (labels: params)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(RESULTS / "acc_vs_n_10_vs_30ep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
