"""Shared readers and plots for comparing one T11-d-layout training run with its reference (T28, T29, T30).

Each run is described by its training JSON (src.train), its entry in an exit_analysis.json (T13 code: per-loop
readout) and its cls_svd JSONs (T5 code: class separation after every block call, test and train).
"""
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASKS = Path(__file__).resolve().parent.parent
SEPARATION_SET = "all classes"
HEAD_CALLS = 2  # T11-d layout: B0, B1, then the loops; loop t = block call t + 2


@dataclass(frozen=True)
class RunFiles:
    """Where one run's results live."""
    label: str
    train_json: Path
    exit_json: Path
    exit_key: str
    cls_svd_prefix: Path  # <prefix>_test.json / <prefix>_train.json
    colour: str


T11D = RunFiles("T11-d (ref)", TASKS / "T11_pretrained_hbt_body_sweep/results/unroll_T11-d.json",
                TASKS / "T13_per_loop_readout_loss/results/exit_analysis.json", "T11-d_unroll_n3_30ep",
                TASKS / "T11_pretrained_hbt_body_sweep/results/cls_svd_T11-d", "tab:red")
T13C = RunFiles("T13-c (ref)", TASKS / "T13_per_loop_readout_loss/results/unroll_T13-c.json",
                TASKS / "T13_per_loop_readout_loss/results/exit_analysis.json", "T13-c_unroll_readout_every_loop",
                TASKS / "T13_per_loop_readout_loss/results/cls_svd_T13-c", "tab:green")


def load_json(path: Path) -> dict:
    """A results JSON; raises with the path if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the task's run.sh first")
    return json.loads(path.read_text())


def train_summary(run: RunFiles) -> dict:
    """Params, GFLOPs, throughput, epochs, final / best top-1, median epoch time from the training JSON."""
    data = load_json(run.train_json)
    times = [h["epoch_time_s"] for h in data["history"]]
    return {"params": data["params"], "gflops": round(data["gflops"], 3),
            "throughput_img_s": round(data["throughput_img_s"]), "epochs": len(data["history"]),
            "final_top1": round(data["final_test_acc"], 2), "best_top1": round(data["best_test_acc"], 2),
            "median_epoch_s": round(statistics.median(times), 1)}


def per_loop_readout(run: RunFiles) -> list[float]:
    """Test top-1 of the readout after loops 1..10."""
    return load_json(run.exit_json)[run.exit_key]["acc_per_loop"]


def separation_per_loop(run: RunFiles, split: str, head_calls: int = HEAD_CALLS, n_loops: int = 10) -> list[float]:
    """Separation (within / between class spread) after loops 1..`n_loops` on `split` (test or train); loop t is
    block call `head_calls` + t (2 head blocks in the T11-d layout, 1 in T9's head-body-tail)."""
    data = load_json(Path(f"{run.cls_svd_prefix}_{split}.json"))
    by_layer = data["separation"][SEPARATION_SET]
    return [round(by_layer[str(head_calls + loop)], 3) for loop in range(1, n_loops + 1)]


def plot_curves(series: dict[str, tuple[list[float], str]], ylabel: str, title: str, out_path: Path) -> None:
    """One line per run over loops 1..L; `series` maps label -> (values, colour)."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for label, (values, colour) in series.items():
        ax.plot(range(1, len(values) + 1), values, color=colour, marker="o", ms=3, label=label)
    ax.set(xlabel="loop t", ylabel=ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def markdown_row(cells: list) -> str:
    """One markdown table row."""
    return "| " + " | ".join(str(cell) for cell in cells) + " |"
