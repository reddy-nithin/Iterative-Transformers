"""T6 plots + table: progressive unrolling at lr 1e-4 vs lr 5e-4 for n = 1..12, against the 12-block baseline.

lr 1e-4 (n=1..12) and lr 5e-4 n=1..4 come from T3; lr 5e-4 n=5..12 are this task's runs. Two panels: test accuracy
vs n and vs parameter count, one fixed colour per lr in both. Also reports where the two curves meet.

Run from the repo root:  python tasks/T6_unroll_lr5e-4_full_curve/plot.py
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TASK_DIR = Path(__file__).resolve().parent
T3_RESULTS = TASK_DIR.parent / "T3_progressive_unrolling" / "results"
RESULTS = TASK_DIR / "results"
BASELINE_ACC = 86.77  # pretrained ViT-Tiny, 12 blocks, lr 1e-4, 10 epochs (T1)
# curve key -> (folders holding unroll_u*.json, legend label, colour). Later folders win for the same n.
CURVES = {
    "lr 1e-4": ([T3_RESULTS / "ft"], "fine-tune lr 1e-4", "tab:orange"),
    "lr 5e-4": ([T3_RESULTS / "lr5e-4" / "ft", RESULTS / "ft"], "fine-tune lr 5e-4", "tab:green"),
}

log = logging.getLogger(__name__)


def load_curve(folders: list[Path]) -> pd.DataFrame:
    """One row per finished run (all epochs present): n, params (M), final test acc. Sorted by n."""
    rows: dict[int, dict] = {}
    for folder in folders:
        for path in sorted(folder.glob("unroll_u*.json")):
            run = json.loads(path.read_text())
            if len(run["history"]) != run["args"]["epochs"]:
                log.warning("skipping unfinished run %s (%d/%d epochs)", path, len(run["history"]),
                            run["args"]["epochs"])
                continue
            n = run["args"]["unique_blocks"]
            rows[n] = {"n": n, "params (M)": round(run["params"] / 1e6, 3),
                       "acc (%)": round(run["final_test_acc"], 2), "lr": run["args"]["lr"]}
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r["n"]),
                        columns=["n", "params (M)", "acc (%)", "lr"])


def meeting_point(low_lr: pd.DataFrame, high_lr: pd.DataFrame) -> int | None:
    """First n where the lr 5e-4 accuracy is at or below the lr 1e-4 accuracy; None if it stays above."""
    merged = low_lr.merge(high_lr, on="n", suffixes=("_low", "_high"))
    below = merged[merged["acc (%)_high"] <= merged["acc (%)_low"]]
    return int(below["n"].min()) if not below.empty else None


def build_table(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """n, params, accuracy at each lr, and the lr 5e-4 gain."""
    table = curves["lr 1e-4"][["n", "params (M)", "acc (%)"]].rename(columns={"acc (%)": "lr 1e-4 acc (%)"})
    table = table.merge(curves["lr 5e-4"][["n", "acc (%)"]].rename(columns={"acc (%)": "lr 5e-4 acc (%)"}),
                        on="n", how="outer")
    table["gain (pts)"] = (table["lr 5e-4 acc (%)"] - table["lr 1e-4 acc (%)"]).round(2)
    return table.sort_values("n").reset_index(drop=True)


def plot(curves: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Left: accuracy vs n. Right: accuracy vs params. Baseline as a dashed line in both."""
    fig, (ax_n, ax_p) = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, df in curves.items():
        _, label, color = CURVES[key]
        ax_n.plot(df["n"], df["acc (%)"], marker="o", lw=2, color=color, label=label)
        ax_p.plot(df["params (M)"], df["acc (%)"], marker="o", lw=2, color=color, label=label)
    for ax in (ax_n, ax_p):
        ax.axhline(BASELINE_ACC, ls="--", color="gray", lw=1, label=f"baseline {BASELINE_ACC:.2f}%")
        ax.set_ylabel("final CIFAR-100 test acc (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    ax_n.set_xticks(range(1, 13))
    ax_n.set_xlabel("n = distinct blocks (blocks 0..n-2 kept, block n-1 repeated)")
    ax_p.set_xlabel("parameters (M)")
    fig.suptitle("T6 progressive unrolling, lr 1e-4 vs 5e-4 (ViT-Tiny, 10 epochs, seed 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> None:
    """Build the table and plot from whatever runs have finished so far."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    out_dir = parser.parse_args().out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = {key: load_curve(folders) for key, (folders, _, _) in CURVES.items()}
    if any(df.empty for df in curves.values()):
        raise FileNotFoundError(f"missing runs: {[k for k, df in curves.items() if df.empty]}")
    table = build_table(curves)
    meet = meeting_point(curves["lr 1e-4"], curves["lr 5e-4"])
    verdict = (f"curves meet at n={meet}" if meet is not None
               else f"lr 5e-4 stays above lr 1e-4 for every n run (up to n={curves['lr 5e-4']['n'].max()})")
    text = table.to_string(index=False) + f"\n\n{verdict}\n"
    print(text)
    (out_dir / "summary.md").write_text(text)
    plot(curves, out_dir / "acc_vs_n_and_params.png")
    log.info("wrote summary.md and acc_vs_n_and_params.png in %s", out_dir)


if __name__ == "__main__":
    main()
