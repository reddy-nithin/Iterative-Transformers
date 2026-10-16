"""T3 plot + table: final test accuracy vs number of distinct blocks n, one line per pass.

Run from the repo root:  python tasks/T3_progressive_unrolling/plot.py [--results-dir DIR]
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
RESULTS = TASK_DIR / "results"
BASELINE_JSON = TASK_DIR.parent / "T1_baseline_vs_shared_block" / "results" / "baseline.json"
PASSES = {"probe": "probe: body lr 1e-6, head lr 1e-3", "ft": "fine-tune: all lr 1e-4"}

log = logging.getLogger(__name__)


def load_pass(results: Path, pass_name: str) -> pd.DataFrame:
    """One row per finished run of a pass: n, params (M), final and best test acc, epochs run."""
    rows = []
    for path in sorted((results / pass_name).glob("unroll_u*.json")):
        run = json.loads(path.read_text())
        rows.append({"n": run["args"]["unique_blocks"], "params (M)": round(run["params"] / 1e6, 3),
                     "final acc (%)": round(run["final_test_acc"], 2), "best acc (%)": round(run["best_test_acc"], 2),
                     "epochs": len(run["history"])})
    if not rows:
        log.warning("no runs found in %s", results / pass_name)
        return pd.DataFrame(columns=["n", "params (M)", "final acc (%)", "best acc (%)", "epochs"])
    return pd.DataFrame(rows).sort_values("n").reset_index(drop=True)


def build_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Side-by-side table: n, params, then final acc for each pass."""
    keyed = [df.set_index("n")[["final acc (%)"]].rename(columns={"final acc (%)": f"{p} acc (%)"})
             for p, df in frames.items() if not df.empty]
    params = pd.concat([df.set_index("n")["params (M)"] for df in frames.values() if not df.empty])
    params = params[~params.index.duplicated()]
    return pd.concat([params, *keyed], axis=1).sort_index().reset_index()


def plot(frames: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Final test acc vs n, one line per pass, baseline (T1) as a dashed reference."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for pass_name, df in frames.items():
        if not df.empty:
            ax.plot(df["n"], df["final acc (%)"], marker="o", label=PASSES[pass_name])
    if BASELINE_JSON.exists():
        base = json.loads(BASELINE_JSON.read_text())["final_test_acc"]
        ax.axhline(base, ls="--", color="gray", lw=1)
        ax.text(1, base, f"T1 baseline {base:.1f}%", va="bottom", fontsize=8, color="gray")
    else:
        log.warning("baseline reference not found at %s", BASELINE_JSON)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("n = number of distinct blocks (blocks 0..n-2 kept, block n-1 repeated)")
    ax.set_ylabel("final CIFAR-100 test acc (%)")
    ax.set_title("T3 progressive unrolling (ViT-Tiny, 10 epochs, seed 0)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)


def main() -> None:
    """Build the table and plot from whatever runs have finished so far."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS, help="folder holding probe/ and ft/")
    results = parser.parse_args().results_dir
    frames = {p: load_pass(results, p) for p in PASSES}
    if all(df.empty for df in frames.values()):
        raise FileNotFoundError(f"no T3 runs under {results}/{{probe,ft}}")
    table = build_table(frames).to_string(index=False)
    print(table)
    (results / "summary.md").write_text(table + "\n")
    plot(frames, results / "acc_vs_n.png")
    log.info("wrote %s and %s", results / "summary.md", results / "acc_vs_n.png")


if __name__ == "__main__":
    main()
