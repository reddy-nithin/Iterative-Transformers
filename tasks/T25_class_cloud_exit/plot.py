"""T25 plots, decision lines and README tables from the outputs of class_cloud.py.

1. ncc_vs_readout_per_loop.png: per loop, NCC top-1 with per-loop and with shared centers, readout top-1, and the
   NCC / readout mismatch; one panel per model.
2. exit_curves_class_cloud.png: accuracy vs mean loops for every R1 / R2 / R3 point (per-loop centers filled, shared
   hollow), the fixed-depth curve, and T13's patience (best on T13-c) and KL rules from T13's exit_analysis.json.

Run from the repo root:  .venv/bin/python tasks/T25_class_cloud_exit/plot.py [--results DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
T13_EXITS = TASK_DIR.parent / "T13_per_loop_readout_loss" / "results" / "exit_analysis.json"
MODELS = {"T13-c": "T13-c_unroll_readout_every_loop", "T11-d": "T11-d_unroll_n3_30ep"}
T13_BEST_GAIN = 0.79  # T13-c, patience m = 1 (T13 exit_analysis.json)
SAME_CENTERS_TOLERANCE = 1.0
SAME_CENTERS_LOOPS = range(4, 11)
RULE_STYLE = {"R1 radius": ("tab:green", "o"), "R2 ratio": ("tab:blue", "s"), "R3 agree": ("tab:purple", "^")}


def load_json(path: Path) -> dict:
    """A results JSON; raises if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run class_cloud.py first")
    return json.loads(path.read_text())


def plot_ncc(results: dict[str, dict], out_path: Path) -> None:
    """NCC (per-loop / shared centers) vs readout per loop, mismatch on a second axis."""
    fig, axes = plt.subplots(1, len(results), figsize=(13, 4.5), constrained_layout=True)
    for ax, (tag, result) in zip(axes, results.items()):
        rows = result["ncc"]
        loops = [row["loop"] for row in rows]
        ax.plot(loops, [row["readout"] for row in rows], "k-o", ms=3, label="readout (classifier)")
        ax.plot(loops, [row["ncc_per_loop"] for row in rows], "-s", color="tab:blue", ms=3,
                label="NCC, per-loop centers")
        ax.plot(loops, [row["ncc_shared"] for row in rows], "--^", color="tab:orange", ms=3,
                label="NCC, shared centers")
        twin = ax.twinx()
        twin.plot(loops, [row["mismatch"] for row in rows], ":", color="tab:red", label="NCC ≠ readout (%)")
        twin.set_ylabel("mismatch (%)", color="tab:red")
        ax.set(xlabel="loop t", ylabel="test top-1 (%)", title=f"{tag}: nearest class center vs readout")
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        ax.legend(handles + twin_handles, labels + twin_labels, fontsize=7, loc="lower right")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_exits(results: dict[str, dict], t13: dict, out_path: Path) -> None:
    """Exit curves of the class-cloud rules against fixed depth and T13's rules."""
    fig, axes = plt.subplots(1, len(results), figsize=(14, 5), constrained_layout=True)
    for ax, (tag, result) in zip(axes, results.items()):
        fixed = result["acc_per_loop"]
        ax.plot(range(1, len(fixed) + 1), fixed, "k-", lw=1.5, label="fixed depth")
        reference = t13[MODELS[tag]]["curves"]
        for name, colour in (("patience", "0.5"), ("kl", "tab:brown")):
            points = sorted(reference[name], key=lambda p: p["mean_loops"])
            ax.plot([p["mean_loops"] for p in points], [p["acc"] for p in points], "--", color=colour, lw=1,
                    label=f"T13 {name} rule")
        for mode, filled in (("per_loop", True), ("shared", False)):
            for rule, points in result["curves"][mode].items():
                colour, marker = RULE_STYLE[rule]
                ax.scatter([p["mean_loops"] for p in points], [p["acc"] for p in points], marker=marker, s=28,
                           color=colour if filled else "none", edgecolor=colour,
                           label=f"{rule}, {mode.replace('_', '-')} centers")
        ax.set(xlabel="mean loops used", ylabel="test top-1 (%)", title=f"{tag}: class-cloud exit")
        ax.set_ylim(bottom=max(0, min(fixed[1:]) - 5))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="lower right")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def best_overall(result: dict) -> tuple[str, dict]:
    """The class-cloud rule point with the largest gain over fixed depth, over all rules and both center modes."""
    return max(result["best_gain"].items(), key=lambda item: item[1]["gain_vs_fixed"])


def decision_lines(result: dict) -> list[str]:
    """The two README decision lines, from T13-c's results."""
    name, point = best_overall(result)
    gain = point["gain_vs_fixed"]
    first = ("class-cloud exit beats T13's best rule" if gain > T13_BEST_GAIN
             else "class-cloud exit does not beat T13's best rule")
    first += f" (best class-cloud gain {gain:+.2f} from {name} at {point}, vs T13 patience +{T13_BEST_GAIN})"
    rows = {row["loop"]: row for row in result["ncc"]}
    parted = [t for t in SAME_CENTERS_LOOPS
              if abs(rows[t]["ncc_shared"] - rows[t]["ncc_per_loop"]) > SAME_CENTERS_TOLERANCE]
    second = ("one center set is enough" if not parted
              else f"loops need their own centers (they part at loop {parted[0]}: per-loop "
                   f"{rows[parted[0]]['ncc_per_loop']:.2f} vs shared {rows[parted[0]]['ncc_shared']:.2f})")
    return [first, second]


def print_summary(results: dict[str, dict]) -> None:
    """Tables for the README."""
    for tag, result in results.items():
        print(f"\n[{tag}] top-1 {result['top1_test']}  (train {result['n_train']}, test {result['n_test']})")
        print("| loop | NCC per-loop | NCC shared | readout | mismatch |")
        for row in result["ncc"]:
            print(f"| {row['loop']} | {row['ncc_per_loop']:.2f} | {row['ncc_shared']:.2f} | {row['readout']:.2f} "
                  f"| {row['mismatch']:.2f} |")
        print("| rule (centers) | best point | acc | mean loops | gain vs fixed |")
        for name, point in result["best_gain"].items():
            setting = ", ".join(f"{k}={v}" for k, v in point.items() if k in ("q", "tau"))
            print(f"| {name} | {setting} | {point['acc']:.2f} | {point['mean_loops']:.2f} | "
                  f"{point['gain_vs_fixed']:+.2f} |")
    for line in decision_lines(results["T13-c"]):
        print(f"\nDECISION (T13-c): {line}")


def main() -> None:
    """Load both models' results, plot, print tables and decision lines."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(TASK_DIR / "results"))
    results_dir = Path(parser.parse_args().results)
    results = {tag: load_json(results_dir / f"class_cloud_{tag}.json") for tag in MODELS}
    plot_ncc(results, results_dir / "ncc_vs_readout_per_loop.png")
    plot_exits(results, load_json(T13_EXITS), results_dir / "exit_curves_class_cloud.png")
    print_summary(results)


if __name__ == "__main__":
    main()
