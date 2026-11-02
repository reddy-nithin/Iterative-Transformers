"""T15 tables: final test top-1 per seed (seed 0 = the existing T11 / T13 run), mean and spread per config, the T13-d
fixed-depth curve per seed, and which last-round gaps are larger than the seed spread. Writes results/table.md.

Run from the repo root:  .venv/bin/python tasks/T15_seed_repeats/results_table.py
"""
import json
import logging
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
RESULTS = TASK_DIR / "results"
CONFIGS = {  # config -> run JSON per seed (seed 0 is the existing run)
    "T11-c": [TASKS / "T11_pretrained_hbt_body_sweep/results/hbt_T11-c.json",
              RESULTS / "hbt_T15-a.json", RESULTS / "hbt_T15-b.json"],
    "T11-d": [TASKS / "T11_pretrained_hbt_body_sweep/results/unroll_T11-d.json",
              RESULTS / "unroll_T15-c.json", RESULTS / "unroll_T15-d.json"],
    "T13-d": [TASKS / "T13_per_loop_readout_loss/results/unroll_T13-d.json",
              RESULTS / "unroll_T15-e.json", RESULTS / "unroll_T15-f.json"],
}
T13D_EXITS = [(TASKS / "T13_per_loop_readout_loss/results/exit_analysis.json", "T13-d_unroll_loss_margin"),
              (RESULTS / "exit_analysis_T15-e.json", "T15-e_t13d_seed1"),
              (RESULTS / "exit_analysis_T15-f.json", "T15-f_t13d_seed2")]
LAST_ROUND_GAPS = [  # (label, config a, config b or None, single-seed value of b) for the "larger than spread" line
    ("T11-d vs T11-c (74.11 vs 68.59)", "T11-d", "T11-c"),
    ("T13-d vs T11-d (73.17 vs 74.11)", "T13-d", "T11-d"),
]
SINGLE_SEED_GAPS = [  # gaps against runs that only have seed 0: compare with the base config's seed spread
    ("T14-a vs T13-d (72.89 vs 73.17)", -0.28, "T13-d"),
    ("T13-a vs T11-c (68.85 vs 68.59)", 0.26, "T11-c"),
    ("T13-c vs T11-d (72.82 vs 74.11)", -1.29, "T11-d"),
]

log = logging.getLogger("t15_table")


def final_acc(path: Path) -> float | None:
    """Final test top-1 of a finished run, None if missing or unfinished."""
    if not path.exists():
        return None
    run = json.loads(path.read_text())
    if len(run["history"]) != run["args"]["epochs"]:
        return None
    return run["history"][-1]["test_acc"]


def seed_stats(values: list[float | None]) -> tuple[float | None, float | None]:
    """(mean, max - min) over the available seeds, None if a seed is missing."""
    if any(v is None for v in values):
        return None, None
    return round(sum(values) / len(values), 2), round(max(values) - min(values), 2)


def fmt(value: float | None) -> str:
    """Two decimals, or 'pending'."""
    return "pending" if value is None else f"{value:.2f}"


def main() -> None:
    """Build results/table.md."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stats, lines = {}, ["| config | seed 0 | seed 1 | seed 2 | mean | spread (max − min) |", "|---|---|---|---|---|---|"]
    for config, paths in CONFIGS.items():
        values = [final_acc(p) for p in paths]
        stats[config] = (values, *seed_stats(values))
        lines.append(f"| {config} | " + " | ".join(fmt(v) for v in values)
                     + f" | {fmt(stats[config][1])} | {fmt(stats[config][2])} |")
    lines += ["", "T13-d fixed-depth curve (top-1 at loop 1..10, full test set, fp32):", "",
              "| seed | " + " | ".join(str(t) for t in range(1, 11)) + " |", "|---|" + "---|" * 10]
    for seed, (path, key) in enumerate(T13D_EXITS):
        if path.exists() and key in json.loads(path.read_text()):
            curve = json.loads(path.read_text())[key]["acc_per_loop"]
            lines.append(f"| {seed} | " + " | ".join(f"{a:.2f}" for a in curve) + " |")
        else:
            lines.append(f"| {seed} | pending |" + " |" * 9)
    lines += ["", "Gaps vs seed spread:", ""]
    for label, a, b in LAST_ROUND_GAPS:
        (va, ma, sa), (vb, mb, sb) = stats[a], stats[b]
        if ma is None or mb is None:
            lines.append(f"- {label}: pending")
            continue
        per_seed = [round(x - y, 2) for x, y in zip(va, vb)]
        lines.append(f"- {label}: mean gap {ma - mb:+.2f}, per seed {per_seed}, spreads {sa:.2f} / {sb:.2f}")
    for label, gap, base in SINGLE_SEED_GAPS:
        spread = stats[base][2]
        lines.append(f"- {label}: gap {gap:+.2f} at seed 0, {base} seed spread "
                     + ("pending" if spread is None else f"{spread:.2f}"))
    (RESULTS / "table.md").write_text("\n".join(lines) + "\n")
    log.info("\n".join(lines))


if __name__ == "__main__":
    main()
