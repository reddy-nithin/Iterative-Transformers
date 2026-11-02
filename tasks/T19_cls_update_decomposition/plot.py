"""T19 plots and tables from results/decomposition.json (and T13's exit_analysis.json for the reference curves).

1. results/update_split_per_loop.png: ||a_t|| / ||d_t|| and ||m_t|| / ||d_t|| (left), mean share and self share (right),
   per loop, 3 models.
2. results/successive_cos_parts.png: cos between successive loops of a, m, d, the mean part and the deviation part,
   one panel per model.
3. results/exit_curves_collinear_rule.png: fixed-depth curve, T13's three exit rules and the collinear halting rule
   (one line per theta across rho), one panel per model.
Also writes results/tables.md (per-loop tables, halting grid, decision lines).

Run from the repo root:  .venv/bin/python tasks/T19_cls_update_decomposition/plot.py
"""
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
T13_EXITS = TASK_DIR.parent / "T13_per_loop_readout_loss" / "results" / "exit_analysis.json"
MODELS = {  # checkpoint stem -> (label, colour)
    "T11-d_unroll_n3_30ep": ("T11-d", "tab:red"),
    "T13-c_unroll_readout_every_loop": ("T13-c", "tab:orange"),
    "T13-d_unroll_loss_margin": ("T13-d", "tab:brown"),
}
RULE_STYLES = {"confidence": ("^", "tab:green"), "kl": ("s", "tab:cyan"), "patience": ("D", "0.35")}
PART_STYLES = {"cos_a": ("attention a_t", "tab:blue"), "cos_m": ("MLP m_t", "tab:purple"),
               "cos_d": ("full update d_t", "black"), "cos_mean": ("attn mean part", "tab:green"),
               "cos_dev": ("attn deviation part", "tab:olive")}
THETA_COLOURS = {0.8: "tab:pink", 0.85: "tab:red", 0.9: "tab:purple", 0.95: "tab:blue"}

log = logging.getLogger("t19_plot")


def load(path: Path) -> dict:
    """A results JSON; raises if missing (the plots need it)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run decompose.py / T13 exit_analysis.py first")
    return json.loads(path.read_text())


def finish(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Axis labels, title, grid, legend."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def plot_update_split(results: dict, out_path: Path) -> None:
    """Attention vs MLP size (left), mean vs self share (right), per loop."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    loops = range(1, 11)
    for stem, (label, colour) in MODELS.items():
        means = results[stem]["means"]
        left.plot(loops, means["attn_frac"], color=colour, marker="o", ms=3, label=f"{label} ||a||/||d||")
        left.plot(loops, means["mlp_frac"], color=colour, ls="--", marker="s", ms=3, label=f"{label} ||m||/||d||")
        right.plot(loops, means["mean_share"], color=colour, marker="o", ms=3, label=f"{label} mean share")
        right.plot(loops, means["self_share"], color=colour, ls=":", marker="^", ms=3, label=f"{label} self share")
    left.axhline(0.3, color="0.5", lw=0.8, ls="--")
    right.axhline(0.5, color="0.5", lw=0.8, ls="--")
    finish(left, "loop t (B2 application)", "share of ||d_t||", "CLS update: attention vs MLP part")
    finish(right, "loop t", "share", "attention part: mean share ||W mean|| / (||W mean|| + ||W dev||), self share")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_successive_cos(results: dict, out_path: Path) -> None:
    """cos(part_t, part_t-1) per loop for every part, one panel per model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True, sharey=True)
    for ax, (stem, (label, _)) in zip(axes, MODELS.items()):
        means = results[stem]["means"]
        for key, (part, colour) in PART_STYLES.items():
            ax.plot(range(2, 11), means[key], color=colour, marker="o", ms=3, label=part)
        ax.set_ylim(0, 1.02)
        finish(ax, "loop t", "cos(part_t, part_t-1)", f"{label}: successive-update cos per part")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_exit_curves(results: dict, t13: dict, out_path: Path) -> None:
    """Fixed depth, T13's rules and the collinear rule on the same axes, one panel per model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, (stem, (label, colour)) in zip(axes, MODELS.items()):
        reference = t13[stem]
        fixed = reference["acc_per_loop"]
        ax.plot(range(1, len(fixed) + 1), fixed, color=colour, lw=2, marker="o", ms=3, label="fixed depth")
        for rule, (marker, rule_colour) in RULE_STYLES.items():
            points = sorted(reference["curves"][rule], key=lambda p: p["mean_loops"])
            ax.plot([p["mean_loops"] for p in points], [p["acc"] for p in points], color=rule_colour,
                    marker=marker, ms=4, lw=1, label=f"T13 {rule}")
        for theta, theta_colour in THETA_COLOURS.items():
            points = sorted((p for p in results[stem]["halting"] if p["theta"] == theta),
                            key=lambda p: p["mean_loops"])
            ax.plot([p["mean_loops"] for p in points], [p["acc"] for p in points], color=theta_colour, marker="x",
                    ms=6, lw=1, ls="--", label=f"collinear θ={theta} (ρ 0.1..none)")
        ax.set_xlim(2.5, 10.3)
        ax.set_ylim(max(0.0, fixed[-1] - 25), max(fixed) + 2)
        finish(ax, "loops used (mean per image for the rules)", "test top-1 (%)", label)
    fig.suptitle("Collinear halting rule vs fixed depth and T13's exit rules (T11-d layout)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_loop_table(label: str, means: dict) -> str:
    """Markdown table: one row per loop."""
    rows = [f"**{label}**", "",
            "| loop | ‖a‖/‖d‖ | ‖m‖/‖d‖ | mean share | self share | ‖d‖/‖CLS‖ | cos a | cos m | cos d | cos mean | cos dev |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in range(1, 11):
        successive = [means[k][t - 2] if t >= 2 else None for k in ("cos_a", "cos_m", "cos_d", "cos_mean", "cos_dev")]
        cells = [f"{means[k][t - 1]:.3f}" for k in ("attn_frac", "mlp_frac", "mean_share", "self_share", "rel_d")]
        cells += ["–" if v is None else f"{v:.3f}" for v in successive]
        rows.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def halting_table(label: str, result: dict, t13: dict) -> str:
    """Markdown table of the theta x rho grid, plus T13's best rule gain for comparison."""
    fixed = result["acc_per_loop"]
    rows = [f"**{label}** (fixed depth: loop 10 = {fixed[-1]:.2f}, peak {max(fixed):.2f} at loop "
            f"{fixed.index(max(fixed)) + 1})", "",
            "| θ | ρ | top-1 | mean loops | gain vs fixed |", "|---|---|---|---|---|"]
    for p in result["halting"]:
        rho = "none" if p["rho"] is None else p["rho"]
        rows.append(f"| {p['theta']} | {rho} | {p['acc']:.2f} | {p['mean_loops']:.2f} | {p['gain_vs_fixed']:+.2f} |")
    t13_best = max(t13["vs_fixed"].values(), key=lambda s: s["best_gain"]["gain_vs_fixed"])["best_gain"]
    ours = result["halting_vs_fixed"]
    best, cheap = ours["best_gain"], ours["cheapest_near_full"]
    cheap_text = "none" if cheap is None else f"{cheap['mean_loops']} loops ({cheap['acc']:.2f}%)"
    rows += ["", f"Best collinear gain vs fixed: {best['gain_vs_fixed']:+.2f} (θ {best['theta']}, ρ {best['rho']}, "
                 f"{best['mean_loops']} loops, top-1 {best['acc']:.2f}). T13's best rule gain: "
                 f"{t13_best['gain_vs_fixed']:+.2f} at {t13_best['mean_loops']} loops. Cheapest collinear point within "
                 f"1 pt of loop 10: {cheap_text} vs fixed depth {ours['fixed_loops_near_full']}."]
    return "\n".join(rows)


def write_tables(results: dict, t13: dict, out_path: Path) -> None:
    """All per-loop and halting tables plus the decision lines, as markdown."""
    parts = ["# T19 tables (generated by plot.py)", ""]
    for stem, (label, _) in MODELS.items():
        r = results[stem]
        parts += [per_loop_table(label, r["means"]), "",
                  f"proj bias norm {r['bias_norm']}; T12 (c) check: max |cos d diff| {r['t12_check']['max_abs_diff']}; "
                  f"decision values: late mean share {r['decision']['late_mean_share']}, late ‖a‖/‖d‖ "
                  f"{r['decision']['late_attn_frac']} → {r['decision']['line']}", ""]
    for stem, (label, _) in MODELS.items():
        parts += [halting_table(label, results[stem], t13[stem]), ""]
    out_path.write_text("\n".join(parts))


def main() -> None:
    """All three plots and the tables."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results, t13 = load(RESULTS / "decomposition.json"), load(T13_EXITS)
    plot_update_split(results, RESULTS / "update_split_per_loop.png")
    plot_successive_cos(results, RESULTS / "successive_cos_parts.png")
    plot_exit_curves(results, t13, RESULTS / "exit_curves_collinear_rule.png")
    write_tables(results, t13, RESULTS / "tables.md")
    log.info("wrote the 3 plots and tables.md in %s", RESULTS)


if __name__ == "__main__":
    main()
