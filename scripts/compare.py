"""Build results table + plots from results/*.json (run: python -m scripts.compare).

Writes results/summary.md, accuracy.png (per epoch), acc_vs_params.png, final_acc_bars.png.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

runs = {p.stem: json.loads(p.read_text()) for p in sorted(Path("results").glob("*.json"))}
rows = [{"model": k, "params (M)": round(v["params"] / 1e6, 3), "GFLOPs": round(v["gflops"], 2),
         "throughput (img/s)": round(v["throughput_img_s"]), "final acc (%)": round(v["final_test_acc"], 2),
         "best acc (%)": round(v["best_test_acc"], 2), "epochs": len(v["history"])} for k, v in runs.items()]
df = pd.DataFrame(rows)
table = df.to_string(index=False)
print(table)
Path("results/summary.md").write_text(table + "\n")

# one fixed colour per run, reused by every plot so a run looks the same everywhere
colors = {k: plt.cm.tab20.colors[i % 20] for i, k in enumerate(runs)}

fig, ax = plt.subplots(figsize=(7, 4))
for k, v in runs.items():
    ax.plot([h["epoch"] for h in v["history"]], [h["test_acc"] for h in v["history"]],
            marker="o", label=k, color=colors[k])
ax.set_xlabel("epoch"); ax.set_ylabel("CIFAR-100 test acc (%)"); ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout(); fig.savefig("results/accuracy.png", dpi=150)

# accuracy vs parameter count; baseline and scratch drawn as dashed reference lines.
# left: all runs (log x). right: zoom on the shared-block runs (< 1.5M params) so their labels don't overlap.
SMALL = 1.5e6  # shared-block runs are below this; they get labels only in the zoom panel
refs = {k: v["final_test_acc"] for k, v in runs.items() if k in ("baseline", "scratch")}
fig, (ax, zoom) = plt.subplots(1, 2, figsize=(11, 4.5))
for k, v in runs.items():
    x, y = v["params"] / 1e6, v["final_test_acc"]
    ax.scatter(x, y, color=colors[k])
    if v["params"] >= SMALL:
        ax.annotate(k, (x, y), textcoords="offset points", xytext=(5, 2), fontsize=7)
    else:
        zoom.scatter(x, y, color=colors[k])
        zoom.annotate(k, (x, y), textcoords="offset points", xytext=(5, 2), fontsize=7)
zoom_top = max(v["final_test_acc"] for v in runs.values() if v["params"] < SMALL) + 3
for k, y in refs.items():
    ax.axhline(y, ls="--", color="gray", lw=1)
    if y < zoom_top:  # baseline sits far above the zoomed range
        zoom.axhline(y, ls="--", color="gray", lw=1)
        zoom.text(0.99, y, k, transform=zoom.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color="gray")
for a in (ax, zoom):
    a.set_ylabel("final CIFAR-100 test acc (%)"); a.grid(alpha=0.3)
ax.set_xscale("log"); ax.minorticks_off()
ax.set_xticks([0.5, 1, 2, 5], ["0.5", "1", "2", "5"])
ax.set_xlabel("params (M, log scale)"); ax.set_title("all runs", fontsize=9)
zoom.set_xlabel("params (M)"); zoom.set_title("shared-block runs (zoom)", fontsize=9)
zoom.set_ylim(top=zoom_top)
fig.tight_layout(); fig.savefig("results/acc_vs_params.png", dpi=150)

# one bar per run, sorted by accuracy and coloured by kind of model: the quickest read of what moved accuracy
FAMILIES = [  # (name prefix, legend label, colour); first match wins
    ("baseline", "pretrained, 12 blocks", "tab:blue"),
    ("scratch", "random init, 12 blocks", "tab:gray"),
    ("iterative_lr", "shared block, higher lr", "tab:green"),
    ("iterative", "shared block", "tab:cyan"),
    ("lora", "shared block + per-loop LoRA", "tab:orange"),
]


def family(name):
    return next(f for f in FAMILIES if name.startswith(f[0]))


order = sorted(runs, key=lambda k: runs[k]["final_test_acc"])
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, k in enumerate(order):
    acc = runs[k]["final_test_acc"]
    ax.barh(i, acc, color=family(k)[2])
    ax.text(acc + 0.8, i, f"{acc:.1f}%  ({runs[k]['params'] / 1e6:.2f}M)", va="center", fontsize=7)
ax.set_yticks(range(len(order)), order, fontsize=8)
ax.set_xlim(0, 100); ax.set_xlabel("final CIFAR-100 test acc (%)"); ax.grid(axis="x", alpha=0.3)
used = {family(k)[1]: family(k)[2] for k in order}
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in used.values()], labels=list(used),
          fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig("results/final_acc_bars.png", dpi=150)
