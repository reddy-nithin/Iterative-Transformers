"""Build results table + plot from results/*.json (run: python -m scripts.compare)."""
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

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_prop_cycle(color=plt.cm.tab20.colors)  # default cycle repeats after 10 runs
for k, v in runs.items():
    ax.plot([h["epoch"] for h in v["history"]], [h["test_acc"] for h in v["history"]], marker="o", label=k)
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
    ax.scatter(x, y)
    if v["params"] >= SMALL:
        ax.annotate(k, (x, y), textcoords="offset points", xytext=(5, 2), fontsize=7)
    else:
        zoom.scatter(x, y)
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
