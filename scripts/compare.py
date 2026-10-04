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
         "best acc (%)": round(v["best_test_acc"], 2)} for k, v in runs.items()]
df = pd.DataFrame(rows)
table = df.to_string(index=False)
print(table)
Path("results/summary.md").write_text(table + "\n")

fig, ax = plt.subplots(figsize=(5, 3.5))
for k, v in runs.items():
    ax.plot([h["epoch"] for h in v["history"]], [h["test_acc"] for h in v["history"]], marker="o", label=k)
ax.set_xlabel("epoch"); ax.set_ylabel("CIFAR-100 test acc (%)"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("results/accuracy.png", dpi=150)
