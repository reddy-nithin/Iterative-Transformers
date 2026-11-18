"""T37 loss and accuracy curves (train and test vs epoch)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("tasks/T37_fair_baseline_30ep/results")


def main():
    h = json.loads((RES / "baseline_T37-a.json").read_text())["history"]
    ep = [r["epoch"] for r in h]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ep, [r["train_loss"] for r in h], label="train loss", color="#1f77b4")
    ax1.plot(ep, [r["test_loss"] for r in h], label="test loss", color="#ff7f0e")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax2.plot(ep, [r["train_acc"] for r in h], label="train acc (augmented, running)", color="#1f77b4")
    ax2.plot(ep, [r["test_acc"] for r in h], label="test top-1", color="#ff7f0e")
    ax2.axhline(73.79, ls="--", color="gray", label="T11-d 3-seed mean (73.79)")
    ax2.axhline(86.77, ls=":", color="gray", label="old baseline 86.77 (10 ep, lr 1e-4)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("top-1 (%)")
    ax2.legend(fontsize=8)
    fig.suptitle("T37-a: 12-block ViT-Tiny on the T11-d recipe (30 ep, lr 5e-4)")
    fig.tight_layout()
    fig.savefig(RES / "loss_curves.png", dpi=150)
    print("saved", RES / "loss_curves.png")


if __name__ == "__main__":
    main()
