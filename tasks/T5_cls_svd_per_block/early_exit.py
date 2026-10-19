"""T5: test accuracy if a model stopped after block / loop k (final norm + head applied to the CLS after k), and
per-image stopping rules.

A direct check of the stopping rule: separation says when classes stop getting tighter; this says when the answer
stops getting better. Two per-image rules are tested against fixed depths:
- patience m: stop at the first loop whose predicted class equals that of the previous m - 1 loops (m loops agree);
- confidence tau: stop at the first loop whose top softmax probability is at least tau.
An image that never meets the rule runs all loops. The head was only trained on the output of the last block, so all
early-exit numbers are lower bounds on what a model trained with per-loop exits could do.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T5_cls_svd_per_block/early_exit.py CKPT [CKPT ...]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from cls_svd import CLSRecord, collect_cls_per_layer, load_model

TASK_DIR = Path(__file__).resolve().parent
PATIENCES = (2, 3, 4)
CONFIDENCES = (0.3, 0.5, 0.7, 0.9)


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="weights saved by src.train --save-checkpoint")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    return p.parse_args()


@torch.no_grad()
def exit_logits(model: nn.Module, record: CLSRecord, device: torch.device) -> torch.Tensor:
    """Logits (depth, N, classes) after each block k = 1..depth, using the model's own final norm and head."""
    return torch.stack([model.head(model.norm(cls.to(device))).float().cpu() for cls in record.cls[1:]])


def exit_accuracies(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    """Top-1 (%) at each exit."""
    return [100.0 * (step.argmax(dim=1) == labels).float().mean().item() for step in logits]


def prediction_changes(logits: torch.Tensor) -> dict[str, list[float]]:
    """Per exit k: % of images whose prediction differs from exit k-1, and % that already match the last exit."""
    predictions = logits.argmax(dim=2)
    changed = [None] + [100.0 * (predictions[k] != predictions[k - 1]).float().mean().item()
                        for k in range(1, len(predictions))]
    agrees_with_last = [100.0 * (step == predictions[-1]).float().mean().item() for step in predictions]
    return {"changed_from_previous": changed, "agrees_with_last": agrees_with_last}


def patience_exit(logits: torch.Tensor, patience: int) -> torch.Tensor:
    """Per image, the exit index (0-based) where `patience` consecutive exits first agree; last exit otherwise."""
    predictions = logits.argmax(dim=2)
    depth, n_images = predictions.shape
    stop = torch.full((n_images,), depth - 1)
    run = torch.ones(n_images, dtype=torch.long)
    done = torch.zeros(n_images, dtype=torch.bool)
    for k in range(1, depth):
        run = torch.where(predictions[k] == predictions[k - 1], run + 1, torch.ones_like(run))
        newly = (run >= patience) & ~done
        stop[newly] = k
        done |= newly
    return stop


def confidence_exit(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    """Per image, the first exit index (0-based) whose top softmax probability is >= threshold; last otherwise."""
    confident = logits.softmax(dim=2).amax(dim=2) >= threshold
    first = confident.float().argmax(dim=0)  # argmax returns the first maximal index
    return torch.where(confident.any(dim=0), first, torch.full_like(first, confident.shape[0] - 1))


def rule_result(logits: torch.Tensor, labels: torch.Tensor, stop: torch.Tensor) -> dict:
    """Accuracy (%) and mean number of loops used when each image stops at its own exit."""
    predictions = logits.argmax(dim=2)[stop, torch.arange(len(stop))]
    return {"acc": round(100.0 * (predictions == labels).float().mean().item(), 2),
            "mean_loops": round((stop + 1).float().mean().item(), 2)}


def evaluate_rules(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, dict]:
    """All stopping rules: patience m and confidence tau."""
    rules = {f"patience {m}": rule_result(logits, labels, patience_exit(logits, m)) for m in PATIENCES}
    rules.update({f"confidence {tau}": rule_result(logits, labels, confidence_exit(logits, tau))
                  for tau in CONFIDENCES})
    return rules


def plot_rules(results: dict, out_path: Path) -> None:
    """Accuracy vs loops used: fixed exits as a line, per-image rules as points, one colour per model."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    colours = plt.get_cmap("tab10").colors
    for index, (name, result) in enumerate(results.items()):
        fixed = result["acc_after_block"]
        ax.plot(range(1, len(fixed) + 1), fixed, color=colours[index], marker="o", markersize=3,
                label=f"{name}: fixed exit")
        for rule, values in result["rules"].items():
            marker = "s" if rule.startswith("patience") else "^"
            ax.scatter(values["mean_loops"], values["acc"], color=colours[index], marker=marker, s=30, zorder=3)
            ax.annotate(rule.split()[1], (values["mean_loops"], values["acc"]), fontsize=7,
                        xytext=(3, -9), textcoords="offset points")
    ax.scatter([], [], color="grey", marker="s", label="patience m (m loops agree)")
    ax.scatter([], [], color="grey", marker="^", label="confidence tau")
    ax.set_xlabel("loops / blocks used (mean per image for the rules)")
    ax.set_ylabel("CIFAR-100 test accuracy (%)")
    ax.set_xlim(6.5, 12.5)
    ax.set_ylim(bottom=30)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Early exit: fixed depth vs per-image stopping rules")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Early-exit accuracy and stopping rules per checkpoint on the full CIFAR-100 test set."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for checkpoint in args.checkpoints:
        model = load_model(checkpoint).to(device).eval()
        transform = create_transform(**resolve_data_config({}, model=model), is_training=False)
        loader = DataLoader(CIFAR100(args.data_root, train=False, download=True, transform=transform),
                            batch_size=args.batch_size, num_workers=args.num_workers)
        record = collect_cls_per_layer(model, loader, device, include_final_norm=False)
        logits = exit_logits(model, record, device)
        name = Path(checkpoint).stem
        results[name] = {"checkpoint": checkpoint, "n_images": len(record.labels),
                         "acc_after_block": [round(a, 2) for a in exit_accuracies(logits, record.labels)],
                         **{key: [None if v is None else round(v, 2) for v in values]
                            for key, values in prediction_changes(logits).items()},
                         "rules": evaluate_rules(logits, record.labels)}
        print(f"[{name}] fixed: " + " ".join(f"{k}:{a:.1f}" for k, a in enumerate(results[name]["acc_after_block"], 1)),
              flush=True)
        print(f"[{name}] changed from previous (%): " + " ".join(
            f"{k}:{v}" for k, v in enumerate(results[name]["changed_from_previous"], 1) if v is not None), flush=True)
        for rule, values in results[name]["rules"].items():
            print(f"[{name}] {rule}: acc {values['acc']} with {values['mean_loops']} loops", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "early_exit.json").write_text(json.dumps(results, indent=1))
    plot_rules(results, out_dir / "early_exit_rules.png")
    print(f"wrote {out_dir / 'early_exit.json'} and early_exit_rules.png", flush=True)


if __name__ == "__main__":
    main()
