"""T5: test accuracy if a model stopped after block / loop k (final norm + head applied to the CLS after k).

A direct check of the stopping rule: separation says when classes stop getting tighter; this says when the answer
stops getting better. The head was only trained on the output of the last block, so early-exit accuracy is a lower
bound on what a model trained with per-loop exits could do.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T5_cls_svd_per_block/early_exit.py CKPT [CKPT ...]
"""
import argparse
import json
from pathlib import Path

import torch
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from cls_svd import CLSRecord, collect_cls_per_layer, load_model

TASK_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="weights saved by src.train --save-checkpoint")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out", default=str(TASK_DIR / "results" / "early_exit.json"))
    return p.parse_args()


@torch.no_grad()
def exit_accuracies(model: nn.Module, record: CLSRecord, device: torch.device) -> list[float]:
    """Top-1 (%) after each block k = 1..depth, using the model's own final norm and head on the CLS after k."""
    accuracies = []
    for cls in record.cls[1:]:  # layer 0 is the input to the first block
        predictions = model.head(model.norm(cls.to(device))).argmax(dim=1).cpu()
        accuracies.append(100.0 * (predictions == record.labels).float().mean().item())
    return accuracies


def main() -> None:
    """Early-exit accuracy per checkpoint on the full CIFAR-100 test set; prints and writes JSON."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for checkpoint in args.checkpoints:
        model = load_model(checkpoint).to(device).eval()
        transform = create_transform(**resolve_data_config({}, model=model), is_training=False)
        loader = DataLoader(CIFAR100(args.data_root, train=False, download=True, transform=transform),
                            batch_size=args.batch_size, num_workers=args.num_workers)
        record = collect_cls_per_layer(model, loader, device, include_final_norm=False)
        accuracies = exit_accuracies(model, record, device)
        results[Path(checkpoint).stem] = {"checkpoint": checkpoint, "n_images": len(record.labels),
                                          "acc_after_block": [round(a, 2) for a in accuracies]}
        print(f"[{Path(checkpoint).stem}] " + " ".join(f"{k}:{a:.1f}" for k, a in enumerate(accuracies, 1)),
              flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
