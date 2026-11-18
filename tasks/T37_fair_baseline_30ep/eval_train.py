"""T37: top-1 of the epoch-30 checkpoint on the full 50k CIFAR-100 training set (eval transform, no augmentation)."""
import json
from pathlib import Path

import torch
import torch.nn as nn
from timm.data import create_dataset, create_transform, resolve_data_config
from torch.utils.data import DataLoader

from src.models import build
from src.train import evaluate

CKPT = "/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/T37-a_baseline_12blk_30ep.pt"
OUT = Path("tasks/T37_fair_baseline_30ep/results/train_set_acc.json")


def main():
    device = torch.device("cuda")
    ckpt = torch.load(CKPT, map_location="cpu")
    model = build(ckpt["model"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # same eval transform as the test loader in src/data.py, applied to the train split
    cfg = resolve_data_config({}, model=model)
    ds = create_dataset("torch/cifar100", root="data", split="train", download=True)
    ds.transform = create_transform(**cfg, is_training=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    acc, loss = evaluate(model, loader, device, True, nn.CrossEntropyLoss(label_smoothing=0.1))
    res = {"checkpoint": CKPT, "epoch": ckpt["epoch"], "n_images": len(ds), "train_top1": acc, "train_loss": loss}
    OUT.write_text(json.dumps(res, indent=2))
    print(res)


if __name__ == "__main__":
    main()
