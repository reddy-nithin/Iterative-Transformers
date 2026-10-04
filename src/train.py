"""Fine-tune baseline or iterative ViT-Tiny on CIFAR-100. Same recipe for both.

Example:  python -m src.train --model baseline --epochs 10
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data import make_loaders
from src.metrics import count_gflops, count_params, measure_throughput
from src.models import build


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "iterative"], required=True)
    p.add_argument("--block-idx", type=int, default=0, help="which pretrained block to reuse (iterative)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subset", type=int, default=0, help="use N train/val images (smoke test)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--tag", default="", help="suffix for output filenames")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    name = f"{args.model}{args.tag}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    kwargs = {"block_idx": args.block_idx} if args.model == "iterative" else {}
    model = build(args.model, **kwargs).to(device)
    train_loader, val_loader = make_loaders(
        model, args.data_root, args.batch_size, args.num_workers, args.subset, args.seed
    )

    info = {
        "args": vars(args),
        "device": str(device),
        "params": count_params(model),
        "gflops": count_gflops(model, device),
        "throughput_img_s": measure_throughput(model, device),
    }
    print(f"[{name}] params={info['params']:,} GFLOPs={info['gflops']:.2f} "
          f"throughput={info['throughput_img_s']:.0f} img/s", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    history, step = [], 0
    for epoch in range(args.epochs):
        model.train()
        t0, loss_sum = time.time(), 0.0
        for x, y in train_loader:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, warmup_steps, args.lr)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
            step += 1
        acc = evaluate(model, val_loader, device, use_amp)
        rec = {"epoch": epoch + 1, "train_loss": loss_sum / steps_per_epoch,
               "test_acc": acc, "epoch_time_s": time.time() - t0}
        history.append(rec)
        print(f"[{name}] epoch {rec['epoch']}/{args.epochs} loss={rec['train_loss']:.4f} "
              f"acc={acc:.2f} ({rec['epoch_time_s']:.0f}s)", flush=True)
        # write every epoch so a crash / dropped session still leaves partial results
        info.update(history=history, final_test_acc=acc, best_test_acc=max(h["test_acc"] for h in history))
        (out_dir / f"{name}.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
