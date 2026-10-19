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
    p.add_argument("--model", choices=["baseline", "scratch", "iterative", "lora", "unroll"], required=True)
    p.add_argument("--block-idx", type=int, default=0, help="which pretrained block to reuse (init=block)")
    p.add_argument("--init", choices=["block", "avg"], default=None,
                   help="shared block: one pretrained block or the mean of all 12 "
                        "(default: block for iterative, avg for lora)")
    p.add_argument("--rank", type=int, default=2, help="LoRA rank per iteration (lora)")
    p.add_argument("--lora-init", choices=["svd", "zero"], default="svd", help="LoRA start (lora)")
    p.add_argument("--unique-blocks", type=int, default=1,
                   help="unroll: number of distinct blocks n (blocks 0..n-2 kept, block n-1 repeated)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=None, help="separate lr for the classifier head (default: --lr)")
    p.add_argument("--lora-lr", type=float, default=None,
                   help="lora: separate lr for the per-loop LoRA factors (default: --lr)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subset", type=int, default=0, help="use N train/val images (smoke test)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--tag", default="", help="suffix for output filenames")
    p.add_argument("--save-checkpoint", default=None,
                   help="write model weights here after every epoch (keep it out of the repo)")
    return p.parse_args()


def model_kwargs(args) -> dict:
    if args.model == "iterative":
        return {"block_idx": args.block_idx, "init": args.init or "block"}
    if args.model == "lora":
        return {"block_idx": args.block_idx, "init": args.init or "avg",
                "rank": args.rank, "lora_init": args.lora_init}
    if args.model == "unroll":
        return {"unique_blocks": args.unique_blocks}
    return {}


def save_checkpoint(path: str, model: nn.Module, model_name: str, kwargs: dict, epoch: int, test_acc: float) -> None:
    """Weights plus what is needed to rebuild the model (`src.models.build(model_name, **kwargs)`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_name, "model_kwargs": kwargs, "state_dict": model.state_dict(),
                "epoch": epoch, "test_acc": test_acc}, path)


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


def is_lora_param(name: str) -> bool:
    """True for the per-loop LoRA factors of `LoRALoopBlocks` (`blocks.lora_A.*`, `blocks.lora_B.*`)."""
    return "lora_A" in name or "lora_B" in name


def make_param_groups(
    model: nn.Module, lr: float, head_lr: float | None = None, lora_lr: float | None = None
) -> list[dict]:
    """Optimizer param groups. With `head_lr` / `lora_lr`, the classifier head / LoRA factors get their own lr;
    with neither, one group.

    Each group stores `base_lr` so the cosine schedule scales every group from its own starting lr.
    """
    if head_lr is None and lora_lr is None:
        return [{"params": list(model.parameters()), "lr": lr, "base_lr": lr}]
    head_ids = {id(p) for p in model.get_classifier().parameters()} if head_lr is not None else set()
    body, head, lora = [], [], []
    for name, p in model.named_parameters():
        if id(p) in head_ids:
            head.append(p)
        elif lora_lr is not None and is_lora_param(name):
            lora.append(p)
        else:
            body.append(p)
    if lora_lr is not None and not lora:
        raise ValueError("--lora-lr was given but the model has no LoRA parameters (use it with --model lora)")
    groups = [(body, lr), (head, head_lr), (lora, lora_lr)]
    return [{"params": ps, "lr": g_lr, "base_lr": g_lr} for ps, g_lr in groups if ps]


@torch.no_grad()
def evaluate(model, loader, device, use_amp, criterion):
    """Test accuracy (%) and mean loss (same criterion as training, so the two losses are comparable)."""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total, loss_sum / total


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    name = f"{args.model}{args.tag}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    model = build(args.model, **model_kwargs(args)).to(device)
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

    groups = make_param_groups(model, args.lr, args.head_lr, args.lora_lr)
    print(f"[{name}] param groups: " + ", ".join(
        f"{sum(p.numel() for p in g['params']):,} @ lr {g['base_lr']:g}" for g in groups), flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    history, step = [], 0
    for epoch in range(args.epochs):
        model.train()
        t0, loss_sum = time.time(), 0.0
        train_correct = train_seen = 0
        for x, y in train_loader:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, warmup_steps, g["base_lr"])
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
            # running train acc: augmented batches, train mode, weights changing within the epoch (no extra pass)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_seen += y.numel()
            step += 1
        acc, test_loss = evaluate(model, val_loader, device, use_amp, criterion)
        rec = {"epoch": epoch + 1, "train_loss": loss_sum / steps_per_epoch, "test_loss": test_loss,
               "train_acc": 100.0 * train_correct / train_seen, "test_acc": acc, "epoch_time_s": time.time() - t0}
        history.append(rec)
        print(f"[{name}] epoch {rec['epoch']}/{args.epochs} loss={rec['train_loss']:.4f} "
              f"test_loss={test_loss:.4f} train_acc={rec['train_acc']:.2f} "
              f"acc={acc:.2f} ({rec['epoch_time_s']:.0f}s)", flush=True)
        # write every epoch so a crash / dropped session still leaves partial results
        info.update(history=history, final_test_acc=acc, best_test_acc=max(h["test_acc"] for h in history))
        (out_dir / f"{name}.json").write_text(json.dumps(info, indent=2))
        if args.save_checkpoint:
            save_checkpoint(args.save_checkpoint, model, args.model, model_kwargs(args), rec["epoch"], acc)


if __name__ == "__main__":
    main()
