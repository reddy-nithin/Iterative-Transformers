"""T5: per-class SVD of the CLS token after every transformer block.

For each chosen class (plus a mixed-class control), collect the CLS vector of every image after each block and after
the final LayerNorm, stack them into an N x 192 matrix per layer, and compute its singular values, raw and
mean-centred. Also: how tight the classes are relative to their distances (separation), and the head's top-1.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T5_cls_svd_per_block/cls_svd.py [--dataset cifar100|imagenet]
        [--split train|test] [--checkpoint PATH] [--model-tag TAG] [--save-cls PATH]
"""
import argparse
import json
import math
from pathlib import Path
from typing import NamedTuple

import timm
import torch
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data_sources import (CONTROL, DEFAULT_CIFAR_CLASSES, DEFAULT_IMAGENET_CLASSES, IMAGENET_ROOT, cifar_subsets,
                          imagenet_separation_sets, imagenet_subsets)
from src.models import MODEL_NAME, build

TASK_DIR = Path(__file__).resolve().parent
ENERGY_FRACTION = 0.9


class CLSRecord(NamedTuple):
    """CLS vectors per layer (L, N, D), the head's logits (N, C) and the labels (N,) of one image group."""
    cls: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar100", "imagenet"], default="cifar100")
    p.add_argument("--split", choices=["train", "test"], default="train", help="CIFAR-100 split (ImageNet: train)")
    p.add_argument("--classes", nargs="+", default=None,
                   help="CIFAR-100 class names, or ImageNet wnids / indices (default: the T5 sets)")
    p.add_argument("--checkpoint", default=None, help="weights saved by src.train --save-checkpoint "
                                                      "(default: the ImageNet-pretrained model as-is)")
    p.add_argument("--per-class", type=int, default=500, help="images per class (CIFAR-100: 500 train, 100 test)")
    p.add_argument("--control-size", type=int, default=500, help="mixed-class control images (0 = off)")
    p.add_argument("--model-tag", default="pretrained", help="names the output file: cls_svd_<tag>.json")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="picks the control images")
    p.add_argument("--data-root", default="data", help="CIFAR-100 folder")
    p.add_argument("--imagenet-root", default=IMAGENET_ROOT)
    p.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    p.add_argument("--save-cls", default=None, help="optional .pt path for the raw CLS tensors (keep out of the repo)")
    return p.parse_args()


def load_model(checkpoint: str | None) -> nn.Module:
    """The ImageNet-pretrained ViT-Tiny with its 1000-class head, or a model saved by `src.train`."""
    if checkpoint is None:
        return timm.create_model(MODEL_NAME, pretrained=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build(saved["model"], **saved["model_kwargs"])
    model.load_state_dict(saved["state_dict"], strict=True)
    print(f"loaded {saved['model']} {saved['model_kwargs']} from {checkpoint} "
          f"(epoch {saved['epoch']}, test acc {saved['test_acc']:.2f})", flush=True)
    return model


@torch.no_grad()
def collect_cls_per_layer(model: nn.Module, loader, device: torch.device, include_final_norm: bool = True) -> CLSRecord:
    """CLS vector of every image at layer 0 (input to the first block), after each block and (optionally) after
    the final LayerNorm, plus the head's logits.

    `cls` is a float32 CPU tensor of shape (depth + 1 [+ 1], N, embed_dim). A pre-hook on the `model.blocks`
    container gives layer 0; forward hooks on each unique block module record outputs in call order, so a shared
    block that runs 12 times (iterative model) still yields 12 layers.
    """
    calls: list[torch.Tensor] = []

    def save_input(module: nn.Module, args: tuple) -> None:
        calls.append(args[0][:, 0, :].float().cpu())

    def save_output(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        calls.append(output[:, 0, :].float().cpu())

    depth = len(model.blocks)
    unique_blocks = list({id(block): block for block in model.blocks}.values())
    handles = [model.blocks.register_forward_pre_hook(save_input)]
    handles += [block.register_forward_hook(save_output) for block in unique_blocks]
    model.eval()
    cls_batches, logit_batches, label_batches = [], [], []
    try:
        for images, labels in loader:
            calls.clear()
            features = model.forward_features(images.to(device, non_blocking=True))
            if len(calls) != depth + 1:
                raise RuntimeError(f"expected {depth + 1} CLS snapshots per batch, got {len(calls)}")
            if include_final_norm:
                calls.append(features[:, 0, :].float().cpu())
            cls_batches.append(torch.stack(calls))
            logit_batches.append(model.forward_head(features).float().cpu())
            label_batches.append(torch.as_tensor(labels))
    finally:
        for handle in handles:
            handle.remove()
    return CLSRecord(torch.cat(cls_batches, dim=1), torch.cat(logit_batches), torch.cat(label_batches))


def singular_values(matrix: torch.Tensor, centre: bool) -> torch.Tensor:
    """Singular values (float64, descending) of an N x D matrix, optionally after subtracting the row mean.

    Values below rounding noise (relative to the raw matrix's largest singular value) are set to exactly 0, so
    identical rows give an all-zero centred spectrum instead of ~1e-16 noise.
    """
    m = matrix.double()
    tolerance = max(m.shape) * torch.finfo(torch.float64).eps * torch.linalg.matrix_norm(m, ord=2)
    if centre:
        m = m - m.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(m)
    return torch.where(s > tolerance, s, torch.zeros_like(s))


def effective_rank(s: torch.Tensor) -> float:
    """Entropy effective rank exp(H(p)), p = s / sum(s) (Roy & Vetterli 2007). 0.0 for an all-zero spectrum."""
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return math.exp(-(p * p.log()).sum().item())


def energy_rank(s: torch.Tensor, fraction: float = ENERGY_FRACTION) -> int:
    """Smallest k such that the top k singular values hold `fraction` of sum(s^2). 0 for an all-zero spectrum."""
    energy = s.double() ** 2
    total = energy.sum()
    if total == 0:
        return 0
    return int((energy.cumsum(0) / total < fraction).sum().item()) + 1


def separation_ratio(matrices: list[torch.Tensor]) -> float | None:
    """tr(within-class covariance) / tr(between-class covariance), classes weighted equally (NC1-style, Papyan et
    al. 2020). Small = classes are tight compared with the distances between them. None if all class means coincide
    (e.g. layer 0, where every image has the same CLS)."""
    class_means = torch.stack([m.double().mean(dim=0) for m in matrices])
    within = torch.stack([((m.double() - mean) ** 2).sum(dim=1).mean() for m, mean in zip(matrices, class_means)])
    between = ((class_means - class_means.mean(dim=0)) ** 2).sum(dim=1).mean()
    scale = class_means.norm(dim=1).max() ** 2
    if between <= torch.finfo(torch.float64).eps * max(scale.item(), 1.0):
        return None
    return (within.mean() / between).item()


def top1(record: CLSRecord) -> float:
    """Top-1 accuracy (%) of the model's own head on one group."""
    return 100.0 * (record.logits.argmax(dim=1) == record.labels).float().mean().item()


def spectrum_stats(s: torch.Tensor) -> dict:
    """JSON-ready summary of one spectrum."""
    return {"singular_values": [float(f"{v:.6g}") for v in s.tolist()],
            "effective_rank": round(effective_rank(s), 4),
            "energy_rank_90": energy_rank(s)}


def analyse(cls_per_layer: torch.Tensor) -> dict[str, dict]:
    """Raw and centred spectrum stats for every layer of one group. Keys are layer numbers as strings."""
    return {str(layer): {"raw": spectrum_stats(singular_values(matrix, centre=False)),
                         "centred": spectrum_stats(singular_values(matrix, centre=True))}
            for layer, matrix in enumerate(cls_per_layer)}


def separation_by_layer(records: dict[str, CLSRecord], sets: dict[str, list[str]]) -> dict[str, dict]:
    """For each named set of classes: {layer: separation ratio}."""
    n_layers = len(next(iter(records.values())).cls)
    return {set_name: {str(layer): separation_ratio([records[name].cls[layer] for name in members])
                       for layer in range(n_layers)}
            for set_name, members in sets.items()}


def load_groups(args: argparse.Namespace, model: nn.Module) -> tuple[dict[str, Dataset], dict[str, list[str]]]:
    """The image groups for the chosen dataset and the class sets to measure separation over."""
    transform = create_transform(**resolve_data_config({}, model=model), is_training=False)
    if args.dataset == "imagenet":
        classes = args.classes or list(DEFAULT_IMAGENET_CLASSES)
        groups = imagenet_subsets(transform, args.imagenet_root, classes, args.per_class, args.control_size,
                                  args.seed)
        return groups, imagenet_separation_sets(classes)
    classes = args.classes or list(DEFAULT_CIFAR_CLASSES)
    groups = cifar_subsets(transform, args.data_root, classes, args.per_class, args.control_size, args.seed,
                           args.split)
    return groups, {"all classes": classes}


def main() -> None:
    """Collect CLS vectors per group, run the SVDs, write JSON (and optionally the raw tensors)."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint).to(device)
    groups, separation_sets = load_groups(args, model)
    # the head's labels match the images unless the 1000-class ImageNet head looks at CIFAR images
    head_matches = args.dataset == "imagenet" or args.checkpoint is not None

    records, results = {}, {}
    for name, dataset in groups.items():
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")
        records[name] = collect_cls_per_layer(model, loader, device)
        results[name] = {"n_images": len(dataset), "top1": round(top1(records[name]), 2) if head_matches else None,
                         "layers": analyse(records[name].cls)}
        centred = [results[name]["layers"][str(layer)]["centred"]["effective_rank"]
                   for layer in range(len(records[name].cls))]
        print(f"[{name}] N={len(dataset)} top1={results[name]['top1']} centred effective rank by layer: "
              + " ".join(f"{r:.1f}" for r in centred), flush=True)
    separation = separation_by_layer(records, separation_sets)
    for set_name, by_layer in separation.items():
        print(f"[separation, {set_name}] " + " ".join("-" if r is None else f"{r:.2f}" for r in by_layer.values()),
              flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cls_svd_{args.model_tag}.json"
    class_names = [name for name in groups if name != CONTROL]
    out_path.write_text(json.dumps({
        "model": MODEL_NAME, "model_tag": args.model_tag, "checkpoint": args.checkpoint, "dataset": args.dataset,
        "split": "train" if args.dataset == "imagenet" else args.split, "classes": class_names,
        "per_class": args.per_class, "control_size": args.control_size, "seed": args.seed,
        "final_norm_layer": len(model.blocks) + 1, "timm_version": timm.__version__, "device": str(device),
        "separation": separation, "groups": results,
    }, indent=1))
    print(f"wrote {out_path}", flush=True)

    if args.save_cls:
        save_path = Path(args.save_cls)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_tag": args.model_tag, "cls": {n: r.cls for n, r in records.items()},
                    "labels": {n: r.labels for n, r in records.items()}}, save_path)
        print(f"wrote {save_path}", flush=True)


if __name__ == "__main__":
    main()
