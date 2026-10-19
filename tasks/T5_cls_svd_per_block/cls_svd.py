"""T5: per-class SVD of the CLS token after every transformer block.

For each chosen CIFAR-100 class (plus a mixed-class control), collect the CLS vector of every image after each
block, stack them into an N x 192 matrix per layer, and compute its singular values, raw and mean-centred.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T5_cls_svd_per_block/cls_svd.py [--per-class 500] [--save-cls PATH]
"""
import argparse
import json
import math
from pathlib import Path

import timm
import torch
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100

from src.models import MODEL_NAME

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_CLASSES = ("apple", "bicycle", "fox", "cloud")  # one each from 4 different CIFAR-100 superclasses
CONTROL = "control (all classes)"
ENERGY_FRACTION = 0.9


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES), help="CIFAR-100 class names")
    p.add_argument("--per-class", type=int, default=500, help="images per class (CIFAR-100 train has 500)")
    p.add_argument("--control-size", type=int, default=500, help="mixed-class control images (0 = off)")
    p.add_argument("--model-tag", default="pretrained", help="names the output file: cls_svd_<tag>.json")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="picks the control images")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    p.add_argument("--save-cls", default=None, help="optional .pt path for the raw CLS tensors (keep out of the repo)")
    return p.parse_args()


def resolve_class_indices(all_classes: list[str], names: list[str]) -> list[int]:
    """Map class names to CIFAR-100 label indices. Raises ValueError on an unknown name."""
    unknown = [name for name in names if name not in all_classes]
    if unknown:
        raise ValueError(f"unknown CIFAR-100 class(es) {unknown}; valid names: {sorted(all_classes)}")
    return [all_classes.index(name) for name in names]


def load_class_subsets(
    model: nn.Module, root: str, class_names: list[str], per_class: int, control_size: int, seed: int
) -> dict[str, Subset]:
    """One Subset of the CIFAR-100 train split per class, plus a mixed-class control.

    Uses the eval transform the pretrained model expects (resize to 224, its mean/std, no augmentation).
    torchvision's CIFAR100 is the same data timm's "torch/cifar100" wraps, and it exposes `.classes`/`.targets`.
    """
    transform = create_transform(**resolve_data_config({}, model=model), is_training=False)
    dataset = CIFAR100(root, train=True, download=True, transform=transform)
    targets = torch.tensor(dataset.targets)
    subsets = {}
    for name, label in zip(class_names, resolve_class_indices(dataset.classes, class_names)):
        indices = (targets == label).nonzero().flatten()[:per_class].tolist()
        subsets[name] = Subset(dataset, indices)
    if control_size:
        generator = torch.Generator().manual_seed(seed)
        subsets[CONTROL] = Subset(dataset, torch.randperm(len(dataset), generator=generator)[:control_size].tolist())
    return subsets


@torch.no_grad()
def collect_cls_per_layer(model: nn.Module, loader, device: torch.device) -> torch.Tensor:
    """CLS vector of every image at layer 0 (input to the first block) and after each block.

    Returns a float32 CPU tensor of shape (depth + 1, N, embed_dim). A pre-hook on the `model.blocks` container
    gives layer 0; forward hooks on each unique block module record outputs in call order, so a shared block
    that runs 12 times (iterative model) still yields 12 layers.
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
    batches = []
    try:
        for images, _ in loader:
            calls.clear()
            model.forward_features(images.to(device, non_blocking=True))
            if len(calls) != depth + 1:
                raise RuntimeError(f"expected {depth + 1} CLS snapshots per batch, got {len(calls)}")
            batches.append(torch.stack(calls))
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(batches, dim=1)


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


def main() -> None:
    """Collect CLS vectors per group, run the SVDs, write JSON (and optionally the raw tensors)."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(MODEL_NAME, pretrained=True).to(device)  # ImageNet weights and head, as-is
    subsets = load_class_subsets(model, args.data_root, args.classes, args.per_class, args.control_size, args.seed)

    cls_by_group, groups = {}, {}
    for name, subset in subsets.items():
        loader = DataLoader(subset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")
        cls_by_group[name] = collect_cls_per_layer(model, loader, device)
        groups[name] = {"n_images": len(subset), "layers": analyse(cls_by_group[name])}
        centred = [groups[name]["layers"][str(layer)]["centred"]["effective_rank"]
                   for layer in range(len(cls_by_group[name]))]
        print(f"[{name}] N={len(subset)} centred effective rank by layer: "
              + " ".join(f"{r:.1f}" for r in centred), flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cls_svd_{args.model_tag}.json"
    out_path.write_text(json.dumps({
        "model": MODEL_NAME, "model_tag": args.model_tag, "classes": args.classes, "per_class": args.per_class,
        "control_size": args.control_size, "seed": args.seed, "timm_version": timm.__version__,
        "device": str(device), "groups": groups,
    }, indent=1))
    print(f"wrote {out_path}", flush=True)

    if args.save_cls:
        save_path = Path(args.save_cls)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_tag": args.model_tag, "cls": cls_by_group}, save_path)
        print(f"wrote {save_path}", flush=True)


if __name__ == "__main__":
    main()
