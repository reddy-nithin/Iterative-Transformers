"""T25: class-cloud exit. Stop looping when the CLS feature reaches a class center (no training).

Per model:
  1. Sanity: the model's own loop-10 top-1 on the full test set must match its logged value.
  2. Centers from all 50k TRAIN images (test transform, fixed order, true labels): per-loop centers mu_c^(t) of
     z_t = final_norm(x_t)[:, 0], shared centers mu_c^(10), radii r_c(q) = q-quantile of cosine distance of class-c
     z_10 to mu_c^(10). Saved as results/centers_<tag>.pt.
  3. On TEST: (a) NCC top-1 per loop (per-loop and shared centers), readout top-1, NCC/readout mismatch;
     (b) exit rules R1 radius, R2 ratio, R3 ratio + agreement, with per-loop and shared centers, scored against the
     fixed-depth curve exactly as T13 / T19 (with_gain, compare_with_fixed).

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T25_class_cloud_exit/class_cloud.py --checkpoint PATH --tag T13-c
        [--limit N --train-limit M]   (smoke)
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import torch
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100

TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
sys.path.insert(0, str(TASKS / "T5_cls_svd_per_block"))
sys.path.insert(0, str(TASKS / "T13_per_loop_readout_loss"))

from early_exit import exit_accuracies, rule_result  # noqa: E402
from exit_analysis import check_top1, compare_with_fixed, load_model, with_gain  # noqa: E402

from src.per_loop_readout import per_loop_logits_and_cls  # noqa: E402

QUANTILES = (0.25, 0.5, 0.75)
RATIOS = (0.5, 0.6, 0.7, 0.8, 0.9)
MIN_EXIT_LOOP = 2  # rules may stop at loop t >= 2
NUM_CLASSES = 100
CENTER_MODES = ("per_loop", "shared")

log = logging.getLogger("t25")


class Features(NamedTuple):
    """Per-loop features of a set of images: z (L, N, D), logits (L, N, C) or None, labels (N,)."""
    z: torch.Tensor
    logits: torch.Tensor | None
    labels: torch.Tensor


def fixed_loader(model: nn.Module, split: str, limit: int, args: argparse.Namespace) -> DataLoader:
    """A CIFAR-100 split (or its first `limit` images) in stored order with the model's TEST transform."""
    transform = create_transform(**resolve_data_config({}, model=model), is_training=False)
    dataset = CIFAR100(args.data_root, train=split == "train", download=True, transform=transform)
    if limit:
        dataset = Subset(dataset, range(limit))
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)


@torch.no_grad()
def collect_features(model: nn.Module, loader, device: torch.device, keep_logits: bool) -> Features:
    """z_t = final_norm(x_t)[:, 0] at every loop (and optionally the per-loop logits), fp32 on the CPU."""
    z_chunks, logit_chunks, labels = [], [], []
    for images, targets in loader:
        logits, cls_states = per_loop_logits_and_cls(model, images.to(device, non_blocking=True))
        z_chunks.append(torch.stack([model.norm(cls) for cls in cls_states[1:]]).float().cpu())
        if keep_logits:
            logit_chunks.append(torch.stack(logits).float().cpu())
        labels.append(torch.as_tensor(targets))
    logits = torch.cat(logit_chunks, dim=1) if keep_logits else None
    return Features(torch.cat(z_chunks, dim=1), logits, torch.cat(labels))


def class_means(z: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Mean feature per class: (C, D) from z (N, D). Raises if a class has no images."""
    counts = torch.bincount(labels, minlength=num_classes)
    if (counts == 0).any():
        raise ValueError(f"classes without images: {torch.nonzero(counts == 0).flatten().tolist()}")
    sums = torch.zeros(num_classes, z.shape[-1], dtype=z.dtype).index_add_(0, labels, z)
    return sums / counts[:, None].to(z.dtype)


def cosine_distance(z: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """1 - cos between every row of z (N, D) and every center (C, D): (N, C)."""
    return 1 - nn.functional.normalize(z, dim=-1) @ nn.functional.normalize(centers, dim=-1).T


def class_radii(z: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor, quantiles: tuple) -> torch.Tensor:
    """r_c(q): q-quantile of the distance of class-c points to their own center, (C, len(quantiles))."""
    own = cosine_distance(z, centers).gather(1, labels[:, None]).squeeze(1)
    q = torch.tensor(quantiles, dtype=own.dtype)
    return torch.stack([torch.quantile(own[labels == c], q) for c in range(centers.shape[0])])


def build_centers(train: Features, num_classes: int = NUM_CLASSES, quantiles: tuple = QUANTILES) -> dict:
    """Per-loop centers (L, C, D), shared centers (C, D) and radii (C, Q) from training features."""
    per_loop = torch.stack([class_means(z_t, train.labels, num_classes) for z_t in train.z])
    shared = per_loop[-1]
    return {"per_loop": per_loop, "shared": shared,
            "radii": class_radii(train.z[-1], train.labels, shared, quantiles), "quantiles": quantiles,
            "feature": "z_t = final_norm(x_t)[:, 0], cosine distance"}


def loop_distances(z: torch.Tensor, centers: dict, mode: str) -> torch.Tensor:
    """Cosine distance of every feature to every center at every loop: (L, N, C)."""
    return torch.stack([cosine_distance(z_t, centers["per_loop"][t] if mode == "per_loop" else centers["shared"])
                        for t, z_t in enumerate(z)])


def nearest_two(distances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(d1, d2, c1) per loop and image from distances (L, N, C)."""
    values, indices = distances.topk(2, dim=-1, largest=False)
    return values[..., 0], values[..., 1], indices[..., 0]


def first_stop(condition: torch.Tensor, min_loop: int) -> torch.Tensor:
    """0-based loop index of the first loop >= min_loop (1-based) where `condition` (L, N) holds; else the last."""
    allowed = condition.clone()
    allowed[: min_loop - 1] = False
    allowed[-1] = True
    return allowed.float().argmax(dim=0)


def radius_condition(d1: torch.Tensor, c1: torch.Tensor, radii: torch.Tensor, q_index: int) -> torch.Tensor:
    """R1: the nearest center is within its class's radius."""
    return d1 <= radii[:, q_index][c1]


def ratio_condition(d1: torch.Tensor, d2: torch.Tensor, tau: float) -> torch.Tensor:
    """R2: nearest / second-nearest distance ratio at most tau."""
    return d1 / d2 <= tau


def percent(mask: torch.Tensor) -> float:
    """Share of True entries in percent, 2 decimals."""
    return round(100.0 * mask.float().mean().item(), 2)


def ncc_table(test: Features, centers: dict) -> list[dict]:
    """Per loop: NCC top-1 (per-loop / shared centers), readout top-1, NCC (per-loop) vs readout mismatch."""
    readout = test.logits.argmax(dim=-1)
    ncc = {mode: loop_distances(test.z, centers, mode).argmin(dim=-1) for mode in CENTER_MODES}
    return [{"loop": t + 1,
             "ncc_per_loop": percent(ncc["per_loop"][t] == test.labels),
             "ncc_shared": percent(ncc["shared"][t] == test.labels),
             "readout": percent(readout[t] == test.labels),
             "mismatch": percent(ncc["per_loop"][t] != readout[t])}
            for t in range(test.z.shape[0])]


def rule_conditions(test: Features, centers: dict, mode: str) -> dict[str, list[tuple[str, float, torch.Tensor]]]:
    """Stop conditions (L, N) for every R1 / R2 / R3 threshold with one center mode."""
    d1, d2, c1 = nearest_two(loop_distances(test.z, centers, mode))
    agrees = c1 == test.logits.argmax(dim=-1)
    return {
        "R1 radius": [("q", q, radius_condition(d1, c1, centers["radii"], i))
                      for i, q in enumerate(centers["quantiles"])],
        "R2 ratio": [("tau", tau, ratio_condition(d1, d2, tau)) for tau in RATIOS],
        "R3 agree": [("tau", tau, ratio_condition(d1, d2, tau) & agrees) for tau in RATIOS],
    }


def rule_points(test: Features, centers: dict, mode: str, fixed: list[float]) -> dict[str, list[dict]]:
    """Accuracy, mean loops and gain vs fixed depth for every threshold of every rule."""
    curves = {}
    for name, points in rule_conditions(test, centers, mode).items():
        curves[name] = [with_gain({key: value, **rule_result(test.logits, test.labels,
                                                             first_stop(condition, MIN_EXIT_LOOP))}, fixed)
                        for key, value, condition in points]
    return curves


def analyse(test: Features, centers: dict) -> dict:
    """Tables (a) and (b) for one model."""
    fixed = [round(acc, 2) for acc in exit_accuracies(test.logits, test.labels)]
    curves = {mode: rule_points(test, centers, mode, fixed) for mode in CENTER_MODES}
    best = {f"{rule} ({mode})": compare_with_fixed(points, fixed)["best_gain"]
            for mode, rules in curves.items() for rule, points in rules.items()}
    return {"acc_per_loop": fixed, "ncc": ncc_table(test, centers), "curves": curves, "best_gain": best}


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True, help="model name used in output files, e.g. T13-c")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N test images only (smoke)")
    parser.add_argument("--train-limit", type=int, default=0, help="first N train images only (smoke)")
    parser.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    return parser.parse_args()


def main() -> None:
    """Sanity check, centers from train, analysis on test, save."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint).to(device).eval()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test = collect_features(model, fixed_loader(model, "test", args.limit, args), device, keep_logits=True)
    top1 = percent(test.logits[-1].argmax(dim=-1) == test.labels)
    check_top1(top1, args.checkpoint, full_test_set=args.limit == 0)

    train = collect_features(model, fixed_loader(model, "train", args.train_limit, args), device, keep_logits=False)
    centers = build_centers(train)
    torch.save(centers, out_dir / f"centers_{args.tag}.pt")

    result = {"tag": args.tag, "checkpoint": args.checkpoint, "top1_test": top1, "n_train": len(train.labels),
              "n_test": len(test.labels), **analyse(test, centers)}
    (out_dir / f"class_cloud_{args.tag}.json").write_text(json.dumps(result, indent=1))
    for row in result["ncc"]:
        log.info("[%s] loop %2d  NCC per-loop %.2f  shared %.2f  readout %.2f  mismatch %.2f", args.tag, row["loop"],
                 row["ncc_per_loop"], row["ncc_shared"], row["readout"], row["mismatch"])
    for name, point in result["best_gain"].items():
        log.info("[%s] best %-22s %s", args.tag, name, point)


if __name__ == "__main__":
    main()
