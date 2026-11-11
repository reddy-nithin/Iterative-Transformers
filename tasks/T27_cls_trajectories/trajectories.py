"""T27: CLS trajectories through the loops (no training).

For T11-d and T13-c, collect the raw CLS of 500 test images (10 classes x 50) at 13 stages (input, after B0, after
B1, after loops 1..10), then compute:
  (a) PCA on all 500 x 13 vectors: variance share of PC1..PC6 and the components (plot.py projects the paths);
  (b) relative distance to the final CLS, ||CLS_s - CLS_12|| / ||CLS_12||, per image and stage;
  (c) per class pair and stage, sep = ||mu_a - mu_b|| / (rho_a + rho_b), and the split step (first stage with sep >= 1).
Before any of that, each model's own top-1 on the full test set must match its logged value.

Outputs (per model tag): results/cls_<tag>.npz (CLS states, labels, class names) and results/trajectories_<tag>.json.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T27_cls_trajectories/trajectories.py --checkpoint PATH --tag T11-d [--limit N]
"""
import argparse
import json
import logging
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from timm.data import create_transform, resolve_data_config
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CIFAR100

TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
sys.path.insert(0, str(TASKS / "T5_cls_svd_per_block"))
sys.path.insert(0, str(TASKS / "T13_per_loop_readout_loss"))

from cls_svd import collect_cls_per_layer, load_model  # noqa: E402
from data_sources import cifar_subsets  # noqa: E402
from exit_analysis import check_top1, collect_logits, test_loader  # noqa: E402

FIXED_CLASSES = ("apple", "bicycle", "cloud", "fox")
EXTRA_CLASS_COUNT = 6
CLASS_SEED = 0
PER_CLASS = 50
HEATMAP_PER_CLASS = 10
N_COMPONENTS = 6
SPLIT_THRESHOLD = 1.0
SETTLED_DISTANCE = 0.1
STAGE_NAMES = ["input", "B0", "B1"] + [f"loop {t}" for t in range(1, 11)]

log = logging.getLogger("t27")


def choose_classes(all_classes: list[str]) -> list[str]:
    """The 4 T5 classes plus 6 drawn with seed 0 from the remaining labels, all in label order."""
    fixed = [all_classes.index(name) for name in FIXED_CLASSES]
    remaining = sorted(set(range(len(all_classes))) - set(fixed))
    extra = random.Random(CLASS_SEED).sample(remaining, EXTRA_CLASS_COUNT)
    return [all_classes[index] for index in sorted(fixed + extra)]


def pca(vectors: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, components (k, D), variance share of each of the first k components) of an (M, D) matrix."""
    mean = vectors.mean(axis=0)
    _, singular, components = np.linalg.svd(vectors - mean, full_matrices=False)
    variance = singular ** 2
    return mean, components[:n_components], variance[:n_components] / variance.sum()


def distance_to_final(cls: np.ndarray) -> np.ndarray:
    """||CLS_s - CLS_last|| / ||CLS_last|| for every stage s and image: (S, N) from (S, N, D)."""
    final = cls[-1]
    return np.linalg.norm(cls - final, axis=-1) / np.linalg.norm(final, axis=-1)


def pair_separation(first: np.ndarray, second: np.ndarray) -> float:
    """||mu_a - mu_b|| / (rho_a + rho_b) for two (n, D) groups; rho = RMS distance to the group's own mean."""
    mean_a, mean_b = first.mean(axis=0), second.mean(axis=0)
    rho_a = np.sqrt(((first - mean_a) ** 2).sum(axis=1).mean())
    rho_b = np.sqrt(((second - mean_b) ** 2).sum(axis=1).mean())
    spread = rho_a + rho_b
    if spread == 0:
        raise ValueError("both groups have zero spread; separation is undefined (stage 0 must be skipped)")
    return float(np.linalg.norm(mean_a - mean_b) / spread)


def first_split(separations: list[float], first_stage: int) -> int | None:
    """First stage (numbered from `first_stage`) whose separation reaches the threshold, or None."""
    for offset, value in enumerate(separations):
        if value >= SPLIT_THRESHOLD:
            return first_stage + offset
    return None


def split_table(cls: np.ndarray, labels: np.ndarray, class_names: list[str]) -> list[dict]:
    """Per class pair: sep at stages 1..last and the split step. Stage 0 is skipped (all images share one CLS)."""
    rows = []
    for a, b in combinations(range(len(class_names)), 2):
        seps = [pair_separation(cls[s, labels == a], cls[s, labels == b]) for s in range(1, cls.shape[0])]
        rows.append({"pair": [class_names[a], class_names[b]], "sep": [round(v, 4) for v in seps],
                     "split_step": first_split(seps, first_stage=1)})
    return rows


def heatmap_rows(labels: np.ndarray, n_classes: int, per_class: int) -> np.ndarray:
    """Indices of the first `per_class` images of each class, grouped by class."""
    return np.concatenate([np.flatnonzero(labels == c)[:per_class] for c in range(n_classes)])


def settle_stage(distances: np.ndarray) -> int | None:
    """First stage from which the median distance to final stays below SETTLED_DISTANCE, or None."""
    median = np.median(distances, axis=1)
    for stage in range(len(median)):
        if np.all(median[stage:] < SETTLED_DISTANCE):
            return stage
    return None


def analyse(cls: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    """All T27 numbers for one model."""
    mean, components, share = pca(cls.reshape(-1, cls.shape[-1]), N_COMPONENTS)
    distances = distance_to_final(cls)
    splits = split_table(cls, labels, class_names)
    seps = np.array([row["sep"] for row in splits])
    return {
        "cls_norm_mean": [round(float(v), 3) for v in np.linalg.norm(cls, axis=-1).mean(axis=1)],
        "sep_median_over_pairs": [None] + [round(float(v), 4) for v in np.median(seps, axis=0)],
        "sep_max_over_pairs": [None] + [round(float(v), 4) for v in seps.max(axis=0)],
        "pca_variance_share": [round(float(v), 4) for v in share],
        "pca_mean": mean.tolist(), "pca_components": components.tolist(),
        "dist_to_final_median": [round(float(v), 4) for v in np.median(distances, axis=1)],
        "dist_to_final_p90": [round(float(v), 4) for v in np.percentile(distances, 90, axis=1)],
        "settle_stage_median_below_0.1": settle_stage(distances),
        "heatmap_rows": heatmap_rows(labels, len(class_names), HEATMAP_PER_CLASS).tolist(),
        "split_steps": splits,
    }


def class_loader(model: torch.nn.Module, class_names: list[str], args: argparse.Namespace) -> DataLoader:
    """The first PER_CLASS test images of each chosen class, concatenated in class order."""
    transform = create_transform(**resolve_data_config({}, model=model), is_training=False)
    groups = cifar_subsets(transform, args.data_root, class_names, PER_CLASS, 0, CLASS_SEED, "test")
    return DataLoader(ConcatDataset([groups[name] for name in class_names]), batch_size=args.batch_size,
                      num_workers=args.num_workers)


def check_model(model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> float:
    """The model's own final top-1 on the test set (or the --limit subset) against its logged value."""
    logits, labels = collect_logits(model, test_loader(model, args), device)
    accuracy = (logits[-1].argmax(dim=1) == labels).float().mean().item() * 100
    check_top1(accuracy, args.checkpoint, full_test_set=args.limit == 0)
    return accuracy


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True, help="model name used in output files, e.g. T11-d")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="top-1 check on the first N test images only (smoke)")
    parser.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    return parser.parse_args()


def main() -> None:
    """Sanity check, collect CLS states, analyse, save."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint).to(device).eval()
    top1 = check_model(model, args, device)

    all_classes = CIFAR100(args.data_root, train=False, download=True).classes
    class_names = choose_classes(all_classes)
    log.info("[%s] classes: %s", args.tag, ", ".join(class_names))
    record = collect_cls_per_layer(model, class_loader(model, class_names, args), device, include_final_norm=False)
    cls = record.cls.numpy()
    if cls.shape[0] != len(STAGE_NAMES):
        raise RuntimeError(f"expected {len(STAGE_NAMES)} stages, got {cls.shape[0]}")
    expected = np.repeat([all_classes.index(name) for name in class_names], PER_CLASS)
    if not np.array_equal(record.labels.numpy(), expected):
        raise RuntimeError("image order does not match the class order")
    labels = np.repeat(np.arange(len(class_names)), PER_CLASS)

    subset_top1 = (record.logits.argmax(dim=1) == record.labels).float().mean().item() * 100
    result = {"tag": args.tag, "checkpoint": args.checkpoint, "top1_test": round(top1, 2),
              "subset_top1": round(subset_top1, 2), "classes": class_names, "stages": STAGE_NAMES,
              **analyse(cls, labels, class_names)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"cls_{args.tag}.npz", cls=cls.astype(np.float16), labels=labels,
                        classes=np.array(class_names))
    (out_dir / f"trajectories_{args.tag}.json").write_text(json.dumps(result, indent=1))
    log.info("[%s] subset top-1 %.2f; PCA shares %s; median dist to final %s; settles at stage %s", args.tag,
             subset_top1, result["pca_variance_share"], result["dist_to_final_median"],
             result["settle_stage_median_below_0.1"])


if __name__ == "__main__":
    main()
