"""T13: per-loop readout accuracy and early-exit rules on the full CIFAR-100 test set (no training).

For each checkpoint (T13-a..d and the references T11-c, T11-d) the readout after every body loop t = 1..10 comes from
`src.per_loop_readout.per_loop_logits`: through the tail on the T11-c layout, final norm + classifier on x_t on the
T11-d layout. Loop 10 is the normal model output, and its top-1 must match the checkpoint's logged test acc.

Per model:
- fixed-depth curve: top-1 when every image stops at loop k, k = 1..10;
- flip rate: % of images whose argmax changes between loop 9 and loop 10;
- exit rules, each swept over thresholds, as (mean loops used, top-1) points; images that never stop use loop 10:
  confidence (first t with max softmax >= tau, tau = 0.3 .. 0.9), KL (first t >= 2 with KL(p_t || p_t-1) < eps, eps
  log-spaced 1e-4 .. 1e-1), patience (stop when the argmax is unchanged for m consecutive loops, m = 1, 2, 3, i.e. m + 1
  loops agree);
- "beats fixed depth": for every rule point, top-1 minus the fixed-depth curve interpolated at the same mean loops,
  and the fewest mean loops that stay within NEAR_FULL_POINTS of loop 10 versus the fixed depth that does the same.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T13_per_loop_readout_loss/exit_analysis.py CKPT [CKPT ...] [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent / "T5_cls_svd_per_block"))  # reuse the T5 / T8 exit helpers
sys.path.insert(0, str(TASK_DIR.parent / "T8_collapse_and_kl_exit"))

from cls_svd import load_model  # noqa: E402
from diagnostics import NEAR_FULL_POINTS, cheapest_near_full, kl_signal, threshold_exit  # noqa: E402
from early_exit import confidence_exit, exit_accuracies, patience_exit, rule_result  # noqa: E402

from src.per_loop_readout import per_loop_logits  # noqa: E402

CONFIDENCES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
KL_THRESHOLDS = tuple(float(f"{eps:.3g}") for eps in np.logspace(-4, -1, 7))  # 1e-4, 3.16e-4, ..., 1e-1
PATIENCES = (1, 2, 3)  # loops with an unchanged argmax
TOP1_TOLERANCE = 0.5  # points; loop 10 must match the logged test acc


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="weights saved by src.train --save-checkpoint")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="use only the first N test images (smoke)")
    p.add_argument("--out", default=str(TASK_DIR / "results" / "exit_analysis.json"))
    return p.parse_args()


def test_loader(model: nn.Module, args: argparse.Namespace) -> DataLoader:
    """The CIFAR-100 test set (or its first `--limit` images) with the model's eval transform."""
    transform = create_transform(**resolve_data_config({}, model=model), is_training=False)
    test_set = CIFAR100(args.data_root, train=False, download=True, transform=transform)
    dataset = Subset(test_set, range(args.limit)) if args.limit else test_set
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(per-loop logits (L, N, classes) in fp32 on the CPU, labels (N,))."""
    chunks, labels = [], []
    for images, targets in loader:
        chunks.append(torch.stack(per_loop_logits(model, images.to(device, non_blocking=True))).float().cpu())
        labels.append(targets)
    return torch.cat(chunks, dim=1), torch.cat(labels)


def flip_rate(logits: torch.Tensor) -> float:
    """% of images whose argmax at the last loop differs from the one before."""
    predictions = logits.argmax(dim=2)
    return round(100.0 * (predictions[-1] != predictions[-2]).float().mean().item(), 2)


def fixed_at(fixed: list[float], loops: float) -> float:
    """The fixed-depth curve (top-1 at loop 1..L) linearly interpolated at a fractional number of loops."""
    return float(np.interp(loops, np.arange(1, len(fixed) + 1), fixed))


def with_gain(point: dict, fixed: list[float]) -> dict:
    """A rule point plus its top-1 gain over the fixed-depth curve at the same mean loops."""
    return {**point, "gain_vs_fixed": round(point["acc"] - fixed_at(fixed, point["mean_loops"]), 2)}


def rule_curves(logits: torch.Tensor, labels: torch.Tensor, fixed: list[float]) -> dict[str, list[dict]]:
    """(threshold, top-1, mean loops, gain vs fixed) for every threshold of the three exit rules."""
    kl = kl_signal(logits)
    curves = {
        "confidence": [{"threshold": tau, **rule_result(logits, labels, confidence_exit(logits, tau))}
                       for tau in CONFIDENCES],
        "kl": [{"threshold": eps, **rule_result(logits, labels, threshold_exit(kl, eps))} for eps in KL_THRESHOLDS],
        "patience": [{"threshold": m, **rule_result(logits, labels, patience_exit(logits, m + 1))}
                     for m in PATIENCES],
    }
    return {name: [with_gain(point, fixed) for point in points] for name, points in curves.items()}


def compare_with_fixed(points: list[dict], fixed: list[float]) -> dict:
    """Best gain over the fixed-depth curve, and the cheapest point within NEAR_FULL_POINTS of loop 10 versus the
    fixed depth needed for the same tolerance (loops saved, as a share of that fixed depth and of full depth)."""
    fixed_near = next(k for k, acc in enumerate(fixed, 1) if acc >= fixed[-1] - NEAR_FULL_POINTS)
    cheapest = cheapest_near_full(points, fixed[-1])
    saved_vs_fixed = None if cheapest is None else round(100.0 * (1 - cheapest["mean_loops"] / fixed_near), 1)
    saved_vs_full = None if cheapest is None else round(100.0 * (1 - cheapest["mean_loops"] / len(fixed)), 1)
    return {"best_gain": max(points, key=lambda p: p["gain_vs_fixed"]), "cheapest_near_full": cheapest,
            "fixed_loops_near_full": fixed_near, "loops_saved_pct_vs_fixed": saved_vs_fixed,
            "loops_saved_pct_vs_full": saved_vs_full}


def analyse(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Fixed-depth curve, flip rate and the three exit rules for one model."""
    fixed = [round(acc, 2) for acc in exit_accuracies(logits, labels)]
    curves = rule_curves(logits, labels, fixed)
    return {"acc_per_loop": fixed, "flip_rate_9_10": flip_rate(logits), "curves": curves,
            "vs_fixed": {name: compare_with_fixed(points, fixed) for name, points in curves.items()}}


def check_top1(acc_last_loop: float, checkpoint: str, full_test_set: bool) -> float:
    """Loop 10's top-1 must match the logged test acc on the full test set; returns the logged value."""
    logged = torch.load(checkpoint, map_location="cpu", weights_only=True)["test_acc"]
    print(f"[{Path(checkpoint).stem}] loop-10 top-1 {acc_last_loop:.2f} (logged {logged:.2f})", flush=True)
    if full_test_set and abs(acc_last_loop - logged) > TOP1_TOLERANCE:
        raise RuntimeError(f"loop-10 top-1 {acc_last_loop:.2f} does not match the logged {logged:.2f}")
    return logged


def report(name: str, result: dict) -> None:
    """One-screen summary of a model's exit analysis."""
    print(f"[{name}] fixed: " + " ".join(f"{t}:{a:.1f}" for t, a in enumerate(result["acc_per_loop"], 1)), flush=True)
    print(f"[{name}] flip rate 9->10: {result['flip_rate_9_10']}%", flush=True)
    for rule, summary in result["vs_fixed"].items():
        best, cheap = summary["best_gain"], summary["cheapest_near_full"]
        cheap_text = "none" if cheap is None else f"{cheap['mean_loops']} loops ({cheap['acc']}%)"
        print(f"[{name}] {rule}: best gain {best['gain_vs_fixed']:+.2f} at {best['mean_loops']} loops; within "
              f"{NEAR_FULL_POINTS} pt of loop 10 with {cheap_text} vs fixed {summary['fixed_loops_near_full']}",
              flush=True)


def main() -> None:
    """Exit analysis for every checkpoint; results merge into the output JSON so models can be added later."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}
    for checkpoint in args.checkpoints:
        name = Path(checkpoint).stem
        model = load_model(checkpoint).to(device).float().eval()
        logits, labels = collect_logits(model, test_loader(model, args), device)
        result = analyse(logits, labels)
        logged = check_top1(result["acc_per_loop"][-1], checkpoint, full_test_set=not args.limit)
        results[name] = {"checkpoint": checkpoint, "n_images": len(labels), "logged_test_acc": round(logged, 2),
                         **result}
        report(name, result)
        out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
