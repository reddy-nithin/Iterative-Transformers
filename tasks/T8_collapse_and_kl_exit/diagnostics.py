"""T8: per-loop diagnostics on trained checkpoints (no training).

1. Token collapse. On a fixed set of test images, take the full token matrix (197 tokens x 192 dims) of every image
   after each loop / block and measure how alike the tokens are: the token-centred singular values (normalised by the
   largest), their effective rank, and the mean pairwise cosine similarity between tokens (near 1 = collapsed).
2. Exit signals. On the full test set, apply the model's own final norm + head to the CLS after each loop, then stop
   each image at the first loop where a signal falls below a threshold: KL(p_l || p_l-1) between successive
   predictions, or the L2 size of the CLS change. Thresholds are swept; each gives (mean loops used, accuracy), compared
   with fixed depth and with the confidence rule from T5.

Heads were trained only on the last loop's output, so every early-exit number is a lower bound.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T8_collapse_and_kl_exit/diagnostics.py CKPT [CKPT ...] [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent / "T5_cls_svd_per_block"))  # reuse the T5 hooks and exit helpers

from cls_svd import CLSRecord, collect_cls_per_layer, load_model  # noqa: E402
from early_exit import confidence_exit, exit_accuracies, exit_logits, rule_result  # noqa: E402

COLLAPSE_IMAGES = 1000
CONFIDENCES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SIGNAL_QUANTILES = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TOP1_TOLERANCE = 0.5  # points; the recomputed top-1 must match the checkpoint's logged test acc
NEAR_FULL_POINTS = 1.0  # "as good as full depth" = within this many points of the last loop's accuracy


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="weights saved by src.train --save-checkpoint")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--collapse-images", type=int, default=COLLAPSE_IMAGES)
    p.add_argument("--limit", type=int, default=0, help="use only the first N test images for the exit part (smoke)")
    p.add_argument("--seed", type=int, default=0, help="picks the collapse images")
    p.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    return p.parse_args()


# ---------- 1. token collapse ----------

def token_stats(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per image of a (B, T, D) batch: normalised token-centred spectrum (B, min(T, D)), effective rank (B,) and mean
    pairwise cosine similarity between distinct tokens (B,).

    Singular values below rounding noise are set to 0, so identical tokens give an all-zero spectrum and effective
    rank 0 (same convention as T5's `singular_values` / `effective_rank`).
    """
    x = tokens.float()
    n_tokens = x.shape[1]
    centred = x - x.mean(dim=1, keepdim=True)
    s = torch.linalg.svdvals(centred)
    tolerance = max(x.shape[1:]) * torch.finfo(torch.float32).eps * torch.linalg.matrix_norm(x).unsqueeze(1)
    s = torch.where(s > tolerance, s, torch.zeros_like(s))
    top = s[:, :1]
    spectrum = torch.where(top > 0, s / top.clamp_min(1e-30), torch.zeros_like(s))
    p = s / s.sum(dim=1, keepdim=True).clamp_min(1e-30)
    entropy = -(p * torch.where(p > 0, p.log(), torch.zeros_like(p))).sum(dim=1)
    eff_rank = torch.where(s.sum(dim=1) > 0, entropy.exp(), torch.zeros_like(entropy))
    unit = F.normalize(x, dim=2)
    gram = unit @ unit.transpose(1, 2)
    cosine = (gram.sum(dim=(1, 2)) - gram.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n_tokens * (n_tokens - 1))
    return {"spectrum": spectrum, "effective_rank": eff_rank, "cosine": cosine}


@torch.no_grad()
def collect_token_stats(model: nn.Module, loader, device: torch.device) -> dict[str, list]:
    """Mean `token_stats` over all images at layer 0 (block input) and after each loop / block.

    Same hook pattern as T5's `collect_cls_per_layer`: a pre-hook on `model.blocks` gives layer 0, forward hooks on
    each unique block record outputs in call order, so a block run 12 times yields 12 layers. Stats are summed on the
    fly, so no full token tensors are kept.
    """
    calls: list[torch.Tensor] = []
    depth = len(model.blocks)
    unique_blocks = list({id(block): block for block in model.blocks}.values())
    handles = [model.blocks.register_forward_pre_hook(lambda module, args: calls.append(args[0]))]
    handles += [block.register_forward_hook(lambda module, args, output: calls.append(output))
                for block in unique_blocks]
    sums: dict[str, list[torch.Tensor]] | None = None
    n_images = 0
    model.eval()
    try:
        for images, _ in loader:
            calls.clear()
            model.forward_features(images.to(device, non_blocking=True))
            if len(calls) != depth + 1:
                raise RuntimeError(f"expected {depth + 1} token snapshots per batch, got {len(calls)}")
            batch = [token_stats(tokens) for tokens in calls]
            if sums is None:
                sums = {key: [torch.zeros_like(layer[key][0]) for layer in batch] for key in batch[0]}
            for key in sums:
                for layer, stats in enumerate(batch):
                    sums[key][layer] += stats[key].sum(dim=0)
            n_images += len(images)
    finally:
        for handle in handles:
            handle.remove()
    if sums is None:
        raise ValueError("empty loader")
    return {key: [(total / n_images).cpu().tolist() for total in layers] for key, layers in sums.items()}


# ---------- 2. exit signals ----------

def kl_signal(logits: torch.Tensor) -> torch.Tensor:
    """(depth, N): KL(p_l || p_l-1) for l >= 1; +inf at the first exit (nothing to compare with)."""
    log_p = logits.float().log_softmax(dim=2)
    kl = (log_p[1:].exp() * (log_p[1:] - log_p[:-1])).sum(dim=2)
    return torch.cat([torch.full_like(kl[:1], float("inf")), kl])


def cls_change_signal(cls: torch.Tensor) -> torch.Tensor:
    """(depth, N): L2 norm of the CLS change made by loop l, ||cls_l - cls_l-1||. `cls` is (depth + 1, N, D) with
    layer 0 = the CLS entering the first block."""
    return (cls[1:] - cls[:-1]).float().norm(dim=2)


def threshold_exit(signal: torch.Tensor, threshold: float) -> torch.Tensor:
    """Per image, the first exit index (0-based) whose signal is below `threshold`; last exit otherwise."""
    below = signal < threshold
    first = below.float().argmax(dim=0)  # argmax returns the first maximal index
    return torch.where(below.any(dim=0), first, torch.full_like(first, below.shape[0] - 1))


def signal_thresholds(signal: torch.Tensor) -> list[float]:
    """Thresholds at fixed quantiles of the signal's finite values, so the sweep fits each signal's own scale."""
    finite = signal[torch.isfinite(signal)].float()
    return sorted({float(f"{v:.4g}") for v in torch.quantile(finite, torch.tensor(SIGNAL_QUANTILES)).tolist()})


def sweep(logits: torch.Tensor, labels: torch.Tensor, signal: torch.Tensor) -> list[dict]:
    """One {threshold, acc, mean_loops} point per threshold."""
    return [{"threshold": t, **rule_result(logits, labels, threshold_exit(signal, t))}
            for t in signal_thresholds(signal)]


def best_within_budget(points: list[dict], budget: float) -> dict | None:
    """The most accurate rule point using at most `budget` loops on average; None if every point uses more."""
    affordable = [p for p in points if p["mean_loops"] <= budget]
    return max(affordable, key=lambda p: p["acc"]) if affordable else None


def cheapest_near_full(points: list[dict], full_acc: float, tolerance: float = NEAR_FULL_POINTS) -> dict | None:
    """The rule point with the fewest mean loops whose accuracy is within `tolerance` of full depth; None if none."""
    close = [p for p in points if p["acc"] >= full_acc - tolerance]
    return min(close, key=lambda p: p["mean_loops"]) if close else None


def compare_with_fixed(points: list[dict], fixed: list[float]) -> dict:
    """The two questions asked of every rule: does it beat fixed depth - 1 loops at no more compute, and how few loops
    does it need to stay within NEAR_FULL_POINTS of full depth (fixed depth needs `fixed_loops_near_full`)."""
    depth = len(fixed)
    best = best_within_budget(points, depth - 1)
    fixed_near = next(k for k, acc in enumerate(fixed, 1) if acc >= fixed[-1] - NEAR_FULL_POINTS)
    return {"fixed_one_less": fixed[depth - 2], "best_within_one_less": best,
            "beats_fixed_one_less": best is not None and best["acc"] > fixed[depth - 2],
            "cheapest_near_full": cheapest_near_full(points, fixed[-1]), "fixed_loops_near_full": fixed_near}


def exit_analysis(model: nn.Module, record: CLSRecord, device: torch.device) -> dict:
    """Fixed depth, confidence, KL and CLS-change exit curves for one model."""
    logits = exit_logits(model, record, device)
    labels = record.labels
    fixed = [round(a, 2) for a in exit_accuracies(logits, labels)]
    confidence = [{"threshold": tau, **rule_result(logits, labels, confidence_exit(logits, tau))}
                  for tau in CONFIDENCES]
    kl = kl_signal(logits)
    cls_change = cls_change_signal(record.cls)
    curves = {"confidence": confidence, "kl": sweep(logits, labels, kl), "cls_change": sweep(logits, labels, cls_change)}
    return {"acc_after_loop": fixed, "curves": curves,
            "vs_fixed": {name: compare_with_fixed(points, fixed) for name, points in curves.items()},
            "median_kl_per_loop": [None] + [round(v, 5) for v in kl[1:].median(dim=1).values.tolist()],
            "median_cls_change_per_loop": [round(v, 4) for v in cls_change.median(dim=1).values.tolist()]}


# ---------- data, checks, plots ----------

def test_loaders(model: nn.Module, args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """(collapse loader: fixed random subset, exit loader: the whole test set or its first `--limit` images)."""
    transform = create_transform(**resolve_data_config({}, model=model), is_training=False)
    test_set = CIFAR100(args.data_root, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(args.seed)
    collapse_ids = torch.randperm(len(test_set), generator=generator)[: args.collapse_images].tolist()
    exit_set = Subset(test_set, range(args.limit)) if args.limit else test_set
    make = lambda dataset: DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)  # noqa: E731
    return make(Subset(test_set, collapse_ids)), make(exit_set)


def check_top1(record: CLSRecord, checkpoint: str, full_test_set: bool) -> float:
    """The model's own top-1 on the exit images; on the full test set it must match the logged test acc."""
    top1 = 100.0 * (record.logits.argmax(dim=1) == record.labels).float().mean().item()
    logged = torch.load(checkpoint, map_location="cpu", weights_only=True)["test_acc"]
    print(f"[{Path(checkpoint).stem}] own top-1 {top1:.2f} (logged at save time {logged:.2f})", flush=True)
    if full_test_set and abs(top1 - logged) > TOP1_TOLERANCE:
        raise RuntimeError(f"top-1 {top1:.2f} does not match the logged {logged:.2f}: wrong weights or labels?")
    return top1


def plot_collapse(results: dict, out_dir: Path) -> None:
    """Spectra per loop (one panel per model) and effective rank / cosine vs loop (one line per model)."""
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4), squeeze=False, constrained_layout=True)
    for ax, (name, result) in zip(axes[0], results.items()):
        spectra = result["collapse"]["spectrum"]
        colours = plt.get_cmap("viridis")(torch.linspace(0, 1, len(spectra)).tolist())
        for layer, (spectrum, colour) in enumerate(zip(spectra, colours)):
            ax.semilogy(range(1, len(spectrum) + 1), [max(v, 1e-6) for v in spectrum], color=colour, lw=1.2,
                        label=f"loop {layer}" if layer in (0, len(spectra) - 1) else None)
        ax.set_title(f"{name}: token spectrum per loop (centred, / largest)", fontsize=9)
        ax.set_xlabel("singular value index")
        ax.set_ylabel("s_i / s_1")
        ax.set_ylim(1e-4, 1.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out_dir / "token_spectra.png", dpi=150)
    plt.close(fig)

    fig, (ax_rank, ax_cos) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, result in results.items():
        layers = range(len(result["collapse"]["effective_rank"]))
        ax_rank.plot(layers, result["collapse"]["effective_rank"], marker="o", ms=3, lw=2, label=name)
        ax_cos.plot(layers, result["collapse"]["cosine"], marker="o", ms=3, lw=2, label=name)
    ax_rank.set_ylabel("effective rank of the token matrix (mean per image)")
    ax_cos.set_ylabel("mean pairwise token cosine (1 = collapsed)")
    for ax in (ax_rank, ax_cos):
        ax.set_xlabel("loop / block (0 = input to the first block)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out_dir / "collapse_vs_loop.png", dpi=150)
    plt.close(fig)


def plot_exits(results: dict, out_path: Path) -> None:
    """Accuracy vs mean loops used, one panel per model: fixed depth and the three threshold rules."""
    styles = {"confidence": ("^", "tab:orange"), "kl": ("o", "tab:blue"), "cls_change": ("s", "tab:green")}
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.2), squeeze=False,
                             constrained_layout=True)
    for ax, (name, result) in zip(axes[0], results.items()):
        fixed = result["exits"]["acc_after_loop"]
        ax.plot(range(1, len(fixed) + 1), fixed, color="black", marker=".", lw=1.5, label="fixed depth")
        for curve, points in result["exits"]["curves"].items():
            marker, colour = styles[curve]
            ordered = sorted(points, key=lambda p: p["mean_loops"])
            ax.plot([p["mean_loops"] for p in ordered], [p["acc"] for p in ordered], marker=marker, color=colour,
                    ms=4, lw=1, label=f"{curve} threshold")
        top = max(fixed)
        ax.set_xlim(max(1, len(fixed) - 6) - 0.5, len(fixed) + 0.3)
        ax.set_ylim(max(0, top - 35), top + 3)
        ax.set_title(f"{name}: early exit (heads trained at the last loop only)", fontsize=9)
        ax.set_xlabel("loops used (mean per image)")
        ax.set_ylabel("CIFAR-100 test acc (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyse_checkpoint(checkpoint: str, args: argparse.Namespace, device: torch.device) -> dict:
    """Both diagnostics for one checkpoint."""
    model = load_model(checkpoint).to(device).eval()
    collapse_loader, exit_loader = test_loaders(model, args)
    record = collect_cls_per_layer(model, exit_loader, device, include_final_norm=False)
    top1 = check_top1(record, checkpoint, full_test_set=not args.limit)
    collapse = collect_token_stats(model, collapse_loader, device)
    exits = exit_analysis(model, record, device)
    name = Path(checkpoint).stem
    print(f"[{name}] effective rank by loop: " + " ".join(f"{v:.1f}" for v in collapse["effective_rank"]), flush=True)
    print(f"[{name}] token cosine by loop:   " + " ".join(f"{v:.3f}" for v in collapse["cosine"]), flush=True)
    print(f"[{name}] fixed exit: " + " ".join(f"{k}:{a:.1f}" for k, a in enumerate(exits["acc_after_loop"], 1)),
          flush=True)
    depth = len(exits["acc_after_loop"])
    for curve, verdict in exits["vs_fixed"].items():
        best, near = verdict["best_within_one_less"], verdict["cheapest_near_full"]
        best_text = f"{best['acc']} @ {best['mean_loops']} loops" if best else "no point"
        near_text = f"{near['mean_loops']} loops ({near['acc']})" if near else "never"
        print(f"[{name}] {curve}: best with <= {depth - 1} loops {best_text} vs fixed {depth - 1} "
              f"{verdict['fixed_one_less']} -> beats: {verdict['beats_fixed_one_less']}; within "
              f"{NEAR_FULL_POINTS:g} pt of full depth at {near_text} vs fixed {verdict['fixed_loops_near_full']}",
              flush=True)
    return {"checkpoint": checkpoint, "top1": round(top1, 2), "n_exit_images": len(record.labels),
            "n_collapse_images": len(collapse_loader.dataset), "collapse": collapse, "exits": exits}


def main() -> None:
    """Run both diagnostics on every checkpoint, write diagnostics.json and the plots."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {Path(c).stem: analyse_checkpoint(c, args, device) for c in args.checkpoints}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    plot_collapse(results, out_dir)
    plot_exits(results, out_dir / "exit_curves.png")
    print(f"wrote diagnostics.json, token_spectra.png, collapse_vs_loop.png, exit_curves.png in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
