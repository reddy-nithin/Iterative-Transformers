"""T19: where does each CLS update of the looped block come from? (no training)

For every loop t of B2 (T11-d layout: B0, B1, B2 x10, no tail) on state x = x_{t-1}:
  (a) d_t = x_t[:, 0] - x_{t-1}[:, 0] = a_t + m_t, a_t = attention sublayer row 0 (after proj), m_t = MLP row 0;
  (b) a_t = W·self + W·mean + W·deviation + bias, split per head over the CLS-row attention weights and values;
  (c) a collinear halting rule: stop at the first t >= 3 with cos(d_t, d_{t-1}) >= theta and
      ||d_t|| / ||x_{t-1}[:, 0]|| <= rho; images that never stop use loop 10.
All 10,000 test images (or the first --limit), eval mode, fp32. The model's forward pass is not changed: T12's body
recorder gives the states, the parts are recomputed from B2's own modules and checked against the real update.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T19_cls_update_decomposition/decompose.py CKPT [CKPT ...] [--limit N]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent / "T12_per_loop_cls_diagnostics"))
sys.path.insert(0, str(TASK_DIR.parent / "T13_per_loop_readout_loss"))

from exit_analysis import check_top1, compare_with_fixed, test_loader, with_gain  # noqa: E402
from loop_diagnostics import BodyRecorder, find_body, loop_states, readout_logits  # noqa: E402

from cls_svd import load_model  # noqa: E402  (T5, put on the path by exit_analysis)
from early_exit import exit_accuracies, rule_result  # noqa: E402
from src.zero_mean_value import ZeroMeanValueBlock, zero_mean_mix  # noqa: E402

SPLIT_TOLERANCE = 1e-4  # parts + bias must equal a_t; a_t + m_t must equal the real CLS update
EPS = 1e-8
THETAS = (0.80, 0.85, 0.90, 0.95)
RHOS = (0.10, 0.20, 0.30, math.inf)  # inf = no limit
FIRST_STOP_LOOP = 3
LATE_LOOPS = range(6, 11)  # decision line: loops 6 to 10
MEAN_SHARE_GATE = 0.5
ATTN_SHARE_GATE = 0.3
PER_LOOP_KEYS = ("attn_frac", "mlp_frac", "mean_share", "self_share", "rel_d", "norm_d", "norm_a", "norm_m",
                 "norm_self", "norm_mean", "norm_dev")
SUCCESSIVE_KEYS = ("cos_a", "cos_m", "cos_d", "cos_mean", "cos_dev")  # t = 2..10
T12_DIAG = {  # T12 (c) per-loop successive-update cos for the same checkpoints (T12's 1,000 test images)
    "T11-d_unroll_n3_30ep": TASK_DIR.parent / "T12_per_loop_cls_diagnostics/results/diag_T11-d_unroll_n3_30ep.json",
    "T13-c_unroll_readout_every_loop":
        TASK_DIR.parent / "T13_per_loop_readout_loss/results/t12_diag/diag_T13-c_unroll_readout_every_loop.json",
    "T13-d_unroll_loss_margin":
        TASK_DIR.parent / "T13_per_loop_readout_loss/results/t12_diag/diag_T13-d_unroll_loss_margin.json",
}
T12_COS_TOLERANCE = 0.02


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="T11-d-layout weights saved by src.train --save-checkpoint")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="use only the first N test images (smoke)")
    p.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    return p.parse_args()


# ---------- the split of one CLS update ----------

def check_plain_block(block: nn.Module) -> None:
    """The split assumes x' = x + attn(norm1(x)), x'' = x' + mlp(norm2(x')): no LayerScale, no drop-path, no
    attention gate or inner norm. Raise if the block has any of them."""
    for name in ("ls1", "ls2", "drop_path1", "drop_path2"):
        if not isinstance(getattr(block, name), nn.Identity):
            raise ValueError(f"block.{name} is {type(getattr(block, name)).__name__}, not Identity")
    if not isinstance(block.attn.norm, nn.Identity) or getattr(block.attn, "gate", None) is not None:
        raise ValueError("attention has an inner norm or gate; the linear split does not apply")


def split_cls_update(block: nn.Module, x: torch.Tensor, zero_mean: bool = False) -> dict[str, torch.Tensor]:
    """CLS update of one block call on state x (B, N, C), split into parts, each (B, C):
    a (attention row 0 after proj), m (MLP row 0), self / mean / dev (proj weight applied, no bias), and the
    proj bias (C,). a = self + mean + dev + bias, d = a + m.
    zero_mean=True (T20 block): CLS reads zero-mean patch values, so the mean part is 0 and a = self + dev + bias."""
    attn = block.attn
    batch, n_tokens, _ = x.shape
    qkv = attn.qkv(block.norm1(x)).reshape(batch, n_tokens, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # (B, heads, N, head_dim)
    q, k = attn.q_norm(q), attn.k_norm(k)
    weights = ((q[:, :, :1] * attn.scale) @ k.transpose(-2, -1)).softmax(dim=-1)[:, :, 0]  # (B, heads, N)
    patch_weights, patch_values = weights[:, :, 1:], v[:, :, 1:]
    n_patches = patch_values.shape[2]
    supply = patch_weights.sum(dim=-1, keepdim=True)  # s_h, (B, heads, 1)
    self_part = weights[:, :, :1] * v[:, :, 0]
    mean_part = supply * patch_values.mean(dim=2)
    deviation = ((patch_weights - supply / n_patches).unsqueeze(-1) * patch_values).sum(dim=2)

    def project(part: torch.Tensor) -> torch.Tensor:  # concatenate heads, proj weight only
        return F.linear(part.reshape(batch, -1), attn.proj.weight)

    parts = {"self": project(self_part), "mean": project(mean_part), "dev": project(deviation)}
    if zero_mean:
        mixed = zero_mean_mix(weights.unsqueeze(2), v)[:, :, 0].reshape(batch, -1)  # the T20 training code's rule
        parts["mean"] = torch.zeros_like(parts["mean"])
    else:
        mixed = (weights.unsqueeze(-1) * v).sum(dim=2).reshape(batch, -1)
    a = attn.proj(mixed)
    m = block.mlp(block.norm2(x[:, 0] + a))
    return {**parts, "a": a, "m": m, "d": a + m, "bias": attn.proj.bias}


def inner_block(body: nn.Module) -> nn.Module:
    """The timm Block of the looped body (unwraps T20's `ZeroMeanValueBlock`)."""
    return body.block if isinstance(body, ZeroMeanValueBlock) else body


def split_body_update(body: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """`split_cls_update` for the looped body, with the zero-mean rule when the body is T20's block."""
    return split_cls_update(inner_block(body), x, zero_mean=isinstance(body, ZeroMeanValueBlock))


def split_error(parts: dict[str, torch.Tensor]) -> float:
    """max |self + mean + dev + bias - a|."""
    return (parts["self"] + parts["mean"] + parts["dev"] + parts["bias"] - parts["a"]).abs().max().item()


# ---------- per-loop metrics ----------

def share(part: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
    """||part|| / (||part|| + sum ||others||), per image."""
    total = part.norm(dim=1) + sum(o.norm(dim=1) for o in others)
    return part.norm(dim=1) / total.clamp_min(EPS)


def loop_metrics(parts: dict[str, torch.Tensor], cls_prev: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-image values (B,) for one loop. self_share = ||W·self|| / (||W·self|| + ||W·mean|| + ||W·dev||)."""
    d_norm = parts["d"].norm(dim=1).clamp_min(EPS)
    return {
        "attn_frac": parts["a"].norm(dim=1) / d_norm,
        "mlp_frac": parts["m"].norm(dim=1) / d_norm,
        "mean_share": share(parts["mean"], parts["dev"]),
        "self_share": share(parts["self"], parts["mean"], parts["dev"]),
        "rel_d": parts["d"].norm(dim=1) / (cls_prev.norm(dim=1) + EPS),
        "norm_d": parts["d"].norm(dim=1), "norm_a": parts["a"].norm(dim=1), "norm_m": parts["m"].norm(dim=1),
        "norm_self": parts["self"].norm(dim=1), "norm_mean": parts["mean"].norm(dim=1),
        "norm_dev": parts["dev"].norm(dim=1),
    }


def successive_cos(parts: dict[str, torch.Tensor], before: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """cos between this loop's and the previous loop's a, m, d, W·mean and W·dev, per image."""
    return {f"cos_{key}": F.cosine_similarity(parts[key], before[key], dim=1, eps=EPS)
            for key in ("a", "m", "d", "mean", "dev")}


def batch_values(body: nn.Module, states: list[torch.Tensor]) -> tuple[dict[str, list[torch.Tensor]], float, float]:
    """All per-loop values for one batch, plus the worst split error and the worst |a + m - real update|."""
    values: dict[str, list[torch.Tensor]] = {key: [] for key in (*PER_LOOP_KEYS, *SUCCESSIVE_KEYS)}
    worst_split = worst_update = 0.0
    before = None
    for t in range(1, len(states)):
        previous, current = states[t - 1], states[t]
        parts = split_body_update(body, previous)
        worst_split = max(worst_split, split_error(parts))
        worst_update = max(worst_update, (parts["d"] - (current[:, 0] - previous[:, 0])).abs().max().item())
        for key, value in loop_metrics(parts, previous[:, 0]).items():
            values[key].append(value)
        if before is not None:
            for key, value in successive_cos(parts, before).items():
                values[key].append(value)
        before = parts
    return values, worst_split, worst_update


# ---------- halting rule ----------

def collinear_exit(cos_d: torch.Tensor, rel_d: torch.Tensor, theta: float, rho: float) -> torch.Tensor:
    """Exit index (0-based loop) per image: first loop t >= 3 with cos(d_t, d_t-1) >= theta and rel_d_t <= rho,
    else the last loop. cos_d is (L-1, N) for t = 2..L, rel_d is (L, N) for t = 1..L."""
    n_loops = rel_d.shape[0]
    first = FIRST_STOP_LOOP - 1  # 0-based index of loop 3
    stop_ok = (cos_d[first - 1:] >= theta) & (rel_d[first:] <= rho)  # rows = loops 3..L
    never = torch.full((stop_ok.shape[1],), n_loops - 1, dtype=torch.long)
    first_hit = stop_ok.float().argmax(dim=0) + first
    return torch.where(stop_ok.any(dim=0), first_hit, never)


def halting_curve(logits: torch.Tensor, labels: torch.Tensor, cos_d: torch.Tensor, rel_d: torch.Tensor,
                  fixed: list[float]) -> list[dict]:
    """(theta, rho, top-1, mean loops, gain vs fixed) for the full theta x rho grid."""
    points = []
    for theta in THETAS:
        for rho in RHOS:
            result = rule_result(logits, labels, collinear_exit(cos_d, rel_d, theta, rho))
            points.append(with_gain({"theta": theta, "rho": None if math.isinf(rho) else rho, **result}, fixed))
    return points


# ---------- collection, summary, decision ----------

@torch.no_grad()
def collect(model: nn.Module, loader, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor,
                                                                    torch.Tensor, float]:
    """({key: (loops, N)} per-image values, per-loop logits (L, N, classes), labels, model's own top-1)."""
    body, tail = find_body(model)
    if tail:
        raise ValueError("T19 is for the T11-d layout (no tail)")
    check_plain_block(inner_block(body))
    batches: dict[str, list[torch.Tensor]] = {}
    logits, labels = [], []
    worst_split = worst_update = 0.0
    correct = 0
    with BodyRecorder(body) as recorder:
        for images, targets in loader:
            recorder.calls.clear()
            images = images.to(device, non_blocking=True)
            correct += (model(images).argmax(dim=1).cpu() == targets).sum().item()
            states = loop_states(recorder.calls)
            values, split_worst, update_worst = batch_values(body, states)
            worst_split, worst_update = max(worst_split, split_worst), max(worst_update, update_worst)
            if max(worst_split, worst_update) >= SPLIT_TOLERANCE:
                raise RuntimeError(f"split error {worst_split:.2e}, update error {worst_update:.2e}")
            for key, per_loop in values.items():
                batches.setdefault(key, []).append(torch.stack(per_loop).cpu())
            logits.append(torch.stack([readout_logits(model, s, [], "norm") for s in states[1:]]).float().cpu())
            labels.append(targets)
    print(f"split check: parts + bias vs a_t {worst_split:.2e}, a_t + m_t vs real update {worst_update:.2e} "
          f"(< {SPLIT_TOLERANCE:g})", flush=True)
    labels_all = torch.cat(labels)
    top1 = 100.0 * correct / len(labels_all)
    return {k: torch.cat(v, dim=1) for k, v in batches.items()}, torch.cat(logits, dim=1), labels_all, top1


def summarise(per_image: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    """Per-loop means over images."""
    return {key: [round(v, 4) for v in values.mean(dim=1).tolist()] for key, values in per_image.items()}


def late_mean(per_loop: list[float], first_loop: int) -> float:
    """Mean over loops 6..10 of a per-loop list whose first entry is loop `first_loop`."""
    return round(sum(per_loop[t - first_loop] for t in LATE_LOOPS) / len(LATE_LOOPS), 4)


def decision(means: dict[str, list[float]]) -> dict:
    """The spec's decision line from one model's per-loop means."""
    mean_share, attn_frac = late_mean(means["mean_share"], 1), late_mean(means["attn_frac"], 1)
    run = mean_share >= MEAN_SHARE_GATE and attn_frac >= ATTN_SHARE_GATE
    return {"late_mean_share": mean_share, "late_attn_frac": attn_frac,
            "line": "run T20" if run else f"skip T20 (mean share {mean_share:.3f}, ||a||/||d|| {attn_frac:.3f})"}


def compare_with_t12(name: str, cos_d: list[float]) -> dict | None:
    """Our cos(d_t, d_t-1) vs T12 (c) for the same checkpoint (loops 2..10)."""
    path = T12_DIAG.get(name)
    if path is None or not path.exists():
        return None
    t12 = json.loads(path.read_text())["means"]["cls_update_cos"]
    worst = max(abs(a - b) for a, b in zip(cos_d, t12))
    status = "ok" if worst <= T12_COS_TOLERANCE else "MISMATCH"
    print(f"[{name}] cos(d_t, d_t-1) vs T12 (c): max |diff| {worst:.4f} ({status}, tol {T12_COS_TOLERANCE})",
          flush=True)
    return {"t12_cls_update_cos": t12, "max_abs_diff": round(worst, 4), "within_tolerance": status == "ok"}


def analyse_checkpoint(checkpoint: str, args: argparse.Namespace, device: torch.device) -> dict:
    """Everything T19 reports for one checkpoint."""
    name = Path(checkpoint).stem
    model = load_model(checkpoint).to(device).float().eval()
    per_image, logits, labels, top1 = collect(model, test_loader(model, args), device)
    check_top1(top1, checkpoint, full_test_set=not args.limit)
    means = summarise(per_image)
    fixed = [round(acc, 2) for acc in exit_accuracies(logits, labels)]
    points = halting_curve(logits, labels, per_image["cos_d"], per_image["rel_d"], fixed)
    body, _ = find_body(model)
    return {"checkpoint": checkpoint, "n_images": len(labels), "own_top1": round(top1, 2), "means": means,
            "bias_norm": round(inner_block(body).attn.proj.bias.norm().item(), 4),
            "acc_per_loop": fixed, "halting": points, "halting_vs_fixed": compare_with_fixed(points, fixed),
            "t12_check": compare_with_t12(name, means["cos_d"]), "decision": decision(means)}


def report(name: str, result: dict) -> None:
    """One-screen summary."""
    means = result["means"]
    for key in ("attn_frac", "mlp_frac", "mean_share", "self_share", "rel_d"):
        print(f"[{name}] {key}: " + " ".join(f"{v:.3f}" for v in means[key]), flush=True)
    for key in SUCCESSIVE_KEYS:
        print(f"[{name}] {key} (t=2..10): " + " ".join(f"{v:.3f}" for v in means[key]), flush=True)
    best = result["halting_vs_fixed"]["best_gain"]
    print(f"[{name}] halting best gain vs fixed {best['gain_vs_fixed']:+.2f} at {best['mean_loops']} loops "
          f"(theta {best['theta']}, rho {best['rho']}); decision: {result['decision']['line']}", flush=True)


def main() -> None:
    """Analyse every checkpoint; results merge into results/decomposition.json so models can be added later."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "decomposition.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    for checkpoint in args.checkpoints:
        name = Path(checkpoint).stem
        results[name] = analyse_checkpoint(checkpoint, args, device)
        report(name, results[name])
        out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
