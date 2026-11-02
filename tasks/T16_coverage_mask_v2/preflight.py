"""T16 pre-flight: does the v2 coverage mask bite, zero-shot, on the trained T13-d model? (no training)

The T13-d checkpoint (B0, B1, B2 x10, no tail) is run on the full CIFAR-100 test set, per batch:
  - unmasked: the model as trained;
  - masked: the v2 mask on B2's CLS query row with fixed eta (c = 1, bias log(c + 1e-6), drain
    eta * max(0, 196 * A_t - 1) after every loop).
Per eta it reports: mean supply after loop 10, share of patches with c < 0.1 after loop 10, late-loop (loops 6 to 10)
CLS attention cos, CLS attention entropy per loop (masked vs unmasked), the output change
||o_mask - o_nomask|| / ||o_nomask|| averaged over loops 6 to 10 (o = CLS row of the attention output after proj; both
from the masked run's input state at that loop, with the current supply vs no mask), test top-1, and the spec's gate.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T16_coverage_mask_v2/preflight.py [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent / "T12_per_loop_cls_diagnostics"))
sys.path.insert(0, str(TASK_DIR.parent / "T13_per_loop_readout_loss"))

import loop_diagnostics as t12  # noqa: E402
from exit_analysis import test_loader  # noqa: E402
from src.coverage_mask import CoverageMaskLoop, masked_block_parts, update_supply_v2  # noqa: E402
from src.per_loop_readout import embed, split_loop  # noqa: E402

CHECKPOINT = "/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/T13-d_unroll_loss_margin.pt"
ETAS = (0.5, 1.0)
LATE_LOOPS = range(6, 11)  # loops 6 to 10
EMPTY_SUPPLY = 0.1
GATE_SUPPLY, GATE_COS, GATE_OUTPUT_CHANGE = 0.8, 0.9, 0.10
COLLAPSE_TOP1 = 5.0
FORWARD_TOLERANCE = 1e-4  # this script's masked loop vs the real CoverageMaskLoop forward (first batch)


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="use only the first N test images (smoke)")
    p.add_argument("--out", default=str(TASK_DIR / "results" / "preflight.json"))
    return p.parse_args()


def loop_values(body: nn.Module, x: torch.Tensor, x_plain: torch.Tensor, supply: torch.Tensor,
                eta: float) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """One loop of the masked and the unmasked run: per-image values, next masked state, next unmasked state, next
    supply. values["attention"] is the masked run's per-head CLS->patch attention (B, heads, P)."""
    masked = masked_block_parts(body, x, supply)
    nomask_same_input = t12.recompute_block(body, x)
    plain = t12.recompute_block(body, x_plain)
    o_mask, o_nomask = masked["attn_out"][:, 0], nomask_same_input["attn_out"][:, 0]
    attention = t12.cls_patch_attention(masked["probs"])
    values = {
        "output_change": (o_mask - o_nomask).norm(dim=1) / o_nomask.norm(dim=1).clamp_min(1e-12),
        "entropy_masked": t12.entropy(attention).mean(dim=1),
        "entropy_unmasked": t12.entropy(t12.cls_patch_attention(plain["probs"])).mean(dim=1),
        "attention": attention,
    }
    next_supply = update_supply_v2(supply, masked["cls_attention"], torch.tensor(eta, device=x.device))
    return values, masked["output"], plain["output"], next_supply


def check_against_real_forward(model: nn.Module, head: list[nn.Module], body: nn.Module, n_loops: int, eta: float,
                               images: torch.Tensor, logits: torch.Tensor) -> None:
    """This script's masked loop must match the training code's CoverageMaskLoop (rule v2) on the same weights."""
    plain_blocks = model.blocks
    model.blocks = CoverageMaskLoop(head, body, n_loops, eta_init=eta, rule="v2").to(images.device)
    try:
        real = model(images)
    finally:
        model.blocks = plain_blocks
    worst = (real - logits).abs().max().item()
    if worst >= FORWARD_TOLERANCE:
        raise RuntimeError(f"masked loop differs from CoverageMaskLoop by {worst:.2e}")
    print(f"[eta {eta}] check: this loop vs CoverageMaskLoop logits max diff {worst:.2e}", flush=True)


def late_mean(per_loop: list[float]) -> float:
    """Mean over loops 6..10 of a per-loop list (entry 0 = loop 1)."""
    return round(sum(per_loop[t - 1] for t in LATE_LOOPS) / len(LATE_LOOPS), 4)


@torch.no_grad()
def run_eta(model: nn.Module, loader, device: torch.device, eta: float) -> dict:
    """All pre-flight numbers for one eta."""
    head, body, tail, n_loops = split_loop(model)
    if tail:
        raise ValueError("expected the T11-d layout (no tail)")
    sums: dict[str, torch.Tensor] = {}
    cos_sum = torch.zeros(n_loops - 1)
    supply_sum = empty = correct = correct_plain = n_images = 0
    for batch_index, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device)
        x = embed(model, images)
        for block in head:
            x = block(x)
        x_plain = x
        supply = torch.ones(x.shape[0], x.shape[1] - 1, device=device)
        attentions = []
        for t in range(n_loops):
            values, x, x_plain, supply = loop_values(body, x, x_plain, supply, eta)
            attentions.append(values.pop("attention"))
            for key, value in values.items():
                sums.setdefault(key, torch.zeros(n_loops))[t] += value.sum().item()
        for t, (current, following) in enumerate(zip(attentions, attentions[1:])):
            cos_sum[t] += t12.attention_cosines(current, following)[0].sum().item()  # per head, head-averaged
        logits = model.forward_head(model.norm(x))
        if batch_index == 0:
            check_against_real_forward(model, head, body, n_loops, eta, images, logits)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        correct_plain += (model.forward_head(model.norm(x_plain)).argmax(dim=1) == labels).sum().item()
        supply_sum += supply.mean(dim=1).sum().item()
        empty += (supply < EMPTY_SUPPLY).float().mean(dim=1).sum().item()
        n_images += len(labels)
    means = {key: [round(v / n_images, 4) for v in total.tolist()] for key, total in sums.items()}
    attn_cos = [round(v / n_images, 4) for v in cos_sum.tolist()]  # entry t-1 = cos(A_t, A_t+1), t = 1..9
    late_pairs = range(LATE_LOOPS.start, LATE_LOOPS.stop - 1)  # pairs (6,7) .. (9,10): both loops in 6..10
    return {"eta": eta, "n_images": n_images, "top1_masked": round(100.0 * correct / n_images, 2),
            "top1_unmasked": round(100.0 * correct_plain / n_images, 2),
            "supply_mean_after_loop10": round(supply_sum / n_images, 4),
            "share_supply_below_0.1_after_loop10": round(empty / n_images, 4),
            "attn_cos_per_pair": attn_cos,
            "late_attn_cos": round(sum(attn_cos[t - 1] for t in late_pairs) / len(late_pairs), 4),
            "entropy_masked_per_loop": means["entropy_masked"], "entropy_unmasked_per_loop": means["entropy_unmasked"],
            "output_change_per_loop": means["output_change"], "late_output_change": late_mean(means["output_change"])}


def gate(results: list[dict]) -> dict:
    """The spec's gate: bites if (supply < 0.8 OR late cos < 0.9) AND output change >= 0.10, for at least one eta."""
    per_eta = {}
    for r in results:
        drained = r["supply_mean_after_loop10"] < GATE_SUPPLY or r["late_attn_cos"] < GATE_COS
        per_eta[str(r["eta"])] = drained and r["late_output_change"] >= GATE_OUTPUT_CHANGE
    collapsed = all(r["top1_masked"] < COLLAPSE_TOP1 for r in results)
    return {"bites_per_eta": per_eta, "passes": any(per_eta.values()), "top1_collapses_at_both_eta": collapsed}


def main() -> None:
    """Pre-flight at both etas; writes results/preflight.json."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = t12.load_model(args.checkpoint).to(device).float().eval()
    loader = test_loader(model, args)
    results = []
    for eta in ETAS:
        result = run_eta(model, loader, device, eta)
        results.append(result)
        print(json.dumps(result), flush=True)
    decision = gate(results)
    print(f"gate: {decision}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "results": results, "gate": decision}, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
