"""T12: per-loop CLS diagnostics on trained checkpoints (no training).

For every loop t of the looped body (x_t = state after loop t, x_0 = the body's first input):
  (a) relative update size of CLS vs patches                         [H3]
  (b) CLS attention over the 196 patches: cos(A_t, A_t+1) and entropy  [H1]
  (c) new-direction share of the CLS attention update, cos of successive CLS updates
  (d) exit curve: top-1 when reading out at loop t (through the tail if the model has one)
(a)-(c) use the same fixed 1,000 test images as T8, (d) all 10,000. Eval mode, fp32, no augmentation.

The model's forward pass is not changed. A forward hook on the body records its input and output at every call;
attention weights and the attention sublayer output are recomputed from the block's own modules, and the recomputed
block output is asserted equal to the hooked one.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T12_per_loop_cls_diagnostics/loop_diagnostics.py CKPT [CKPT ...] [--limit N]
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent / "T5_cls_svd_per_block"))  # T5 checkpoint loader
sys.path.insert(0, str(TASK_DIR.parent / "T8_collapse_and_kl_exit"))  # T8's fixed 1,000-image selection

from cls_svd import load_model  # noqa: E402
from diagnostics import test_loaders  # noqa: E402  (T8's module)

RECOMPUTE_TOLERANCE = 1e-4  # max abs difference between recomputed and real block output (fp32)
TOP1_TOLERANCE = 0.5  # points; full-depth top-1 must match the checkpoint's logged test acc
LATE_LOOP_START = 4  # decision rule: "late loops" = loops >= 4
COVERAGE_COS = 0.8  # decision rule thresholds
ADAPTER_RATIO = 0.3
MAX_ENTROPY = math.log(196)
MODEL_COLOURS = {  # checkpoint stem -> colour, fixed so every plot (and later reruns) use the same one
    "T5-b_iter_b0": "tab:green",
    "T11-a_hbt_body_b0": "tab:blue",
    "T11-b_hbt_body_b6": "tab:orange",
    "T11-c_hbt_body_b1": "tab:purple",
    "T11-d_unroll_n3_30ep": "tab:red",
}
PER_LOOP_KEYS = ("rel_cls", "rel_patch", "attn_entropy", "attn_entropy_headavg", "new_direction_share")
PAIR_KEYS = ("attn_cos", "attn_cos_headavg")  # defined for t = 1 .. L-1 (loop t vs t+1)
SUCCESSIVE_KEYS = ("cls_update_cos",)  # defined for t = 2 .. L (update t vs t-1)


def parse_args() -> argparse.Namespace:
    """Command-line options (the data options are the ones T8's `test_loaders` reads)."""
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="weights saved by src.train --save-checkpoint")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--collapse-images", type=int, default=1000, help="images for (a)-(c), same selection as T8")
    p.add_argument("--limit", type=int, default=0, help="use only the first N test images for (d) (smoke)")
    p.add_argument("--seed", type=int, default=0, help="picks the (a)-(c) images (T8 used 0)")
    p.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    return p.parse_args()


# ---------- model structure ----------

def find_body(model: nn.Module) -> tuple[nn.Module, list[nn.Module]]:
    """(the looped body = the block object that appears most often in `model.blocks`, the blocks after its last
    call = the tail). Raises ValueError if no block repeats."""
    blocks = list(model.blocks)
    body_id, repeats = Counter(id(block) for block in blocks).most_common(1)[0]
    if repeats < 2:
        raise ValueError("no block is repeated: this is not a looped model")
    last = max(i for i, block in enumerate(blocks) if id(block) == body_id)
    return blocks[last], blocks[last + 1:]


def readout_kind(model_name: str, tail: list[nn.Module]) -> str | None:
    """How (d) reads out at loop t: "tail" (hbt: classifier(norm(tail(x_t)))), "norm" (iterative and unrolled models,
    no tail: classifier(norm(x_t)); for unrolled models decided 2026-10-30), or None (no readout defined)."""
    if model_name == "hbt" and tail:
        return "tail"
    if model_name in ("iterative", "unroll") and not tail:
        return "norm"
    return None


def recompute_block(block: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """One timm `Block` (timm 1.0.30, eval mode) recomputed from its own modules, keeping what the metrics need:
    output (B, N, C), attention probabilities (B, heads, N, N) and attention sublayer output after proj (B, N, C)."""
    attn = block.attn
    if getattr(attn, "gate", None) is not None:
        raise ValueError("gated attention is not supported by this recomputation")
    batch, n_tokens, _ = x.shape
    qkv = attn.qkv(block.norm1(x)).reshape(batch, n_tokens, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    probs = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    mixed = (probs @ v).transpose(1, 2).reshape(batch, n_tokens, -1)
    sublayer = attn.proj(attn.norm(mixed))
    after_attn = x + block.ls1(sublayer)
    output = after_attn + block.ls2(block.mlp(block.norm2(after_attn)))
    return {"output": output, "probs": probs, "attn_out": sublayer}


# ---------- metrics (each returns one value per image) ----------

def relative_update(previous: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(rel_cls, rel_patch): ||Δ CLS|| / ||CLS_prev|| and the mean over patches of ||Δ patch_i|| / ||patch_i_prev||."""
    ratio = (current - previous).norm(dim=2) / previous.norm(dim=2).clamp_min(1e-12)
    return ratio[:, 0], ratio[:, 1:].mean(dim=1)


def cls_patch_attention(probs: torch.Tensor) -> torch.Tensor:
    """(B, heads, P): CLS query's attention over the patch keys, CLS->CLS dropped, renormalised to sum 1."""
    patches = probs[:, :, 0, 1:]
    return patches / patches.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def entropy(distribution: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) over the last dim; 0·log 0 = 0."""
    logs = torch.where(distribution > 0, distribution.log(), torch.zeros_like(distribution))
    return -(distribution * logs).sum(dim=-1)


def attention_cosines(current: torch.Tensor, following: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos per head averaged over heads, cos of the head-averaged distributions), each (B,)."""
    per_head = F.cosine_similarity(current, following, dim=-1).mean(dim=1)
    head_avg = F.cosine_similarity(current.mean(dim=1), following.mean(dim=1), dim=-1)
    return per_head, head_avg


def new_direction_share(attn_out_cls: torch.Tensor, cls_prev: torch.Tensor) -> torch.Tensor:
    """||o_perp|| / ||o||, with o_perp = o minus its projection on the previous CLS."""
    projection = (attn_out_cls * cls_prev).sum(dim=1, keepdim=True) / (cls_prev * cls_prev).sum(dim=1, keepdim=True)
    perpendicular = attn_out_cls - projection * cls_prev
    return perpendicular.norm(dim=1) / attn_out_cls.norm(dim=1).clamp_min(1e-12)


# ---------- collection ----------

class BodyRecorder:
    """Context manager: forward hook on the body that records (input, output) of every call, in call order."""

    def __init__(self, body: nn.Module) -> None:
        self.body, self.calls, self.handle = body, [], None

    def __enter__(self) -> "BodyRecorder":
        self.handle = self.body.register_forward_hook(
            lambda module, args, output: self.calls.append((args[0], output)))
        return self

    def __exit__(self, *exc) -> None:
        self.handle.remove()


def loop_states(calls: list[tuple[torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
    """[x_0, x_1, ..., x_L] from the body's recorded calls; checks each call's input is the previous output."""
    if not calls:
        raise RuntimeError("the body was never called")
    states = [calls[0][0]]
    for index, (inputs, outputs) in enumerate(calls):
        if index and not torch.equal(inputs, states[-1]):
            raise RuntimeError(f"body call {index + 1} did not get the previous call's output as input")
        states.append(outputs)
    return states


def batch_metrics(body: nn.Module, states: list[torch.Tensor]) -> tuple[dict[str, list[torch.Tensor]], float]:
    """All (a)-(c) values for one batch, as {key: [per-loop tensor (B,)]}, plus the worst recompute difference."""
    values: dict[str, list[torch.Tensor]] = {key: [] for key in (*PER_LOOP_KEYS, *PAIR_KEYS, *SUCCESSIVE_KEYS)}
    attentions, worst = [], 0.0
    for t in range(1, len(states)):
        previous, current = states[t - 1], states[t]
        parts = recompute_block(body, previous)
        worst = max(worst, (parts["output"] - current).abs().max().item())
        rel_cls, rel_patch = relative_update(previous, current)
        attention = cls_patch_attention(parts["probs"])
        attentions.append(attention)
        values["rel_cls"].append(rel_cls)
        values["rel_patch"].append(rel_patch)
        values["attn_entropy"].append(entropy(attention).mean(dim=1))
        values["attn_entropy_headavg"].append(entropy(attention.mean(dim=1)))
        values["new_direction_share"].append(new_direction_share(parts["attn_out"][:, 0], previous[:, 0]))
        if t >= 2:
            delta, delta_before = current[:, 0] - previous[:, 0], previous[:, 0] - states[t - 2][:, 0]
            values["cls_update_cos"].append(F.cosine_similarity(delta, delta_before, dim=1))
    for current, following in zip(attentions, attentions[1:]):
        per_head, head_avg = attention_cosines(current, following)
        values["attn_cos"].append(per_head)
        values["attn_cos_headavg"].append(head_avg)
    return values, worst


@torch.no_grad()
def collect_loop_metrics(model: nn.Module, loader, device: torch.device) -> dict[str, torch.Tensor]:
    """{key: (n_loops_for_key, N) per-image values} over the loader; asserts the recompute matches every loop."""
    body, _ = find_body(model)
    batches: dict[str, list[torch.Tensor]] = {}
    worst = 0.0
    with BodyRecorder(body) as recorder:
        for images, _ in loader:
            recorder.calls.clear()
            model.forward_features(images.to(device, non_blocking=True))
            values, batch_worst = batch_metrics(body, loop_states(recorder.calls))
            worst = max(worst, batch_worst)
            if worst >= RECOMPUTE_TOLERANCE:
                raise RuntimeError(f"recomputed block output differs from the real one by {worst:.2e}")
            for key, per_loop in values.items():
                batches.setdefault(key, []).append(torch.stack(per_loop).cpu())
    print(f"recompute check: max |recomputed - real| block output = {worst:.2e} (< {RECOMPUTE_TOLERANCE:g})",
          flush=True)
    return {key: torch.cat(parts, dim=1) for key, parts in batches.items()}


def readout_logits(model: nn.Module, state: torch.Tensor, tail: list[nn.Module], kind: str) -> torch.Tensor:
    """Logits when stopping at this state: through the tail ("tail") or straight to the final norm ("norm")."""
    if kind == "tail":
        for block in tail:
            state = block(state)
    return model.forward_head(model.norm(state))


@torch.no_grad()
def collect_exit_accuracy(model: nn.Module, loader, device: torch.device, kind: str) -> dict:
    """Top-1 (%) at each loop t = 1..L with the readout `kind`, plus the model's own full-depth top-1."""
    body, tail = find_body(model)
    correct: torch.Tensor | None = None
    full_correct = total = 0
    with BodyRecorder(body) as recorder:
        for images, labels in loader:
            recorder.calls.clear()
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            full_correct += (model(images).argmax(dim=1) == labels).sum().item()
            states = loop_states(recorder.calls)[1:]
            hits = torch.stack([(readout_logits(model, s, tail, kind).argmax(dim=1) == labels).sum() for s in states])
            correct = hits if correct is None else correct + hits
            total += len(labels)
    if correct is None:
        raise ValueError("empty loader")
    return {"acc_per_loop": [round(100.0 * c / total, 2) for c in correct.tolist()],
            "full_depth_top1": round(100.0 * full_correct / total, 2), "n_images": total}


# ---------- summary, decision, outputs ----------

def summarise(per_image: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    """Per-loop means over images, plus the per-loop ratio mean(rel_cls) / mean(rel_patch)."""
    means = {key: [round(v, 4) for v in values.mean(dim=1).tolist()] for key, values in per_image.items()}
    means["ratio"] = [round(c / p, 4) for c, p in zip(means["rel_cls"], means["rel_patch"])]
    return means


def decision(means: dict[str, list[float]]) -> dict:
    """The T12 decision rule on one model's per-loop means. attn_cos[t-1] = cos(A_t, A_t+1)."""
    late = means["attn_cos"][LATE_LOOP_START - 1:]
    late_headavg = means["attn_cos_headavg"][LATE_LOOP_START - 1:]
    late_cos = sum(late) / len(late)
    mean_ratio = sum(means["ratio"]) / len(means["ratio"])
    coverage, adapters = late_cos > COVERAGE_COS, mean_ratio < ADAPTER_RATIO
    if coverage and adapters:
        pick = "coverage mask first, then CLS adapters"
    elif coverage:
        pick = "coverage mask"
    elif adapters:
        pick = "CLS adapters"
    else:
        pick = "per-loop LayerNorm"
    return {"late_loop_mean_attn_cos": round(late_cos, 4),
            "late_loop_mean_attn_cos_headavg": round(sum(late_headavg) / len(late_headavg), 4),
            "mean_ratio": round(mean_ratio, 4), "pick": pick}


def markdown_table(means: dict[str, list[float]], exits: dict | None) -> str:
    """Per-loop results table (README format). Pair metrics are listed at loop t for (t, t+1)."""
    n_loops = len(means["rel_cls"])
    rows = ["| loop t | rel_cls | rel_patch | ratio | cos(A_t, A_t+1) | H(A_t) | new-direction share | "
            "cos(ΔCLS_t, ΔCLS_t-1) | readout acc |", "|" + "---|" * 9]
    for t in range(1, n_loops + 1):
        cos = f"{means['attn_cos'][t - 1]:.3f}" if t < n_loops else "–"
        update_cos = f"{means['cls_update_cos'][t - 2]:.3f}" if t >= 2 else "–"
        acc = f"{exits['acc_per_loop'][t - 1]:.2f}" if exits else "–"
        rows.append(f"| {t} | {means['rel_cls'][t - 1]:.4f} | {means['rel_patch'][t - 1]:.4f} | "
                    f"{means['ratio'][t - 1]:.3f} | {cos} | {means['attn_entropy'][t - 1]:.3f} | "
                    f"{means['new_direction_share'][t - 1]:.3f} | {update_cos} | {acc} |")
    return "\n".join(rows)


def load_all_results(out_dir: Path) -> dict[str, dict]:
    """Every model's saved summary (diag_<stem>.json) in the results folder, so plots include earlier runs."""
    return {path.stem.removeprefix("diag_"): json.loads(path.read_text())
            for path in sorted(out_dir.glob("diag_*.json"))}


def colour(name: str) -> str:
    """Fixed colour per checkpoint (grey for anything not in MODEL_COLOURS)."""
    return MODEL_COLOURS.get(name, "0.4")


def plot_lines(ax, results: dict[str, dict], key: str, first_loop: int = 1, suffix: str = "",
               linestyle: str = "-", linewidth: float = 2.0) -> None:
    """One line per model of `means[key]` vs loop, starting at loop `first_loop`."""
    for name, result in results.items():
        values = result["means"][key]
        ax.plot(range(first_loop, first_loop + len(values)), values, color=colour(name), marker="o", ms=3,
                ls=linestyle, lw=linewidth, label=f"{name}{suffix}")


def finish(ax, xlabel: str, ylabel: str, title: str, reference: float | None = None) -> None:
    """Axis labels, optional dotted reference line, grid and legend."""
    if reference is not None:
        ax.axhline(reference, color="0.5", ls=":", lw=1)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def plot_all(results: dict[str, dict], out_dir: Path) -> None:
    """The four T12 plots, one line per model, same colour per model in every plot."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    plot_lines(ax, results, "rel_cls", suffix=" CLS")
    plot_lines(ax, results, "rel_patch", suffix=" patches", linestyle="--", linewidth=1.5)
    finish(ax, "loop t", "||x_t - x_t-1|| / ||x_t-1|| (mean)", "(a) relative update: CLS (solid) vs patches (dashed)")
    fig.savefig(out_dir / "rel_update.png", dpi=150)
    plt.close(fig)

    fig, (ax_cos, ax_ent) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    plot_lines(ax_cos, results, "attn_cos")
    finish(ax_cos, "loop t", "cos(A_t, A_t+1), per head then mean", "(b) does CLS read the same patches next loop?",
           COVERAGE_COS)
    plot_lines(ax_ent, results, "attn_entropy")
    finish(ax_ent, "loop t", "entropy of A_t (nats, max 5.28)", "(b) how spread is CLS attention?", MAX_ENTROPY)
    fig.savefig(out_dir / "attn_cos_entropy.png", dpi=150)
    plt.close(fig)

    fig, (ax_new, ax_succ) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    plot_lines(ax_new, results, "new_direction_share")
    finish(ax_new, "loop t", "||o_perp|| / ||o||", "(c) share of the CLS attention update off the CLS direction")
    plot_lines(ax_succ, results, "cls_update_cos", first_loop=2)
    finish(ax_succ, "loop t", "cos(ΔCLS_t, ΔCLS_t-1)", "(c) are successive CLS updates orthogonal?", 0.0)
    fig.savefig(out_dir / "new_direction.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for name, result in results.items():
        if result.get("exits"):
            acc = result["exits"]["acc_per_loop"]
            readout = "tail" if result["readout"] == "tail" else "final norm + head"
            ax.plot(range(1, len(acc) + 1), acc, color=colour(name), lw=2, marker="o", ms=3,
                    label=f"{name} ({readout} readout)")
    finish(ax, "loop t (stop here, read out)", "CIFAR-100 test top-1 (%)",
           "(d) exit curve (heads trained at the last loop only)")
    fig.savefig(out_dir / "exit_tail_readout.png", dpi=150)
    plt.close(fig)


def check_top1(exits: dict, checkpoint: str, full_test_set: bool) -> None:
    """The model's own full-depth top-1 must match the logged test acc (wrong weights / labels otherwise)."""
    logged = torch.load(checkpoint, map_location="cpu", weights_only=True)["test_acc"]
    print(f"[{Path(checkpoint).stem}] own top-1 {exits['full_depth_top1']:.2f} (logged {logged:.2f})", flush=True)
    if full_test_set and abs(exits["full_depth_top1"] - logged) > TOP1_TOLERANCE:
        raise RuntimeError(f"top-1 {exits['full_depth_top1']:.2f} does not match the logged {logged:.2f}")


def analyse_checkpoint(checkpoint: str, args: argparse.Namespace, device: torch.device, out_dir: Path) -> dict:
    """(a)-(d) for one checkpoint; writes per_image_<stem>.pt, diag_<stem>.json and table_<stem>.md."""
    name = Path(checkpoint).stem
    model_name = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model = load_model(checkpoint).to(device).float().eval()
    _, tail = find_body(model)
    kind = readout_kind(model_name, tail)
    loop_loader, exit_loader = test_loaders(model, args)
    exits = None
    if kind is None:
        print(f"[{name}] (d) skipped: no tail and no readout defined for model '{model_name}'", flush=True)
    else:
        exits = collect_exit_accuracy(model, exit_loader, device, kind)
        check_top1(exits, checkpoint, full_test_set=not args.limit)
        print(f"[{name}] (d) acc per loop: "
              + " ".join(f"{t}:{a:.1f}" for t, a in enumerate(exits["acc_per_loop"], 1)), flush=True)
    per_image = collect_loop_metrics(model, loop_loader, device)
    means = summarise(per_image)
    result = {"checkpoint": checkpoint, "model": model_name, "readout": kind,
              "n_loop_images": per_image["rel_cls"].shape[1], "means": means, "exits": exits,
              "decision": decision(means)}
    torch.save(per_image, out_dir / f"per_image_{name}.pt")
    (out_dir / f"diag_{name}.json").write_text(json.dumps(result, indent=1))
    (out_dir / f"table_{name}.md").write_text(markdown_table(means, exits) + "\n")
    print(f"[{name}] decision: {result['decision']}", flush=True)
    return result


def main() -> None:
    """Analyse every checkpoint, then redraw the plots from all saved summaries."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.checkpoints:
        analyse_checkpoint(checkpoint, args, device, out_dir)
    plot_all(load_all_results(out_dir), out_dir)
    print(f"wrote diag_*.json, per_image_*.pt, table_*.md and the 4 plots in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
