"""T14: T12 metrics (a)-(c) and the exit curve (d) on the coverage-mask checkpoint (no training).

Same metrics, images and outputs as T12 `loop_diagnostics.py` (its functions are reused). The only difference: the
looped block's attention has the coverage mask, so the per-loop recompute replays the mask (c = 1, then
`src.coverage_mask.masked_block_parts` and `update_supply` every loop) and is asserted equal to the real body output.
Metric (b) uses the attention CLS actually used (after the mask). Also saved: mean supply left after each loop and the
learned eta.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T14_addon_from_diagnostics/diagnostics.py CKPT [--limit N]
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent / "T12_per_loop_cls_diagnostics"))

import loop_diagnostics as t12  # noqa: E402  (also puts the T5 / T8 helpers on the path)
from src.coverage_mask import CoverageMaskBody, masked_block_parts  # noqa: E402

DEFAULT_OUT = TASK_DIR / "results" / "t12_diag"


def masked_batch_metrics(body: CoverageMaskBody, states: list[torch.Tensor]) -> tuple[dict, float, list[torch.Tensor]]:
    """T12 `batch_metrics` for the masked body: {key: [per-loop (B,)]}, worst recompute difference, and the supply
    left after each loop (B, P)."""
    values: dict[str, list[torch.Tensor]] = {key: [] for key in (*t12.PER_LOOP_KEYS, *t12.PAIR_KEYS,
                                                                  *t12.SUCCESSIVE_KEYS)}
    attentions, supplies, worst = [], [], 0.0
    supply = torch.ones(states[0].shape[0], states[0].shape[1] - 1, device=states[0].device)
    for t in range(1, len(states)):
        previous, current = states[t - 1], states[t]
        parts = masked_block_parts(body.block, previous, supply)
        worst = max(worst, (parts["output"] - current).abs().max().item())
        supply = body.next_supply(supply, parts["cls_attention"])  # T14 rule v1, or T16's v2
        supplies.append(supply)
        rel_cls, rel_patch = t12.relative_update(previous, current)
        attention = t12.cls_patch_attention(parts["probs"])
        attentions.append(attention)
        values["rel_cls"].append(rel_cls)
        values["rel_patch"].append(rel_patch)
        values["attn_entropy"].append(t12.entropy(attention).mean(dim=1))
        values["attn_entropy_headavg"].append(t12.entropy(attention.mean(dim=1)))
        values["new_direction_share"].append(t12.new_direction_share(parts["attn_out"][:, 0], previous[:, 0]))
        if t >= 2:
            delta, delta_before = current[:, 0] - previous[:, 0], previous[:, 0] - states[t - 2][:, 0]
            values["cls_update_cos"].append(F.cosine_similarity(delta, delta_before, dim=1))
    for current, following in zip(attentions, attentions[1:]):
        per_head, head_avg = t12.attention_cosines(current, following)
        values["attn_cos"].append(per_head)
        values["attn_cos_headavg"].append(head_avg)
    return values, worst, supplies


@torch.no_grad()
def collect_masked_metrics(model: nn.Module, loader, device: torch.device) -> dict[str, torch.Tensor]:
    """{key: (n_loops_for_key, N)} per-image values, plus "supply_mean" (mean c over patches after each loop)."""
    body, _ = t12.find_body(model)
    if not isinstance(body, CoverageMaskBody):
        raise ValueError("this script is for coverage-mask checkpoints; use T12 loop_diagnostics.py for the others")
    batches: dict[str, list[torch.Tensor]] = {}
    worst = 0.0
    with t12.BodyRecorder(body) as recorder:
        for images, _ in loader:
            recorder.calls.clear()
            model.forward_features(images.to(device, non_blocking=True))
            values, batch_worst, supplies = masked_batch_metrics(body, t12.loop_states(recorder.calls))
            worst = max(worst, batch_worst)
            if worst >= t12.RECOMPUTE_TOLERANCE:
                raise RuntimeError(f"recomputed masked block output differs from the real one by {worst:.2e}")
            values["supply_mean"] = [supply.mean(dim=1) for supply in supplies]
            for key, per_loop in values.items():
                batches.setdefault(key, []).append(torch.stack(per_loop).cpu())
    print(f"recompute check (with mask): max |recomputed - real| block output = {worst:.2e} "
          f"(< {t12.RECOMPUTE_TOLERANCE:g})", flush=True)
    return {key: torch.cat(parts, dim=1) for key, parts in batches.items()}


def analyse(checkpoint: str, args, device: torch.device, out_dir: Path) -> dict:
    """(a)-(d) for one coverage-mask checkpoint; writes per_image_<stem>.pt, diag_<stem>.json, table_<stem>.md."""
    name = Path(checkpoint).stem
    model_name = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model = t12.load_model(checkpoint).to(device).float().eval()
    body, tail = t12.find_body(model)
    kind = t12.readout_kind(model_name, tail)
    loop_loader, exit_loader = t12.test_loaders(model, args)
    exits = t12.collect_exit_accuracy(model, exit_loader, device, kind)
    t12.check_top1(exits, checkpoint, full_test_set=not args.limit)
    per_image = collect_masked_metrics(model, loop_loader, device)
    supply = per_image.pop("supply_mean")
    means = t12.summarise(per_image)
    means["supply_mean"] = [round(v, 4) for v in supply.mean(dim=1).tolist()]
    result = {"checkpoint": checkpoint, "model": model_name, "readout": kind, "eta": round(body.eta.item(), 5),
              "n_loop_images": per_image["rel_cls"].shape[1], "means": means, "exits": exits,
              "decision": t12.decision(means)}
    torch.save({**per_image, "supply_mean": supply}, out_dir / f"per_image_{name}.pt")
    (out_dir / f"diag_{name}.json").write_text(json.dumps(result, indent=1))
    (out_dir / f"table_{name}.md").write_text(t12.markdown_table(means, exits) + "\n")
    print(f"[{name}] eta {result['eta']}; (d) acc per loop: "
          + " ".join(f"{t}:{a:.1f}" for t, a in enumerate(exits["acc_per_loop"], 1)), flush=True)
    print(f"[{name}] supply left per loop: {means['supply_mean']}", flush=True)
    print(f"[{name}] decision-rule values: {result['decision']}", flush=True)
    return result


def main() -> None:
    """Analyse every given checkpoint (T12's command-line options; default output in this task's results)."""
    if "--out-dir" not in sys.argv:
        sys.argv += ["--out-dir", str(DEFAULT_OUT)]
    args = t12.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in args.checkpoints:
        analyse(checkpoint, args, device, out_dir)
    print(f"wrote diag_*.json, per_image_*.pt, table_*.md in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
