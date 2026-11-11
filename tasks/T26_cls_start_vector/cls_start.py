"""T26: does the CLS starting vector matter? Learned vs zero vs random (no training).

Only the learned `cls_token` (before the position embedding) is replaced. timm concatenates the CLS token and then
adds the position embedding, so token 0 after `_pos_embed` is cls_token + pos_embed[0]; `cls_start` wraps
`model._pos_embed` and adds an offset to that row, which is the same as replacing the token (tested). Starts:
learned, zeros, noise N(0, sigma^2) (seeds 0..4), learned + noise (seeds 0..4), and the other model's learned token.

Per model and start, on the TEST images: final and per-loop readout top-1, and the gap
g_s = mean ||CLS_s(alt) - CLS_s(learned)|| / ||CLS_s(learned)|| after B0, after B1 and after loops 1..10.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T26_cls_start_vector/cls_start.py [--limit N]
"""
import argparse
import json
import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
sys.path.insert(0, str(TASKS / "T13_per_loop_readout_loss"))

from exit_analysis import check_top1, load_model, test_loader  # noqa: E402

from src.per_loop_readout import embed, readout, split_loop  # noqa: E402

CKPT_DIR = Path("/storage/INSLAB/Data/nrs3qy/checkpoints/itertf")
MODELS = {"T11-d": CKPT_DIR / "T11-d_unroll_n3_30ep.pt", "T13-c": CKPT_DIR / "T13-c_unroll_readout_every_loop.pt"}
NOISE_SEEDS = (0, 1, 2, 3, 4)

log = logging.getLogger("t26")

# offset(batch_size, index of the batch's first image, device) -> tensor (B, D) added to token 0
Offset = Callable[[int, int, torch.device], torch.Tensor]


@contextmanager
def cls_start(model: nn.Module, offset: Offset | None) -> Iterator[None]:
    """Add `offset` to token 0 after `_pos_embed` while inside the block (None: no change). Images are counted
    from 0 at entry, so a per-image offset stays tied to the same image across passes. Restores the method on exit."""
    if offset is None:
        yield
        return
    original = model._pos_embed
    seen = [0]

    def shifted(x: torch.Tensor) -> torch.Tensor:
        out = original(x)
        shift = offset(out.shape[0], seen[0], out.device).to(out.dtype)
        seen[0] += out.shape[0]
        return torch.cat([out[:, :1] + shift[:, None, :], out[:, 1:]], dim=1)

    model._pos_embed = shifted
    try:
        yield
    finally:
        del model._pos_embed  # back to the class method


def constant_offset(vector: torch.Tensor) -> Offset:
    """The same offset (D,) for every image."""
    return lambda batch, start, device: vector.to(device).expand(batch, -1)


def noise_table(seed: int, n_images: int, dim: int, sigma: float) -> torch.Tensor:
    """N(0, sigma^2) per image and dimension, (n_images, dim), fixed by `seed`."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_images, dim, generator=generator) * sigma


def table_offset(table: torch.Tensor, minus: torch.Tensor | None) -> Offset:
    """Per-image offset: row i of `table` (minus a constant vector, if given) for the i-th image of the pass."""
    def offset(batch: int, start: int, device: torch.device) -> torch.Tensor:
        rows = table[start:start + batch]
        if len(rows) != batch:
            raise IndexError(f"noise table has {len(table)} rows, the pass asked for {start + batch}")
        return (rows if minus is None else rows - minus).to(device)
    return offset


def learned_token(model: nn.Module) -> torch.Tensor:
    """The model's learned CLS token as a (D,) CPU tensor."""
    return model.cls_token.detach().flatten().cpu()


def start_offsets(model: nn.Module, other: nn.Module, n_images: int) -> dict[str, Offset | None]:
    """Every start as an offset on token 0 (None = learned, unchanged)."""
    cls = learned_token(model)
    sigma = cls.std().item()
    starts: dict[str, Offset | None] = {"learned": None, "zeros": constant_offset(-cls)}
    for seed in NOISE_SEEDS:
        table = noise_table(seed, n_images, cls.numel(), sigma)
        starts[f"noise s{seed}"] = table_offset(table, minus=cls)
        starts[f"learned+noise s{seed}"] = table_offset(table, minus=None)
    starts["swap"] = constant_offset(learned_token(other) - cls)
    return starts


def run_stages(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(per-loop logits (L, B, C), raw CLS after each head block then after each loop (H + L, B, D))."""
    head, body, tail, n_loops = split_loop(model)
    x = embed(model, images)
    cls_states, logits = [], []
    for block in head:
        x = block(x)
        cls_states.append(x[:, 0])
    for _ in range(n_loops):
        x = body(x)
        cls_states.append(x[:, 0])
        logits.append(readout(model, x, tail))
    return torch.stack(logits), torch.stack(cls_states)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, offset: Offset | None,
             reference: torch.Tensor | None) -> dict:
    """Per-loop top-1, plus either the CLS states (reference=None) or the per-stage gap to `reference` (S, N, D)."""
    correct, gap_sum, states, total = None, None, [], 0
    with cls_start(model, offset):
        for images, labels in loader:
            logits, cls = run_stages(model, images.to(device, non_blocking=True))
            hits = (logits.argmax(dim=-1).cpu() == labels).sum(dim=1).double()
            correct = hits if correct is None else correct + hits
            cls = cls.float().cpu()
            if reference is None:
                states.append(cls)
            else:
                ref = reference[:, total:total + len(labels)]
                gap = (torch.linalg.vector_norm(cls - ref, dim=-1) / torch.linalg.vector_norm(ref, dim=-1)).sum(1)
                gap_sum = gap.double() if gap_sum is None else gap_sum + gap.double()
            total += len(labels)
    result = {"acc_per_loop": [round(100.0 * c / total, 2) for c in correct.tolist()]}
    if reference is None:
        result["states"] = torch.cat(states, dim=1)
    else:
        result["gap_per_stage"] = [round(g / total, 4) for g in gap_sum.tolist()]
    return result


def seed_summary(values: list[float]) -> dict:
    """Mean and spread (max - min) over seeds."""
    return {"mean": round(sum(values) / len(values), 2), "spread": round(max(values) - min(values), 2)}


def path_line(result: dict) -> str:
    """'path independent' / 'start matters' / 'partly' for one model, as defined in the task."""
    reference = result["starts"]["learned"]["final"]
    zero_cost = reference - result["starts"]["zeros"]["final"]
    noise_cost = reference - result["noise_final"]["mean"]
    zero_g10, noise_g10 = result["starts"]["zeros"]["gap_per_stage"][-1], result["noise_gap_last_mean"]
    numbers = (f"zeros cost {zero_cost:+.2f} pt, noise cost {noise_cost:+.2f} pt (mean of 5 seeds), "
               f"g_10 zeros {zero_g10:.3f}, noise {noise_g10:.3f}")
    if zero_cost <= 1 and noise_cost <= 1 and zero_g10 < 0.1 and noise_g10 < 0.1:
        return f"path independent ({numbers})"
    if zero_cost > 3 or noise_cost > 3:
        return f"start matters ({numbers})"
    return f"partly ({numbers})"


def analyse_model(tag: str, model: nn.Module, other: nn.Module, args: argparse.Namespace,
                  device: torch.device) -> dict:
    """All starts for one model."""
    loader = test_loader(model, args)
    learned = evaluate(model, loader, device, None, None)
    check_top1(learned["acc_per_loop"][-1], str(MODELS[tag]), full_test_set=args.limit == 0)
    reference = learned.pop("states")
    starts = {"learned": {**learned, "final": learned["acc_per_loop"][-1],
                          "gap_per_stage": [0.0] * reference.shape[0]}}
    for name, offset in start_offsets(model, other, len(loader.dataset)).items():
        if offset is None:
            continue
        outcome = evaluate(model, loader, device, offset, reference)
        starts[name] = {**outcome, "final": outcome["acc_per_loop"][-1]}
        log.info("[%s] %-18s final %.2f  g: %s", tag, name, starts[name]["final"],
                 " ".join(f"{g:.3f}" for g in outcome["gap_per_stage"]))
    result = {"tag": tag, "sigma": round(learned_token(model).std().item(), 5), "n_images": len(loader.dataset),
              "stages": ["B0", "B1"] + [f"loop {t}" for t in range(1, reference.shape[0] - 1)], "starts": starts}
    for kind in ("noise", "learned+noise"):
        result[f"{kind}_final"] = seed_summary([starts[f"{kind} s{seed}"]["final"] for seed in NOISE_SEEDS])
    noise_last = [starts[f"noise s{seed}"]["gap_per_stage"][-1] for seed in NOISE_SEEDS]
    result["noise_gap_last_mean"] = round(sum(noise_last) / len(noise_last), 4)
    result["line"] = path_line(result)
    return result


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N test images only (smoke)")
    parser.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    return parser.parse_args()


def main() -> None:
    """Both models, all starts, saved to results/cls_start.json."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {tag: load_model(str(path)).to(device).eval() for tag, path in MODELS.items()}
    results = {}
    for tag, model in models.items():
        other = next(m for t, m in models.items() if t != tag)
        results[tag] = analyse_model(tag, model, other, args, device)
        log.info("[%s] %s", tag, results[tag]["line"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cls_start.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
