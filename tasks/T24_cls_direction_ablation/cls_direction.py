"""T24: is there one repeated CLS direction in late loops, does it carry class information, can it be cut?

Per model (T11-d, T13-c):
  (a) D = d_t for t = 6..10 on the first 10,000 train images; uncentred SVD -> u1 (and u2), energy shares.
  (b) Fisher ratio of CLS_10 . u1 on the test set vs 200 random unit directions; mean ‖d_t‖ and |d_t . u1| per loop.
  (c) test top-1 with the update's u1 component removed in loops 6..10 / 1..10, and the random / u2 controls.

Outputs: results/u1_<model>.npy, results/u2_<model>.npy, results/cls_direction.json.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T24_cls_direction_ablation/cls_direction.py [--train-limit N] [--test-limit N]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(TASKS / "T25_class_cloud_exit"))
sys.path.insert(0, str(TASKS / "T5_cls_svd_per_block"))
sys.path.insert(0, str(TASKS / "T13_per_loop_readout_loss"))

from class_cloud import fixed_loader  # noqa: E402
from cls_svd import load_model  # noqa: E402
from exit_analysis import check_top1  # noqa: E402
from projection import edited_per_loop, fisher_ratios, random_unit_directions, uncentred_direction  # noqa: E402

from src.per_loop_readout import per_loop_logits_and_cls  # noqa: E402

CHECKPOINTS = Path("/storage/INSLAB/Data/nrs3qy/checkpoints/itertf")
MODELS = {"T11-d": CHECKPOINTS / "T11-d_unroll_n3_30ep.pt",
          "T13-c": CHECKPOINTS / "T13-c_unroll_readout_every_loop.pt"}
LATE_LOOPS = range(6, 11)
ALL_LOOPS = range(1, 11)
N_RANDOM_FISHER = 200
FISHER_SEED = 0
CONTROL_SEEDS = (0, 1, 2, 3, 4)
SPECTRUM_TOP = 20
SUPPORTED_ENERGY = 0.5
REFUTED_ENERGY = 0.3
ALLOWED_COST = 0.5

log = logging.getLogger("t24")


@torch.no_grad()
def collect_cls(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(raw CLS states (L + 1, N, D) as float64: CLS_0 = body input .. CLS_L, labels (N,))."""
    chunks, labels = [], []
    for images, targets in loader:
        _, cls_states = per_loop_logits_and_cls(model, images.to(device, non_blocking=True))
        chunks.append(torch.stack(cls_states).double().cpu())
        labels.append(targets)
    return torch.cat(chunks, dim=1).numpy(), torch.cat(labels).numpy()


def late_updates(states: np.ndarray) -> np.ndarray:
    """D: the rows d_t = CLS_t - CLS_{t-1} for t = 6..10, stacked to (5 N, D)."""
    updates = states[1:] - states[:-1]  # updates[t - 1] = d_t
    return updates[[t - 1 for t in LATE_LOOPS]].reshape(-1, states.shape[-1])


def direction_part(states: np.ndarray) -> dict:
    """(a): u1, u2 and the energy numbers from the train-set late updates."""
    result = uncentred_direction(late_updates(states))
    energy = result["energy"]
    return {"u1": result["u1"], "u2": result["u2"], "energy_share": float(energy[0]),
            "cumulative_top5": np.cumsum(energy[:5]).round(4).tolist(),
            "spectrum_top20": energy[:SPECTRUM_TOP].round(5).tolist(),
            "cos_u1_mean": round(result["cos_u1_mean"], 4),
            "centred_energy_share": round(result["centred_energy_share"], 4)}


def class_information_part(states: np.ndarray, labels: np.ndarray, u1: np.ndarray) -> dict:
    """(b): Fisher ratio of CLS_10 . u1 vs random directions; push along u1 per loop."""
    final = states[-1]
    randoms = random_unit_directions(N_RANDOM_FISHER, final.shape[1], FISHER_SEED).numpy()
    ratios = fisher_ratios(final @ np.column_stack([u1, randoms.T]), labels)
    fisher_u1, random_ratios = float(ratios[0]), ratios[1:]
    updates = states[1:] - states[:-1]
    return {"fisher_u1": round(fisher_u1, 4), "random_ratios": random_ratios.round(5).tolist(),
            "random_median": round(float(np.median(random_ratios)), 4),
            "random_p95": round(float(np.percentile(random_ratios, 95)), 4),
            "u1_percentile": round(float((random_ratios < fisher_u1).mean() * 100), 1),
            "d_norm_per_loop": np.linalg.norm(updates, axis=2).mean(axis=1).round(4).tolist(),
            "push_u1_per_loop": np.abs(updates @ u1).mean(axis=1).round(4).tolist()}


@torch.no_grad()
def edited_accuracy(model: nn.Module, loader, device: torch.device, direction: np.ndarray,
                    loops: range | tuple) -> dict:
    """Test top-1 per loop and final (%) with `direction` removed from the CLS update in `loops`."""
    vector = torch.as_tensor(direction, dtype=torch.float32, device=device)
    correct = None
    seen = 0
    for images, targets in loader:
        logits, _ = edited_per_loop(model, images.to(device, non_blocking=True), vector, set(loops))
        hits = torch.stack([(loop_logits.argmax(dim=1).cpu() == targets).sum() for loop_logits in logits])
        correct = hits if correct is None else correct + hits
        seen += len(targets)
    per_loop = (100.0 * correct.double() / seen).round(decimals=2).tolist()
    return {"final": per_loop[-1], "acc_per_loop": per_loop}


def ablation_part(model: nn.Module, loader, device: torch.device, u1: np.ndarray, u2: np.ndarray) -> dict:
    """(c): the five settings (random control over 5 seeds)."""
    zero = np.zeros_like(u1)
    settings = {"no edit": edited_accuracy(model, loader, device, zero, ()),
                "u1 loops 6-10": edited_accuracy(model, loader, device, u1, LATE_LOOPS),
                "u1 loops 1-10": edited_accuracy(model, loader, device, u1, ALL_LOOPS),
                "u2 loops 6-10": edited_accuracy(model, loader, device, u2, LATE_LOOPS)}
    randoms = [edited_accuracy(model, loader, device,
                               random_unit_directions(1, len(u1), seed)[0].numpy(), LATE_LOOPS)
               for seed in CONTROL_SEEDS]
    finals = [run["final"] for run in randoms]
    settings["random loops 6-10"] = {"final": round(float(np.mean(finals)), 2), "finals": finals,
                                     "spread": round(max(finals) - min(finals), 2),
                                     "acc_per_loop": np.mean([r["acc_per_loop"] for r in randoms], axis=0)
                                     .round(2).tolist()}
    return settings


def supported_line(energy_share: float, no_edit: float, late_cut: float) -> str:
    """The README supported line."""
    numbers = f"energy share {energy_share:.3f}, top-1 {late_cut:.2f} with u1 removed in loops 6-10 vs {no_edit:.2f}"
    if energy_share < REFUTED_ENERGY:
        return f"H11 refuted ({numbers})"
    if energy_share >= SUPPORTED_ENERGY and late_cut >= no_edit - ALLOWED_COST:
        return f"H11 supported ({numbers})"
    return f"H11 mixed ({numbers})"


def analyse_model(tag: str, checkpoint: Path, args: argparse.Namespace, device: torch.device, out_dir: Path) -> dict:
    """(a), (b), (c) and the supported line for one model."""
    model = load_model(str(checkpoint)).to(device).float().eval()
    train_states, _ = collect_cls(model, fixed_loader(model, "train", args.train_limit, args), device)
    direction = direction_part(train_states)
    u1, u2 = direction.pop("u1"), direction.pop("u2")
    np.save(out_dir / f"u1_{tag}.npy", u1)
    np.save(out_dir / f"u2_{tag}.npy", u2)
    test = fixed_loader(model, "test", args.test_limit, args)
    test_states, labels = collect_cls(model, test, device)
    ablation = ablation_part(model, test, device, u1, u2)
    check_top1(ablation["no edit"]["final"], str(checkpoint), full_test_set=args.test_limit == 0)
    line = supported_line(direction["energy_share"], ablation["no edit"]["final"], ablation["u1 loops 6-10"]["final"])
    log.info("[%s] energy share %.3f; LINE: %s", tag, direction["energy_share"], line)
    return {"checkpoint": str(checkpoint), "n_train": train_states.shape[1], "n_test": test_states.shape[1],
            "a": direction, "b": class_information_part(test_states, labels, u1), "c": ablation, "line": line}


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=10_000)
    parser.add_argument("--test-limit", type=int, default=0, help="first N test images only (smoke)")
    parser.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    return parser.parse_args()


def main() -> None:
    """Both models."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {tag: analyse_model(tag, checkpoint, args, device, out_dir) for tag, checkpoint in MODELS.items()}
    (out_dir / "cls_direction.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
