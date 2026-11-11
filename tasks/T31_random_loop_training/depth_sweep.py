"""T31: test top-1 at a fixed number of loops r = 1..20 (T31-a and T11-d), then T13's exit rules and the T8 KL rule
on T31-a with a 20-loop budget (images that never stop use loop 20).

The readout after loop r is the model's normal readout (final norm + classifier on CLS) of the state after r body
loops (`src.random_loops.per_loop_logits_upto`). Loop 10 is the normal model output and must match the logged top-1.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T31_random_loop_training/depth_sweep.py [--limit N]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch import nn

TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
sys.path.insert(0, str(TASKS / "T5_cls_svd_per_block"))
sys.path.insert(0, str(TASKS / "T13_per_loop_readout_loss"))

from cls_svd import load_model  # noqa: E402
from early_exit import exit_accuracies  # noqa: E402
from exit_analysis import check_top1, compare_with_fixed, flip_rate, rule_curves, test_loader  # noqa: E402

from src.random_loops import MAX_LOOPS, per_loop_logits_upto  # noqa: E402

CHECKPOINTS = Path("/storage/INSLAB/Data/nrs3qy/checkpoints/itertf")
MODELS = {"T31-a": CHECKPOINTS / "T31-a_random_loops.pt", "T11-d": CHECKPOINTS / "T11-d_unroll_n3_30ep.pt"}
TRAINED_LOOPS = 10
EXIT_RULE_MODEL = "T31-a"

log = logging.getLogger("t31_depth")


@torch.no_grad()
def collect_logits(model: nn.Module, loader, device: torch.device, n_loops: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(logits after loops 1..n_loops (L, N, classes) in fp32 on the CPU, labels (N,))."""
    chunks, labels = [], []
    for images, targets in loader:
        logits = per_loop_logits_upto(model, images.to(device, non_blocking=True), n_loops)
        chunks.append(torch.stack(logits).float().cpu())
        labels.append(targets)
    return torch.cat(chunks, dim=1), torch.cat(labels)


def exit_rules(logits: torch.Tensor, labels: torch.Tensor, fixed: list[float]) -> dict:
    """Confidence, KL (T8) and patience rules over the 20-loop budget, compared with the fixed-depth curve."""
    curves = rule_curves(logits, labels, fixed)
    return {"curves": curves, "vs_fixed": {name: compare_with_fixed(points, fixed) for name, points in curves.items()}}


def main() -> None:
    """Both models' depth curves, then the exit rules on T31-a."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="first N test images only (smoke)")
    parser.add_argument("--out", default=str(TASK_DIR / "results" / "depth_sweep.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {}
    for tag, checkpoint in MODELS.items():
        model = load_model(str(checkpoint)).to(device).float().eval()
        logits, labels = collect_logits(model, test_loader(model, args), device, MAX_LOOPS)
        fixed = [round(acc, 2) for acc in exit_accuracies(logits, labels)]
        check_top1(fixed[TRAINED_LOOPS - 1], str(checkpoint), full_test_set=args.limit == 0)
        result[tag] = {"checkpoint": str(checkpoint), "n_images": len(labels), "acc_per_loops": fixed,
                       "flip_rate_19_20": flip_rate(logits)}
        log.info("[%s] top-1 at r = 1..20: %s", tag, fixed)
        if tag == EXIT_RULE_MODEL:
            result[tag]["exit_rules_20_loops"] = exit_rules(logits, labels, fixed)
            for name, summary in result[tag]["exit_rules_20_loops"]["vs_fixed"].items():
                log.info("[%s] %s best gain: %s", tag, name, summary["best_gain"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
