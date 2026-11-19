"""T34: do the late loops only repeat the same update? Replace loops k+1..10 by copies of the loop-k update.

Layout 2+10 (B0, B1, then B2 applied 10 times, no tail). x_0 = tokens after B1, x_t = B2(x_{t-1}).
Readout at loop t: classifier(final_norm(x_t)[:, 0]) with the checkpoint's own modules.

Per checkpoint (T11-d, T13-c, T31-a), on the full CIFAR-100 test set:
- frozen(k), k = 3..9: run B2 to loop k, d_k = x_k - x_{k-1}, then x_t = x_{t-1} + d_k for t = k+1..10 (no B2 call),
  read out at loop 10;
- normal pass to loop 20: top-1 at every loop, mean 1 - cos(final_norm(x_t), final_norm(x_{t-1})) for CLS and for the
  patches (per-token cos, averaged over tokens 1..196), and the mean raw L2 norm of the CLS token.
Sanity: loop-10 top-1 must match the logged number within 0.05 pt, or the checkpoint is skipped.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T34_frozen_update_test/frozen_update.py [--limit N] [--out PATH]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
sys.path.insert(0, str(TASKS / "T5_cls_svd_per_block"))
sys.path.insert(0, str(TASKS / "T13_per_loop_readout_loss"))

from cls_svd import load_model  # noqa: E402
from exit_analysis import test_loader  # noqa: E402

from src.per_loop_readout import embed, readout, split_loop  # noqa: E402

CHECKPOINT_DIR = Path("/storage/INSLAB/Data/nrs3qy/checkpoints/itertf")
CHECKPOINTS = {
    "T11-d": CHECKPOINT_DIR / "T11-d_unroll_n3_30ep.pt",
    "T13-c": CHECKPOINT_DIR / "T13-c_unroll_readout_every_loop.pt",
    "T31-a": CHECKPOINT_DIR / "T31-a_random_loops.pt",
}
LOGGED_TOP1_LOOP10 = {"T11-d": 74.11, "T13-c": 72.82, "T31-a": 71.77}  # task file, "Logged reference numbers"
LOGGED_DEPTH_SOURCES = {  # exact logged per-loop top-1 (fp32 eval), used for the per-loop check
    "T11-d": (TASKS / "T31_random_loop_training/results/depth_sweep.json", "T11-d", "acc_per_loops"),
    "T31-a": (TASKS / "T31_random_loop_training/results/depth_sweep.json", "T31-a", "acc_per_loops"),
    "T13-c": (TASKS / "T13_per_loop_readout_loss/results/exit_analysis.json", "T13-c_unroll_readout_every_loop",
              "acc_per_loop"),
}
TOLERANCE = 0.05  # points
TRAINED_LOOPS = 10
DEPTH = 20
FROZEN_KS = tuple(range(3, 10))

# Verdict rule, fixed in advance in the task file.
CONFIRM_LOOSE = ("T13-c", "T31-a")  # within 1 pt of true-10 for some k <= 4
CONFIRM_LOOSE_POINTS, CONFIRM_LOOSE_MAX_K = 1.0, 4
CONFIRM_STRICT = "T11-d"  # within 2 pt of true-10 for some k <= 6
CONFIRM_STRICT_POINTS, CONFIRM_STRICT_MAX_K = 2.0, 6
KILL_K, KILL_BELOW = 5, 61.9  # T11-d frozen(5) below 61.9 (closer to the loop-5 readout than to true-10)

log = logging.getLogger("t34")


def plain_loop_parts(model: nn.Module) -> tuple[list[nn.Module], nn.Module, int]:
    """(head blocks, looped block B2, number of loops) of a 2+r model; raises ValueError if it has a tail."""
    head, body, tail, n_loops = split_loop(model)
    if tail:
        raise ValueError(f"T34 needs a model without tail blocks, found {len(tail)}")
    if hasattr(body, "reset"):
        raise ValueError("stateful bodies are not supported")
    return head, body, n_loops


def head_state(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """x_0: the tokens after the distinct head blocks (B0, B1), shape (B, 197, C)."""
    head, _, _ = plain_loop_parts(model)
    x = embed(model, images)
    for block in head:
        x = block(x)
    return x


def frozen_final_state(model: nn.Module, x0: torch.Tensor, k: int, n_loops: int = TRAINED_LOOPS) -> torch.Tensor:
    """x_{n_loops} when B2 runs for loops 1..k and loops k+1..n_loops add d_k = x_k - x_{k-1} instead of calling B2."""
    if not 1 <= k <= n_loops:
        raise ValueError(f"k must be in 1..{n_loops}, got {k}")
    _, body, _ = plain_loop_parts(model)
    previous, x = x0, x0
    for _ in range(k):
        previous, x = x, body(x)
    update = x - previous
    for _ in range(k + 1, n_loops + 1):
        x = x + update
    return x


def frozen_logits(model: nn.Module, x0: torch.Tensor, k: int, n_loops: int = TRAINED_LOOPS) -> torch.Tensor:
    """Loop-`n_loops` readout of the frozen path (see `frozen_final_state`)."""
    return readout(model, frozen_final_state(model, x0, k, n_loops), [])


def normal_pass(model: nn.Module, x0: torch.Tensor, n_loops: int = DEPTH) -> dict[str, list[torch.Tensor]]:
    """Per loop t = 1..n_loops of the normal model: logits (B, classes), CLS and patch angular step (B,) and raw CLS
    norm (B,). Angular step = 1 - cos(final_norm(x_t), final_norm(x_{t-1})); patches are averaged over tokens."""
    _, body, _ = plain_loop_parts(model)
    out: dict[str, list[torch.Tensor]] = {"logits": [], "cls_step": [], "patch_step": [], "cls_norm": []}
    x, normed_previous = x0, model.norm(x0)
    for _ in range(n_loops):
        x = body(x)
        normed = model.norm(x)
        step = 1.0 - F.cosine_similarity(normed.float(), normed_previous.float(), dim=-1)  # (B, tokens)
        out["logits"].append(readout(model, x, []))
        out["cls_step"].append(step[:, 0])
        out["patch_step"].append(step[:, 1:].mean(dim=1))
        out["cls_norm"].append(x[:, 0].float().norm(dim=-1))
        normed_previous = normed
    return out


def count_correct(logits: torch.Tensor, labels: torch.Tensor) -> int:
    """Number of top-1 hits in a batch."""
    return int((logits.argmax(dim=-1) == labels).sum())


@torch.no_grad()
def analyse_checkpoint(model: nn.Module, loader, device: torch.device) -> dict:
    """Frozen(k) top-1 for k in FROZEN_KS and the normal 1..DEPTH statistics, over the whole loader."""
    frozen_hits = {k: 0 for k in FROZEN_KS}
    normal_hits = torch.zeros(DEPTH, dtype=torch.float64)
    sums = {name: torch.zeros(DEPTH, dtype=torch.float64) for name in ("cls_step", "patch_step", "cls_norm")}
    n_images = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        x0 = head_state(model, images)
        normal = normal_pass(model, x0, DEPTH)
        for t in range(DEPTH):
            normal_hits[t] += count_correct(normal["logits"][t], labels)
            for name, total in sums.items():
                total[t] += float(normal[name][t].double().sum())
        for k in FROZEN_KS:
            frozen_hits[k] += count_correct(frozen_logits(model, x0, k, TRAINED_LOOPS), labels)
        n_images += len(labels)
    acc_per_loop = [round(100.0 * float(h) / n_images, 2) for h in normal_hits]
    return {
        "n_images": n_images,
        "true_10": acc_per_loop[TRAINED_LOOPS - 1],
        "frozen": {str(k): round(100.0 * frozen_hits[k] / n_images, 2) for k in FROZEN_KS},
        "loop_k_readout": {str(k): acc_per_loop[k - 1] for k in FROZEN_KS},
        "acc_per_loop_1_20": acc_per_loop,
        "angular_step_cls_1_20": [float(v) / n_images for v in sums["cls_step"]],
        "angular_step_patches_1_20": [float(v) / n_images for v in sums["patch_step"]],
        "cls_norm_1_20": [float(v) / n_images for v in sums["cls_norm"]],
    }


def logged_depth_curve(tag: str) -> list[float]:
    """The logged per-loop top-1 for `tag` (20 loops for T11-d and T31-a, 10 for T13-c)."""
    path, key, field = LOGGED_DEPTH_SOURCES[tag]
    return json.loads(path.read_text())[key][field]


def depth_check(tag: str, acc_per_loop: list[float]) -> dict:
    """Largest per-loop difference to the logged depth curve, and whether every loop is within TOLERANCE."""
    logged = logged_depth_curve(tag)
    diffs = [round(abs(ours - theirs), 2) for ours, theirs in zip(acc_per_loop, logged)]
    return {"loops_checked": len(logged), "max_abs_diff": max(diffs), "passed": max(diffs) <= TOLERANCE + 1e-9}


def within(frozen: dict, true_10: float, points: float, max_k: int) -> list[int]:
    """The k <= max_k whose frozen(k) is within `points` of true-10."""
    return [k for k in FROZEN_KS if k <= max_k and abs(frozen[str(k)] - true_10) <= points + 1e-9]


def verdict(results: dict) -> dict:
    """The task's fixed verdict rule. Needs all three checkpoints; otherwise 'incomplete'."""
    missing = [tag for tag in CHECKPOINTS if results.get(tag, {}).get("status") != "ok"]
    if missing:
        return {"verdict": "incomplete", "missing": missing}
    broken = []
    for tag in CONFIRM_LOOSE:
        if not within(results[tag]["frozen"], results[tag]["true_10"], CONFIRM_LOOSE_POINTS, CONFIRM_LOOSE_MAX_K):
            broken.append(f"{tag}: no k <= {CONFIRM_LOOSE_MAX_K} within {CONFIRM_LOOSE_POINTS} pt of true-10")
    strict = results[CONFIRM_STRICT]
    if not within(strict["frozen"], strict["true_10"], CONFIRM_STRICT_POINTS, CONFIRM_STRICT_MAX_K):
        broken.append(f"{CONFIRM_STRICT}: no k <= {CONFIRM_STRICT_MAX_K} within {CONFIRM_STRICT_POINTS} pt of true-10")
    confirmed = not broken
    killed = strict["frozen"][str(KILL_K)] < KILL_BELOW
    if confirmed and killed:
        label = "ambiguous: both 'confirmed' and 'killed' hold; the rule gives no precedence (see gotcha)"
    elif confirmed:
        label = "story confirmed"
    elif killed:
        label = "story killed"
    else:
        label = "mixed"
    return {"verdict": label, "confirmed": confirmed, "killed": killed, "broken_parts": broken,
            "t11d_frozen_5": strict["frozen"][str(KILL_K)], "kill_threshold": KILL_BELOW}


def run_one(tag: str, args: argparse.Namespace, device: torch.device) -> dict:
    """Load one checkpoint, analyse it and apply the sanity checks (full test set only)."""
    checkpoint = CHECKPOINTS[tag]
    model = load_model(str(checkpoint)).to(device).float().eval()
    result = {"checkpoint": str(checkpoint), **analyse_checkpoint(model, test_loader(model, args), device)}
    full_test_set = args.limit == 0
    logged = LOGGED_TOP1_LOOP10[tag]
    result["sanity"] = {"logged_loop_10": logged, "ours_loop_10": result["true_10"],
                        "abs_diff": round(abs(result["true_10"] - logged), 2), "full_test_set": full_test_set}
    result["depth_check"] = depth_check(tag, result["acc_per_loop_1_20"]) if full_test_set else None
    if full_test_set and result["sanity"]["abs_diff"] > TOLERANCE + 1e-9:
        log.error("[%s] SANITY FAILED: loop-10 top-1 %.2f vs logged %.2f; checkpoint not analysed",
                  tag, result["true_10"], logged)
        return {"checkpoint": str(checkpoint), "status": "sanity_failed", "sanity": result["sanity"]}
    result["status"] = "ok"
    return result


def report(tag: str, result: dict) -> None:
    """One-screen summary for the log."""
    if result["status"] != "ok":
        return
    sanity, depth = result["sanity"], result["depth_check"]
    log.info("[%s] loop-10 top-1 %.2f (logged %.2f, diff %.2f)", tag, result["true_10"], sanity["logged_loop_10"],
             sanity["abs_diff"])
    if depth is not None:
        log.info("[%s] depth check over %d loops: max diff %.2f, %s", tag, depth["loops_checked"],
                 depth["max_abs_diff"], "passed" if depth["passed"] else "FAILED")
    log.info("[%s] frozen(k) / loop-k readout: %s", tag, "  ".join(
        f"k{k}: {result['frozen'][str(k)]:.2f}/{result['loop_k_readout'][str(k)]:.2f}" for k in FROZEN_KS))
    log.info("[%s] top-1 at t = 1..20: %s", tag, result["acc_per_loop_1_20"])


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="first N test images only (smoke; no sanity checks)")
    parser.add_argument("--out", default=str(TASK_DIR / "results" / "frozen_update.json"))
    return parser.parse_args()


def main() -> None:
    """All three checkpoints, then the verdict; the JSON is rewritten after each checkpoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for tag in CHECKPOINTS:
        results[tag] = run_one(tag, args, device)
        report(tag, results[tag])
        out.write_text(json.dumps(results, indent=1))
    results["verdict"] = verdict(results)
    log.info("verdict: %s", results["verdict"])
    out.write_text(json.dumps(results, indent=1))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
