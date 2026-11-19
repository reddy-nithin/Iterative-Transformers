"""T35: how much could per-image early stopping ever gain on T31-a? Best-case (oracle) test, no training.

Step 1  store per-loop outputs of T31-a for t = 1..10 (logits fp16, z_t = final_norm(x_t)[:, 0] fp16, raw CLS norm,
        labels) and check top-1 per loop from the STORED logits against the logged T31-a values (0.05 pt).
Step 2  per-loop table: top-1, mean cross-entropy (no label smoothing), drop in mean CE from t-1 to t.
Step 3  settling loop: smallest t whose prediction and every later one equal the loop-10 prediction.
Step 4  oracle: e = first loop whose prediction is correct; greedy curve (mean loops, accuracy); accuracy at mean loops
        2, 3, 4; mean loops to reach 68.3 and 70.9; headroom at 3 = oracle accuracy at mean loops 3 - 68.3.
Step 5  AUROC of five signals for "the loop-t prediction is correct", t = 2..10 and pooled over t = 2..10.
All steps after step 1 read the stored file, so they can be redone from it. No gate is built or trained.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T35_halting_headroom/halting.py [--limit N] [--outputs PATH] [--out PATH]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
sys.path.insert(0, str(TASKS / "T5_cls_svd_per_block"))
sys.path.insert(0, str(TASKS / "T13_per_loop_readout_loss"))

from cls_svd import load_model  # noqa: E402
from exit_analysis import test_loader  # noqa: E402

from src.per_loop_readout import embed, split_loop  # noqa: E402

CHECKPOINT_DIR = Path("/storage/INSLAB/Data/nrs3qy/checkpoints/itertf")
CHECKPOINT = CHECKPOINT_DIR / "T31-a_random_loops.pt"
OUTPUTS_FILE = CHECKPOINT_DIR / "T35_T31-a_per_loop_outputs.pt"
LOGGED_DEPTH_JSON = TASKS / "T31_random_loop_training/results/depth_sweep.json"  # exact T31-a top-1, loops 1..20
N_LOOPS = 10
TOLERANCE = 0.05  # points
BUDGETS = (2, 3, 4)  # mean loops
FIXED_DEPTH_TARGETS = {"fixed_depth_3": 68.3, "fixed_depth_4": 70.9}  # task numbers, used as written
HEADROOM_BUDGET, HEADROOM_REFERENCE, HEADROOM_THRESHOLD = 3, 68.3, 3.0
DECISION_NO_ROOM = "Halting: no room to train a gate on this model; keep as an explained negative."
DECISION_WORTH_IT = "Halting: worth a gate (next: fit a small gate on the stored outputs)."
SIGNALS = ("margin", "max_softmax", "neg_kl", "neg_angular_step", "log_cls_norm")
EPS = 1e-9

log = logging.getLogger("t35")


# ---------- step 1: collect and store ----------

@torch.no_grad()
def collect_outputs(model: nn.Module, loader, device: torch.device, n_loops: int = N_LOOPS) -> dict[str, torch.Tensor]:
    """Per-loop logits (L, N, classes) fp16, z (L, N, C) fp16, raw CLS norm (L, N) fp32 and labels (N,)."""
    head, body, tail, _ = split_loop(model)
    if tail or hasattr(body, "reset"):
        raise ValueError("T35 needs a plain 2+r model (no tail, no stateful body)")
    logits, z, cls_norm, labels = [], [], [], []
    for images, targets in loader:
        x = embed(model, images.to(device, non_blocking=True))
        for block in head:
            x = block(x)
        batch_logits, batch_z, batch_norm = [], [], []
        for _ in range(n_loops):
            x = body(x)
            normed = model.norm(x)
            batch_logits.append(model.forward_head(normed).half().cpu())
            batch_z.append(normed[:, 0].half().cpu())
            batch_norm.append(x[:, 0].float().norm(dim=-1).cpu())
        logits.append(torch.stack(batch_logits))
        z.append(torch.stack(batch_z))
        cls_norm.append(torch.stack(batch_norm))
        labels.append(targets)
    return {"logits": torch.cat(logits, dim=1), "z": torch.cat(z, dim=1), "cls_norm": torch.cat(cls_norm, dim=1),
            "labels": torch.cat(labels)}


def logged_top1() -> list[float]:
    """T31-a's logged top-1 at loops 1..N_LOOPS (fp32 eval, T31 depth sweep)."""
    return json.loads(LOGGED_DEPTH_JSON.read_text())["T31-a"]["acc_per_loops"][:N_LOOPS]


def sanity_check(top1: list[float]) -> dict:
    """Per-loop top-1 from the stored logits against the logged values."""
    logged = logged_top1()
    diffs = [round(abs(ours - theirs), 2) for ours, theirs in zip(top1, logged)]
    return {"logged": logged, "ours": top1, "abs_diff": diffs, "passed": max(diffs) <= TOLERANCE + EPS}


# ---------- steps 2 to 4: pure functions on the stored tensors ----------

def correct_per_loop(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """(L, N) bool: is the loop-t prediction correct."""
    return logits.argmax(dim=-1) == labels.unsqueeze(0)


def per_loop_table(logits: torch.Tensor, labels: torch.Tensor) -> list[dict]:
    """Top-1, mean CE (no label smoothing, fp32) and the CE drop from t-1 to t, for t = 1..L."""
    rows, previous_ce = [], None
    for t in range(logits.shape[0]):
        ce = float(F.cross_entropy(logits[t].float(), labels))
        top1 = 100.0 * float((logits[t].argmax(dim=-1) == labels).float().mean())
        drop = None if previous_ce is None else round(previous_ce - ce, 4)
        rows.append({"loop": t + 1, "top1": round(top1, 2), "mean_ce": round(ce, 4), "ce_drop": drop})
        previous_ce = ce
    return rows


def settling_loops(predictions: torch.Tensor) -> torch.Tensor:
    """(N,) smallest t (1-based) such that the predictions at loops t..L all equal the loop-L prediction."""
    n_loops = predictions.shape[0]
    disagree = predictions != predictions[-1].unsqueeze(0)  # (L, N)
    loop_numbers = torch.arange(1, n_loops + 1).unsqueeze(1)
    last_disagreement = (disagree.long() * loop_numbers).max(dim=0).values  # 0 if never
    return last_disagreement + 1


def settling_summary(settle: torch.Tensor, correct_last: torch.Tensor, n_loops: int = N_LOOPS) -> dict:
    """Histogram over 1..L, mean, standard deviation (population), mean for images correct / wrong at loop L."""
    settle_f = settle.double()
    return {
        "histogram": {str(t): int((settle == t).sum()) for t in range(1, n_loops + 1)},
        "mean": round(float(settle_f.mean()), 3),
        "std": round(float(settle_f.std(unbiased=False)), 3),
        "mean_correct_at_last": round(float(settle_f[correct_last].mean()), 3),
        "mean_wrong_at_last": round(float(settle_f[~correct_last].mean()), 3),
    }


def first_correct_loop(correct: torch.Tensor) -> torch.Tensor:
    """(N,) e = earliest loop (1-based) whose prediction is correct; 0 means never correct."""
    n_loops = correct.shape[0]
    loop_numbers = torch.arange(1, n_loops + 1).unsqueeze(1).expand_as(correct)
    never = n_loops + 1
    earliest = torch.where(correct, loop_numbers, torch.full_like(loop_numbers, never)).min(dim=0).values
    return torch.where(earliest == never, torch.zeros_like(earliest), earliest)


def oracle_curve(first_correct: torch.Tensor) -> list[tuple[float, float]]:
    """Greedy oracle curve [(mean loops over all images, accuracy %)]: every image starts at loop 1 (e = 1 images are
    correct for free); rescuable images are then added in ascending e, each costing e - 1 extra loops."""
    n_images = len(first_correct)
    total_loops, hits = float(n_images), int((first_correct == 1).sum())
    curve = [(total_loops / n_images, 100.0 * hits / n_images)]
    for e in sorted(int(v) for v in first_correct[first_correct > 1]):
        total_loops += e - 1
        hits += 1
        curve.append((total_loops / n_images, 100.0 * hits / n_images))
    return curve


def accuracy_at_budget(curve: list[tuple[float, float]], budget: float) -> float:
    """Accuracy of the last curve point whose mean loops is <= budget."""
    eligible = [acc for loops, acc in curve if loops <= budget + EPS]
    if not eligible:
        raise ValueError(f"no oracle point within {budget} mean loops")
    return eligible[-1]


def loops_to_reach(curve: list[tuple[float, float]], target: float) -> float | None:
    """Mean loops of the first curve point with accuracy >= target; None if the oracle never reaches it."""
    for loops, acc in curve:
        if acc >= target - EPS:
            return loops
    return None


def oracle_summary(first_correct: torch.Tensor) -> dict:
    """Oracle accuracy, accuracy at the budgets, loops to reach the fixed-depth numbers, headroom and decision."""
    curve = oracle_curve(first_correct)
    headroom = accuracy_at_budget(curve, HEADROOM_BUDGET) - HEADROOM_REFERENCE
    reach = {name: loops_to_reach(curve, target) for name, target in FIXED_DEPTH_TARGETS.items()}
    return {
        "oracle_accuracy": round(100.0 * float((first_correct > 0).double().mean()), 2),
        "first_correct_histogram": {str(e): int((first_correct == e).sum()) for e in range(0, N_LOOPS + 1)},
        "accuracy_at_mean_loops": {str(b): round(accuracy_at_budget(curve, b), 2) for b in BUDGETS},
        "mean_loops_to_reach": {name: None if loops is None else round(loops, 3) for name, loops in reach.items()},
        "headroom_at_3": round(headroom, 2),
        "decision": DECISION_WORTH_IT if headroom >= HEADROOM_THRESHOLD - EPS else DECISION_NO_ROOM,
        "curve": [[round(loops, 5), round(acc, 3)] for loops, acc in curve],
    }


# ---------- step 5: signals and AUROC ----------

def signal_scores(logits: torch.Tensor, z: torch.Tensor, cls_norm: torch.Tensor, t: int) -> dict[str, torch.Tensor]:
    """The five scores at loop index t (0-based, t >= 1), oriented so that higher should mean 'correct'."""
    current, previous = logits[t].float(), logits[t - 1].float()
    top2 = current.topk(2, dim=-1).values
    log_p, log_p_previous = F.log_softmax(current, dim=-1), F.log_softmax(previous, dim=-1)
    kl = (log_p.exp() * (log_p - log_p_previous)).sum(dim=-1)  # KL(p_t || p_{t-1})
    angular_step = 1.0 - F.cosine_similarity(z[t].float(), z[t - 1].float(), dim=-1)
    return {
        "margin": top2[:, 0] - top2[:, 1],
        "max_softmax": log_p.exp().max(dim=-1).values,
        "neg_kl": -kl,
        "neg_angular_step": -angular_step,
        "log_cls_norm": cls_norm[t].float().log(),
    }


def auroc(scores: torch.Tensor, correct: torch.Tensor) -> float | None:
    """sklearn AUROC of `scores` for target `correct`; None if only one class is present."""
    if bool(correct.all()) or not bool(correct.any()):
        return None
    return float(roc_auc_score(correct.numpy().astype(int), scores.double().numpy()))


def auroc_table(outputs: dict[str, torch.Tensor]) -> dict:
    """AUROC per signal for t = 2..L and pooled over t = 2..L (equal images per loop, so equal weight per loop)."""
    logits, labels = outputs["logits"], outputs["labels"]
    correct = correct_per_loop(logits, labels)
    per_loop: dict[str, dict] = {name: {} for name in SIGNALS}
    pooled_scores: dict[str, list] = {name: [] for name in SIGNALS}
    for t in range(1, logits.shape[0]):
        scores = signal_scores(logits, outputs["z"], outputs["cls_norm"], t)
        for name in SIGNALS:
            value = auroc(scores[name], correct[t])
            per_loop[name][str(t + 1)] = None if value is None else round(value, 4)
            pooled_scores[name].append(scores[name])
    pooled_target = correct[1:].reshape(-1)
    pooled = {}
    for name in SIGNALS:
        value = auroc(torch.cat(pooled_scores[name]), pooled_target)
        pooled[name] = None if value is None else round(value, 4)
    return {"per_loop": per_loop, "pooled_2_10": pooled}


# ---------- driver ----------

def analyse(outputs: dict[str, torch.Tensor]) -> dict:
    """Steps 2 to 5 on the stored outputs."""
    logits, labels = outputs["logits"], outputs["labels"]
    correct = correct_per_loop(logits, labels)
    return {
        "per_loop": per_loop_table(logits, labels),
        "settling": settling_summary(settling_loops(logits.argmax(dim=-1)), correct[-1]),
        "oracle": oracle_summary(first_correct_loop(correct)),
        "auroc": auroc_table(outputs),
    }


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="first N test images only (smoke; no sanity check)")
    parser.add_argument("--outputs", default=str(OUTPUTS_FILE), help="where the per-loop outputs are stored")
    parser.add_argument("--out", default=str(TASK_DIR / "results" / "halting_headroom.json"))
    return parser.parse_args()


def main() -> None:
    """Store the outputs, reload them, check them against the log, then run steps 2 to 5."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(CHECKPOINT)).to(device).float().eval()
    outputs_path = Path(args.outputs)
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(collect_outputs(model, test_loader(model, args), device), outputs_path)
    outputs = torch.load(outputs_path, map_location="cpu", weights_only=True)
    log.info("stored %s images x %d loops in %s", len(outputs["labels"]), N_LOOPS, outputs_path)

    top1 = [row["top1"] for row in per_loop_table(outputs["logits"], outputs["labels"])]
    result = {"checkpoint": str(CHECKPOINT), "outputs_file": str(outputs_path), "n_images": len(outputs["labels"]),
              "sanity": sanity_check(top1) if args.limit == 0 else None}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if result["sanity"] is not None:
        log.info("sanity (stored logits vs logged, loops 1..10): max diff %.2f, %s", max(result["sanity"]["abs_diff"]),
                 "passed" if result["sanity"]["passed"] else "FAILED")
        if not result["sanity"]["passed"]:
            out.write_text(json.dumps(result, indent=1))
            raise RuntimeError(f"T31-a per-loop top-1 {top1} does not match the logged {result['sanity']['logged']}")
    result.update(analyse(outputs))
    out.write_text(json.dumps(result, indent=1))
    oracle = result["oracle"]
    log.info("per-loop top-1: %s", top1)
    log.info("settling: mean %.2f, std %.2f", result["settling"]["mean"], result["settling"]["std"])
    log.info("oracle acc %.2f; at mean loops 2/3/4: %s; loops to reach: %s", oracle["oracle_accuracy"],
             oracle["accuracy_at_mean_loops"], oracle["mean_loops_to_reach"])
    log.info("headroom at 3: %.2f -> %s", oracle["headroom_at_3"], oracle["decision"])
    log.info("pooled AUROC: %s", result["auroc"]["pooled_2_10"])
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
