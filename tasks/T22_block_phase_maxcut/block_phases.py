"""T22: block-to-block similarity of a 12-block ViT-Tiny and contiguous max-cut into phases (no training).

For (A) the pretrained timm model and (B) the fine-tuned baseline, on the CIFAR-100 test images:
  S_tok[l, m] = mean over images and all 197 tokens of cos(h_l, h_m), h_l = residual state after block l;
  S_cls[l, m] = the same on token 0 only.
Then every contiguous split into k = 2, 3, 4 groups is scored (maxcut.py), and our hand-picked T11-c / T11-d cuts are
ranked among the 55 k = 3 options.

Outputs: results/S_{tok,cls}_{pretrained,baseline}.npy and results/block_phases.json.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T22_block_phase_maxcut/block_phases.py [--limit N]
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
sys.path.insert(0, str(TASKS / "T13_per_loop_readout_loss"))

from exit_analysis import check_top1, load_model, test_loader  # noqa: E402
from maxcut import rank_of, ranked_cuts  # noqa: E402

BASELINE = "/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/T5-a_baseline_ft.pt"
HAND_CUTS = {"T11-c": (1, 11), "T11-d": (1, 2)}  # {0},{1..10},{11} and {0},{1},{2..11}
K_VALUES = (2, 3, 4)
TOP_K3 = 5
RANK_THRESHOLD = 6

log = logging.getLogger("t22")


@torch.no_grad()
def similarity_matrices(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """(S_tok, S_cls, top-1 of the model's own head in % on the loader) from running sums over batches."""
    outputs: list[torch.Tensor] = []
    handles = [block.register_forward_hook(lambda module, args, out: outputs.append(out)) for block in model.blocks]
    n_blocks = len(model.blocks)
    tok_sum = torch.zeros(n_blocks, n_blocks, dtype=torch.float64, device=device)
    cls_sum = torch.zeros_like(tok_sum)
    tokens, images_seen, correct = 0, 0, 0
    try:
        for images, labels in loader:
            outputs.clear()
            logits = model(images.to(device, non_blocking=True))
            if len(outputs) != n_blocks:
                raise RuntimeError(f"expected {n_blocks} block outputs, got {len(outputs)}")
            states = nn.functional.normalize(torch.stack(outputs).float(), dim=-1)  # (L, B, T, D)
            tok_sum += torch.einsum("lbtd,mbtd->lm", states, states).double()
            cls_sum += torch.einsum("lbd,mbd->lm", states[:, :, 0], states[:, :, 0]).double()
            tokens += states.shape[1] * states.shape[2]
            images_seen += states.shape[1]
            correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
    finally:
        for handle in handles:
            handle.remove()
    return ((tok_sum / tokens).cpu().numpy(), (cls_sum / images_seen).cpu().numpy(),
            100.0 * correct / images_seen)


def analyse_matrix(similarity: np.ndarray) -> dict:
    """Top-5 k = 3 cuts, best k = 2 / k = 4 cuts and the ranks of our hand-picked cuts."""
    ranked = {k: ranked_cuts(similarity, k) for k in K_VALUES}
    hand = {name: next(row for row in ranked[3] if tuple(row["cut"]) == cut) for name, cut in HAND_CUTS.items()}
    return {"k3_top5": ranked[3][:TOP_K3], "k3_count": len(ranked[3]), "k2_best": ranked[2][0],
            "k4_best": ranked[4][0], "hand_cuts": {name: {**row, "rank": rank_of(ranked[3], HAND_CUTS[name])}
                                                   for name, row in hand.items()}}


def describe(groups: list[list[int]]) -> str:
    """Groups as text, e.g. '{0}, {1..10}, {11}'."""
    return ", ".join(f"{{{g[0]}..{g[-1]}}}" if len(g) > 1 else f"{{{g[0]}}}" for g in groups)


def decision_line(analysis: dict) -> str:
    """The README decision line, from model (A) S_tok."""
    best = analysis["k3_top5"][0]
    rank_c = analysis["hand_cuts"]["T11-c"]["rank"]
    rank_d = analysis["hand_cuts"]["T11-d"]["rank"]
    ranks = f"T11-c cut rank {rank_c}/55, T11-d cut rank {rank_d}/55"
    if tuple(best["cut"]) != HAND_CUTS["T11-c"] and rank_c >= RANK_THRESHOLD:
        return f"run H8 step 2 with groups {describe(best['groups'])} ({ranks})"
    return f"skip H8 step 2 ({ranks}; best {describe(best['groups'])})"


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N test images only (smoke)")
    parser.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    return parser.parse_args()


def main() -> None:
    """Both models: matrices, max-cut tables, decision line."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for tag, checkpoint in (("pretrained", None), ("baseline", BASELINE)):
        model = load_model(checkpoint).to(device).eval()
        s_tok, s_cls, top1 = similarity_matrices(model, test_loader(model, args), device)
        if checkpoint is not None:
            check_top1(top1, checkpoint, full_test_set=args.limit == 0)
        np.save(out_dir / f"S_tok_{tag}.npy", s_tok)
        np.save(out_dir / f"S_cls_{tag}.npy", s_cls)
        result[tag] = {"checkpoint": checkpoint or "timm vit_tiny_patch16_224 (pretrained)",
                       "top1_cifar100": round(top1, 2) if checkpoint else None,
                       "S_tok": analyse_matrix(s_tok), "S_cls": analyse_matrix(s_cls)}
        for kind in ("S_tok", "S_cls"):
            info = result[tag][kind]
            log.info("[%s %s] best k=3 %s (cut score %.3f); T11-c rank %d, T11-d rank %d", tag, kind,
                     describe(info["k3_top5"][0]["groups"]), info["k3_top5"][0]["cut_score"],
                     info["hand_cuts"]["T11-c"]["rank"], info["hand_cuts"]["T11-d"]["rank"])
    result["decision"] = decision_line(result["pretrained"]["S_tok"])
    log.info("DECISION: %s", result["decision"])
    (out_dir / "block_phases.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
