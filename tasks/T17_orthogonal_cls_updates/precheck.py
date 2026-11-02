"""T17 check before training: the loss code's ||d_t|| / ||CLS_{t-1}|| on the T13-c checkpoint must match T12's rel_cls
for T13-c within 0.01 per loop (same 1,000 test images as T12, eval mode, fp32). Also prints the two T17 terms at the
T13-c weights, so their size next to the CE terms is known before training.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T17_orthogonal_cls_updates/precheck.py
"""
import json
import sys
from pathlib import Path

import torch

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent / "T12_per_loop_cls_diagnostics"))

import loop_diagnostics as t12  # noqa: E402
from src.cls_update_loss import (cls_updates, orthogonal_penalty, relative_sizes, size_floor_penalty,  # noqa: E402
                                 successive_cos)
from src.per_loop_readout import per_loop_logits_and_cls  # noqa: E402

CHECKPOINT = "/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/T13-c_unroll_readout_every_loop.pt"
T12_DIAG = TASK_DIR.parent / "T13_per_loop_readout_loss/results/t12_diag/diag_T13-c_unroll_readout_every_loop.json"
TOLERANCE = 0.01


@torch.no_grad()
def main() -> None:
    """Per-loop rel size and successive cos from the loss code, compared with T12; writes results/precheck.json."""
    sys.argv = [sys.argv[0], CHECKPOINT]  # T12's options (1,000 images, seed 0) with their defaults
    args = t12.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = t12.load_model(CHECKPOINT).to(device).float().eval()
    loader, _ = t12.test_loaders(model, args)
    rel, cos, orth, size, n = 0.0, 0.0, 0.0, 0.0, 0
    for images, _ in loader:
        _, cls_states = per_loop_logits_and_cls(model, images.to(device))
        updates = cls_updates(cls_states)
        batch = images.shape[0]
        rel = rel + relative_sizes(cls_states, updates).sum(dim=1).cpu()
        cos = cos + successive_cos(updates).sum(dim=1).cpu()
        orth += orthogonal_penalty(updates).item() * batch
        size += size_floor_penalty(cls_states, updates).item() * batch
        n += batch
    rel_ours = [round(v / n, 4) for v in rel.tolist()]
    t12_means = json.loads(T12_DIAG.read_text())["means"]
    worst = max(abs(a - b) for a, b in zip(rel_ours, t12_means["rel_cls"]))
    result = {"checkpoint": CHECKPOINT, "n_images": n, "rel_cls_ours": rel_ours, "rel_cls_t12": t12_means["rel_cls"],
              "max_abs_diff": round(worst, 4), "matches": worst <= TOLERANCE,
              "successive_cos_ours": [round(v / n, 4) for v in cos.tolist()],
              "successive_cos_t12": t12_means["cls_update_cos"],
              "orthogonal_term_at_t13c": round(orth / n, 4), "size_floor_term_at_t13c": round(size / n, 4)}
    print(json.dumps(result, indent=1), flush=True)
    (TASK_DIR / "results").mkdir(exist_ok=True)
    (TASK_DIR / "results" / "precheck.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
