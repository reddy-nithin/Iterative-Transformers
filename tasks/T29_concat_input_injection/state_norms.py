"""T29: mean state norm ‖x_t‖ per loop (T9's problem was growth) and the adapter's weight split.

Models: T29-a (concat + adapter), T11-d (no injection) and T9-c (plain-add injection, head 1 / body x10 / tail 1).
For each, forward hooks record the head output x_0 and the looped block's output x_t after loops t = 1..10 on the
CIFAR-100 test set; ‖x_t‖ is the L2 norm of a token, averaged over all 197 tokens and all images (CLS-only mean too).
The model's own top-1 is checked against its logged value first. For T29-a, ‖W_state‖ and ‖W_e‖ (Frobenius norms of
the two halves of the adapter weight) at the end of training, and at initialisation (sqrt(192) = 13.86 and 0).

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T29_concat_input_injection/state_norms.py [--limit N]
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
from exit_analysis import check_top1, test_loader  # noqa: E402

from src.concat_inject import identity_adapter  # noqa: E402
from src.per_loop_readout import split_loop  # noqa: E402

CHECKPOINTS = Path("/storage/INSLAB/Data/nrs3qy/checkpoints/itertf")
MODELS = {"T29-a": CHECKPOINTS / "T29-a_concat_inject.pt",
          "T11-d": CHECKPOINTS / "T11-d_unroll_n3_30ep.pt",
          "T9-c": CHECKPOINTS / "T9-c_hbt_inject.pt"}

log = logging.getLogger("t29_norms")


class NormRecorder:
    """Sums of per-token norms of x_0 (head output) and of every body output, over all batches."""

    def __init__(self, model: nn.Module):
        head, body, _, n_loops = split_loop(model)
        self.n_states = n_loops + 1
        self.token_sums = torch.zeros(self.n_states, dtype=torch.float64)
        self.cls_sums = torch.zeros(self.n_states, dtype=torch.float64)
        self.images = 0
        self.calls: list[torch.Tensor] = []
        self.handles = [head[-1].register_forward_hook(self.save), body.register_forward_hook(self.save)]

    def save(self, module: nn.Module, args: tuple, output: torch.Tensor) -> None:
        """Keep the state's per-token norms (B, N)."""
        self.calls.append(output.float().norm(dim=-1).double().cpu())

    def end_batch(self) -> None:
        """Add this batch's norms to the sums."""
        if len(self.calls) != self.n_states:
            raise RuntimeError(f"expected {self.n_states} recorded states per batch, got {len(self.calls)}")
        norms = torch.stack(self.calls)  # (states, B, N)
        self.token_sums += norms.mean(dim=2).sum(dim=1)
        self.cls_sums += norms[:, :, 0].sum(dim=1)
        self.images += norms.shape[1]
        self.calls.clear()

    def remove(self) -> None:
        """Remove the hooks."""
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def model_norms(model: nn.Module, loader, device: torch.device) -> dict:
    """Top-1 and mean ‖x_t‖ (all tokens, CLS) for t = 0..10."""
    recorder = NormRecorder(model)
    correct = 0
    try:
        for images, labels in loader:
            recorder.calls.clear()
            logits = model(images.to(device, non_blocking=True))
            recorder.end_batch()
            correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
    finally:
        recorder.remove()
    return {"top1": round(100.0 * correct / recorder.images, 2),
            "mean_token_norm": (recorder.token_sums / recorder.images).round(decimals=3).tolist(),
            "mean_cls_norm": (recorder.cls_sums / recorder.images).round(decimals=3).tolist()}


def adapter_split(model: nn.Module) -> dict:
    """‖W_state‖, ‖W_e‖ and ‖bias‖ of the trained adapter, and the same at initialisation."""
    body = split_loop(model)[1]
    init = identity_adapter(body.adapter.out_features)
    dim = init.out_features
    return {"trained": {**body.adapter_norms(), "bias": round(body.adapter.bias.norm().item(), 4)},
            "init": {"w_state": init.weight[:, :dim].norm().item(), "w_e": init.weight[:, dim:].norm().item(),
                     "bias": 0.0}}


def main() -> None:
    """All three models, then the adapter split."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="first N test images only (smoke)")
    parser.add_argument("--out", default=str(TASK_DIR / "results" / "state_norms.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {}
    for tag, checkpoint in MODELS.items():
        model = load_model(str(checkpoint)).to(device).float().eval()
        norms = model_norms(model, test_loader(model, args), device)
        check_top1(norms["top1"], str(checkpoint), full_test_set=args.limit == 0)
        result[tag] = norms
        if tag == "T29-a":
            result["adapter"] = adapter_split(model)
        log.info("[%s] top-1 %.2f  mean token norm x_0..x_10: %s", tag, norms["top1"], norms["mean_token_norm"])
    log.info("adapter: %s", result["adapter"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
