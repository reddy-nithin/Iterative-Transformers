"""T23: per-loop gradient norms of the shared block for b0-init, b0-trained, d-init, d-trained (no training).

For each model: top-1 sanity check (trained checkpoints), the gate (sum of per-loop gradients equals the shared
gradient, unrolled logits equal normal logits), then 20 batches of 128 train images: ‖g_t‖ per loop for the whole block
and each parameter group, spread / dominance / agreement / pairwise cos, and ‖x_t‖ per loop.

Output: results/grad_norms.json.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python tasks/T23_per_loop_grad_norms/grad_norms.py [--batches N] [--test-limit N]
"""
import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100

TASK_DIR = Path(__file__).resolve().parent
TASKS = TASK_DIR.parent
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(TASKS / "T5_cls_svd_per_block"))
sys.path.insert(0, str(TASKS / "T13_per_loop_readout_loss"))

from cls_svd import load_model  # noqa: E402
from exit_analysis import check_top1, test_loader  # noqa: E402
from unroll import WHOLE_BLOCK, gate, group_metrics, parameter_groups, per_loop_gradients, shared_block, summarise  # noqa: E402,E501

from src.models import build  # noqa: E402
from src.train import set_seed  # noqa: E402

CHECKPOINTS = Path("/storage/INSLAB/Data/nrs3qy/checkpoints/itertf")
TRAIN_SIZE = 50_000
DATA_SEED = 0
INIT_SEED = 0  # src/train.py --seed default; set_seed runs right before build there
LABEL_SMOOTHING = 0.1
GATE_TOLERANCE = 1e-4
LOGITS_TOLERANCE = 1e-4
WHOLE_SPREAD_THRESHOLD = 3.0
LAYERNORM_SPREAD_THRESHOLD = 5.0
INIT_MODELS = ("b0-init", "d-init")
LAYERNORM_GROUPS = ("norm1", "norm2")

log = logging.getLogger("t23")


def init_builder(name: str, **kwargs) -> Callable[[], nn.Module]:
    """A fresh model whose new classifier comes from set_seed(INIT_SEED) right before build, as in src/train.py."""
    def make() -> nn.Module:
        set_seed(INIT_SEED)
        return build(name, **kwargs)
    return make


MODELS: dict[str, tuple[Callable[[], nn.Module], str | None]] = {
    "b0-init": (init_builder("iterative", block_idx=0, init="block"), None),
    "b0-trained": (lambda: load_model(str(CHECKPOINTS / "T5-b_iter_b0.pt")), str(CHECKPOINTS / "T5-b_iter_b0.pt")),
    "d-init": (init_builder("unroll", unique_blocks=3), None),
    "d-trained": (lambda: load_model(str(CHECKPOINTS / "T11-d_unroll_n3_30ep.pt")),
                  str(CHECKPOINTS / "T11-d_unroll_n3_30ep.pt")),
}


def check_init_classifier_reproducible(make: Callable[[], nn.Module]) -> None:
    """Two builds after the same seed must give a bit-identical classifier (else the 'init' is not what training saw)."""
    first, second = make().head, make().head
    if not (torch.equal(first.weight, second.weight) and torch.equal(first.bias, second.bias)):
        raise RuntimeError("set_seed + build does not reproduce the classifier bit for bit")


def train_batches(model: nn.Module, args: argparse.Namespace) -> DataLoader:
    """The first `batches` x `batch_size` images of a seed-0 permutation of the train set, test transform."""
    transform = create_transform(**resolve_data_config({}, model=model), is_training=False)
    dataset = CIFAR100(args.data_root, train=True, download=True, transform=transform)
    order = torch.randperm(TRAIN_SIZE, generator=torch.Generator().manual_seed(DATA_SEED))
    subset = Subset(dataset, order[: args.batches * args.batch_size].tolist())
    return DataLoader(subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)


@torch.no_grad()
def test_top1(model: nn.Module, args: argparse.Namespace, device: torch.device) -> float:
    """Top-1 (%) of the model's own head on the test set (or its first --test-limit images)."""
    loader_args = argparse.Namespace(data_root=args.data_root, batch_size=256, num_workers=args.num_workers,
                                     limit=args.test_limit)
    correct, seen = 0, 0
    for images, labels in test_loader(model, loader_args):
        correct += (model(images.to(device)).argmax(dim=1).cpu() == labels).sum().item()
        seen += len(labels)
    return 100.0 * correct / seen


def run_gate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, tag: str) -> dict:
    """The one-batch gate; raises (T23 stops) if it fails."""
    images, labels = next(iter(loader))
    result = gate(model, images.to(device), labels.to(device), criterion)
    log.info("[%s] gate: grad rel error %.2e, logits max diff %.2e", tag, result["grad_rel_error"],
             result["logits_max_abs_diff"])
    if result["grad_rel_error"] > GATE_TOLERANCE or result["logits_max_abs_diff"] > LOGITS_TOLERANCE:
        raise RuntimeError(f"[{tag}] gate failed: {result}")
    return result


def measure(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    """Per-group summaries and ‖x_t‖ per loop over all batches."""
    shared, positions = shared_block(list(model.blocks))
    groups = parameter_groups(shared)
    per_group: dict[str, list[dict]] = {group: [] for group in groups}
    residual: list[list[float]] = []
    for images, labels in loader:
        per_loop, norms, _ = per_loop_gradients(model, images.to(device), labels.to(device), criterion)
        residual.append(norms)
        for group, names in groups.items():
            per_group[group].append(group_metrics(per_loop, names))
    return {"loops": len(positions), "shared_positions": positions,
            "residual_norm_per_loop": torch.tensor(residual).mean(dim=0).tolist(),
            "groups": {group: summarise(batches) for group, batches in per_group.items()}}


def decision_line(results: dict) -> str:
    """The README decision line from the two init models."""
    parts, run = [], False
    for tag in INIT_MODELS:
        groups = results[tag]["groups"]
        whole = groups[WHOLE_BLOCK]["spread"]
        layernorm_group = max(LAYERNORM_GROUPS, key=lambda group: groups[group]["spread"])
        layernorm = groups[layernorm_group]["spread"]
        run = run or whole >= WHOLE_SPREAD_THRESHOLD or layernorm >= LAYERNORM_SPREAD_THRESHOLD
        parts.append(f"{tag}: whole-block spread {whole:.2f}, largest LayerNorm spread {layernorm:.2f} "
                     f"({layernorm_group})")
    return f"{'run' if run else 'skip'} H10 step 2 ({'; '.join(parts)})"


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-limit", type=int, default=0, help="first N test images for the top-1 check (smoke)")
    parser.add_argument("--out-dir", default=str(TASK_DIR / "results"))
    return parser.parse_args()


def main() -> None:
    """All four models, then the decision line."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    results = {}
    for tag, (make, checkpoint) in MODELS.items():
        if checkpoint is None:
            check_init_classifier_reproducible(make)
        model = make().to(device).float().eval()
        top1 = None
        if checkpoint is not None:
            top1 = test_top1(model, args, device)
            check_top1(top1, checkpoint, full_test_set=args.test_limit == 0)
        loader = train_batches(model, args)
        gate_result = run_gate(model, loader, criterion, device, tag)
        results[tag] = {"checkpoint": checkpoint, "top1": top1, "gate": gate_result,
                        **measure(model, loader, criterion, device)}
        whole = results[tag]["groups"][WHOLE_BLOCK]
        log.info("[%s] %d loops; whole block ‖g_t‖ %s; spread %.2f, dominance %.3f", tag, results[tag]["loops"],
                 " ".join(f"{value:.3g}" for value in whole["norm_per_loop"]), whole["spread"], whole["dominance"])
        del model
        torch.cuda.empty_cache()
    results["decision"] = decision_line(results)
    log.info("DECISION: %s", results["decision"])
    (out_dir / "grad_norms.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
