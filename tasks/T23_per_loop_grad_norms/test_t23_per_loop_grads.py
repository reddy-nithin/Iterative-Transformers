"""Tests for T23's unrolled per-loop gradients."""
import sys
from pathlib import Path

import pytest
import timm
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unroll import (  # noqa: E402
    apply_blocks, gate, group_metrics, loop_copies, parameter_groups, shared_block, summarise, vit_unrolled_logits,
)

from src.models import MODEL_NAME, build_iterative, build_unrolled  # noqa: E402


class ResidualMLP(nn.Module):
    """x + MLP(x): a toy looped block."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.Tanh(), nn.Linear(2 * width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


def tiny_vit(depth: int = 4) -> nn.Module:
    """A small random ViT-Tiny (32 px, 4 patches) for fast tests."""
    torch.manual_seed(0)
    return timm.create_model(MODEL_NAME, pretrained=False, num_classes=10, img_size=32, depth=depth).eval()


def test_toy_two_loop_sum_of_per_loop_grads_equals_shared_grad() -> None:
    torch.manual_seed(0)
    block = ResidualMLP(6).double()
    x = torch.randn(5, 6, dtype=torch.float64)
    names, params = zip(*block.named_parameters())
    shared = torch.autograd.grad(block(block(x)).pow(2).sum(), params)
    copies = loop_copies(block, 2)
    out, _ = apply_blocks([block, block], block, x, copies)
    flat = [p for copy in copies for p in copy.values()]
    grads = torch.autograd.grad(out.pow(2).sum(), flat)
    for index, reference in enumerate(shared):
        assert torch.allclose(grads[index] + grads[index + len(names)], reference, atol=1e-5)


def test_unrolled_vit_logits_equal_normal_logits_and_gate_passes() -> None:
    model = build_iterative(source=tiny_vit())
    images, labels = torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2])
    shared, positions = shared_block(list(model.blocks))
    logits, norms = vit_unrolled_logits(model, images, shared, loop_copies(shared, len(positions)))
    assert (logits - model(images)).abs().max() <= 1e-5
    assert len(norms) == 4
    result = gate(model, images, labels, nn.CrossEntropyLoss(label_smoothing=0.1))
    assert result["grad_rel_error"] <= 1e-4 and result["logits_max_abs_diff"] <= 1e-5


def test_shared_block_positions_in_unrolled_layout() -> None:
    model = build_unrolled(unique_blocks=2, source=tiny_vit())
    shared, positions = shared_block(list(model.blocks))
    assert positions == [1, 2, 3] and shared is model.blocks[1]


def test_groups_cover_block_and_layerscale_becomes_seventh_group() -> None:
    plain = parameter_groups(tiny_vit().blocks[0])
    assert len(plain) == 7 and len(plain["block"]) == 12  # 6 groups + whole block
    with_ls = timm.create_model(MODEL_NAME, pretrained=False, img_size=32, depth=1, init_values=1e-5).blocks[0]
    assert "layerscale" in parameter_groups(with_ls)
    with pytest.raises(ValueError):
        parameter_groups(nn.Linear(2, 2))


def test_metrics_on_known_gradients() -> None:
    direction = torch.tensor([1.0, 2.0, 2.0])
    per_loop = [{"w": 2 * direction}, {"w": direction}, {"w": direction}]
    batch = group_metrics(per_loop, ["w"])
    assert batch["agreement"] == pytest.approx(1.0) and batch["pairwise_cos"] == pytest.approx(1.0)
    summary = summarise([batch])
    assert summary["spread"] == pytest.approx(2.0)
    assert summary["dominance"] == pytest.approx(9.0 / 12.0)
