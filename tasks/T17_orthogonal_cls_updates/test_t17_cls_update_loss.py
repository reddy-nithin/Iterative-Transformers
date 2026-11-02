"""T17 checks: the two terms give known values on hand-made CLS paths, gradients reach the looped block, the per-loop
CLS states are the body's real inputs/outputs, and the default (none) leaves the T13 loss unchanged."""
import pytest
import timm
import torch
from torch import nn

from src.cls_update_loss import (LAMBDA_ORTH, LAMBDA_SIZE, R_MIN, cls_update_penalty, cls_updates,
                                 orthogonal_penalty, relative_sizes, size_floor_penalty, successive_cos)
from src.models import MODEL_NAME, build_unrolled
from src.per_loop_readout import per_loop_logits, per_loop_logits_and_cls, readout_loss
from src.train import forward_and_loss

DIM = 24


def tiny_unrolled() -> nn.Module:
    """T11-d layout (b0, b1, b2 x10) on a small random ViT with the ViT-Tiny structure."""
    torch.manual_seed(0)
    source = timm.create_model(MODEL_NAME, pretrained=False, depth=12, embed_dim=DIM, num_heads=3,
                               img_size=32, patch_size=8, num_classes=10)
    return build_unrolled(3, source=source)


def path(*points: list[float]) -> list[torch.Tensor]:
    """CLS states for a batch of 1 from a list of 2-d points."""
    return [torch.tensor([p]) for p in points]


def test_straight_line_is_fully_collinear() -> None:
    states = path([1, 0], [2, 0], [3, 0], [4, 0])
    updates = cls_updates(states)
    assert torch.allclose(successive_cos(updates), torch.ones(2, 1))
    assert orthogonal_penalty(updates).item() == pytest.approx(1.0)


def test_right_angle_turns_cost_nothing_and_reversal_costs_fully() -> None:
    assert orthogonal_penalty(cls_updates(path([1, 0], [2, 0], [2, 1], [1, 1]))).item() == pytest.approx(0.0, abs=1e-6)
    assert orthogonal_penalty(cls_updates(path([1, 0], [2, 0], [1, 0]))).item() == pytest.approx(1.0)  # cos -1, squared


def test_size_floor() -> None:
    states = path([4, 0], [5, 0], [5.5, 0])  # rel sizes 1/4 and 0.5/5
    updates = cls_updates(states)
    assert torch.allclose(relative_sizes(states, updates), torch.tensor([[0.25], [0.1]]))
    assert size_floor_penalty(states, updates).item() == pytest.approx((0 + (R_MIN - 0.1)) / 2)


def test_penalty_kinds() -> None:
    states = path([4, 0], [5, 0], [5.5, 0])
    updates = cls_updates(states)
    orth, size = orthogonal_penalty(updates).item(), size_floor_penalty(states, updates).item()
    assert cls_update_penalty(states, "orth").item() == pytest.approx(LAMBDA_ORTH * orth)
    assert cls_update_penalty(states, "orth_large").item() == pytest.approx(LAMBDA_ORTH * orth + LAMBDA_SIZE * size)
    with pytest.raises(ValueError):
        cls_update_penalty(states, "none")


def test_cls_states_match_the_body_and_logits_unchanged() -> None:
    model = tiny_unrolled().eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        logits, cls_states = per_loop_logits_and_cls(model, images)
        x = model.blocks[1](model.blocks[0](model.norm_pre(model._pos_embed(model.patch_embed(images)))))
        assert len(cls_states) == 11 and torch.allclose(cls_states[0], x[:, 0])
        assert torch.allclose(cls_states[1], model.blocks[2](x)[:, 0], atol=1e-6)
        assert all(torch.equal(a, b) for a, b in zip(logits, per_loop_logits(model, images)))


def test_gradient_reaches_the_looped_block() -> None:
    model = tiny_unrolled().train()
    _, cls_states = per_loop_logits_and_cls(model, torch.randn(2, 3, 32, 32))
    cls_update_penalty(cls_states, "orth_large").backward()
    assert model.blocks[2].attn.qkv.weight.grad.abs().sum() > 0


def test_default_none_keeps_the_t13_loss() -> None:
    model = tiny_unrolled().eval()
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    with torch.no_grad():
        _, loss_default, _ = forward_and_loss(model, x, y, criterion, "every_loop")
        expected = readout_loss(per_loop_logits(model, x), y, "every_loop")
        _, loss_orth, _ = forward_and_loss(model, x, y, criterion, "every_loop", "orth")
    assert torch.allclose(loss_default, expected)
    assert loss_orth > loss_default
    with pytest.raises(ValueError):
        forward_and_loss(model, x, y, criterion, "none", "orth")
