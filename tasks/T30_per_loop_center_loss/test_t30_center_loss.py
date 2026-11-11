"""T30 checks: prototypes get no gradient, the EMA update matches a hand computation, alpha = 0 gives exactly the T13-c
(every_loop) loss, and the flag checks reject unsupported combinations."""
from types import SimpleNamespace

import pytest
import timm
import torch
from torch import nn

from src.center_loss import center_loss, ema_update, loop_features
from src.models import MODEL_NAME, build_unrolled
from src.per_loop_readout import per_loop_logits_and_cls
from src.train import check_options, forward_and_center_loss, forward_and_loss

DIM = 24
IMAGES = torch.randn(4, 3, 32, 32)
LABELS = torch.tensor([0, 1, 1, 3])


def tiny_unrolled() -> nn.Module:
    """Small random T11-d-style model (B0, B1, B2 x10)."""
    torch.manual_seed(0)
    source = timm.create_model(MODEL_NAME, pretrained=False, depth=12, embed_dim=DIM, num_heads=3,
                               img_size=32, patch_size=8, num_classes=10)
    return build_unrolled(3, source=source).eval()


def test_prototypes_get_no_gradient() -> None:
    model = tiny_unrolled()
    prototypes = torch.randn(10, DIM)
    _, cls_states = per_loop_logits_and_cls(model, IMAGES)
    loss = center_loss(loop_features(model, cls_states), LABELS, prototypes)
    loss.backward()
    assert not prototypes.requires_grad and prototypes.grad is None
    assert list(model.blocks)[2].attn.qkv.weight.grad.abs().sum() > 0


def test_ema_matches_hand_computation() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    features = torch.tensor([[3.0, 1.0], [1.0, 1.0], [5.0, 3.0]])
    labels = torch.tensor([0, 0, 1])
    updated = ema_update(prototypes, features, labels)
    expected = torch.tensor([[0.9 * 1 + 0.1 * 2.0, 0.9 * 0 + 0.1 * 1.0],
                             [0.9 * 0 + 0.1 * 5.0, 0.9 * 1 + 0.1 * 3.0],
                             [2.0, 2.0]])  # class 2 absent: unchanged
    assert torch.allclose(updated, expected)
    assert torch.equal(prototypes, torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]))  # input not changed


def test_center_loss_by_hand() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    features = [torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([[0.0, 2.0], [0.0, 3.0]])]
    labels = torch.tensor([0, 1])
    # loop 1: cos 1 and 0 -> mean(0, 1) = 0.5; loop 2: cos 0 and 1 -> 0.5; mean 0.5
    assert center_loss(features, labels, prototypes).item() == pytest.approx(0.5)


@torch.no_grad()
def test_alpha_zero_equals_t13c_loss() -> None:
    model = tiny_unrolled()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    _, reference, _ = forward_and_loss(model, IMAGES, LABELS, criterion, "every_loop")
    _, loss, _, center, z_final = forward_and_center_loss(model, IMAGES, LABELS, criterion, "every_loop",
                                                          torch.randn(10, DIM), alpha=0.0)
    assert torch.equal(loss, reference)
    assert center > 0 and z_final.shape == (4, DIM)


@torch.no_grad()
def test_alpha_adds_weighted_center_term() -> None:
    model = tiny_unrolled()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    prototypes = torch.randn(10, DIM)
    _, reference, _ = forward_and_loss(model, IMAGES, LABELS, criterion, "every_loop")
    _, loss, _, center, _ = forward_and_center_loss(model, IMAGES, LABELS, criterion, "every_loop", prototypes, 0.1)
    assert loss.item() == pytest.approx(reference.item() + 0.1 * center.item(), rel=1e-6)


def options(**changes) -> SimpleNamespace:
    """Default train options with `changes`."""
    base = dict(model="unroll", cls_only_loop=False, concat_inject=False, random_loops=False,
                center_loss_weight=0.0, readout_loss="none", cls_update_loss="none")
    return SimpleNamespace(**{**base, **changes})


def test_center_loss_needs_every_loop_readout() -> None:
    check_options(options(center_loss_weight=0.1, readout_loss="every_loop"))
    with pytest.raises(ValueError):
        check_options(options(center_loss_weight=0.1))
    with pytest.raises(ValueError):
        check_options(options(center_loss_weight=0.1, readout_loss="every_loop", cls_update_loss="orth"))
