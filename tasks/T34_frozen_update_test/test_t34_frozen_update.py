"""T34 checks on a tiny random 2+10 model: k = 10 gives the normal output, k = 9 adds d_9 once, and the frozen path
never calls B2 after loop k.

Run from the repo root:  PYTHONPATH=. .venv/bin/python -m pytest tasks/T34_frozen_update_test -q
"""
import sys
from pathlib import Path

import pytest
import timm
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_update import FROZEN_KS, frozen_final_state, frozen_logits, head_state, normal_pass  # noqa: E402

from src.models import MODEL_NAME, build_unrolled  # noqa: E402
from src.per_loop_readout import split_loop  # noqa: E402

IMAGES = torch.randn(2, 3, 32, 32)


def tiny_unrolled() -> nn.Module:
    """Small random T11-d-style model (B0, B1, B2 x10)."""
    torch.manual_seed(0)
    source = timm.create_model(MODEL_NAME, pretrained=False, depth=12, embed_dim=24, num_heads=3,
                               img_size=32, patch_size=8, num_classes=10)
    return build_unrolled(3, source=source).eval()


@torch.no_grad()
def test_k10_equals_normal_loop_10() -> None:
    model = tiny_unrolled()
    x0 = head_state(model, IMAGES)
    frozen = frozen_logits(model, x0, 10)
    assert torch.allclose(frozen, normal_pass(model, x0, 10)["logits"][9], atol=1e-5, rtol=0)
    assert torch.allclose(frozen, model(IMAGES), atol=1e-5, rtol=0)


@torch.no_grad()
def test_k9_adds_d9_once() -> None:
    model = tiny_unrolled()
    body = split_loop(model)[1]
    x0 = head_state(model, IMAGES)
    states = [x0]
    for _ in range(9):
        states.append(body(states[-1]))
    expected = states[9] + (states[9] - states[8])
    assert torch.allclose(frozen_final_state(model, x0, 9), expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("k", FROZEN_KS)
@torch.no_grad()
def test_no_body_calls_after_k(k: int) -> None:
    model = tiny_unrolled()
    body = split_loop(model)[1]
    x0 = head_state(model, IMAGES)
    calls = []
    handle = body.register_forward_hook(lambda module, args, output: calls.append(1))
    try:
        frozen_logits(model, x0, k)
    finally:
        handle.remove()
    assert len(calls) == k
