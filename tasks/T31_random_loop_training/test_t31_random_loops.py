"""T31 checks: the loop-count sampler's mean and cap, flag off means 10 loops, and running the body r = 10 times
equals the normal model output.

The task text says the mean of 10,000 draws is 10 +- 0.4. Its own formula (tau ~ N(log 10 - 0.125, 0.5),
r = 1 + Poisson(e^tau), cap 20) has mean about 10.55 (1M-draw simulation, 2026-11-11), so the test checks
10.55 +- 0.4 (Nithin's decision, gotcha 2026-11-11)."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import timm
import torch
from torch import nn

from src.models import MODEL_NAME, build_unrolled
from src.random_loops import MAX_LOOPS, forward_loops, per_loop_logits_upto, sample_loops
from src.train import check_options, parse_args

EXPECTED_MEAN = 10.55
IMAGES = torch.randn(2, 3, 32, 32)


def tiny_unrolled(**options) -> nn.Module:
    """Small random T11-d-style model (B0, B1, B2 x10)."""
    torch.manual_seed(0)
    source = timm.create_model(MODEL_NAME, pretrained=False, depth=12, embed_dim=24, num_heads=3,
                               img_size=32, patch_size=8, num_classes=10)
    return build_unrolled(3, source=source, **options).eval()


def test_sampler_mean_and_cap() -> None:
    rng = np.random.default_rng(0)
    draws = np.array([sample_loops(rng) for _ in range(10_000)])
    assert abs(draws.mean() - EXPECTED_MEAN) <= 0.4
    assert draws.max() <= MAX_LOOPS and draws.min() >= 1


def test_sampler_is_reproducible() -> None:
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    assert [sample_loops(rng_a) for _ in range(50)] == [sample_loops(rng_b) for _ in range(50)]


def test_flag_off_means_ten_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--model", "unroll", "--unique-blocks", "3"])
    assert parse_args().random_loops is False
    assert len(list(tiny_unrolled().blocks)) == 12  # B0, B1, B2 x10: model(x) always runs 10 loops


@torch.no_grad()
def test_ten_loops_equal_model_output() -> None:
    model = tiny_unrolled()
    assert torch.equal(forward_loops(model, IMAGES, 10), model(IMAGES))


@torch.no_grad()
def test_per_loop_logits_upto_extends_past_ten() -> None:
    model = tiny_unrolled()
    logits = per_loop_logits_upto(model, IMAGES, 20)
    assert len(logits) == 20 and torch.equal(logits[9], model(IMAGES))
    assert torch.equal(logits[4], forward_loops(model, IMAGES, 5))


def test_stateful_body_rejected() -> None:
    with pytest.raises(ValueError):
        forward_loops(tiny_unrolled(concat_inject=True), IMAGES, 12)


def test_random_loops_needs_final_only_unroll() -> None:
    base = dict(model="unroll", cls_only_loop=False, concat_inject=False, random_loops=True,
                center_loss_weight=0.0, readout_loss="none", cls_update_loss="none")
    check_options(SimpleNamespace(**base))
    with pytest.raises(ValueError):
        check_options(SimpleNamespace(**{**base, "readout_loss": "every_loop"}))
    with pytest.raises(ValueError):
        check_options(SimpleNamespace(**{**base, "model": "hbt"}))
