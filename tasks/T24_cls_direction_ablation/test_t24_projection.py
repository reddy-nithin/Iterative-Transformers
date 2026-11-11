"""Tests for T24's direction removal and maths."""
import sys
from pathlib import Path

import numpy as np
import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from projection import (  # noqa: E402
    edited_per_loop, fisher_ratios, random_unit_directions, remove_direction, uncentred_direction,
)

from src.models import MODEL_NAME, build_unrolled  # noqa: E402
from src.per_loop_readout import per_loop_logits_and_cls  # noqa: E402


def tiny_unrolled() -> torch.nn.Module:
    """A small random T11-d-style model: B0, B1, then B2 x4 (32 px images)."""
    torch.manual_seed(0)
    source = timm.create_model(MODEL_NAME, pretrained=False, num_classes=10, img_size=32, depth=6)
    return build_unrolled(unique_blocks=3, source=source).eval()


def test_edited_update_has_no_component_along_direction() -> None:
    torch.manual_seed(0)
    previous, new = torch.randn(4, 5, 8), torch.randn(4, 5, 8)
    direction = random_unit_directions(1, 8, seed=3)[0].float()
    edited = remove_direction(previous, new, direction)
    assert ((edited[:, 0] - previous[:, 0]) @ direction).abs().max() <= 1e-6


def test_patch_rows_unchanged_by_edit() -> None:
    torch.manual_seed(1)
    previous, new = torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    edited = remove_direction(previous, new, random_unit_directions(1, 8, seed=0)[0].float())
    assert torch.equal(edited[:, 1:], new[:, 1:])


@torch.no_grad()
def test_zero_direction_edit_equals_no_edit() -> None:
    model = tiny_unrolled()
    images = torch.randn(3, 3, 32, 32)
    reference_logits, _ = per_loop_logits_and_cls(model, images)
    logits, cls_states = edited_per_loop(model, images, torch.zeros(192), edit_loops={1, 2, 3, 4})
    assert len(logits) == 4 and len(cls_states) == 5
    for ours, theirs in zip(logits, reference_logits):
        assert (ours - theirs).abs().max() <= 1e-6
    unedited, _ = edited_per_loop(model, images, torch.zeros(192), edit_loops=set())
    assert torch.equal(unedited[-1], model(images))


def test_direction_sign_and_energy() -> None:
    rng = np.random.default_rng(0)
    push = np.array([3.0, 4.0, 0.0]) / 5.0
    updates = 5.0 * push + 0.1 * rng.standard_normal((500, 3))
    result = uncentred_direction(updates)
    assert result["energy"][0] > 0.95 and abs(result["u1"] @ push) > 0.99
    assert (updates @ result["u1"]).mean() > 0 and result["cos_u1_mean"] > 0.99


def test_fisher_ratio_high_for_class_coding_low_for_noise() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(10), 50)
    coding = labels + 0.1 * rng.standard_normal(500)
    noise = rng.standard_normal(500)
    ratios = fisher_ratios(np.stack([coding, noise], axis=1), labels)
    assert ratios[0] > 100 and ratios[1] < 0.5
