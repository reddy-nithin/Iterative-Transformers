"""Tests for T26's CLS-start swap: identity, parameters untouched, equivalence with replacing cls_token."""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cls_start import cls_start, constant_offset, learned_token, run_stages, table_offset  # noqa: E402

from src.models import build_unrolled  # noqa: E402
from src.per_loop_readout import per_loop_logits_and_cls  # noqa: E402


def tiny_model() -> torch.nn.Module:
    """A randomly initialised T11-d-layout model (unrolling n = 3), eval mode."""
    torch.manual_seed(0)
    model = build_unrolled(3, pretrained=False).eval()
    with torch.no_grad():
        model.cls_token.normal_()  # timm inits it near zero; make the swap tests meaningful
    return model


IMAGES = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(1))


@torch.no_grad()
def test_zero_offset_gives_identical_logits() -> None:
    model = tiny_model()
    reference = model(IMAGES)
    with cls_start(model, constant_offset(torch.zeros(192))):
        shifted = model(IMAGES)
    assert torch.allclose(reference, shifted, atol=1e-6)


@torch.no_grad()
def test_parameters_bit_identical_after_use() -> None:
    model = tiny_model()
    before = copy.deepcopy(model.state_dict())
    with cls_start(model, constant_offset(-learned_token(model))):
        model(IMAGES)
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert "_pos_embed" not in vars(model)  # the class method is back


@torch.no_grad()
def test_zeros_offset_equals_setting_cls_token_to_zero() -> None:
    model = tiny_model()
    with cls_start(model, constant_offset(-learned_token(model))):
        shifted = model(IMAGES)
    zeroed = copy.deepcopy(model)
    zeroed.cls_token.zero_()
    assert torch.allclose(shifted, zeroed(IMAGES), atol=1e-5)


@torch.no_grad()
def test_per_image_offset_follows_image_index() -> None:
    model = tiny_model()
    table = torch.randn(2, 192, generator=torch.Generator().manual_seed(2))
    with cls_start(model, table_offset(table, minus=None)):
        both = model(IMAGES)
    for i in range(2):
        single = copy.deepcopy(model)
        single.cls_token += table[i]
        assert torch.allclose(both[i], single(IMAGES[i:i + 1])[0], atol=1e-5)


@torch.no_grad()
def test_run_stages_matches_per_loop_readout() -> None:
    model = tiny_model()
    logits, cls = run_stages(model, IMAGES)
    reference_logits, reference_cls = per_loop_logits_and_cls(model, IMAGES)
    assert cls.shape == (12, 2, 192)
    assert torch.allclose(logits, torch.stack(reference_logits), atol=1e-5)
    assert torch.allclose(cls[1:], torch.stack(reference_cls), atol=1e-5)  # after B1 = body input, then loops
