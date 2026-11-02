"""T20 checks: with the CLS row's attention uniform over the patches and 0 on CLS, the patches add nothing to the CLS
output; patch rows are identical with the change on or off; no new parameters; the flag is added only when on."""
from types import SimpleNamespace

import pytest
import timm
import torch
from torch import nn

from src.models import MODEL_NAME, build_unrolled
from src.train import model_kwargs
from src.zero_mean_value import ZeroMeanValueBlock, zero_mean_block_parts, zero_mean_mix

DIM = 24


def tiny_vit() -> nn.Module:
    """Small random ViT with the ViT-Tiny structure (17 tokens: CLS + 16 patches, 3 heads)."""
    torch.manual_seed(0)
    return timm.create_model(MODEL_NAME, pretrained=False, depth=12, embed_dim=DIM, num_heads=3,
                             img_size=32, patch_size=8, num_classes=10).eval()


def test_uniform_patch_read_adds_zero_to_cls() -> None:
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(2, 3, 17, 17), dim=-1)
    probs[:, :, 0] = 0.0
    probs[:, :, 0, 1:] = 1 / 16  # CLS row: uniform over the 16 patches, 0 on CLS
    v = torch.randn(2, 3, 17, 8)
    mixed = zero_mean_mix(probs, v)
    assert mixed[:, :, 0].abs().max() < 1e-5
    assert torch.equal(mixed[:, :, 1:], (probs @ v)[:, :, 1:])  # patch query rows unchanged


def test_patch_rows_identical_cls_row_changed() -> None:
    block = tiny_vit().blocks[2]
    x = torch.randn(4, 17, DIM)
    with torch.no_grad():
        on, off = zero_mean_block_parts(block, x)["output"], block(x)
    assert (on[:, 1:] - off[:, 1:]).abs().max() < 1e-5
    assert (on[:, 0] - off[:, 0]).abs().max() > 1e-3


def test_cls_self_value_kept() -> None:
    probs = torch.zeros(1, 1, 3, 3)
    probs[:, :, :, 0] = 1.0  # every row reads only CLS
    v = torch.randn(1, 1, 3, 4)
    assert torch.allclose(zero_mean_mix(probs, v)[:, :, 0], v[:, :, 0])


def test_model_has_no_new_parameters_and_wraps_only_the_looped_block() -> None:
    plain = build_unrolled(3, source=tiny_vit())
    model = build_unrolled(3, source=tiny_vit(), zero_mean_value=True)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in plain.parameters())
    blocks = list(model.blocks)
    assert not isinstance(blocks[0], ZeroMeanValueBlock) and not isinstance(blocks[1], ZeroMeanValueBlock)
    assert all(b is blocks[2] and isinstance(b, ZeroMeanValueBlock) for b in blocks[2:]) and len(blocks) == 12


def test_two_options_at_once_is_an_error() -> None:
    with pytest.raises(ValueError):
        build_unrolled(3, source=tiny_vit(), zero_mean_value=True, coverage_mask=True)


def test_flag_only_added_when_set() -> None:
    args = SimpleNamespace(model="unroll", unique_blocks=3, coverage_mask=False, coverage_mask_v2_eta=None,
                           zero_mean_value=False)
    assert model_kwargs(args) == {"unique_blocks": 3}
    args.zero_mean_value = True
    assert model_kwargs(args) == {"unique_blocks": 3, "zero_mean_value": True}
