"""T28 checks: with the CLS-only loop the patch rows equal P (the head output) after every loop, loop 1's CLS equals a
normal block pass's CLS row, the flag off rebuilds the plain T11-d model, no new parameters, kwargs only when on."""
from types import SimpleNamespace

import pytest
import timm
import torch
from torch import nn

from src.cls_only_loop import CLSOnlyBlock
from src.models import MODEL_NAME, build_unrolled
from src.per_loop_readout import embed, split_loop
from src.train import model_kwargs

DIM = 24


def tiny_vit() -> nn.Module:
    """Small random ViT with the ViT-Tiny structure (17 tokens: CLS + 16 patches, 3 heads)."""
    torch.manual_seed(0)
    return timm.create_model(MODEL_NAME, pretrained=False, depth=12, embed_dim=DIM, num_heads=3,
                             img_size=32, patch_size=8, num_classes=10).eval()


IMAGES = torch.randn(3, 3, 32, 32)


@torch.no_grad()
def test_patch_rows_stay_p_after_every_loop() -> None:
    model = build_unrolled(3, source=tiny_vit(), cls_only_loop=True)
    head, body, _, n_loops = split_loop(model)
    x = embed(model, IMAGES)
    for block in head:
        x = block(x)
    patches = x[:, 1:].clone()
    for _ in range(n_loops):
        x = body(x)
        assert torch.equal(x[:, 1:], patches)


@torch.no_grad()
def test_loop_one_cls_matches_normal_block() -> None:
    plain = build_unrolled(3, source=tiny_vit())
    cls_only = build_unrolled(3, source=tiny_vit(), cls_only_loop=True)
    x = embed(plain, IMAGES)
    for block in list(plain.blocks)[:2]:
        x = block(x)
    normal = list(plain.blocks)[2](x)
    ours = list(cls_only.blocks)[2](x)
    assert (ours[:, 0] - normal[:, 0]).abs().max() <= 1e-5


@torch.no_grad()
def test_flag_off_reproduces_plain_model() -> None:
    reference = build_unrolled(3, source=tiny_vit())
    rebuilt = build_unrolled(3, source=tiny_vit(), cls_only_loop=False)
    assert (rebuilt(IMAGES) - reference(IMAGES)).abs().max() <= 1e-6


def test_no_new_parameters_and_only_looped_block_wrapped() -> None:
    plain = build_unrolled(3, source=tiny_vit())
    model = build_unrolled(3, source=tiny_vit(), cls_only_loop=True)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in plain.parameters())
    blocks = list(model.blocks)
    assert not isinstance(blocks[0], CLSOnlyBlock) and not isinstance(blocks[1], CLSOnlyBlock)
    assert all(b is blocks[2] and isinstance(b, CLSOnlyBlock) for b in blocks[2:]) and len(blocks) == 12


def test_gradients_reach_head_blocks_through_patches() -> None:
    model = build_unrolled(3, source=tiny_vit(), cls_only_loop=True).train()
    model(IMAGES).sum().backward()
    assert list(model.blocks)[1].attn.qkv.weight.grad.abs().sum() > 0


def test_two_options_at_once_is_an_error() -> None:
    with pytest.raises(ValueError):
        build_unrolled(3, source=tiny_vit(), cls_only_loop=True, zero_mean_value=True)


def test_flag_only_added_when_set() -> None:
    args = SimpleNamespace(model="unroll", unique_blocks=3, cls_only_loop=False)
    assert model_kwargs(args) == {"unique_blocks": 3}
    args.cls_only_loop = True
    assert model_kwargs(args) == {"unique_blocks": 3, "cls_only_loop": True}
