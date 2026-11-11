"""T29 checks: at init the concat-inject model equals the plain T11-d model, it adds 2C*C + C parameters (73,920 for
ViT-Tiny), the flag off changes nothing, no state leaks between passes, the per-loop readout's last loop is the model
output, and the kwargs key is added only when on."""
from types import SimpleNamespace

import timm
import torch
from torch import nn

from src.concat_inject import ConcatInjectBody
from src.models import MODEL_NAME, build_unrolled
from src.per_loop_readout import per_loop_logits_and_cls
from src.train import model_kwargs

DIM = 24
IMAGES = torch.randn(3, 3, 32, 32)


def tiny_vit() -> nn.Module:
    """Small random ViT with the ViT-Tiny structure (17 tokens: CLS + 16 patches, 3 heads)."""
    torch.manual_seed(0)
    return timm.create_model(MODEL_NAME, pretrained=False, depth=12, embed_dim=DIM, num_heads=3,
                             img_size=32, patch_size=8, num_classes=10).eval()


def count(model: nn.Module) -> int:
    """Unique parameters."""
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def test_init_equals_plain_model() -> None:
    plain = build_unrolled(3, source=tiny_vit())
    injected = build_unrolled(3, source=tiny_vit(), concat_inject=True)
    assert (injected(IMAGES) - plain(IMAGES)).abs().max() <= 1e-5


def test_vit_tiny_parameter_count() -> None:
    plain = build_unrolled(3, source=timm.create_model(MODEL_NAME, pretrained=False, num_classes=100))
    injected = build_unrolled(3, source=timm.create_model(MODEL_NAME, pretrained=False, num_classes=100),
                              concat_inject=True)
    assert count(injected) == count(plain) + 73_920


@torch.no_grad()
def test_flag_off_changes_nothing() -> None:
    reference = build_unrolled(3, source=tiny_vit())
    rebuilt = build_unrolled(3, source=tiny_vit(), concat_inject=False)
    assert (rebuilt(IMAGES) - reference(IMAGES)).abs().max() <= 1e-6
    assert count(rebuilt) == count(reference)


@torch.no_grad()
def test_no_state_leak_between_passes() -> None:
    model = build_unrolled(3, source=tiny_vit(), concat_inject=True)
    model.blocks[2].adapter.weight.add_(0.05 * torch.randn_like(model.blocks[2].adapter.weight))
    first = model(IMAGES)
    model(torch.randn(5, 3, 32, 32))  # different batch in between
    assert torch.equal(model(IMAGES), first)
    assert model.blocks[2].injected is None and model.blocks[2].calls == 0


@torch.no_grad()
def test_per_loop_readout_last_loop_is_model_output() -> None:
    model = build_unrolled(3, source=tiny_vit(), concat_inject=True)
    model.blocks[2].adapter.weight.add_(0.05 * torch.randn_like(model.blocks[2].adapter.weight))
    logits, cls_states = per_loop_logits_and_cls(model, IMAGES)
    assert len(logits) == 10 and len(cls_states) == 11
    assert (logits[-1] - model(IMAGES)).abs().max() <= 1e-6


def test_injection_is_the_head_output_and_not_detached() -> None:
    model = build_unrolled(3, source=tiny_vit(), concat_inject=True).train()
    with torch.no_grad():
        model.blocks[2].adapter.weight[:, DIM:].fill_(0.01)
    model(IMAGES).sum().backward()
    assert model.blocks[2].adapter.weight.grad[:, DIM:].abs().sum() > 0
    assert list(model.blocks)[1].attn.qkv.weight.grad.abs().sum() > 0


def test_only_looped_block_wrapped() -> None:
    blocks = list(build_unrolled(3, source=tiny_vit(), concat_inject=True).blocks)
    assert all(b is blocks[2] and isinstance(b, ConcatInjectBody) for b in blocks[2:]) and len(blocks) == 12


def test_flag_only_added_when_set() -> None:
    args = SimpleNamespace(model="unroll", unique_blocks=3, concat_inject=False)
    assert model_kwargs(args) == {"unique_blocks": 3}
    args.concat_inject = True
    assert model_kwargs(args) == {"unique_blocks": 3, "concat_inject": True}
