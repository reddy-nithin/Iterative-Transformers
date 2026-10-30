"""T11 checks: the pretrained head + body x10 + tail option builds three distinct copies of the chosen pretrained
blocks, calls the body 10 times, leaves the T9 (option off) model unchanged, and survives the checkpoint round trip."""
from argparse import Namespace

import pytest
import torch
from torch import nn

from src.models import build, build_baseline, build_head_body_tail
from src.train import model_kwargs

torch.manual_seed(0)
IMAGES = torch.randn(2, 3, 224, 224)
T11_LAYOUTS = [(0, 0, 11), (0, 6, 11), (0, 1, 11)]  # (head, body, tail) of T11-a, T11-b, T11-c


@pytest.fixture(scope="module")
def pretrained() -> nn.Module:
    """The pretrained ViT-Tiny with the 100-class head, loaded once (the reference blocks)."""
    return build_baseline(pretrained=True).eval()


def pretrained_hbt(source: nn.Module, head: int, body: int, tail: int) -> nn.Module:
    """T11 model built from a fresh copy of `source`, so the fixture itself is never modified."""
    clone = build_baseline(pretrained=False)
    clone.load_state_dict(source.state_dict())
    return build_head_body_tail(source=clone, pretrained_init=True, head_idx=head, body_idx=body, tail_idx=tail)


@pytest.mark.parametrize("head, body, tail", T11_LAYOUTS)
def test_three_distinct_objects_and_body_repeated(pretrained: nn.Module, head: int, body: int, tail: int) -> None:
    model = pretrained_hbt(pretrained, head, body, tail)
    ids = [id(block) for block in model.blocks]
    head_block, body_block, tail_block = model.blocks[0], model.blocks[1], model.blocks[11]
    assert len(ids) == 12 and len({id(head_block), id(body_block), id(tail_block)}) == 3
    assert ids.count(id(body_block)) == 10


def test_body_is_called_ten_times_per_forward(pretrained: nn.Module) -> None:
    model = pretrained_hbt(pretrained, 0, 0, 11).eval()
    calls = []
    handle = model.blocks[1].register_forward_hook(lambda module, args, output: calls.append(1))
    with torch.no_grad():
        model(IMAGES)
    handle.remove()
    assert len(calls) == 10


@pytest.mark.parametrize("head, body, tail", T11_LAYOUTS)
def test_copied_blocks_start_equal_to_their_pretrained_block(
    pretrained: nn.Module, head: int, body: int, tail: int
) -> None:
    model = pretrained_hbt(pretrained, head, body, tail)
    for copied, source_idx in ((model.blocks[0], head), (model.blocks[1], body), (model.blocks[11], tail)):
        reference = pretrained.blocks[source_idx].state_dict()
        for name, tensor in copied.state_dict().items():
            assert torch.equal(tensor, reference[name]), f"block {source_idx} {name} differs"
    for name in ("cls_token", "pos_embed"):
        assert torch.equal(getattr(model, name), getattr(pretrained, name))
    assert torch.equal(model.patch_embed.proj.weight, pretrained.patch_embed.proj.weight)
    assert torch.equal(model.norm.weight, pretrained.norm.weight)
    assert model.get_classifier().out_features == 100


def test_head_and_body_from_the_same_block_train_independently(pretrained: nn.Module) -> None:
    model = pretrained_hbt(pretrained, 0, 0, 11)
    head, body = model.blocks[0], model.blocks[1]
    assert head.mlp.fc1.weight.data_ptr() != body.mlp.fc1.weight.data_ptr()
    with torch.no_grad():
        body.mlp.fc1.weight.add_(1.0)
    assert not torch.equal(head.mlp.fc1.weight, body.mlp.fc1.weight)


def test_param_count_matches_three_blocks(pretrained: nn.Module) -> None:
    model = pretrained_hbt(pretrained, 0, 6, 11)
    block = sum(p.numel() for p in pretrained.blocks[0].parameters())
    total = sum(p.numel() for p in pretrained.parameters())
    assert sum(p.numel() for p in model.parameters()) == total - 9 * block


def test_option_off_is_the_t9_model() -> None:
    torch.manual_seed(0)
    old = build_head_body_tail().eval()
    torch.manual_seed(0)
    explicit_off = build_head_body_tail(pretrained_init=False).eval()
    with torch.no_grad():
        assert torch.equal(old(IMAGES), explicit_off(IMAGES))
    assert model_kwargs(Namespace(model="hbt", head_blocks=1, tail_blocks=1, inject=False)) == \
        {"head_blocks": 1, "tail_blocks": 1, "inject": False}


@pytest.mark.parametrize("bad", [{"tail_blocks": 0}, {"inject": True}, {"body_idx": 12}])
def test_unsupported_layouts_are_rejected(bad: dict) -> None:
    kwargs = {"pretrained_init": True, "source": build_baseline(pretrained=False), **bad}
    with pytest.raises(ValueError):
        build_head_body_tail(**kwargs)


def test_checkpoint_round_trip() -> None:
    args = Namespace(model="hbt", head_blocks=1, tail_blocks=1, inject=False, hbt_pretrained=True,
                     head_idx=0, body_idx=6, tail_idx=11)
    kwargs = model_kwargs(args)
    assert kwargs["pretrained_init"] and kwargs["body_idx"] == 6
    model = build("hbt", **kwargs).eval()
    rebuilt = build("hbt", **kwargs)
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.allclose(model(IMAGES), rebuilt.eval()(IMAGES))
