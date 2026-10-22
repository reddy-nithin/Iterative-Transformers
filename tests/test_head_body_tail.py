"""T9 checks: head + body + tail is three distinct blocks with the body reused, gradients reach all of them, the
injection flag off is the plain stack, and the checkpoint round trip rebuilds the same model."""
from argparse import Namespace

import pytest
import torch
from torch import nn

from src.head_body_tail import InjectedLoop
from src.models import build, build_baseline, build_head_body_tail
from src.train import model_kwargs

torch.manual_seed(0)
IMAGES = torch.randn(2, 3, 224, 224)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def seeded(**kwargs) -> nn.Module:
    """Head + body + tail from a fixed seed, so two calls get the same random weights."""
    torch.manual_seed(0)
    return build_head_body_tail(**kwargs).eval()


def test_head_body_tail_are_three_objects_and_body_runs_ten_times() -> None:
    model = build_head_body_tail()
    ids = [id(block) for block in model.blocks]
    head, body, tail = model.blocks[0], model.blocks[1], model.blocks[11]
    assert len(ids) == 12 and len({id(head), id(body), id(tail)}) == 3
    assert ids.count(id(body)) == 10 and ids.count(id(head)) == 1 and ids.count(id(tail)) == 1


def test_param_count_is_three_blocks_plus_the_rest() -> None:
    base, model = build_baseline(pretrained=False), build_head_body_tail()
    block = sum(p.numel() for p in base.blocks[0].parameters())
    assert n_params(base) - n_params(model) == 9 * block


def test_head_only_and_loop_only_layouts() -> None:
    head_only = build_head_body_tail(tail_blocks=0)
    ids = [id(b) for b in head_only.blocks]
    assert len(set(ids)) == 2 and ids.count(id(head_only.blocks[1])) == 11
    loop_only = build_head_body_tail(head_blocks=0, tail_blocks=0)
    assert len({id(b) for b in loop_only.blocks}) == 1 and len(loop_only.blocks) == 12


def test_gradients_reach_head_body_and_tail() -> None:
    model = build_head_body_tail()
    model(IMAGES).sum().backward()
    for block in (model.blocks[0], model.blocks[1], model.blocks[11]):
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())


def test_inject_off_equals_a_hand_built_stack() -> None:
    model = seeded()
    head, body, tail = model.blocks[0], model.blocks[1], model.blocks[11]
    with torch.no_grad():
        features = model.forward_features(IMAGES)
        x = head(model.norm_pre(model._pos_embed(model.patch_embed(IMAGES))))
        for _ in range(10):
            x = body(x)
        expected = model.norm(tail(x))
    assert torch.allclose(features, expected, atol=1e-5)


def test_inject_on_differs_with_the_same_weights_and_keeps_twelve_steps() -> None:
    plain, injected = seeded(), seeded(inject=True)
    assert isinstance(injected.blocks, InjectedLoop) and len(injected.blocks) == 12
    assert len(list(injected.blocks)) == 12 and n_params(injected) == n_params(plain)
    assert torch.equal(plain.blocks[1].mlp.fc1.weight, injected.blocks.body.mlp.fc1.weight)
    with torch.no_grad():
        assert not torch.allclose(plain(IMAGES), injected(IMAGES))


def test_injection_needs_a_head() -> None:
    with pytest.raises(ValueError):
        build_head_body_tail(head_blocks=0, inject=True)


def test_needs_at_least_one_body_step() -> None:
    with pytest.raises(ValueError):
        build_head_body_tail(head_blocks=6, tail_blocks=6)


@pytest.mark.parametrize("inject", [False, True])
def test_checkpoint_round_trip(inject: bool) -> None:
    kwargs = model_kwargs(Namespace(model="hbt", head_blocks=1, tail_blocks=1, inject=inject))
    model = build("hbt", **kwargs).eval()
    rebuilt = build("hbt", **kwargs)
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.allclose(model(IMAGES), rebuilt.eval()(IMAGES))
