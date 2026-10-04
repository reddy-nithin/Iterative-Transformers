import torch

from src.models import build_baseline, build_iterative


def test_blocks_are_same_object():
    m = build_iterative(pretrained=False)
    assert len({id(b) for b in m.blocks}) == 1 and len(m.blocks) == 12


def test_param_count_drops():
    base, it = build_baseline(pretrained=False), build_iterative(pretrained=False)
    n = lambda m: sum(p.numel() for p in m.parameters())
    block = sum(p.numel() for p in base.blocks[0].parameters())
    assert n(base) - n(it) == 11 * block


def test_gradients_flow_into_shared_block():
    m = build_iterative(pretrained=False)
    m(torch.randn(2, 3, 224, 224)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.blocks[0].parameters())


def test_forward_shape():
    m = build_iterative(pretrained=False)
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 100)
