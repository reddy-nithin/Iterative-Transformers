"""T3 checks: progressive-unrolling structure, parameter counts, and the separate head lr."""
import pytest
import torch

from src.models import build_baseline, build_iterative, build_unrolled
from src.train import make_param_groups


def n_params(model: torch.nn.Module) -> int:
    """Unique parameters (shared blocks counted once)."""
    return sum(p.numel() for p in model.parameters())


def test_one_unique_block_matches_iterative() -> None:
    assert n_params(build_unrolled(1, pretrained=False)) == n_params(build_iterative(pretrained=False))


def test_twelve_unique_blocks_matches_baseline() -> None:
    assert n_params(build_unrolled(12, pretrained=False)) == n_params(build_baseline(pretrained=False))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 12])
def test_slot_layout(n: int) -> None:
    """Slots 0..n-2 are distinct; slots n-1..11 are one shared object; 12 slots; params grow by one block per n."""
    m = build_unrolled(n, pretrained=False)
    ids = [id(b) for b in m.blocks]
    assert len(ids) == 12
    assert len(set(ids[: n - 1])) == n - 1
    assert len(set(ids[n - 1:])) == 1
    assert len(set(ids)) == n
    block = sum(p.numel() for p in m.blocks[0].parameters())
    assert n_params(m) == n_params(build_iterative(pretrained=False)) + (n - 1) * block


def test_forward_shape() -> None:
    assert build_unrolled(3, pretrained=False)(torch.randn(2, 3, 224, 224)).shape == (2, 100)


@pytest.mark.parametrize("n", [0, 13, -1])
def test_out_of_range_raises(n: int) -> None:
    with pytest.raises(ValueError):
        build_unrolled(n, pretrained=False)


def test_head_lr_groups_cover_every_param_once() -> None:
    m = build_unrolled(3, pretrained=False)
    body, head = make_param_groups(m, lr=1e-6, head_lr=1e-3)
    grouped = [id(p) for g in (body, head) for p in g["params"]]
    assert sorted(grouped) == sorted(id(p) for p in m.parameters())
    assert {id(p) for p in head["params"]} == {id(p) for p in m.get_classifier().parameters()}
    assert (body["lr"], body["base_lr"], head["lr"], head["base_lr"]) == (1e-6, 1e-6, 1e-3, 1e-3)


def test_no_head_lr_keeps_one_group() -> None:
    m = build_unrolled(2, pretrained=False)
    (group,) = make_param_groups(m, lr=1e-4, head_lr=None)
    assert len(group["params"]) == len(list(m.parameters())) and group["base_lr"] == 1e-4
