"""T36 checks for `--body-loops`: the default builds the same model as before, N = 4 calls the looped block exactly
4 times, and the parameter count is 1,539,940 for N = 1, 4 and 10 (looping adds no parameters)."""
import copy

import pytest
import torch
from torch import nn

from src.metrics import count_params
from src.models import build_baseline, build_unrolled
from src.train import check_options, model_kwargs, parse_args

T11D_PARAMS = 1_539_940
UNIQUE_BLOCKS = 3  # T11-d layout: B0, B1, then B2 looped


@pytest.fixture(scope="module")
def source() -> nn.Module:
    """Real ViT-Tiny architecture with random weights (no download), fixed seed."""
    torch.manual_seed(0)
    return build_baseline(pretrained=False).eval()


@pytest.fixture(scope="module")
def images() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 3, 224, 224)


def old_layout(source: nn.Module) -> nn.Module:
    """The pre-T36 code path for unique_blocks = 3, written out by hand: B0, B1, B2 x10 in an nn.Sequential."""
    model = copy.deepcopy(source)
    blocks = list(model.blocks)
    model.blocks = nn.Sequential(blocks[0], blocks[1], *[blocks[2]] * 10)
    return model


@torch.no_grad()
def test_default_matches_old_layout(source: nn.Module, images: torch.Tensor) -> None:
    new = build_unrolled(UNIQUE_BLOCKS, source=copy.deepcopy(source)).eval()
    old = old_layout(source).eval()
    assert (new(images) - old(images)).abs().max().item() <= 1e-5


@torch.no_grad()
def test_four_loops_call_the_block_four_times(source: nn.Module, images: torch.Tensor) -> None:
    model = build_unrolled(UNIQUE_BLOCKS, source=copy.deepcopy(source), body_loops=4).eval()
    looped = list(model.blocks)[2]
    calls = []
    handle = looped.register_forward_hook(lambda module, inputs, output: calls.append(1))
    model(images)
    handle.remove()
    assert len(calls) == 4


@pytest.mark.parametrize("n_loops", [1, 4, 10])
def test_param_count_does_not_depend_on_loops(source: nn.Module, n_loops: int) -> None:
    model = build_unrolled(UNIQUE_BLOCKS, source=copy.deepcopy(source), body_loops=n_loops)
    assert count_params(model) == T11D_PARAMS


def test_flag_off_keeps_old_checkpoint_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the flag, the saved model kwargs are exactly the old ones (old checkpoints rebuild unchanged)."""
    monkeypatch.setattr("sys.argv", ["train", "--model", "unroll", "--unique-blocks", "3"])
    args = parse_args()
    check_options(args)
    assert model_kwargs(args) == {"unique_blocks": 3}
