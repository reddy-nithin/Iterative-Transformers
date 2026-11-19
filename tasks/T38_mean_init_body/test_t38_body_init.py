"""T38 checks for `--body-init-range START END`: with 2 2 the looped block equals pretrained block 2 exactly, with 2 11
one tensor equals the hand-computed mean of blocks 2..11, the parameter count stays 1,539,940, B0 and B1 are unchanged,
and without the flag the saved model kwargs are the old ones."""
import copy

import pytest
import torch
from torch import nn

from src.metrics import count_params
from src.models import build_baseline, build_unrolled
from src.train import check_options, model_kwargs, parse_args

T11D_PARAMS = 1_539_940
UNIQUE_BLOCKS = 3  # T11-d layout: B0, B1, then the looped block
MEAN_RANGE = (2, 11)  # 0-based, inclusive: the ten blocks the looped block replaces
TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def source() -> nn.Module:
    """Real ViT-Tiny architecture with random weights (no download), fixed seed."""
    torch.manual_seed(0)
    return build_baseline(pretrained=False).eval()


def looped_block(model: nn.Module) -> nn.Module:
    """The repeated block of an unrolled model (slot 2 for unique_blocks = 3)."""
    return list(model.blocks)[UNIQUE_BLOCKS - 1]


def test_range_2_2_equals_block_2(source: nn.Module) -> None:
    model = build_unrolled(UNIQUE_BLOCKS, source=copy.deepcopy(source), body_init_range=(2, 2))
    expected = source.blocks[2].state_dict()
    actual = looped_block(model).state_dict()
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor), name


@pytest.mark.parametrize("tensor_name", ["attn.qkv.weight", "norm1.bias"])
def test_range_2_11_is_hand_mean(source: nn.Module, tensor_name: str) -> None:
    model = build_unrolled(UNIQUE_BLOCKS, source=copy.deepcopy(source), body_init_range=MEAN_RANGE)
    start, end = MEAN_RANGE
    total = sum(source.blocks[i].state_dict()[tensor_name] for i in range(start, end + 1))
    hand_mean = total / (end - start + 1)
    actual = looped_block(model).state_dict()[tensor_name]
    assert (actual - hand_mean).abs().max().item() <= TOLERANCE


def test_looped_block_is_one_shared_object(source: nn.Module) -> None:
    model = build_unrolled(UNIQUE_BLOCKS, source=copy.deepcopy(source), body_init_range=MEAN_RANGE)
    body = list(model.blocks)[UNIQUE_BLOCKS - 1:]
    assert len(body) == 10
    assert all(block is body[0] for block in body)


@pytest.mark.parametrize("block_range", [None, MEAN_RANGE])
def test_param_count_unchanged(source: nn.Module, block_range: tuple[int, int] | None) -> None:
    model = build_unrolled(UNIQUE_BLOCKS, source=copy.deepcopy(source), body_init_range=block_range)
    assert count_params(model) == T11D_PARAMS


def test_head_blocks_unchanged(source: nn.Module) -> None:
    model = build_unrolled(UNIQUE_BLOCKS, source=copy.deepcopy(source), body_init_range=MEAN_RANGE)
    for idx in (0, 1):
        expected = source.blocks[idx].state_dict()
        actual = list(model.blocks)[idx].state_dict()
        for name, tensor in expected.items():
            assert torch.equal(actual[name], tensor), f"B{idx}.{name}"


@pytest.mark.parametrize("block_range", [(3, 2), (-1, 5), (2, 12)])
def test_bad_range_is_rejected(source: nn.Module, block_range: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        build_unrolled(UNIQUE_BLOCKS, source=copy.deepcopy(source), body_init_range=block_range)


def test_flag_off_keeps_old_checkpoint_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the flag, the saved model kwargs are exactly the old ones (old checkpoints rebuild unchanged)."""
    monkeypatch.setattr("sys.argv", ["train", "--model", "unroll", "--unique-blocks", "3"])
    args = parse_args()
    check_options(args)
    assert model_kwargs(args) == {"unique_blocks": 3}


def test_flag_on_is_saved_in_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["train", "--model", "unroll", "--unique-blocks", "3",
                                     "--body-init-range", "2", "11"])
    args = parse_args()
    check_options(args)
    assert model_kwargs(args) == {"unique_blocks": 3, "body_init_range": (2, 11)}


def test_flag_needs_unroll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["train", "--model", "iterative", "--body-init-range", "2", "11"])
    with pytest.raises(ValueError):
        check_options(parse_args())
