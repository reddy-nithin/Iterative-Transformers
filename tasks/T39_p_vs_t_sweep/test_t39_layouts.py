"""T39 checks for the 7 h+r layouts built with `--unique-blocks (h+1) --body-loops r`.

For each layout: the parameter count matches the formula (P distinct blocks + embeddings + final_norm + classifier)
and the looped block (pretrained block h) is called exactly r times per forward pass. A last check confirms the two
flags combine on the command line the way `run.sh` uses them.
"""
import copy

import pytest
import torch
from torch import nn

from src.metrics import count_params
from src.models import build_baseline, build_unrolled
from src.train import check_options, model_kwargs, parse_args

BLOCK_PARAMS = 444_864  # one ViT-Tiny block (attn qkv+proj, MLP fc1+fc2, two LayerNorms)
NON_BLOCK_PARAMS = 205_348  # patch embed 147,648 + CLS 192 + pos 37,824 + final_norm 384 + classifier 19,300

# run name -> (h distinct head blocks, r loops of block h)
LAYOUTS = {
    "T39-a": (2, 2),
    "T39-b": (2, 8),
    "T39-c": (2, 16),
    "T39-d": (3, 1),
    "T39-e": (3, 2),
    "T39-f": (3, 4),
    "T39-g": (3, 8),
}


@pytest.fixture(scope="module")
def source() -> nn.Module:
    """Real ViT-Tiny architecture with random weights (no download), fixed seed."""
    torch.manual_seed(0)
    return build_baseline(pretrained=False).eval()


def expected_params(stored_blocks: int) -> int:
    """Formula: P stored blocks plus everything that is not a block."""
    return NON_BLOCK_PARAMS + stored_blocks * BLOCK_PARAMS


def build_layout(source: nn.Module, head: int, loops: int) -> nn.Module:
    """h+r layout exactly as run.sh builds it: --unique-blocks h+1 --body-loops r."""
    return build_unrolled(head + 1, source=copy.deepcopy(source), body_loops=loops).eval()


@pytest.mark.parametrize("run", list(LAYOUTS))
def test_param_count_matches_formula(source: nn.Module, run: str) -> None:
    head, loops = LAYOUTS[run]
    model = build_layout(source, head, loops)
    stored_blocks = head + 1
    count = count_params(model)
    print(f"{run} {head}+{loops}: P = {stored_blocks}W, params = {count:,}")
    assert count == expected_params(stored_blocks)


@pytest.mark.parametrize("run", list(LAYOUTS))
@torch.no_grad()
def test_looped_block_called_r_times(source: nn.Module, run: str) -> None:
    head, loops = LAYOUTS[run]
    model = build_layout(source, head, loops)
    looped = list(model.blocks)[head]
    calls = []
    handle = looped.register_forward_hook(lambda module, inputs, output: calls.append(1))
    torch.manual_seed(1)
    logits = model(torch.randn(2, 3, 224, 224))
    handle.remove()
    assert len(calls) == loops
    assert logits.shape == (2, 100)


@pytest.mark.parametrize("run", list(LAYOUTS))
def test_flags_combine_on_command_line(monkeypatch: pytest.MonkeyPatch, run: str) -> None:
    head, loops = LAYOUTS[run]
    argv = ["train", "--model", "unroll", "--unique-blocks", str(head + 1), "--body-loops", str(loops)]
    monkeypatch.setattr("sys.argv", argv)
    args = parse_args()
    check_options(args)
    assert model_kwargs(args) == {"unique_blocks": head + 1, "body_loops": loops}
