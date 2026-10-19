"""T4 checks: the separate LoRA lr (`--lora-lr`), test loss from `evaluate`, and the plateau rule."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from src.models import build_iterative, build_lora_loop
from src.train import evaluate, is_lora_param, make_param_groups, parse_args

PLOT_PATH = Path(__file__).resolve().parent / "plot.py"


def load_plot_module():
    """Import the T4 plot.py by path (the task folder is not a package)."""
    spec = importlib.util.spec_from_file_location("t4_plot", PLOT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_ids(group: dict) -> set[int]:
    return {id(p) for p in group["params"]}


def test_no_extra_lrs_gives_one_group_as_before() -> None:
    m = build_lora_loop(rank=2, pretrained=False, init="block", block_idx=0)
    (group,) = make_param_groups(m, lr=5e-4)
    assert group_ids(group) == {id(p) for p in m.parameters()}
    assert group["lr"] == group["base_lr"] == 5e-4


def test_lora_lr_splits_lora_factors_into_their_own_group() -> None:
    m = build_lora_loop(rank=2, pretrained=False, init="block", block_idx=0)
    body, lora = make_param_groups(m, lr=5e-4, lora_lr=5e-3)
    lora_ids = {id(p) for n, p in m.named_parameters() if is_lora_param(n)}
    assert lora_ids and group_ids(lora) == lora_ids
    assert lora["base_lr"] == 5e-3 and body["base_lr"] == 5e-4
    assert group_ids(body) | group_ids(lora) == {id(p) for p in m.parameters()}
    assert not group_ids(body) & group_ids(lora)
    assert len(lora_ids) == 4 * 12 * 2  # 4 linears x 12 loops x (A, B)


def test_head_and_lora_lr_together_cover_every_param_once() -> None:
    m = build_lora_loop(rank=2, pretrained=False, init="block", block_idx=0)
    groups = make_param_groups(m, lr=5e-4, head_lr=1e-3, lora_lr=5e-3)
    assert [g["base_lr"] for g in groups] == [5e-4, 1e-3, 5e-3]
    ids = [i for g in groups for i in group_ids(g)]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))


def test_lora_lr_on_model_without_lora_raises() -> None:
    with pytest.raises(ValueError):
        make_param_groups(build_iterative(pretrained=False), lr=5e-4, lora_lr=5e-3)


def test_cli_lora_lr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--model", "lora", "--lora-lr", "5e-3"])
    assert parse_args().lora_lr == 5e-3
    monkeypatch.setattr(sys, "argv", ["train", "--model", "lora"])
    assert parse_args().lora_lr is None


def test_evaluate_returns_acc_and_mean_loss() -> None:
    """A fixed linear 'model' on a fake loader: acc and loss must match a direct computation."""
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    batches = [(torch.randn(5, 4), torch.randint(0, 3, (5,))) for _ in range(3)]
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    acc, loss = evaluate(model, batches, torch.device("cpu"), use_amp=False, criterion=criterion)
    x = torch.cat([b[0] for b in batches]); y = torch.cat([b[1] for b in batches])
    with torch.no_grad():
        logits = model(x)
    assert acc == pytest.approx(100.0 * (logits.argmax(1) == y).float().mean().item())
    assert loss == pytest.approx(criterion(logits, y).item(), rel=1e-5)


@pytest.mark.parametrize("accs, expected", [
    ([10, 20, 30, 40, 50, 60, 61, 61.2], "still climbing"),
    ([10, 20, 30, 40, 50, 60, 60.1, 60.2, 60.3, 60.4, 60.4], "plateaued"),
    ([10, 20, 30], "too short"),
])
def test_plateau_verdict(accs: list[float], expected: str) -> None:
    plot = load_plot_module()
    history = [{"epoch": i + 1, "test_acc": a} for i, a in enumerate(accs)]
    assert plot.verdict(history) == expected
