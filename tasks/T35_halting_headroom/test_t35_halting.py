"""T35 checks on toy tensors with hand-computed answers: settling loop, first-correct loop, oracle curve, budget
lookup, headroom, and the AUROC orientation.

Run from the repo root:  PYTHONPATH=. .venv/bin/python -m pytest tasks/T35_halting_headroom -q
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from halting import (  # noqa: E402
    accuracy_at_budget,
    auroc,
    first_correct_loop,
    loops_to_reach,
    oracle_curve,
    oracle_summary,
    settling_loops,
)

# 4 loops x 3 images. Columns: image 0 settles at loop 3, image 1 never changes, image 2 flips until the end.
PREDICTIONS = torch.tensor([[1, 5, 1], [2, 5, 2], [3, 5, 1], [3, 5, 2]])
LABELS = torch.tensor([3, 7, 1])  # image 0 first right at loop 3, image 1 never, image 2 right at loop 1


def test_settling_loop() -> None:
    assert settling_loops(PREDICTIONS).tolist() == [3, 1, 4]


def test_first_correct_loop() -> None:
    correct = PREDICTIONS == LABELS.unsqueeze(0)
    assert first_correct_loop(correct).tolist() == [3, 0, 1]


def test_oracle_curve_and_budget_lookup() -> None:
    curve = oracle_curve(torch.tensor([3, 0, 1]))
    # start: 3 images at loop 1, image 2 free -> (1, 33.3); add image 0 (e = 3, +2 loops) -> (5/3, 66.7)
    assert curve[0] == pytest.approx((1.0, 100 / 3))
    assert curve[1] == pytest.approx((5 / 3, 200 / 3))
    assert len(curve) == 2
    assert accuracy_at_budget(curve, 1.5) == pytest.approx(100 / 3)
    assert accuracy_at_budget(curve, 2) == pytest.approx(200 / 3)
    assert loops_to_reach(curve, 50.0) == pytest.approx(5 / 3)
    assert loops_to_reach(curve, 90.0) is None


def test_headroom_and_decision() -> None:
    # 10 images: 6 right at loop 1, 3 right at loop 2, 1 never -> 90% at 1.3 mean loops, so headroom = 90 - 68.3
    summary = oracle_summary(torch.tensor([1] * 6 + [2] * 3 + [0]))
    assert summary["oracle_accuracy"] == pytest.approx(90.0)
    assert summary["accuracy_at_mean_loops"]["3"] == pytest.approx(90.0)
    assert summary["headroom_at_3"] == pytest.approx(21.7)
    assert summary["decision"].startswith("Halting: worth a gate")
    low = oracle_summary(torch.tensor([1] * 6 + [0] * 4))  # 60% -> headroom -8.3
    assert low["decision"].startswith("Halting: no room")


def test_auroc_orientation() -> None:
    correct = torch.tensor([True, True, False, False])
    assert auroc(torch.tensor([0.9, 0.8, 0.2, 0.1]), correct) == pytest.approx(1.0)
    assert auroc(torch.tensor([0.1, 0.2, 0.8, 0.9]), correct) == pytest.approx(0.0)
    assert auroc(torch.tensor([0.5, 0.5]), torch.tensor([True, True])) is None
