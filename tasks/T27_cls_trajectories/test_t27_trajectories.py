"""Tests for T27's trajectory analysis: separation, split step, PCA, distance to final, and the 13 stages."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models import build_unrolled  # noqa: E402
from src.per_loop_readout import per_loop_logits_and_cls  # noqa: E402
from trajectories import (choose_classes, collect_cls_per_layer, distance_to_final, first_split,  # noqa: E402
                          heatmap_rows, pair_separation, pca, split_table)


def two_clusters(distance: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two 50-point unit-variance Gaussian clusters in 8-d whose means are `distance` apart."""
    rng = np.random.default_rng(seed)
    first = rng.normal(size=(50, 8))
    second = rng.normal(size=(50, 8))
    second[:, 0] += distance
    return first, second


def test_separation_of_far_clusters_is_above_one() -> None:
    first, second = two_clusters(distance=20.0)
    assert pair_separation(first, second) > 1


def test_split_step_of_identical_clusters_is_none() -> None:
    first, _ = two_clusters(distance=0.0)
    seps = [pair_separation(first, first.copy()) for _ in range(12)]
    assert first_split(seps, first_stage=1) is None


def test_split_step_is_first_stage_reaching_threshold() -> None:
    assert first_split([0.2, 0.9, 1.0, 3.0], first_stage=1) == 3


def test_pca_shares_sum_to_at_most_one_and_are_sorted() -> None:
    vectors = np.random.default_rng(0).normal(size=(200, 16)) * np.arange(1, 17)
    _, components, share = pca(vectors, 6)
    assert share.sum() <= 1 + 1e-9
    assert np.all(np.diff(share) <= 0)
    assert np.allclose(components @ components.T, np.eye(6), atol=1e-8)


def test_distance_to_final_is_zero_at_last_stage() -> None:
    cls = np.random.default_rng(0).normal(size=(13, 5, 4))
    distances = distance_to_final(cls)
    assert distances.shape == (13, 5)
    assert np.allclose(distances[-1], 0)


def test_split_table_has_45_pairs_for_10_classes() -> None:
    cls = np.random.default_rng(0).normal(size=(13, 100, 4))
    labels = np.repeat(np.arange(10), 10)
    rows = split_table(cls, labels, [f"c{i}" for i in range(10)])
    assert len(rows) == 45
    assert all(len(row["sep"]) == 12 for row in rows)


def test_heatmap_rows_group_by_class() -> None:
    labels = np.repeat(np.arange(3), 5)
    assert heatmap_rows(labels, 3, 2).tolist() == [0, 1, 5, 6, 10, 11]


def test_choose_classes_keeps_the_t5_four_and_is_seeded() -> None:
    names = [f"class{i}" for i in range(100)]
    names[0], names[8], names[23], names[34] = "apple", "bicycle", "cloud", "fox"
    chosen = choose_classes(names)
    assert len(chosen) == 10 and len(set(chosen)) == 10
    assert {"apple", "bicycle", "cloud", "fox"} <= set(chosen)
    assert chosen == choose_classes(names)


def test_thirteen_stages_and_last_matches_per_loop_cls() -> None:
    torch.manual_seed(0)
    model = build_unrolled(3, pretrained=False).eval()
    images = torch.randn(2, 3, 224, 224)
    loader = [(images, torch.tensor([0, 1]))]
    record = collect_cls_per_layer(model, loader, torch.device("cpu"), include_final_norm=False)
    assert record.cls.shape == (13, 2, 192)
    with torch.no_grad():
        _, cls_states = per_loop_logits_and_cls(model, images)
    assert torch.allclose(record.cls[-1], cls_states[-1].float(), atol=1e-5)
    assert torch.allclose(record.cls[2], cls_states[0].float(), atol=1e-5)  # stage 2 (after B1) = body input
