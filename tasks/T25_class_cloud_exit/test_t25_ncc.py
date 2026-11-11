"""Tests for T25's class-cloud exit: NCC, distance ratio, radius rule, stop index, center shapes."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from class_cloud import (Features, build_centers, class_means, cosine_distance, first_stop,  # noqa: E402
                         loop_distances, nearest_two, radius_condition)


def three_clusters(per_class: int = 40, loops: int = 3, seed: int = 0) -> Features:
    """3 well separated Gaussian clusters in 8-d, the same features at every loop."""
    generator = torch.Generator().manual_seed(seed)
    means = torch.eye(8)[:3] * 10
    labels = torch.arange(3).repeat_interleave(per_class)
    z = means[labels] + torch.randn(len(labels), 8, generator=generator) * 0.5
    return Features(z.expand(loops, -1, -1).clone(), None, labels)


def test_ncc_with_class_means_is_perfect_on_separated_clusters() -> None:
    data = three_clusters()
    centers = class_means(data.z[0], data.labels, 3)
    assert (cosine_distance(data.z[0], centers).argmin(dim=1) == data.labels).all()


def test_distance_ratio_is_between_zero_and_one() -> None:
    data = three_clusters()
    centers = build_centers(data, num_classes=3, quantiles=(0.5,))
    d1, d2, _ = nearest_two(loop_distances(data.z, centers, "per_loop"))
    ratio = d1 / d2
    assert (ratio >= 0).all() and (ratio <= 1).all()


def test_radius_rule_with_q1_stops_every_training_point_at_the_first_loop() -> None:
    data = three_clusters()
    centers = build_centers(data, num_classes=3, quantiles=(1.0,))
    d1, _, c1 = nearest_two(loop_distances(data.z, centers, "shared"))
    stop = first_stop(radius_condition(d1, c1, centers["radii"], 0), min_loop=1)
    assert (stop == 0).all()


def test_first_stop_respects_min_loop_and_falls_back_to_last() -> None:
    condition = torch.tensor([[True, False], [False, False], [True, False]])
    assert first_stop(condition, min_loop=2).tolist() == [2, 2]
    assert first_stop(condition, min_loop=1).tolist() == [0, 2]


def test_center_shapes() -> None:
    data = three_clusters(loops=4)
    centers = build_centers(data, num_classes=3, quantiles=(0.25, 0.5, 0.75))
    assert centers["per_loop"].shape == (4, 3, 8)
    assert torch.equal(centers["shared"], centers["per_loop"][-1])
    assert centers["radii"].shape == (3, 3)
    assert (centers["radii"].diff(dim=1) >= 0).all()
