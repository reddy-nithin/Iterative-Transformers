"""Tests for T22's contiguous max-cut."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from maxcut import contiguous_cuts, cut_score, groups_from_cut, rank_of, ranked_cuts, within_score  # noqa: E402


def two_block_matrix() -> np.ndarray:
    """6x6 similarity: 1 inside {0,1,2} and {3,4,5}, 0.2 across."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    return np.where(labels[:, None] == labels[None, :], 1.0, 0.2)


def test_k2_cut_on_two_obvious_blocks_is_at_3() -> None:
    assert ranked_cuts(two_block_matrix(), 2)[0]["cut"] == [3]


def test_number_of_k3_options_for_12_blocks_is_55() -> None:
    assert len(contiguous_cuts(12, 3)) == 55
    assert len(contiguous_cuts(12, 2)) == 11
    assert len(contiguous_cuts(12, 4)) == 165


def test_within_score_skips_single_block_groups() -> None:
    similarity = two_block_matrix()
    # groups {0}, {1,2}, {3,4,5}: the singleton is left out, both other groups have mean S = 1
    assert within_score(similarity, (1, 3)) == 1.0
    assert within_score(np.eye(3), (1, 2)) is None


def test_cut_score_counts_cross_pairs_only() -> None:
    similarity = two_block_matrix()
    assert np.isclose(cut_score(similarity, (3,)), 9 * 0.8)
    assert groups_from_cut((1, 11), 12) == [[0], list(range(1, 11)), [11]]


def test_rank_of_finds_the_hand_cut() -> None:
    rows = ranked_cuts(np.random.default_rng(0).uniform(size=(12, 12)), 3)
    assert {rank_of(rows, (1, 11)), rank_of(rows, (1, 2))} <= set(range(1, 56))
    assert [row["rank"] for row in rows] == list(range(1, 56))
