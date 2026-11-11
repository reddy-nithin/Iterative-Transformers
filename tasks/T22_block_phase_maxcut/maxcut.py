"""Contiguous max-cut of a block-to-block similarity matrix (T22).

A split of n blocks into k contiguous groups is given by k - 1 cut points 0 < c1 < ... < c_{k-1} < n; group i holds
blocks c_{i-1} .. c_i - 1. Scores (our definition, see the T22 README):
  cut score    = sum over pairs (l < m) in different groups of (1 - S[l, m])   (higher = better split)
  within score = mean over groups with >= 2 blocks of the mean S[l, m] over pairs l < m inside the group
"""
from itertools import combinations

import numpy as np

Cut = tuple[int, ...]


def contiguous_cuts(n_blocks: int, k_groups: int) -> list[Cut]:
    """All cut-point tuples that split n_blocks into k_groups non-empty contiguous groups."""
    if not 1 <= k_groups <= n_blocks:
        raise ValueError(f"cannot split {n_blocks} blocks into {k_groups} groups")
    return list(combinations(range(1, n_blocks), k_groups - 1))


def groups_from_cut(cut: Cut, n_blocks: int) -> list[list[int]]:
    """The block indices of each group."""
    edges = [0, *cut, n_blocks]
    return [list(range(start, end)) for start, end in zip(edges[:-1], edges[1:])]


def across_mask(cut: Cut, n_blocks: int) -> np.ndarray:
    """Boolean (n, n) mask of pairs l < m that the cut puts in different groups."""
    labels = np.zeros(n_blocks, dtype=int)
    for index, group in enumerate(groups_from_cut(cut, n_blocks)):
        labels[group] = index
    upper = np.triu(np.ones((n_blocks, n_blocks), dtype=bool), k=1)
    return upper & (labels[:, None] != labels[None, :])


def cut_score(similarity: np.ndarray, cut: Cut) -> float:
    """Sum of (1 - S) over block pairs l < m that the cut puts in different groups."""
    return float((1 - similarity)[across_mask(cut, similarity.shape[0])].sum())


def within_score(similarity: np.ndarray, cut: Cut) -> float | None:
    """Mean over groups of >= 2 blocks of their mean within-group S (pairs l < m); None if every group is single."""
    means = [float(np.mean([similarity[l, m] for l, m in combinations(group, 2)]))
             for group in groups_from_cut(cut, similarity.shape[0]) if len(group) >= 2]
    return float(np.mean(means)) if means else None


def across_mean(similarity: np.ndarray, cut: Cut) -> float:
    """Mean S over block pairs in different groups (for the within-vs-across sharpness comparison)."""
    return float(similarity[across_mask(cut, similarity.shape[0])].mean())


def ranked_cuts(similarity: np.ndarray, k_groups: int) -> list[dict]:
    """Every contiguous k-group cut with its scores, best cut score first (rank 1)."""
    n_blocks = similarity.shape[0]
    rows = [{"cut": list(cut), "groups": groups_from_cut(cut, n_blocks), "cut_score": cut_score(similarity, cut),
             "within_score": within_score(similarity, cut), "across_mean": across_mean(similarity, cut)}
            for cut in contiguous_cuts(n_blocks, k_groups)]
    rows.sort(key=lambda row: row["cut_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def rank_of(rows: list[dict], cut: Cut) -> int:
    """Rank of a given cut in a ranked list."""
    for row in rows:
        if tuple(row["cut"]) == tuple(cut):
            return row["rank"]
    raise KeyError(f"cut {cut} not in the ranking")
