"""T24 core: remove one direction from the CLS update of chosen loops, and the direction / class-information maths.

Edit at loop t (in the edit range): after the body block computes x_t, its CLS row becomes
    CLS_{t-1} + d_t - (d_t . u) u,   d_t = CLS_t - CLS_{t-1},
patch rows unchanged; later loops run on the edited state.
"""
import numpy as np
import torch
from torch import nn

from src.per_loop_readout import embed, readout, split_loop


def remove_direction(previous: torch.Tensor, new: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """`new` (B, T, D) with its CLS row's update relative to `previous` stripped of its component along `direction`."""
    update = new[:, 0] - previous[:, 0]
    edited_cls = previous[:, 0] + update - (update @ direction)[:, None] * direction
    return torch.cat([edited_cls[:, None], new[:, 1:]], dim=1)


def edited_per_loop(model: nn.Module, images: torch.Tensor, direction: torch.Tensor,
                    edit_loops: set[int]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """(logits after every loop t = 1..L, raw CLS [CLS_0..CLS_L]) with the edit applied in loops `edit_loops`."""
    head, body, tail, n_loops = split_loop(model)
    x = embed(model, images)
    for block in head:
        x = block(x)
    logits, cls_states = [], [x[:, 0]]
    for loop in range(1, n_loops + 1):
        new = body(x)
        x = remove_direction(x, new, direction) if loop in edit_loops else new
        logits.append(readout(model, x, tail))
        cls_states.append(x[:, 0])
    return logits, cls_states


def uncentred_direction(updates: np.ndarray) -> dict:
    """SVD of the update matrix D (rows = d_t) without centering: u1 (sign: mean(D . u1) > 0), u2, energy shares."""
    _, singular, right = np.linalg.svd(updates, full_matrices=False)
    energy = singular ** 2 / np.sum(singular ** 2)
    u1, u2 = right[0], right[1]
    if (updates @ u1).mean() < 0:
        u1 = -u1
    mean_row = updates.mean(axis=0)
    centred_singular = np.linalg.svd(updates - mean_row, compute_uv=False)
    return {"u1": u1, "u2": u2, "energy": energy,
            "cos_u1_mean": float(u1 @ mean_row / np.linalg.norm(mean_row)),
            "centred_energy_share": float(centred_singular[0] ** 2 / np.sum(centred_singular ** 2))}


def fisher_ratios(projections: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Between-class variance of the class means / mean within-class variance, per column of `projections` (N, K)."""
    classes = np.unique(labels)
    means = np.stack([projections[labels == c].mean(axis=0) for c in classes])
    within = np.stack([projections[labels == c].var(axis=0) for c in classes]).mean(axis=0)
    return means.var(axis=0) / within


def random_unit_directions(count: int, dim: int, seed: int) -> torch.Tensor:
    """`count` Gaussian directions normalised to unit length (rows), from torch.Generator().manual_seed(seed)."""
    vectors = torch.randn(count, dim, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return vectors / vectors.norm(dim=1, keepdim=True)
