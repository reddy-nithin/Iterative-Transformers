"""T17: Dr. Uddin's H2 in its original form: each loop should push CLS in a NEW direction, and by a real amount.

CLS_t = x_t[:, 0] (raw state, before the final norm), t = 0..L, CLS_0 = the body's input (output of the last head
block). d_t = CLS_t - CLS_{t-1}, t = 1..L. Two extra terms on top of the per-loop readout loss:
  orthogonal: mean over batch and t = 2..L of cos(d_t, d_{t-1})^2        (squared: pushing back is penalised too)
  large:      mean over batch and t = 1..L of max(0, R_MIN - ||d_t|| / (||CLS_{t-1}|| + EPS))   (a floor, not a reward)
Kinds: "orth" = readout loss + LAMBDA_ORTH * orthogonal (T17-a); "orth_large" = that + LAMBDA_SIZE * large (T17-b).
"""
import torch

CLS_UPDATE_LOSSES = ("none", "orth", "orth_large")
LAMBDA_ORTH = 0.5
LAMBDA_SIZE = 1.0
R_MIN = 0.25
EPS = 1e-8


def cls_updates(cls_states: list[torch.Tensor]) -> torch.Tensor:
    """(L, B, C) d_t = CLS_t - CLS_{t-1} in fp32, from [CLS_0, ..., CLS_L] (each (B, C))."""
    states = torch.stack(cls_states).float()
    return states[1:] - states[:-1]


def successive_cos(updates: torch.Tensor) -> torch.Tensor:
    """(L - 1, B) cos(d_t, d_{t-1}) for t = 2..L, with EPS added to each norm."""
    current, previous = updates[1:], updates[:-1]
    norms = (current.norm(dim=-1) + EPS) * (previous.norm(dim=-1) + EPS)
    return (current * previous).sum(dim=-1) / norms


def relative_sizes(cls_states: list[torch.Tensor], updates: torch.Tensor) -> torch.Tensor:
    """(L, B) ||d_t|| / (||CLS_{t-1}|| + EPS) for t = 1..L (T12's rel_cls)."""
    previous = torch.stack(cls_states[:-1]).float()
    return updates.norm(dim=-1) / (previous.norm(dim=-1) + EPS)


def orthogonal_penalty(updates: torch.Tensor) -> torch.Tensor:
    """mean over batch and t = 2..L of cos(d_t, d_{t-1})^2."""
    return successive_cos(updates).pow(2).mean()


def size_floor_penalty(cls_states: list[torch.Tensor], updates: torch.Tensor) -> torch.Tensor:
    """mean over batch and t = 1..L of max(0, R_MIN - ||d_t|| / (||CLS_{t-1}|| + EPS))."""
    return torch.relu(R_MIN - relative_sizes(cls_states, updates)).mean()


def cls_update_penalty(cls_states: list[torch.Tensor], kind: str) -> torch.Tensor:
    """The term added to the readout loss for `kind` ("orth" or "orth_large")."""
    updates = cls_updates(cls_states)
    if kind == "orth":
        return LAMBDA_ORTH * orthogonal_penalty(updates)
    if kind == "orth_large":
        return LAMBDA_ORTH * orthogonal_penalty(updates) + LAMBDA_SIZE * size_floor_penalty(cls_states, updates)
    raise ValueError(f"unknown CLS-update loss {kind!r}; choose from {list(CLS_UPDATE_LOSSES[1:])}")
