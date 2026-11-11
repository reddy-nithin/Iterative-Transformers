"""Per-loop center loss with EMA class prototypes (T30; DE3-BERT / intermediate-layer center loss).

    L_center = alpha * mean_t mean_batch (1 - cos(z_t, p_y)),   t = 1..L,  z_t = final_norm(x_t)[:, 0]

Prototypes p_c (num_classes x C) are a plain tensor, not a parameter: no gradient. After every training step:
    p_c <- 0.9 p_c + 0.1 * (batch mean of z_L over the images of class c), only for classes in the batch.
They start at 0. They are not part of the model's state_dict (the model rebuilds from its kwargs unchanged); training
saves them under their own checkpoint key.
"""
import torch
import torch.nn.functional as F
from torch import nn

EMA_KEEP = 0.9


def loop_features(model: nn.Module, cls_states: list[torch.Tensor]) -> list[torch.Tensor]:
    """z_t = final_norm(CLS_t) for t = 1..L (`cls_states[0]` is the body input and is skipped)."""
    return [model.norm(cls) for cls in cls_states[1:]]


def center_loss(features: list[torch.Tensor], labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """mean over loops of mean over the batch of (1 - cos(z_t, p_y)), in fp32. No alpha."""
    targets = prototypes[labels]
    per_loop = [(1.0 - F.cosine_similarity(z.float(), targets, dim=-1)).mean() for z in features]
    return torch.stack(per_loop).mean()


@torch.no_grad()
def ema_update(prototypes: torch.Tensor, final_features: torch.Tensor, labels: torch.Tensor,
               keep: float = EMA_KEEP) -> torch.Tensor:
    """New prototypes: classes present in the batch move to keep * p_c + (1 - keep) * batch mean of z_L."""
    num_classes = prototypes.shape[0]
    features = final_features.detach().float()
    sums = torch.zeros_like(prototypes).index_add_(0, labels, features)
    counts = torch.bincount(labels, minlength=num_classes).to(prototypes.dtype)
    present = counts > 0
    updated = prototypes.clone()
    updated[present] = keep * prototypes[present] + (1.0 - keep) * sums[present] / counts[present, None]
    return updated
