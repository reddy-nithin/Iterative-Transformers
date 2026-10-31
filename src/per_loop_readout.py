"""Per-loop readout and the T13 losses for looped ViT-Tiny models (head blocks -> one body block xN -> tail blocks).

At every body loop t the current state x_t (all 197 tokens) is read out with the model's own modules:
    logits_t = forward_head(final_norm(tail(x_t)))
For a model with a tail (T11-c layout: head b0, body b1 x10, tail b11) this is the readout through the tail; for a model
without one (T11-d layout: unrolling n = 3, blocks 0, 1 then block 2 x10) the tail is empty and it is final norm +
classifier on x_t. No new parameters, and the last loop's logits are exactly the normal model output.

Losses (T13): "every_loop" = sum_t w_t mean(CE_t) / sum_t w_t with w_t = 0.85^(L - t); "loss_margin" adds
lambda * mean over t >= 2 of mean_batch[max(0, CE_t - CE_{t-1}.detach() + m) * 1(CE_{t-1}.detach() > floor)].
"""
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn

READOUT_LOSSES = ("none", "every_loop", "loss_margin")
LABEL_SMOOTHING = 0.1  # same as the main loss in src.train
LOOP_DECAY = 0.85  # w_t = LOOP_DECAY ** (n_loops - t): last loop 1.0, loop 1 of 10 about 0.23
MARGIN = 0.02  # loop t must beat loop t-1 by this much CE ...
MARGIN_WEIGHT = 1.0  # ... with this weight (lambda) ...
MARGIN_FLOOR = 1.0  # ... unless loop t-1 already has CE <= floor (image already solved)


def split_loop(model: nn.Module) -> tuple[list[nn.Module], nn.Module, list[nn.Module], int]:
    """(head blocks, body block, tail blocks, number of body loops) of a looped `model.blocks`.

    The body is the block object that appears most often; head = blocks before its first call, tail = blocks after its
    last call. Raises ValueError if no block repeats or the body calls are not consecutive.
    """
    blocks = list(model.blocks)
    body_id, repeats = Counter(id(block) for block in blocks).most_common(1)[0]
    if repeats < 2:
        raise ValueError("no block is repeated: this is not a looped model")
    calls = [i for i, block in enumerate(blocks) if id(block) == body_id]
    if calls != list(range(calls[0], calls[-1] + 1)):
        raise ValueError(f"body calls are not consecutive: {calls}")
    return blocks[: calls[0]], blocks[calls[0]], blocks[calls[-1] + 1:], repeats


def embed(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Tokens entering the first block, exactly as timm 1.0.30 `VisionTransformer.forward_features` builds them."""
    x = model.patch_embed(images)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    return model.norm_pre(x)


def readout(model: nn.Module, state: torch.Tensor, tail: list[nn.Module]) -> torch.Tensor:
    """Logits when stopping at `state`: the tail blocks (if any), then final norm and the classifier on CLS."""
    for block in tail:
        state = block(state)
    return model.forward_head(model.norm(state))


def per_loop_logits(model: nn.Module, images: torch.Tensor) -> list[torch.Tensor]:
    """Logits (B, classes) after every body loop t = 1..L; the last entry equals `model(images)`."""
    head, body, tail, n_loops = split_loop(model)
    if hasattr(body, "reset"):  # stateful body (T14 coverage mask): start every pass with full supply
        body.reset()
    x = embed(model, images)
    for block in head:
        x = block(x)
    logits = []
    for _ in range(n_loops):
        x = body(x)
        logits.append(readout(model, x, tail))
    return logits


def loop_weights(n_loops: int) -> torch.Tensor:
    """w_t = LOOP_DECAY ** (n_loops - t) for t = 1..n_loops (last loop 1.0)."""
    return LOOP_DECAY ** torch.arange(n_loops - 1, -1, -1, dtype=torch.float32)


def per_loop_ce(logits_per_loop: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """(L, B) per-sample cross-entropy with label smoothing, computed in fp32."""
    return torch.stack([F.cross_entropy(logits.float(), labels, reduction="none", label_smoothing=LABEL_SMOOTHING)
                        for logits in logits_per_loop])


def every_loop_loss(ce: torch.Tensor) -> torch.Tensor:
    """sum_t w_t mean(CE_t) / sum_t w_t."""
    weights = loop_weights(ce.shape[0]).to(ce.device)
    return (weights * ce.mean(dim=1)).sum() / weights.sum()


def margin_penalty(ce: torch.Tensor) -> torch.Tensor:
    """(1 / (L - 1)) sum_{t>=2} mean_batch[max(0, CE_t - CE_{t-1}.detach() + MARGIN) * 1(CE_{t-1}.detach() > FLOOR)]."""
    previous = ce[:-1].detach()
    active = (previous > MARGIN_FLOOR).float()
    return (F.relu(ce[1:] - previous + MARGIN) * active).mean(dim=1).mean()


def readout_loss(logits_per_loop: list[torch.Tensor], labels: torch.Tensor, kind: str) -> torch.Tensor:
    """The T13 training loss of kind "every_loop" (H5) or "loss_margin" (H2, readout version)."""
    ce = per_loop_ce(logits_per_loop, labels)
    if kind == "every_loop":
        return every_loop_loss(ce)
    if kind == "loss_margin":
        return every_loop_loss(ce) + MARGIN_WEIGHT * margin_penalty(ce)
    raise ValueError(f"unknown readout loss {kind!r}; choose from {list(READOUT_LOSSES[1:])}")
