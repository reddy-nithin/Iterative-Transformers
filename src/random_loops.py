"""Random loop count per training batch (T31, Huginn recipe) and running a looped model for any number of loops.

    tau ~ N(log(MEAN) - SIGMA^2 / 2, SIGMA),  r = 1 + Poisson(e^tau),  capped at MAX_LOOPS

With MEAN = 10 and SIGMA = 0.5 the shift is 0.125 (the task's formula). The +1 and the cap make the mean of r about
10.55, not 10 (gotcha 2026-11-11).
"""
import math

import numpy as np
import torch
from torch import nn

from src.per_loop_readout import embed, readout, split_loop

MEAN_LOOPS = 10
SIGMA = 0.5
MAX_LOOPS = 20


def sample_loops(rng: np.random.Generator) -> int:
    """One loop count r for a training batch."""
    tau = rng.normal(math.log(MEAN_LOOPS) - SIGMA ** 2 / 2, SIGMA)
    return int(min(1 + rng.poisson(math.exp(tau)), MAX_LOOPS))


def forward_loops(model: nn.Module, images: torch.Tensor, n_loops: int) -> torch.Tensor:
    """Logits of a looped model run with its body applied `n_loops` times (head and tail as usual)."""
    return per_loop_logits_upto(model, images, n_loops, every_loop=False)[-1]


def per_loop_logits_upto(model: nn.Module, images: torch.Tensor, n_loops: int,
                         every_loop: bool = True) -> list[torch.Tensor]:
    """Readout logits after loops 1..`n_loops` (or only the last with `every_loop=False`)."""
    if n_loops < 1:
        raise ValueError(f"n_loops must be >= 1, got {n_loops}")
    head, body, tail, _ = split_loop(model)
    if hasattr(body, "reset"):
        raise ValueError("stateful bodies (coverage mask, concat injection) are not supported with a variable depth")
    x = embed(model, images)
    for block in head:
        x = block(x)
    logits = []
    for step in range(1, n_loops + 1):
        x = body(x)
        if every_loop or step == n_loops:
            logits.append(readout(model, x, tail))
    return logits
