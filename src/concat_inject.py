"""Huginn-style input injection: concat + linear adapter before every loop of the repeated block (T29).

e = the state entering the first loop (output of the head blocks, all tokens, fixed for the pass, not detached).
Before each application of the repeated block:  x_in = A([x_{t-1}; e]),  A = Linear(2C -> C) per token.
A starts at weight [I, 0], bias 0, so at initialisation x_in = x_{t-1} and the model equals the plain unrolled model.
New parameters: 2C * C + C (73,920 for ViT-Tiny, C = 192). Recurrent Depth / Huginn: Geiping et al., arXiv 2502.05171.

The body is stateful within one forward pass, like `src.coverage_mask.CoverageMaskBody`: the first call after a reset
stores e, and after `n_loops` calls the state is cleared. Code that runs the body loop by loop calls `reset()` first
(`src.per_loop_readout.per_loop_logits_and_cls` does).
"""
import torch
from torch import nn


def identity_adapter(dim: int) -> nn.Linear:
    """Linear(2 dim -> dim) with weight [I, 0] and bias 0: passes the state through and ignores e."""
    adapter = nn.Linear(2 * dim, dim)
    with torch.no_grad():
        adapter.weight.zero_()
        adapter.weight[:, :dim].copy_(torch.eye(dim))
        adapter.bias.zero_()
    return adapter


class ConcatInjectBody(nn.Module):
    """The repeated block with concat + adapter injection; stateful within one pass (stores e on the first call)."""

    def __init__(self, block: nn.Module, n_loops: int, dim: int):
        super().__init__()
        if n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {n_loops}")
        self.block = block
        self.adapter = identity_adapter(dim)
        self.n_loops = n_loops
        self.injected: torch.Tensor | None = None
        self.calls = 0

    def reset(self) -> None:
        """Forget e (start of a new pass)."""
        self.injected, self.calls = None, 0

    def adapter_norms(self) -> dict[str, float]:
        """Frobenius norms of the adapter's state half and e half of the weight."""
        dim = self.adapter.out_features
        weight = self.adapter.weight.detach()
        return {"w_state": weight[:, :dim].norm().item(), "w_e": weight[:, dim:].norm().item()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.injected is None:
            self.injected = x
        if self.injected.shape != x.shape:
            raise RuntimeError(f"stale injection state {tuple(self.injected.shape)} vs input {tuple(x.shape)}; "
                               "call reset() before a new pass")
        out = self.block(self.adapter(torch.cat([x, self.injected], dim=-1)))
        self.calls += 1
        if self.calls == self.n_loops:
            self.reset()
        return out
