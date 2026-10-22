"""Head + repeated body + tail with input injection (T9, variant b).

The body gets the head's output again at every loop:  e = head(x),  x <- body(x + e)  (repeated),  then the tail.
This is the input injection of the Recurrent Depth / Huginn paper (Geiping et al., arXiv 2502.05171), here with plain
addition instead of their concatenation + linear adapter. Without injection the model is a plain `nn.Sequential`
(see `src.models.build_head_body_tail`), so this module is only used when injection is on.
"""
from collections.abc import Iterator

import torch
from torch import nn


class InjectedLoop(nn.Module):
    """Drop-in replacement for `VisionTransformer.blocks`: head blocks, one body block run `repeats` times on
    (state + head output), then tail blocks.

    `len()` and iteration give the blocks in call order (body listed `repeats` times), like an `nn.Sequential` of the
    same layout, so code that hooks every unique block and counts calls sees one snapshot per step.
    """

    def __init__(self, head: list[nn.Module], body: nn.Module, tail: list[nn.Module], repeats: int):
        super().__init__()
        if not head:
            raise ValueError("input injection needs at least one head block (its output is what gets injected)")
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        self.head = nn.ModuleList(head)
        self.body = body
        self.tail = nn.ModuleList(tail)
        self.repeats = repeats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.head:
            x = block(x)
        injected = x
        for _ in range(self.repeats):
            x = self.body(x + injected)
        for block in self.tail:
            x = block(x)
        return x

    def __len__(self) -> int:
        return len(self.head) + self.repeats + len(self.tail)

    def __iter__(self) -> Iterator[nn.Module]:
        return iter([*self.head, *[self.body] * self.repeats, *self.tail])
