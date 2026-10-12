"""One shared transformer block applied `depth` times, with a tiny low-rank update per iteration.

Iteration i uses, for each of the block's four big linears, the weight  W_shared + B_i @ A_i
(A_i: rank x in, B_i: out x rank). This is "idea 2" from the advisor meetings and the
vision analogue of Relaxed Recursive Transformers (Bae et al., ICLR 2025).
"""
import math

import torch
from torch import nn
from torch.func import functional_call

LINEARS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def _key(name: str) -> str:
    """ParameterDict / ModuleDict keys cannot contain '.'."""
    return name.replace(".", "_")


def _linear(block: nn.Module, name: str) -> nn.Linear:
    return dict(block.named_modules())[name]


def svd_factors(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-`rank` factorisation delta ~ B @ A (Eckart-Young), singular values split evenly."""
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    root = s[:rank].sqrt()
    return u[:, :rank] * root, root[:, None] * vh[:rank]


class LoRALoopBlocks(nn.Module):
    """Drop-in replacement for `VisionTransformer.blocks`: loops one shared block with per-loop LoRA."""

    def __init__(
        self,
        shared: nn.Module,
        originals: list[nn.Module] | None,
        rank: int,
        depth: int = 12,
        lora_init: str = "svd",
    ):
        super().__init__()
        if lora_init not in ("svd", "zero"):
            raise ValueError(f"lora_init must be 'svd' or 'zero', got {lora_init!r}")
        if lora_init == "svd" and (originals is None or len(originals) != depth):
            raise ValueError("svd init needs the `depth` original blocks to factor their residuals")
        self.shared, self.depth, self.rank = shared, depth, rank
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        for name in LINEARS:
            lin = _linear(shared, name)
            a_list, b_list = nn.ParameterList(), nn.ParameterList()
            for i in range(depth):
                if lora_init == "svd":
                    delta = (_linear(originals[i], name).weight - lin.weight).detach().float()
                    b, a = svd_factors(delta, rank)
                else:
                    a = torch.empty(rank, lin.in_features)
                    nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                    b = torch.zeros(lin.out_features, rank)
                a_list.append(nn.Parameter(a.to(lin.weight.dtype)))
                b_list.append(nn.Parameter(b.to(lin.weight.dtype)))
            self.lora_A[_key(name)], self.lora_B[_key(name)] = a_list, b_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.depth):
            overrides = {
                f"{name}.weight": _linear(self.shared, name).weight
                + self.lora_B[_key(name)][i] @ self.lora_A[_key(name)][i]
                for name in LINEARS
            }
            x = functional_call(self.shared, overrides, (x,), strict=False)
        return x
