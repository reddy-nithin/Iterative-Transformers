"""Zero-mean value readout for CLS in the looped block (T20, Gemini Round 3 Brainstorm 1).

In the wrapped block's attention, for the CLS query row only, every patch value V_{h,i} (i = 1..196) is replaced by
V_{h,i} - mean_i V_{h,i} (per head, per image, per call). The CLS->CLS value, all patch query rows, the MLP and the
weights are unchanged; no new parameters. A uniform read of the patches then adds nothing to CLS: the only way to
change CLS through attention is to attend selectively.

With s_h = sum_i A_{h,0,i} (CLS weight on the patches), the CLS row becomes
    sum_j A_{h,0,j} V_{h,j} - s_h * mean_i V_{h,i},
which is what `zero_mean_block_parts` computes.
"""
import torch
from torch import nn


def zero_mean_mix(probs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """probs @ v (B, heads, N, head_dim), except the CLS query row reads V_i - mean_i V_i for the patch keys."""
    mixed = probs @ v
    patch_mean = v[:, :, 1:].mean(dim=2)  # (B, heads, head_dim)
    patch_weight = probs[:, :, 0, 1:].sum(dim=-1, keepdim=True)  # s_h, (B, heads, 1)
    cls_row = mixed[:, :, :1] - (patch_weight * patch_mean).unsqueeze(2)
    return torch.cat([cls_row, mixed[:, :, 1:]], dim=2)


def zero_mean_block_parts(block: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """One timm `Block` (timm 1.0.30) computed from its own modules with the CLS row reading zero-mean patch values.

    Returns output (B, N, C), probs (B, heads, N, N) and the attention sublayer output after proj (B, N, C), the same
    keys as T12's `recompute_block`.
    """
    attn = block.attn
    if getattr(attn, "gate", None) is not None:
        raise ValueError("gated attention is not supported by the zero-mean value readout")
    batch, n_tokens, _ = x.shape
    qkv = attn.qkv(block.norm1(x)).reshape(batch, n_tokens, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    probs = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    mixed = zero_mean_mix(attn.attn_drop(probs), v).transpose(1, 2).reshape(batch, n_tokens, -1)
    sublayer = attn.proj_drop(attn.proj(attn.norm(mixed)))
    after_attn = x + block.drop_path1(block.ls1(sublayer))
    output = after_attn + block.drop_path2(block.ls2(block.mlp(block.norm2(after_attn))))
    return {"output": output, "probs": probs, "attn_out": sublayer}


class ZeroMeanValueBlock(nn.Module):
    """Wraps a timm `Block`; forward = the block with the CLS row reading zero-mean patch values."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return zero_mean_block_parts(self.block, x)["output"]
