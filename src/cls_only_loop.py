"""Frozen-patch CLS-only loop (T28, Dr. Uddin's reading of Huginn).

After the head blocks the patch tokens P = x[:, 1:] are fixed. Each loop of the repeated block updates ONLY the CLS
row: the CLS query attends to keys and values from norm1([CLS; P]), then CLS gets the attention output and the MLP on
its own row, exactly as in a normal block pass. The patch rows are returned unchanged, so P stays the head output in
every loop (not detached: gradients still reach the head blocks through P). No new parameters.

Only the CLS query is computed (one row instead of 197). Keys and values of the patches are recomputed in every loop:
they are the same every loop, so caching them would give the same numbers more cheaply, but this keeps the module
stateless.
"""
import torch
import torch.nn.functional as F
from torch import nn


def cls_query_and_keys_values(attn: nn.Module, normed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(q for the CLS row (B, heads, 1, head_dim), k and v for all tokens (B, heads, N, head_dim)) from the block's
    fused qkv weight: rows [0, C) are the query, [C, 3C) keys and values."""
    batch, n_tokens, dim = normed.shape
    weight, bias = attn.qkv.weight, attn.qkv.bias
    q = F.linear(normed[:, :1], weight[:dim], None if bias is None else bias[:dim])
    kv = F.linear(normed, weight[dim:], None if bias is None else bias[dim:])
    q = q.reshape(batch, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
    k, v = kv.reshape(batch, n_tokens, 2, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    return attn.q_norm(q), attn.k_norm(k), v


def cls_only_block_forward(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """One timm `Block` (timm 1.0.30) applied to the CLS row only; patch rows of `x` are returned unchanged."""
    attn = block.attn
    if getattr(attn, "gate", None) is not None:
        raise ValueError("gated attention is not supported by the CLS-only loop")
    batch, _, dim = x.shape
    q, k, v = cls_query_and_keys_values(attn, block.norm1(x))
    dropout = attn.attn_drop.p if block.training else 0.0
    mixed = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)  # (B, heads, 1, head_dim)
    mixed = mixed.transpose(1, 2).reshape(batch, 1, dim)
    cls = x[:, :1] + block.drop_path1(block.ls1(attn.proj_drop(attn.proj(attn.norm(mixed)))))
    cls = cls + block.drop_path2(block.ls2(block.mlp(block.norm2(cls))))
    return torch.cat([cls, x[:, 1:]], dim=1)


class CLSOnlyBlock(nn.Module):
    """Wraps a timm `Block`; forward updates only the CLS row (`cls_only_block_forward`)."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cls_only_block_forward(self.block, x)
