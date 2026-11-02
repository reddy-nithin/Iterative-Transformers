"""Coverage mask on the looped block's CLS attention (T14, Gemini B1).

The looped body keeps a per-image supply vector c over the 196 patches, c = 1 before loop 1. In every loop, log(c + 1e-6)
is added to the CLS query's attention logits over the patches (all heads; the CLS->CLS logit and every patch query row
are unchanged), and afterwards the patches CLS just read are used up:
    c <- clamp(c - eta * A_t, min 0),   A_t = CLS->patch attention averaged over heads (CLS->CLS dropped, renormalised).
eta is one learnable scalar. The idea (H1): CLS gets a memory of which patches it already read, so later loops look
elsewhere.

Rule "v2" (T16) keeps the same bias but drains only patches read more than their fair share, with a FIXED eta (a
buffer, not a parameter, so the model has no new parameters):
    c <- clamp(c - eta * max(0, P * A_t - 1), 0, 1),   P = 196 patches.
A patch read at exactly the uniform rate 1/P loses nothing; one read at 4.9x uniform loses eta * 3.9.

`CoverageMaskLoop` replaces `VisionTransformer.blocks` of an unrolled model (head blocks, then the body x n_loops, no
tail). Like `src.head_body_tail.InjectedLoop`, iterating it gives the blocks in call order, so code that walks
`model.blocks` or hooks every unique block (per-loop readout, T5 separation, T12 recorder) sees the body once per loop.
"""
from collections.abc import Iterator

import torch
from torch import nn

SUPPLY_EPS = 1e-6  # log(c + eps): a used-up patch gets about -13.8 added to its logit
ETA_INIT = 0.5


def cls_patch_attention(probs: torch.Tensor) -> torch.Tensor:
    """(B, P) CLS query's attention over the patch keys: CLS->CLS dropped, renormalised, averaged over heads."""
    patches = probs[:, :, 0, 1:]
    patches = patches / patches.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    return patches.mean(dim=1)


def supply_bias(logits: torch.Tensor, supply: torch.Tensor) -> torch.Tensor:
    """`logits` (B, heads, N, N) with log(supply + eps) added to row 0 (CLS query), columns 1: (patch keys) only."""
    patch_bias = torch.log(supply + SUPPLY_EPS).to(logits.dtype)[:, None, None, :]  # (B, 1, 1, P)
    cls_row = logits[:, :, :1]
    cls_row = torch.cat([cls_row[..., :1], cls_row[..., 1:] + patch_bias], dim=-1)
    return torch.cat([cls_row, logits[:, :, 1:]], dim=2)


def update_supply(supply: torch.Tensor, cls_attention: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """c <- clamp(c - eta * A_t, min 0), exactly as in the T14 spec (no upper clamp)."""
    return torch.clamp(supply - eta * cls_attention, min=0.0)


def update_supply_v2(supply: torch.Tensor, cls_attention: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """c <- clamp(c - eta * max(0, P * A_t - 1), 0, 1) (T16): only above-fair-share reads use up supply."""
    over_read = torch.relu(cls_attention.shape[-1] * cls_attention - 1.0)
    return torch.clamp(supply - eta * over_read, min=0.0, max=1.0)


SUPPLY_RULES = {"v1": update_supply, "v2": update_supply_v2}


def masked_block_parts(block: nn.Module, x: torch.Tensor, supply: torch.Tensor) -> dict[str, torch.Tensor]:
    """One timm `Block` (timm 1.0.30) computed from its own modules with the supply bias on the CLS query row.

    Returns output (B, N, C), logits and probs (B, heads, N, N) after the bias, attention sublayer output after proj
    (B, N, C) and the head-averaged CLS->patch attention A_t (B, P).
    """
    attn = block.attn
    if getattr(attn, "gate", None) is not None:
        raise ValueError("gated attention is not supported by the coverage mask")
    batch, n_tokens, _ = x.shape
    qkv = attn.qkv(block.norm1(x)).reshape(batch, n_tokens, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    logits = supply_bias((q * attn.scale) @ k.transpose(-2, -1), supply)
    probs = logits.softmax(dim=-1)
    mixed = (attn.attn_drop(probs) @ v).transpose(1, 2).reshape(batch, n_tokens, -1)
    sublayer = attn.proj_drop(attn.proj(attn.norm(mixed)))
    after_attn = x + block.drop_path1(block.ls1(sublayer))
    output = after_attn + block.drop_path2(block.ls2(block.mlp(block.norm2(after_attn))))
    return {"output": output, "logits": logits, "probs": probs, "attn_out": sublayer,
            "cls_attention": cls_patch_attention(probs)}


class CoverageMaskBody(nn.Module):
    """The looped block with a coverage mask. Stateful within one forward pass: the first call sets c = 1, each call
    applies the mask and uses up supply, and after `n_loops` calls the state is cleared for the next batch.

    rule "v1" (T14): learnable eta, c <- clamp(c - eta * A_t, min 0). rule "v2" (T16): fixed eta (buffer),
    c <- clamp(c - eta * max(0, P * A_t - 1), 0, 1)."""

    def __init__(self, block: nn.Module, n_loops: int, eta_init: float = ETA_INIT, rule: str = "v1"):
        super().__init__()
        if n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {n_loops}")
        if rule not in SUPPLY_RULES:
            raise ValueError(f"rule must be one of {sorted(SUPPLY_RULES)}, got {rule!r}")
        self.block = block
        self.n_loops = n_loops
        self.rule = rule
        if rule == "v1":
            self.eta = nn.Parameter(torch.tensor(float(eta_init)))
        else:
            self.register_buffer("eta", torch.tensor(float(eta_init)))
        self.supply: torch.Tensor | None = None
        self.calls = 0

    def next_supply(self, supply: torch.Tensor, cls_attention: torch.Tensor) -> torch.Tensor:
        """Supply after one loop under this body's rule."""
        return SUPPLY_RULES[self.rule](supply, cls_attention, self.eta)

    def reset(self) -> None:
        """Forget the supply (start of a new pass)."""
        self.supply, self.calls = None, 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.supply is None:
            self.supply = torch.ones(x.shape[0], x.shape[1] - 1, device=x.device, dtype=torch.float32)
        parts = masked_block_parts(self.block, x, self.supply)
        self.supply = self.next_supply(self.supply, parts["cls_attention"].float())
        self.calls += 1
        if self.calls == self.n_loops:
            self.reset()
        return parts["output"]


class CoverageMaskLoop(nn.Module):
    """Drop-in replacement for `VisionTransformer.blocks`: head blocks, then the masked body run `n_loops` times.

    `len()` and iteration give the blocks in call order (body listed `n_loops` times), like the `nn.Sequential` of
    the unmasked unrolled model.
    """

    def __init__(self, head: list[nn.Module], body: nn.Module, n_loops: int, eta_init: float = ETA_INIT,
                 rule: str = "v1"):
        super().__init__()
        self.head = nn.ModuleList(head)
        self.body = CoverageMaskBody(body, n_loops, eta_init, rule)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.body.reset()
        for block in self.head:
            x = block(x)
        for _ in range(self.body.n_loops):
            x = self.body(x)
        return x

    def __len__(self) -> int:
        return len(self.head) + self.body.n_loops

    def __iter__(self) -> Iterator[nn.Module]:
        return iter([*self.head, *[self.body] * self.body.n_loops])
