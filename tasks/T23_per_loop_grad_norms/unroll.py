"""Unroll a looped (shared) block so every loop has its own parameter copy, and measure each loop's gradient (T23).

The shared block is one module object that appears several times in `model.blocks`. Unrolled, loop t runs that module
through `torch.func.functional_call` with its own leaf copies W_t of the parameters (same values). Then
g_t = dL/dW_t is loop t's share of the gradient, and sum_t g_t equals the normal shared gradient.
"""
from collections.abc import Sequence

import torch
from torch import nn
from torch.func import functional_call

ParamDict = dict[str, torch.Tensor]

LAYER_GROUPS = ("norm1", "attn.qkv", "attn.proj", "norm2", "mlp.fc1", "mlp.fc2")
LAYERSCALE_PREFIXES = ("ls1.", "ls2.")
WHOLE_BLOCK = "block"


def shared_block(blocks: Sequence[nn.Module]) -> tuple[nn.Module, list[int]]:
    """The one module that appears more than once in `blocks`, and the positions where it appears."""
    ids = [id(block) for block in blocks]
    repeated = {block_id for block_id in ids if ids.count(block_id) > 1}
    if len(repeated) != 1:
        raise ValueError(f"expected exactly one repeated block, found {len(repeated)}")
    positions = [index for index, block_id in enumerate(ids) if block_id in repeated]
    return blocks[positions[0]], positions


def parameter_groups(block: nn.Module) -> dict[str, list[str]]:
    """Parameter names per group: the 6 layer groups, LayerScale if present, and the whole block.

    Raises if a parameter belongs to no group, so a block layout we did not expect cannot be silently half-measured.
    """
    names = [name for name, _ in block.named_parameters()]
    groups = {group: [name for name in names if name.startswith(group + ".")] for group in LAYER_GROUPS}
    layerscale = [name for name in names if name.startswith(LAYERSCALE_PREFIXES)]
    if layerscale:
        groups["layerscale"] = layerscale
    grouped = {name for members in groups.values() for name in members}
    missing = [name for name in names if name not in grouped]
    if missing:
        raise ValueError(f"parameters in no group: {missing}")
    groups[WHOLE_BLOCK] = names
    return groups


def loop_copies(block: nn.Module, n_loops: int) -> list[ParamDict]:
    """One dict of leaf copies (same values, requires_grad) of the block's parameters per loop."""
    return [{name: param.detach().clone().requires_grad_(True) for name, param in block.named_parameters()}
            for _ in range(n_loops)]


def apply_blocks(blocks: Sequence[nn.Module], shared: nn.Module, x: torch.Tensor,
                 copies: list[ParamDict]) -> tuple[torch.Tensor, list[float]]:
    """Run `blocks` in order; the t-th use of `shared` runs with `copies[t]`. Returns the output and ‖x_t‖ after
    every use of the shared block (mean over all leading dims of the last-dim norm)."""
    residual_norms, loop = [], 0
    for block in blocks:
        if block is shared:
            x = functional_call(shared, copies[loop], (x,))
            residual_norms.append(x.detach().norm(dim=-1).mean().item())
            loop += 1
        else:
            x = block(x)
    if loop != len(copies):
        raise ValueError(f"{len(copies)} parameter copies for {loop} uses of the shared block")
    return x, residual_norms


def vit_unrolled_logits(model: nn.Module, images: torch.Tensor, shared: nn.Module,
                        copies: list[ParamDict]) -> tuple[torch.Tensor, list[float]]:
    """timm VisionTransformer.forward with the loop unrolled (same steps as forward_features + forward_head)."""
    x = model.patch_embed(images)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)
    x, residual_norms = apply_blocks(list(model.blocks), shared, x, copies)
    return model.forward_head(model.norm(x)), residual_norms


def per_loop_gradients(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                       criterion: nn.Module) -> tuple[list[ParamDict], list[float], torch.Tensor]:
    """(g_t per loop as {name: grad}, ‖x_t‖ per loop, logits) for one batch. No weight is changed."""
    shared, positions = shared_block(list(model.blocks))
    copies = loop_copies(shared, len(positions))
    logits, residual_norms = vit_unrolled_logits(model, images, shared, copies)
    flat = [param for copy in copies for param in copy.values()]
    grads = torch.autograd.grad(criterion(logits, labels), flat)
    names = list(copies[0])
    per_param = len(names)
    per_loop = [dict(zip(names, grads[t * per_param:(t + 1) * per_param])) for t in range(len(copies))]
    return per_loop, residual_norms, logits.detach()


def shared_gradient(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                    criterion: nn.Module) -> tuple[ParamDict, torch.Tensor]:
    """The normal gradient of the shared block (one parameter set used every loop), and the normal logits."""
    shared, _ = shared_block(list(model.blocks))
    names, params = zip(*shared.named_parameters())
    logits = model(images)
    grads = torch.autograd.grad(criterion(logits, labels), params)
    return dict(zip(names, grads)), logits.detach()


def gate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, criterion: nn.Module) -> dict:
    """Check on one batch: sum_t g_t vs the shared gradient (relative error) and unrolled vs normal logits."""
    per_loop, _, unrolled_logits = per_loop_gradients(model, images, labels, criterion)
    shared, normal_logits = shared_gradient(model, images, labels, criterion)
    summed = torch.cat([sum(loop[name] for loop in per_loop).flatten() for name in shared])
    reference = torch.cat([grad.flatten() for grad in shared.values()])
    return {"grad_rel_error": ((summed - reference).norm() / reference.norm()).item(),
            "logits_max_abs_diff": (unrolled_logits - normal_logits).abs().max().item()}


def group_metrics(per_loop: list[ParamDict], names: list[str]) -> dict:
    """One batch, one parameter group: ‖g_t‖ per loop, agreement with the sum, mean pairwise cos between loops."""
    vectors = torch.stack([torch.cat([loop[name].flatten() for name in names]) for loop in per_loop]).double()
    norms = vectors.norm(dim=1)
    total = vectors.sum(dim=0)
    agreement = (vectors @ total / (norms * total.norm())).mean()
    unit = vectors / norms[:, None]
    cosines = unit @ unit.T
    upper = torch.triu(torch.ones_like(cosines, dtype=torch.bool), diagonal=1)
    return {"norms": norms.tolist(), "agreement": agreement.item(), "pairwise_cos": cosines[upper].mean().item()}


def summarise(batches: list[dict]) -> dict:
    """Average one group's per-batch metrics; spread and dominance from the batch-averaged norms."""
    norms = torch.tensor([batch["norms"] for batch in batches], dtype=torch.float64).mean(dim=0)
    top_two = norms.sort(descending=True).values[:2].sum()
    return {"norm_per_loop": norms.tolist(),
            "spread": (norms.max() / norms.min()).item(),
            "dominance": (top_two / norms.sum()).item(),
            "agreement": sum(batch["agreement"] for batch in batches) / len(batches),
            "pairwise_cos": sum(batch["pairwise_cos"] for batch in batches) / len(batches)}
