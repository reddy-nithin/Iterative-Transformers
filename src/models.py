"""Model builders: ViT-Tiny baselines, the iterative (shared-block) variant, shared block + per-loop LoRA, and
head + repeated body + tail."""
import copy

import timm
import torch
from torch import nn

from src.head_body_tail import InjectedLoop
from src.lora_loop import LoRALoopBlocks

MODEL_NAME = "vit_tiny_patch16_224"
NUM_CLASSES = 100
INITS = ("block", "avg")


def build_baseline(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Normal ViT-Tiny: 12 different blocks. Head is re-initialised for `num_classes`."""
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)


def build_scratch(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Same architecture as the baseline with random weights (the 'zeroed weights' control)."""
    return build_baseline(num_classes, pretrained=False)


def _average_block(blocks: nn.ModuleList) -> nn.Module:
    """A fresh copy of blocks[0] whose every parameter is the mean over all blocks."""
    avg = copy.deepcopy(blocks[0])
    with torch.no_grad():
        for name, p in avg.named_parameters():
            p.copy_(torch.stack([dict(b.named_parameters())[name] for b in blocks]).mean(0))
    return avg


def _shared_block(source: nn.Module, init: str, block_idx: int) -> nn.Module:
    if init not in INITS:
        raise ValueError(f"unknown init {init!r}; choose from {list(INITS)}")
    return source.blocks[block_idx] if init == "block" else _average_block(source.blocks)


def build_iterative(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    block_idx: int = 0,
    init: str = "block",
    source: nn.Module | None = None,
) -> nn.Module:
    """ViT-Tiny whose 12 blocks are ONE block applied 12 times.

    `init="block"` reuses pretrained block `block_idx`; `init="avg"` uses the mean of all 12.
    `[block] * depth` puts the *same object* in the Sequential 12 times, so there is one
    set of weights and `.parameters()` counts them once. `source` lets tests pass in the
    model whose blocks to use.
    """
    model = source if source is not None else build_baseline(num_classes, pretrained)
    shared = _shared_block(model, init, block_idx)
    model.blocks = nn.Sequential(*[shared] * len(model.blocks))
    return model


def build_unrolled(
    unique_blocks: int,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    source: nn.Module | None = None,
) -> nn.Module:
    """Progressive unrolling: the first `unique_blocks - 1` pretrained blocks keep their own weights, and
    pretrained block `unique_blocks - 1` fills every remaining slot (one shared object).

    unique_blocks=1 is block 0 x12 (same as `build_iterative`); unique_blocks=12 is the normal ViT-Tiny.
    """
    model = source if source is not None else build_baseline(num_classes, pretrained)
    blocks = list(model.blocks)
    depth = len(blocks)
    if not 1 <= unique_blocks <= depth:
        raise ValueError(f"unique_blocks must be in 1..{depth}, got {unique_blocks}")
    distinct, repeated = blocks[: unique_blocks - 1], blocks[unique_blocks - 1]
    model.blocks = nn.Sequential(*distinct, *[repeated] * (depth - unique_blocks + 1))
    return model


def build_lora_loop(
    rank: int,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    init: str = "avg",
    block_idx: int = 0,
    lora_init: str = "svd",
    source: nn.Module | None = None,
) -> nn.Module:
    """Shared block (mean or single pretrained block) + rank-`rank` LoRA per iteration.

    `lora_init="svd"` starts each iteration's LoRA at the best rank-r fit of
    (original block i - shared block); `"zero"` starts identical to the plain shared model.
    """
    model = source if source is not None else build_baseline(num_classes, pretrained)
    originals = list(model.blocks)
    shared = _shared_block(model, init, block_idx)
    model.blocks = LoRALoopBlocks(shared, originals, rank, depth=len(originals), lora_init=lora_init)
    return model


def build_head_body_tail(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    head_blocks: int = 1,
    tail_blocks: int = 1,
    inject: bool = False,
    source: nn.Module | None = None,
) -> nn.Module:
    """Head + body + tail (T9): `head_blocks` free blocks, ONE body block repeated to fill the remaining steps, then
    `tail_blocks` free blocks before the final norm and classifier. Depth (execution steps) stays 12.

    Random init by default (trained from scratch). The model's first blocks supply distinct weights: head =
    blocks[0 : head_blocks], body = blocks[head_blocks], tail = the next `tail_blocks`. head 1 / tail 1 is the T3 meeting
    design (body x10); head 1 / tail 0 is the head-only control (body x11); head 0 / tail 0 is one random block x12.
    `inject=True` feeds the head output into every body step (`src.head_body_tail.InjectedLoop`); off, the blocks are a
    plain `nn.Sequential` with the body object repeated, as in `build_iterative`.
    """
    model = source if source is not None else build_baseline(num_classes, pretrained)
    blocks = list(model.blocks)
    depth = len(blocks)
    repeats = depth - head_blocks - tail_blocks
    if head_blocks < 0 or tail_blocks < 0 or repeats < 1:
        raise ValueError(f"need head_blocks, tail_blocks >= 0 and at least one body step; got head {head_blocks}, "
                         f"tail {tail_blocks}, depth {depth}")
    head = blocks[:head_blocks]
    body = blocks[head_blocks]
    tail = blocks[head_blocks + 1: head_blocks + 1 + tail_blocks]
    if inject:
        model.blocks = InjectedLoop(head, body, tail, repeats)
    else:
        model.blocks = nn.Sequential(*head, *[body] * repeats, *tail)
    return model


def build(name: str, **kwargs) -> nn.Module:
    builders = {
        "baseline": build_baseline,
        "scratch": build_scratch,
        "iterative": build_iterative,
        "lora": build_lora_loop,
        "unroll": build_unrolled,
        "hbt": build_head_body_tail,
    }
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; choose from {list(builders)}")
    return builders[name](**kwargs)
