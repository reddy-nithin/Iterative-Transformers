"""Model builders: ViT-Tiny baselines, the iterative (shared-block) variant, and shared block + per-loop LoRA."""
import copy

import timm
import torch
from torch import nn

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


def build(name: str, **kwargs) -> nn.Module:
    builders = {
        "baseline": build_baseline,
        "scratch": build_scratch,
        "iterative": build_iterative,
        "lora": build_lora_loop,
    }
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; choose from {list(builders)}")
    return builders[name](**kwargs)
