"""Model builders: pretrained ViT-Tiny baseline and the iterative (shared-block) variant."""
import timm
from torch import nn

MODEL_NAME = "vit_tiny_patch16_224"
NUM_CLASSES = 100


def build_baseline(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Normal ViT-Tiny: 12 different blocks. Head is re-initialised for `num_classes`."""
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)


def build_iterative(
    num_classes: int = NUM_CLASSES, pretrained: bool = True, block_idx: int = 0
) -> nn.Module:
    """ViT-Tiny whose 12 blocks are ONE pretrained block (`block_idx`) applied 12 times.

    `[block] * depth` puts the *same object* in the Sequential 12 times, so there is one
    set of weights and `.parameters()` counts them once.
    """
    model = build_baseline(num_classes, pretrained)
    depth = len(model.blocks)
    shared = model.blocks[block_idx]
    model.blocks = nn.Sequential(*[shared] * depth)
    return model


def build(name: str, **kwargs) -> nn.Module:
    builders = {"baseline": build_baseline, "iterative": build_iterative}
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; choose from {list(builders)}")
    return builders[name](**kwargs)
