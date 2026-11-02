"""T19 checks: the CLS-update split adds up on a real timm block, uniform attention has no deviation part, the
halting rule stops where it should, and the decision line follows the spec's thresholds."""
import math

import timm
import torch

from decompose import collinear_exit, decision, split_cls_update, split_error
from src.models import MODEL_NAME

DIM = 24


def tiny_block() -> torch.nn.Module:
    """One block of a small random ViT with the ViT-Tiny structure (17 tokens: CLS + 16 patches, 3 heads)."""
    torch.manual_seed(0)
    model = timm.create_model(MODEL_NAME, pretrained=False, depth=3, embed_dim=DIM, num_heads=3,
                              img_size=32, patch_size=8, num_classes=10).eval()
    return model.blocks[2]


def test_parts_add_up_to_the_real_cls_update() -> None:
    block = tiny_block()
    x = torch.randn(4, 17, DIM)
    with torch.no_grad():
        parts = split_cls_update(block, x)
        real_update = block(x)[:, 0] - x[:, 0]
    assert split_error(parts) < 1e-5
    assert (parts["d"] - real_update).abs().max() < 1e-5
    assert (parts["a"] + parts["m"] - parts["d"]).abs().max() < 1e-6


def test_uniform_attention_has_no_deviation_part() -> None:
    block = tiny_block()
    with torch.no_grad():  # zero query weights and bias: every logit is 0, attention is uniform over the 17 keys
        block.attn.qkv.weight[:DIM] = 0
        block.attn.qkv.bias[:DIM] = 0
        parts = split_cls_update(block, torch.randn(2, 17, DIM))
    assert parts["dev"].abs().max() < 1e-6
    assert parts["mean"].norm() > 0


def test_collinear_exit() -> None:
    # 3 images, 5 loops: cos rows are loops 2..5, rel rows loops 1..5.
    cos_d = torch.tensor([[0.99, 0.99, 0.10], [0.99, 0.10, 0.10], [0.99, 0.95, 0.10], [0.99, 0.95, 0.10]])
    rel_d = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05],
                          [0.05, 0.05, 0.05]])
    stop = collinear_exit(cos_d, rel_d, theta=0.9, rho=0.1)
    # image 0: loop 3 qualifies (index 2); image 1: loop 3 cos 0.10 fails, loop 4 ok (index 3); image 2: never -> last
    assert stop.tolist() == [2, 3, 4]
    assert collinear_exit(cos_d, rel_d, theta=0.9, rho=0.01).tolist() == [4, 4, 4]  # rho too tight: nobody stops
    assert collinear_exit(cos_d, rel_d, theta=0.9, rho=math.inf).tolist() == [2, 3, 4]


def test_decision_line() -> None:
    late = lambda value: [0.0] * 5 + [value] * 5  # noqa: E731  (loops 1..10, only 6..10 matter)
    assert decision({"mean_share": late(0.6), "attn_frac": late(0.4)})["line"] == "run T20"
    assert decision({"mean_share": late(0.4), "attn_frac": late(0.4)})["line"].startswith("skip T20")
    assert decision({"mean_share": late(0.6), "attn_frac": late(0.2)})["line"].startswith("skip T20")
