"""T12 checks: the block recomputation matches timm's own forward, the body/tail finder handles every looped layout,
and each per-loop metric gives the known answer on hand-made tensors."""
import math

import pytest
import timm
import torch
from torch import nn

from loop_diagnostics import (BodyRecorder, attention_cosines, cls_patch_attention, collect_exit_accuracy,
                              collect_loop_metrics, decision, entropy, find_body, loop_states, new_direction_share,
                              readout_kind, recompute_block, relative_update)
from src.models import MODEL_NAME

DIM = 24


def tiny_vit(depth: int = 12) -> nn.Module:
    """Small random ViT with the ViT-Tiny structure (17 tokens: CLS + 16 patches, 3 heads)."""
    torch.manual_seed(0)
    return timm.create_model(MODEL_NAME, pretrained=False, depth=depth, embed_dim=DIM, num_heads=3,
                             img_size=32, patch_size=8, num_classes=10).eval()


def looped(model: nn.Module, layout: str) -> nn.Module:
    """Rewire `model.blocks` as iter (block 0 x12), hbt (head, body x10, tail) or unroll n=3 (b0, b1, b2 x10)."""
    b = list(model.blocks)
    model.blocks = {"iter": lambda: nn.Sequential(*[b[0]] * 12),
                    "hbt": lambda: nn.Sequential(b[0], *[b[1]] * 10, b[2]),
                    "unroll": lambda: nn.Sequential(b[0], b[1], *[b[2]] * 10)}[layout]()
    return model


def test_recompute_matches_the_real_block() -> None:
    block = tiny_vit().blocks[0]
    x = torch.randn(4, 17, DIM)
    with torch.no_grad():
        parts = recompute_block(block, x)
        assert (parts["output"] - block(x)).abs().max() < 1e-5
        assert torch.allclose(parts["probs"].sum(dim=-1), torch.ones(4, 3, 17), atol=1e-5)


@pytest.mark.parametrize("layout, body_pos, tail_len", [("iter", 0, 0), ("hbt", 1, 1), ("unroll", 2, 0)])
def test_find_body_and_tail(layout: str, body_pos: int, tail_len: int) -> None:
    model = looped(tiny_vit(), layout)
    body, tail = find_body(model)
    assert body is model.blocks[body_pos] and len(tail) == tail_len


def test_find_body_rejects_unlooped_models() -> None:
    with pytest.raises(ValueError):
        find_body(tiny_vit())


def test_readout_kind() -> None:
    assert readout_kind("hbt", [nn.Identity()]) == "tail"
    assert readout_kind("iterative", []) == "norm"
    assert readout_kind("unroll", []) == "norm"
    assert readout_kind("baseline", []) is None


def test_relative_update_known_values() -> None:
    previous = torch.ones(1, 3, 4)
    current = previous.clone()
    current[0, 0] *= 1.5  # CLS grows by half its norm
    current[0, 1] *= 2.0  # one patch doubles, the other is unchanged
    rel_cls, rel_patch = relative_update(previous, current)
    assert torch.allclose(rel_cls, torch.tensor([0.5])) and torch.allclose(rel_patch, torch.tensor([0.5]))


def test_cls_attention_drops_cls_and_renormalises() -> None:
    probs = torch.zeros(1, 1, 3, 3)
    probs[0, 0, 0] = torch.tensor([0.5, 0.25, 0.25])
    assert torch.allclose(cls_patch_attention(probs), torch.tensor([[[0.5, 0.5]]]))


def test_entropy_and_cosine_extremes() -> None:
    uniform = torch.full((1, 196), 1 / 196)
    assert math.isclose(entropy(uniform).item(), math.log(196), rel_tol=1e-5)
    one_hot = torch.zeros(1, 196)
    one_hot[0, 3] = 1.0
    assert entropy(one_hot).item() == 0.0
    a = torch.softmax(torch.randn(2, 3, 196), dim=-1)
    per_head, head_avg = attention_cosines(a, a)
    assert torch.allclose(per_head, torch.ones(2)) and torch.allclose(head_avg, torch.ones(2))


def test_new_direction_share() -> None:
    cls = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(new_direction_share(torch.tensor([[3.0, 0.0]]), cls), torch.tensor([0.0]))
    assert torch.allclose(new_direction_share(torch.tensor([[0.0, 2.0]]), cls), torch.tensor([1.0]))
    assert torch.allclose(new_direction_share(torch.tensor([[1.0, 1.0]]), cls), torch.tensor([math.sqrt(0.5)]))


def test_loop_states_from_recorded_calls() -> None:
    model = looped(tiny_vit(), "hbt")
    body, _ = find_body(model)
    with torch.no_grad(), BodyRecorder(body) as recorder:
        model.forward_features(torch.randn(2, 3, 32, 32))
    states = loop_states(recorder.calls)
    assert len(states) == 11  # x_0 = head output, then 10 loops


def test_collect_loop_metrics_shapes() -> None:
    model = looped(tiny_vit(), "iter")
    loader = [(torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))] * 2
    per_image = collect_loop_metrics(model, loader, torch.device("cpu"))
    assert per_image["rel_cls"].shape == (12, 8) and per_image["attn_cos"].shape == (11, 8)
    assert per_image["cls_update_cos"].shape == (11, 8)


@pytest.mark.parametrize("layout, kind", [("iter", "norm"), ("hbt", "tail")])
def test_last_loop_readout_equals_the_model_output(layout: str, kind: str) -> None:
    model = looped(tiny_vit(), layout)
    images = torch.randn(6, 3, 32, 32)
    with torch.no_grad():
        labels = model(images).argmax(dim=1)
    exits = collect_exit_accuracy(model, [(images, labels)], torch.device("cpu"), kind)
    assert exits["full_depth_top1"] == 100.0 and exits["acc_per_loop"][-1] == 100.0


def test_decision_rule() -> None:
    n = 12
    base = {"attn_cos_headavg": [0.0] * (n - 1)}
    pick = lambda cos, ratio: decision({**base, "attn_cos": [cos] * (n - 1), "ratio": [ratio] * n})["pick"]  # noqa: E731
    assert pick(0.9, 0.1) == "coverage mask first, then CLS adapters"
    assert pick(0.9, 0.5) == "coverage mask"
    assert pick(0.5, 0.1) == "CLS adapters"
    assert pick(0.5, 0.5) == "per-loop LayerNorm"
    early_only = decision({**base, "attn_cos": [0.99] * 3 + [0.1] * (n - 4), "ratio": [0.5] * n})
    assert early_only["pick"] == "per-loop LayerNorm"  # loops 1-3 do not count
