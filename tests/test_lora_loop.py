import copy

import pytest
import torch

from src.models import build_baseline, build_iterative, build_lora_loop

LINEARS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
IMG = torch.randn(2, 3, 224, 224)


def _n(m):
    return sum(p.numel() for p in m.parameters())


def _linear_dims(model):
    block = model.blocks[0] if not hasattr(model.blocks, "shared") else model.blocks.shared
    mods = dict(block.named_modules())
    return [(mods[n].in_features, mods[n].out_features) for n in LINEARS]


def test_single_shared_block():
    m = build_lora_loop(rank=2, pretrained=False)
    assert len({id(p) for p in m.blocks.shared.parameters()}) == len(list(m.blocks.shared.parameters()))
    assert m.blocks.depth == 12


@pytest.mark.parametrize("rank", [1, 2, 8])
def test_param_count_formula(rank):
    shared = build_iterative(pretrained=False)
    lora = build_lora_loop(rank=rank, pretrained=False)
    extra = 12 * rank * sum(i + o for i, o in _linear_dims(lora))
    assert _n(lora) - _n(shared) == extra


def test_gradients_reach_shared_and_every_lora_factor():
    m = build_lora_loop(rank=2, pretrained=False, lora_init="svd")
    m(IMG).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.blocks.shared.parameters())
    lora_params = [p for n, p in m.named_parameters() if "lora_" in n]
    assert len(lora_params) == 12 * 2 * len(LINEARS)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_params)


def test_zero_init_matches_plain_shared_model():
    torch.manual_seed(0)
    plain = build_iterative(pretrained=False, init="avg").eval()
    lora = build_lora_loop(rank=2, pretrained=False, lora_init="zero").eval()
    lora.load_state_dict({k: v for k, v in plain.state_dict().items() if "blocks" not in k}, strict=False)
    lora.blocks.shared.load_state_dict(plain.blocks[0].state_dict())
    with torch.no_grad():
        assert torch.allclose(plain(IMG), lora(IMG), atol=1e-5)


def test_avg_init_is_mean_of_pretrained_blocks():
    base = build_baseline(pretrained=False)
    originals = copy.deepcopy(base.blocks)  # builders swap source.blocks in place
    m = build_iterative(pretrained=False, init="avg", source=base)
    assert not torch.allclose(originals[0].mlp.fc1.weight, originals[1].mlp.fc1.weight)
    want = torch.stack([b.mlp.fc1.weight for b in originals]).mean(0)
    assert torch.allclose(m.blocks[0].mlp.fc1.weight, want, atol=1e-6)


def test_svd_init_exact_at_full_rank():
    base = build_baseline(pretrained=False)
    originals = copy.deepcopy(base.blocks)
    # rank 192 = full rank for every ViT-Tiny linear: shared W + B@A must rebuild each original block
    m = build_lora_loop(rank=192, pretrained=False, lora_init="svd", source=base)
    shared_w = m.blocks.shared.attn.proj.weight
    for i, b in enumerate(originals):
        delta = m.blocks.lora_B["attn_proj"][i] @ m.blocks.lora_A["attn_proj"][i]
        assert torch.allclose(shared_w + delta, b.attn.proj.weight, atol=1e-4)


def test_svd_init_beats_zero_init_at_low_rank():
    base = build_baseline(pretrained=False)
    originals = copy.deepcopy(base.blocks)
    m = build_lora_loop(rank=4, pretrained=False, lora_init="svd", source=base)
    shared_w = m.blocks.shared.mlp.fc1.weight
    for i, b in enumerate(originals):
        target = b.mlp.fc1.weight
        delta = m.blocks.lora_B["mlp_fc1"][i] @ m.blocks.lora_A["mlp_fc1"][i]
        assert (shared_w + delta - target).norm() < (shared_w - target).norm()


def test_unknown_options_rejected():
    with pytest.raises(ValueError):
        build_iterative(pretrained=False, init="nope")
    with pytest.raises(ValueError):
        build_lora_loop(rank=2, pretrained=False, lora_init="nope")
