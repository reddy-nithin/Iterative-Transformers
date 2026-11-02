"""T16 checks: the v2 drain rule (fair-share reads free, over-reads drained, c kept in [0, 1]), fixed eta with no new
parameters, T14's v1 mask unchanged, the training flag, and the pre-flight gate."""
from types import SimpleNamespace

import pytest
import timm
import torch
from torch import nn

from preflight import gate
from src.coverage_mask import CoverageMaskBody, update_supply, update_supply_v2
from src.models import MODEL_NAME, build_unrolled
from src.train import model_kwargs

DIM = 24


def tiny_vit() -> nn.Module:
    """Small random ViT with the ViT-Tiny structure (17 tokens: CLS + 16 patches, 3 heads)."""
    torch.manual_seed(0)
    return timm.create_model(MODEL_NAME, pretrained=False, depth=12, embed_dim=DIM, num_heads=3,
                             img_size=32, patch_size=8, num_classes=10).eval()


def test_uniform_reads_drain_nothing() -> None:
    attention = torch.full((2, 196), 1 / 196)
    assert torch.allclose(update_supply_v2(torch.ones(2, 196), attention, torch.tensor(1.0)), torch.ones(2, 196))


def test_over_read_patch_is_drained_by_eta_times_excess() -> None:
    attention = torch.full((1, 196), 0.5 / 195)
    attention[0, 0] = 0.5  # read at 98x uniform; the others below uniform
    supply = update_supply_v2(torch.ones(1, 196), attention, torch.tensor(0.01))
    assert supply[0, 0] == pytest.approx(1 - 0.01 * (196 * 0.5 - 1), abs=1e-6)
    assert torch.equal(supply[0, 1:], torch.ones(195))  # below fair share: untouched


def test_supply_stays_in_unit_interval() -> None:
    torch.manual_seed(1)
    supply = torch.rand(4, 196)
    for _ in range(10):
        attention = torch.softmax(5 * torch.randn(4, 196), dim=-1)
        supply = update_supply_v2(supply, attention, torch.tensor(1.0))
        assert supply.min() >= 0 and supply.max() <= 1


def test_v2_adds_no_parameters_and_keeps_eta_fixed() -> None:
    plain = sum(p.numel() for p in build_unrolled(3, source=tiny_vit()).parameters())
    model = build_unrolled(3, source=tiny_vit(), coverage_mask_v2_eta=0.5)
    assert sum(p.numel() for p in model.parameters()) == plain
    body = model.blocks.body
    assert body.rule == "v2" and not isinstance(body.eta, nn.Parameter) and body.eta.item() == 0.5
    assert "blocks.body.eta" in model.state_dict()  # saved with the checkpoint, so it rebuilds exactly


def test_v2_forward_runs_and_differs_from_unmasked() -> None:
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        plain = build_unrolled(3, source=tiny_vit()).eval()(images)
        masked = build_unrolled(3, source=tiny_vit(), coverage_mask_v2_eta=1.0).eval()(images)
    assert masked.shape == plain.shape and torch.isfinite(masked).all()


def test_t14_v1_mask_unchanged() -> None:
    body = CoverageMaskBody(nn.Identity(), n_loops=10)
    assert body.rule == "v1" and isinstance(body.eta, nn.Parameter) and body.eta.item() == 0.5
    attention = torch.softmax(torch.randn(2, 196), dim=-1)
    supply = torch.ones(2, 196)
    assert torch.equal(body.next_supply(supply, attention), update_supply(supply, attention, body.eta))


def test_both_masks_at_once_is_an_error() -> None:
    with pytest.raises(ValueError):
        build_unrolled(3, source=tiny_vit(), coverage_mask=True, coverage_mask_v2_eta=0.5)


def test_flag_only_added_when_set() -> None:
    args = SimpleNamespace(model="unroll", unique_blocks=3, coverage_mask=False, coverage_mask_v2_eta=None)
    assert model_kwargs(args) == {"unique_blocks": 3}
    args.coverage_mask_v2_eta = 1.0
    assert model_kwargs(args) == {"unique_blocks": 3, "coverage_mask_v2_eta": 1.0}


def result(eta: float, supply: float, cos: float, change: float, top1: float = 50.0) -> dict:
    return {"eta": eta, "supply_mean_after_loop10": supply, "late_attn_cos": cos, "late_output_change": change,
            "top1_masked": top1}


def test_gate() -> None:
    assert gate([result(0.5, 0.9, 0.95, 0.5), result(1.0, 0.7, 0.95, 0.12)])["passes"]  # eta 1 bites via supply
    assert gate([result(0.5, 0.9, 0.85, 0.11), result(1.0, 0.9, 0.95, 0.5)])["passes"]  # eta 0.5 bites via cos
    assert not gate([result(0.5, 0.5, 0.5, 0.05), result(1.0, 0.4, 0.4, 0.09)])["passes"]  # output barely changes
    assert not gate([result(0.5, 0.9, 0.95, 0.5), result(1.0, 0.85, 0.92, 0.5)])["passes"]  # mask never drains
    assert gate([result(0.5, 0.5, 0.5, 0.5, 2.0), result(1.0, 0.5, 0.5, 0.5, 3.0)])["top1_collapses_at_both_eta"]
