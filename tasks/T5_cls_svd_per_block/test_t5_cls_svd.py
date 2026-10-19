"""T5 checks: CLS collection hooks (normal and shared-block models) and the rank measures."""
import pytest
import timm
import torch
from torch import nn

from cls_svd import (analyse, collect_cls_per_layer, effective_rank, energy_rank, resolve_class_indices,
                     singular_values)
from src.models import MODEL_NAME, build_iterative

DEPTH, DIM, N_IMAGES = 3, 32, 6


def tiny_vit() -> nn.Module:
    """Small random ViT with the same structure as ViT-Tiny (CLS token, pre-norm blocks, final norm)."""
    torch.manual_seed(0)
    return timm.create_model(MODEL_NAME, pretrained=False, depth=DEPTH, embed_dim=DIM, num_heads=2,
                             img_size=32, patch_size=8, num_classes=10)


def fake_loader(batches: int = 2) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """A few (images, labels) batches; collection only needs something iterable."""
    torch.manual_seed(1)
    return [(torch.randn(N_IMAGES // batches, 3, 32, 32), torch.zeros(N_IMAGES // batches)) for _ in range(batches)]


def test_shape_is_layers_by_images_by_dim() -> None:
    cls = collect_cls_per_layer(tiny_vit(), fake_loader(), torch.device("cpu"))
    assert cls.shape == (DEPTH + 1, N_IMAGES, DIM)


def test_last_layer_matches_forward_features_before_final_norm() -> None:
    model = tiny_vit().eval()
    loader = fake_loader(batches=1)
    cls = collect_cls_per_layer(model, loader, torch.device("cpu"))
    model.norm = nn.Identity()
    with torch.no_grad():
        expected = model.forward_features(loader[0][0])[:, 0, :]
    torch.testing.assert_close(cls[-1], expected)


def test_shared_block_still_gives_one_snapshot_per_slot() -> None:
    model = build_iterative(source=tiny_vit())
    assert len({id(b) for b in model.blocks}) == 1
    cls = collect_cls_per_layer(model, fake_loader(), torch.device("cpu"))
    assert cls.shape == (DEPTH + 1, N_IMAGES, DIM)
    assert not torch.allclose(cls[1], cls[2])  # same weights, different inputs -> different outputs


def test_hooks_are_removed() -> None:
    model = tiny_vit()
    collect_cls_per_layer(model, fake_loader(), torch.device("cpu"))
    assert not model.blocks._forward_pre_hooks
    assert all(not block._forward_hooks for block in model.blocks)


def test_layer_zero_is_identical_so_centred_rank_is_zero() -> None:
    cls = collect_cls_per_layer(tiny_vit(), fake_loader(), torch.device("cpu"))
    layer0 = analyse(cls)["0"]
    assert layer0["centred"]["effective_rank"] == 0.0
    assert layer0["centred"]["energy_rank_90"] == 0
    assert layer0["raw"]["energy_rank_90"] == 1  # every row is the same vector -> rank 1


def test_centring_identical_rows_gives_exact_zeros() -> None:
    rows = torch.randn(1, DIM).repeat(50, 1)
    s = singular_values(rows, centre=True)
    assert torch.all(s == 0) and not torch.isnan(s).any()


def test_effective_rank_of_known_matrices() -> None:
    rank_one = torch.outer(torch.randn(40, dtype=torch.float64), torch.randn(DIM, dtype=torch.float64))
    assert effective_rank(singular_values(rank_one, centre=False)) == pytest.approx(1.0)
    assert effective_rank(singular_values(torch.eye(8), centre=False)) == pytest.approx(8.0)
    assert effective_rank(torch.zeros(5)) == 0.0


def test_energy_rank_of_known_spectrum() -> None:
    s = torch.tensor([3.0, 1.0, 0.0])  # energies 9, 1, 0 -> top 1 holds 90%
    assert energy_rank(s, 0.9) == 1
    assert energy_rank(s, 0.95) == 2
    assert energy_rank(torch.zeros(3)) == 0


def test_unknown_class_name_raises() -> None:
    assert resolve_class_indices(["apple", "fox"], ["fox"]) == [1]
    with pytest.raises(ValueError, match="unicorn"):
        resolve_class_indices(["apple", "fox"], ["unicorn"])
