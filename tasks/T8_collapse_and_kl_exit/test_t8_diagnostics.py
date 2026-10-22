"""T8 checks: token collapse measures, exit signals and the threshold rule, and the token hooks on a looped model."""
import math

import timm
import torch

from diagnostics import (cls_change_signal, collect_token_stats, compare_with_fixed, kl_signal, threshold_exit,
                         token_stats)
from cls_svd import effective_rank, singular_values
from src.models import MODEL_NAME, build_iterative

DEPTH, DIM = 3, 32


def tiny_vit() -> torch.nn.Module:
    """Small random ViT with the ViT-Tiny structure (17 tokens: CLS + 16 patches)."""
    torch.manual_seed(0)
    return timm.create_model(MODEL_NAME, pretrained=False, depth=DEPTH, embed_dim=DIM, num_heads=2,
                             img_size=32, patch_size=8, num_classes=10)


def test_identical_tokens_are_fully_collapsed() -> None:
    tokens = torch.randn(2, 1, DIM).expand(2, 17, DIM)
    stats = token_stats(tokens)
    assert torch.allclose(stats["cosine"], torch.ones(2), atol=1e-5)
    assert torch.all(stats["effective_rank"] == 0)
    assert torch.all(stats["spectrum"] == 0)


def test_effective_rank_matches_t5() -> None:
    torch.manual_seed(1)
    tokens = torch.randn(1, 17, DIM)
    expected = effective_rank(singular_values(tokens[0], centre=True))
    assert math.isclose(token_stats(tokens)["effective_rank"].item(), expected, rel_tol=1e-4)


def test_orthogonal_tokens_have_zero_cosine() -> None:
    assert token_stats(torch.eye(DIM)[:4].unsqueeze(0))["cosine"].abs().item() < 1e-6


def test_kl_is_zero_for_identical_predictions_and_inf_at_first_exit() -> None:
    logits = torch.randn(1, 5, 10).expand(4, 5, 10)
    kl = kl_signal(logits)
    assert torch.isinf(kl[0]).all()
    assert kl[1:].abs().max() < 1e-6


def test_cls_change_signal() -> None:
    cls = torch.zeros(3, 2, 4)
    cls[1, :, 0] = 3.0
    cls[2, :, 0] = 3.0
    cls[2, :, 1] = 4.0
    assert torch.allclose(cls_change_signal(cls), torch.tensor([[3.0, 3.0], [4.0, 4.0]]))


def test_threshold_exit_by_hand() -> None:
    signal = torch.tensor([[float("inf"), float("inf"), float("inf")],
                           [0.5, 2.0, 2.0],
                           [0.1, 0.5, 3.0],
                           [0.1, 0.1, 3.0]])
    assert threshold_exit(signal, 1.0).tolist() == [1, 2, 3]  # image 3 never drops below: last exit


def test_threshold_exit_never_beyond_last() -> None:
    signal = torch.rand(12, 50) + 1.0
    assert threshold_exit(signal, 0.5).max().item() == 11
    assert threshold_exit(signal, 0.0).max().item() <= 11


def test_compare_with_fixed() -> None:
    fixed = [10.0, 20.0, 29.5, 30.0]
    points = [{"acc": 29.8, "mean_loops": 2.9}, {"acc": 25.0, "mean_loops": 2.0}, {"acc": 30.0, "mean_loops": 3.5}]
    verdict = compare_with_fixed(points, fixed)
    assert verdict["best_within_one_less"]["acc"] == 29.8 and verdict["beats_fixed_one_less"]
    assert verdict["cheapest_near_full"]["mean_loops"] == 2.9
    assert verdict["fixed_loops_near_full"] == 3


def test_compare_with_fixed_when_no_rule_is_cheap_enough() -> None:
    verdict = compare_with_fixed([{"acc": 50.0, "mean_loops": 4.0}], [10.0, 20.0, 30.0, 40.0])
    assert verdict["best_within_one_less"] is None and not verdict["beats_fixed_one_less"]


def test_token_hooks_count_every_loop_of_a_shared_block() -> None:
    model = build_iterative(source=tiny_vit())
    loader = [(torch.randn(3, 3, 32, 32), torch.zeros(3))]
    stats = collect_token_stats(model, loader, torch.device("cpu"))
    assert len(stats["cosine"]) == DEPTH + 1
    assert len(stats["spectrum"][0]) == 17  # min(tokens, dim)
