"""T5 checks: CLS collection hooks (normal and shared-block models), the rank and separation measures, the ImageNet
file lists and the checkpoint round trip."""
from argparse import Namespace

import pytest
import timm
import torch
from PIL import Image
from timm.data import ImageNetInfo
from torch import nn
from torchvision import transforms

from cls_svd import (analyse, collect_cls_per_layer, effective_rank, energy_rank, load_model, separation_ratio,
                     singular_values)
from early_exit import confidence_exit, exit_accuracies, exit_logits, patience_exit, rule_result
from data_sources import (CONTROL, imagenet_separation_sets, imagenet_subsets, resolve_class_indices,
                          resolve_imagenet_classes)
from src.models import MODEL_NAME, build_iterative
from src.train import model_kwargs, save_checkpoint

DEPTH, DIM, N_IMAGES = 3, 32, 6


def tiny_vit(seed: int = 0) -> nn.Module:
    """Small random ViT with the same structure as ViT-Tiny (CLS token, pre-norm blocks, final norm)."""
    torch.manual_seed(seed)
    return timm.create_model(MODEL_NAME, pretrained=False, depth=DEPTH, embed_dim=DIM, num_heads=2,
                             img_size=32, patch_size=8, num_classes=10)


def fake_loader(batches: int = 2) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """A few (images, labels) batches; collection only needs something iterable."""
    torch.manual_seed(1)
    per_batch = N_IMAGES // batches
    return [(torch.randn(per_batch, 3, 32, 32), torch.arange(per_batch)) for _ in range(batches)]


def test_shape_is_layers_by_images_by_dim() -> None:
    record = collect_cls_per_layer(tiny_vit(), fake_loader(), torch.device("cpu"))
    assert record.cls.shape == (DEPTH + 2, N_IMAGES, DIM)  # layer 0, blocks 1..DEPTH, final norm
    assert record.logits.shape == (N_IMAGES, 10)
    assert record.labels.shape == (N_IMAGES,)
    without_norm = collect_cls_per_layer(tiny_vit(), fake_loader(), torch.device("cpu"), include_final_norm=False)
    assert without_norm.cls.shape == (DEPTH + 1, N_IMAGES, DIM)


def test_layers_match_forward_features_before_and_after_final_norm() -> None:
    model = tiny_vit().eval()
    loader = fake_loader(batches=1)
    record = collect_cls_per_layer(model, loader, torch.device("cpu"))
    with torch.no_grad():
        with_norm = model.forward_features(loader[0][0])
        logits = model.forward_head(with_norm)
        model.norm = nn.Identity()
        before_norm = model.forward_features(loader[0][0])
    torch.testing.assert_close(record.cls[DEPTH], before_norm[:, 0, :])
    torch.testing.assert_close(record.cls[DEPTH + 1], with_norm[:, 0, :])
    torch.testing.assert_close(record.logits, logits)


def test_shared_block_still_gives_one_snapshot_per_slot() -> None:
    model = build_iterative(source=tiny_vit())
    assert len({id(b) for b in model.blocks}) == 1
    record = collect_cls_per_layer(model, fake_loader(), torch.device("cpu"))
    assert record.cls.shape == (DEPTH + 2, N_IMAGES, DIM)
    assert not torch.allclose(record.cls[1], record.cls[2])  # same weights, different inputs -> different outputs


def test_hooks_are_removed() -> None:
    model = tiny_vit()
    collect_cls_per_layer(model, fake_loader(), torch.device("cpu"))
    assert not model.blocks._forward_pre_hooks
    assert all(not block._forward_hooks for block in model.blocks)


def test_layer_zero_is_identical_so_centred_rank_is_zero() -> None:
    record = collect_cls_per_layer(tiny_vit(), fake_loader(), torch.device("cpu"))
    layer0 = analyse(record.cls)["0"]
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


def test_separation_ratio() -> None:
    torch.manual_seed(2)
    tight_far = [torch.randn(50, DIM) * 0.01 + 10, torch.randn(50, DIM) * 0.01 - 10]
    assert separation_ratio(tight_far) < 1e-4
    # unit noise per dim -> within = DIM; means at +1 and -1 in every dim -> between = DIM
    spread = [torch.randn(4000, DIM) + 1, torch.randn(4000, DIM) - 1]
    assert separation_ratio(spread) == pytest.approx(1.0, rel=0.1)
    same_rows = torch.ones(5, DIM)
    assert separation_ratio([same_rows, same_rows.clone()]) is None  # layer 0: no between-class spread


def test_unknown_class_name_raises() -> None:
    assert resolve_class_indices(["apple", "fox"], ["fox"]) == [1]
    with pytest.raises(ValueError, match="unicorn"):
        resolve_class_indices(["apple", "fox"], ["unicorn"])


def fake_imagenet(root, wnids: dict[str, int]) -> None:
    """A tiny ImageNet-style tree: <root>/<wnid>/<wnid>_<k>.JPEG, plus a stray non-class file at the top."""
    for wnid, count in wnids.items():
        (root / wnid).mkdir()
        for k in range(count):
            Image.new("RGB", (8, 8), color=(k * 30, 0, 0)).save(root / wnid / f"{wnid}_{k}.JPEG")
    (root / "words.txt").write_text("not a class")


def test_imagenet_subsets_labels_and_control(tmp_path) -> None:
    fake_imagenet(tmp_path, {"n01440764": 4, "n02119022": 3, "n07742313": 2})
    groups = imagenet_subsets(transforms.ToTensor(), str(tmp_path), ["n02119022", "948"], per_class=2,
                              control_size=3, seed=0)
    assert list(groups) == ["red fox", "Granny Smith", CONTROL]
    assert len(groups["red fox"]) == 2
    assert [groups["red fox"][i][1] for i in range(2)] == [277, 277]
    assert groups["Granny Smith"][0][1] == 948
    assert groups["red fox"][0][0].shape == (3, 8, 8)
    assert sorted(groups[CONTROL].labels) == [0, 277, 948]  # one image from each class folder present
    with pytest.raises(ValueError, match="needs as many classes"):
        imagenet_subsets(transforms.ToTensor(), str(tmp_path), ["277"], 1, control_size=4, seed=0)


def test_imagenet_class_resolution_and_sets() -> None:
    assert resolve_imagenet_classes(ImageNetInfo(), ["n02119022", "948"]) == [277, 948]
    with pytest.raises(ValueError, match="n99999999"):
        resolve_imagenet_classes(ImageNetInfo(), ["n99999999"])
    sets = imagenet_separation_sets(["n02110063", "n02110185", "n02119022"])
    assert sets == {"all classes": ["malamute", "Siberian husky", "red fox"],
                    "malamute vs Siberian husky": ["malamute", "Siberian husky"]}


def test_early_exit_at_last_block_equals_normal_prediction() -> None:
    model = tiny_vit().eval()
    record = collect_cls_per_layer(model, fake_loader(), torch.device("cpu"), include_final_norm=False)
    logits = exit_logits(model, record, torch.device("cpu"))
    assert logits.shape == (DEPTH, N_IMAGES, 10)
    torch.testing.assert_close(logits[-1], record.logits)
    accuracies = exit_accuracies(logits, record.labels)
    expected = 100.0 * (record.logits.argmax(dim=1) == record.labels).float().mean().item()
    assert accuracies[-1] == pytest.approx(expected)


def one_hot_logits(predictions: list[list[int]], confidence: list[list[float]] | None = None) -> torch.Tensor:
    """(exits, images, 3) logits whose argmax is `predictions[k][i]`; optional softmax-ish peak sizes."""
    logits = torch.zeros(len(predictions), len(predictions[0]), 3)
    for k, row in enumerate(predictions):
        for i, cls in enumerate(row):
            logits[k, i, cls] = 10.0 if confidence is None else confidence[k][i]
    return logits


def test_patience_exit() -> None:
    # image 0: 0,1,1,1 -> 2 agree at exit 2, 3 agree at exit 3; image 1: never 2 in a row -> last exit
    logits = one_hot_logits([[0, 0], [1, 1], [1, 0], [1, 1]])
    assert patience_exit(logits, 2).tolist() == [2, 3]
    assert patience_exit(logits, 3).tolist() == [3, 3]
    result = rule_result(logits, torch.tensor([1, 1]), patience_exit(logits, 2))
    assert result == {"acc": 100.0, "mean_loops": 3.5}


def test_confidence_exit() -> None:
    # peak logit 0 -> uniform over 3 classes (p = 1/3); 10 -> p ~ 1
    logits = one_hot_logits([[0, 0], [2, 0], [0, 0]], confidence=[[0.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
    assert confidence_exit(logits, 0.9).tolist() == [1, 2]  # image 1 is never confident -> last exit


def test_checkpoint_round_trip(tmp_path, monkeypatch) -> None:
    model = tiny_vit().eval()
    path = tmp_path / "sub" / "ckpt.pt"
    args = Namespace(model="baseline")
    save_checkpoint(str(path), model, args.model, model_kwargs(args), epoch=3, test_acc=12.5)
    # rebuild a tiny model with different weights instead of the real ViT-Tiny, so the test proves the load
    monkeypatch.setattr("cls_svd.build", lambda name, **kwargs: tiny_vit(seed=123))
    loaded = load_model(str(path)).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))
