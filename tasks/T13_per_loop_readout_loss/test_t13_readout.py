"""T13 checks: the per-loop readout works on both layouts (T11-c: through the tail; T11-d: final norm + classifier on
x_t), its last loop is exactly the normal model output, it adds no parameters, and the two losses match their
definitions."""
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.models import build_baseline, build_head_body_tail, build_unrolled
from src.per_loop_readout import (
    LABEL_SMOOTHING, MARGIN, every_loop_loss, loop_weights, margin_penalty, per_loop_ce, per_loop_logits,
    readout_loss, split_loop,
)
from src.train import forward_and_loss

torch.manual_seed(0)
IMAGES = torch.randn(2, 3, 224, 224)
LABELS = torch.tensor([3, 7])
N_LOOPS = 10


@pytest.fixture(scope="module")
def pretrained() -> nn.Module:
    """The pretrained ViT-Tiny with the 100-class head, loaded once."""
    return build_baseline(pretrained=True).eval()


def fresh_copy(source: nn.Module) -> nn.Module:
    """A copy of `source` with its weights, so builders never modify the fixture."""
    clone = build_baseline(pretrained=False)
    clone.load_state_dict(source.state_dict())
    return clone


def t11c_layout(source: nn.Module) -> nn.Module:
    """T11-c / T13-a,b: head b0, body b1 x10, tail b11 (pretrained copies)."""
    return build_head_body_tail(source=fresh_copy(source), pretrained_init=True, head_idx=0, body_idx=1,
                                tail_idx=11).eval()


def t11d_layout(source: nn.Module) -> nn.Module:
    """T11-d / T13-c,d: unrolling n = 3 (blocks 0, 1 distinct, block 2 x10, no tail)."""
    return build_unrolled(3, source=fresh_copy(source)).eval()


LAYOUTS = {"T11-c": t11c_layout, "T11-d": t11d_layout}


def body_outputs(model: nn.Module, images: torch.Tensor) -> list[torch.Tensor]:
    """x_1..x_10 from a forward hook on the body during a normal forward pass (independent of per_loop_logits)."""
    _, body, _, _ = split_loop(model)
    states = []
    handle = body.register_forward_hook(lambda module, args, output: states.append(output))
    try:
        model(images)
    finally:
        handle.remove()
    return states


@pytest.mark.parametrize("layout", LAYOUTS)
def test_last_loop_equals_normal_output(pretrained: nn.Module, layout: str) -> None:
    model = LAYOUTS[layout](pretrained)
    with torch.no_grad():
        logits = per_loop_logits(model, IMAGES)
        expected = model(IMAGES)
    assert len(logits) == N_LOOPS
    assert torch.allclose(logits[-1], expected, atol=1e-5)


def test_split_loop_layouts(pretrained: nn.Module) -> None:
    head, _, tail, loops = split_loop(t11c_layout(pretrained))
    assert (len(head), len(tail), loops) == (1, 1, 10)
    head, _, tail, loops = split_loop(t11d_layout(pretrained))
    assert (len(head), len(tail), loops) == (2, 0, 10)


def test_t11c_readout_goes_through_the_tail(pretrained: nn.Module) -> None:
    model = t11c_layout(pretrained)
    tail = model.blocks[11]
    with torch.no_grad():
        logits = per_loop_logits(model, IMAGES)
        states = body_outputs(model, IMAGES)
        expected = [model.head(model.norm(tail(x))[:, 0]) for x in states]
    for got, want in zip(logits, expected):
        assert torch.allclose(got, want, atol=1e-5)


def test_t11d_readout_is_final_norm_and_classifier(pretrained: nn.Module) -> None:
    model = t11d_layout(pretrained)
    with torch.no_grad():
        logits = per_loop_logits(model, IMAGES)
        states = body_outputs(model, IMAGES)
        expected = [model.head(model.norm(x)[:, 0]) for x in states]
    assert len(states) == N_LOOPS
    for got, want in zip(logits, expected):
        assert torch.allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_no_new_parameters(pretrained: nn.Module, layout: str) -> None:
    model = LAYOUTS[layout](pretrained)
    before = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        per_loop_logits(model, IMAGES)
    assert sum(p.numel() for p in model.parameters()) == before == 1_539_940


def test_loop_weights() -> None:
    weights = loop_weights(N_LOOPS)
    assert weights[-1].item() == pytest.approx(1.0)
    assert weights[0].item() == pytest.approx(0.85 ** 9, rel=1e-6)
    assert weights[0].item() == pytest.approx(0.2316, abs=1e-4)


def test_every_loop_with_identical_logits_is_plain_ce() -> None:
    logits = torch.randn(4, 100)
    labels = torch.randint(0, 100, (4,))
    plain = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)
    assert readout_loss([logits] * N_LOOPS, labels, "every_loop").item() == pytest.approx(plain.item(), rel=1e-6)


def test_every_loop_weighting_by_hand() -> None:
    ce = torch.tensor([[2.0, 4.0], [1.0, 1.0]])  # 2 loops, 2 images: means 3.0 and 1.0, weights 0.85 and 1.0
    assert every_loop_loss(ce).item() == pytest.approx((0.85 * 3.0 + 1.0 * 1.0) / 1.85)


def test_margin_penalty_by_hand() -> None:
    # image 0: CE goes 2.0 -> 2.5 (got worse, previous above floor) -> penalty 0.5 + 0.02
    # image 1: CE goes 0.9 -> 1.5 (previous below floor 1.0) -> rule off, 0
    # image 2: CE goes 3.0 -> 2.0 (improved by more than the margin) -> 0
    ce = torch.tensor([[2.0, 0.9, 3.0], [2.5, 1.5, 2.0]])
    assert margin_penalty(ce).item() == pytest.approx((0.5 + MARGIN) / 3)


def test_margin_gradient_only_reaches_the_later_loop() -> None:
    ce = torch.tensor([[2.0], [2.5]], requires_grad=True)
    margin_penalty(ce).backward()
    assert ce.grad[0, 0].item() == 0.0 and ce.grad[1, 0].item() == pytest.approx(1.0)


def test_loss_margin_adds_to_every_loop() -> None:
    torch.manual_seed(1)
    logits = [torch.randn(4, 100) for _ in range(N_LOOPS)]
    labels = torch.randint(0, 100, (4,))
    ce = per_loop_ce(logits, labels)
    expected = every_loop_loss(ce) + margin_penalty(ce)
    assert readout_loss(logits, labels, "loss_margin").item() == pytest.approx(expected.item(), rel=1e-6)


def test_unknown_loss_rejected() -> None:
    with pytest.raises(ValueError):
        readout_loss([torch.randn(2, 100)], LABELS, "bogus")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_forward_and_loss_final_logits_match_model(pretrained: nn.Module, layout: str) -> None:
    model = LAYOUTS[layout](pretrained)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    with torch.no_grad():
        plain_logits, plain_loss, _ = forward_and_loss(model, IMAGES, LABELS, criterion, "none")
        logits, _, ce_final = forward_and_loss(model, IMAGES, LABELS, criterion, "every_loop")
    assert torch.allclose(logits, plain_logits, atol=1e-5)
    assert ce_final.item() == pytest.approx(plain_loss.item(), rel=1e-5)


def test_readout_loss_default_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.train import parse_args
    monkeypatch.setattr("sys.argv", ["train", "--model", "hbt"])
    assert parse_args().readout_loss == "none"
