"""T14 checks: the coverage mask (src.coverage_mask) on the T13-d layout (unrolling n = 3: blocks 0, 1, then block 2
x10). (1) with eta = 0 the model gives the T13-d-layout output; (2) the supply c stays in [0, 1] and only the
CLS->patch logits change. Plus: +1 parameter, eta gets a gradient, the per-loop readout still ends at the normal
output, and the model rebuilds from a checkpoint's kwargs."""
import pytest
import torch
from torch import nn

from src.coverage_mask import (
    CoverageMaskBody, CoverageMaskLoop, masked_block_parts, supply_bias, update_supply,
)
from src.metrics import count_params
from src.models import build, build_baseline, build_unrolled
from src.per_loop_readout import embed, per_loop_logits, readout_loss, split_loop
from src.train import model_kwargs

torch.manual_seed(0)
IMAGES = torch.randn(2, 3, 224, 224)
LABELS = torch.tensor([3, 7])
N_LOOPS = 10
N_PATCHES = 196


@pytest.fixture(scope="module")
def pretrained() -> nn.Module:
    """The pretrained ViT-Tiny with the 100-class head, loaded once."""
    return build_baseline(pretrained=True).eval()


def fresh_copy(source: nn.Module) -> nn.Module:
    """A copy of `source` with its weights, so builders never modify the fixture."""
    clone = build_baseline(pretrained=False)
    clone.load_state_dict(source.state_dict())
    return clone


def t13d_layout(source: nn.Module) -> nn.Module:
    """T13-d / T11-d layout without the mask."""
    return build_unrolled(3, source=fresh_copy(source)).eval()


def t14_model(source: nn.Module, eta: float | None = None) -> nn.Module:
    """T14-a: the same layout with the coverage mask; `eta` overrides the 0.5 init."""
    model = build_unrolled(3, source=fresh_copy(source), coverage_mask=True).eval()
    if eta is not None:
        with torch.no_grad():
            model.blocks.body.eta.fill_(eta)
    return model


def replay_supplies(model: nn.Module, images: torch.Tensor) -> list[torch.Tensor]:
    """c after every loop, replayed with the body's own functions; each replayed loop output is checked against the
    real body output recorded by a forward hook during a normal forward pass."""
    body = model.blocks.body
    outputs = []
    handle = body.register_forward_hook(lambda module, args, output: outputs.append(output))
    try:
        model(images)
    finally:
        handle.remove()
    x = embed(model, images)
    for block in model.blocks.head:
        x = block(x)
    supply, supplies = torch.ones(images.shape[0], N_PATCHES), []
    for real in outputs:
        parts = masked_block_parts(body.block, x, supply)
        assert torch.allclose(parts["output"], real, atol=1e-5)
        supply = update_supply(supply, parts["cls_attention"], body.eta)
        supplies.append(supply)
        x = parts["output"]
    return supplies


# ---------- (1) eta = 0 gives the T13-d-layout model ----------

def test_eta_zero_equals_t13d_layout(pretrained: nn.Module) -> None:
    base, masked = t13d_layout(pretrained), t14_model(pretrained, eta=0.0)
    with torch.no_grad():
        expected, got = base(IMAGES), masked(IMAGES)
        per_loop_base, per_loop_masked = per_loop_logits(base, IMAGES), per_loop_logits(masked, IMAGES)
    assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()
    for a, b in zip(per_loop_masked, per_loop_base):
        assert torch.allclose(a, b, atol=1e-5)


def test_eta_zero_keeps_full_supply(pretrained: nn.Module) -> None:
    with torch.no_grad():
        supplies = replay_supplies(t14_model(pretrained, eta=0.0), IMAGES)
    assert all(torch.equal(c, torch.ones(2, N_PATCHES)) for c in supplies)


# ---------- (2) c in [0, 1], only CLS -> patch logits change ----------

@pytest.mark.parametrize("eta", [0.5, 5.0, 200.0])
def test_supply_stays_in_unit_interval(pretrained: nn.Module, eta: float) -> None:
    model = t14_model(pretrained, eta=eta)
    with torch.no_grad():
        supplies = replay_supplies(model, IMAGES)
    assert len(supplies) == N_LOOPS
    for previous, current in zip([torch.ones_like(supplies[0])] + supplies[:-1], supplies):
        assert current.shape == (2, N_PATCHES)
        assert current.min() >= 0 and current.max() <= 1
        assert torch.all(current <= previous + 1e-7)  # supply only gets used up
    if eta >= 200.0:
        assert (supplies[-1] == 0).any()  # the clamp at 0 is reached


def test_bias_only_on_cls_to_patch_logits() -> None:
    logits = torch.randn(2, 3, N_PATCHES + 1, N_PATCHES + 1)
    supply = torch.rand(2, N_PATCHES)
    biased = supply_bias(logits, supply)
    changed = biased != logits
    assert not changed[:, :, 1:].any()  # every patch query row unchanged
    assert not changed[:, :, 0, 0].any()  # CLS -> CLS unchanged
    expected = logits[:, :, 0, 1:] + torch.log(supply + 1e-6)[:, None, :]
    assert torch.allclose(biased[:, :, 0, 1:], expected)


def test_masked_block_changes_only_cls_attention_row(pretrained: nn.Module) -> None:
    block = t14_model(pretrained).blocks.body.block
    x = torch.randn(2, N_PATCHES + 1, 192)
    supply = torch.rand(2, N_PATCHES)
    with torch.no_grad():
        plain, masked = masked_block_parts(block, x, torch.ones(2, N_PATCHES)), masked_block_parts(block, x, supply)
    assert torch.allclose(masked["probs"][:, :, 1:], plain["probs"][:, :, 1:])
    assert not torch.allclose(masked["probs"][:, :, 0], plain["probs"][:, :, 0])
    assert torch.allclose(masked["cls_attention"].sum(dim=1), torch.ones(2))


# ---------- structure, training signal, checkpoints ----------

def test_one_extra_parameter_and_layout(pretrained: nn.Module) -> None:
    base, masked = t13d_layout(pretrained), t14_model(pretrained)
    assert count_params(masked) == count_params(base) + 1
    assert isinstance(masked.blocks, CoverageMaskLoop) and isinstance(masked.blocks.body, CoverageMaskBody)
    assert masked.blocks.body.eta.item() == 0.5
    head, body, tail, loops = split_loop(masked)
    assert (len(head), len(tail), loops) == (2, 0, N_LOOPS)
    assert body is masked.blocks.body
    for got, want in zip(masked.blocks.body.block.state_dict().values(), base.blocks[2].state_dict().values()):
        assert torch.equal(got, want)  # body starts from pretrained block 2


def test_last_loop_equals_normal_output(pretrained: nn.Module) -> None:
    model = t14_model(pretrained)
    with torch.no_grad():
        logits = per_loop_logits(model, IMAGES)
        expected = model(IMAGES)
    assert len(logits) == N_LOOPS
    assert torch.allclose(logits[-1], expected, atol=1e-5)


def test_eta_gets_a_gradient(pretrained: nn.Module) -> None:
    model = t14_model(pretrained).train()
    loss = readout_loss(per_loop_logits(model, IMAGES), LABELS, "loss_margin")
    loss.backward()
    grad = model.blocks.body.eta.grad
    assert grad is not None and torch.isfinite(grad) and grad.abs() > 0


def test_mask_changes_the_output_at_init(pretrained: nn.Module) -> None:
    with torch.no_grad():
        assert not torch.allclose(t14_model(pretrained)(IMAGES), t13d_layout(pretrained)(IMAGES), atol=1e-4)


def test_rebuild_from_checkpoint_kwargs(pretrained: nn.Module) -> None:
    class Args:
        model, unique_blocks, coverage_mask = "unroll", 3, True
    kwargs = model_kwargs(Args)
    assert kwargs == {"unique_blocks": 3, "coverage_mask": True}
    model = t14_model(pretrained, eta=0.3)
    rebuilt = build("unroll", pretrained=False, **kwargs).eval()
    rebuilt.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.allclose(rebuilt(IMAGES), model(IMAGES))


def test_flag_off_kwargs_unchanged() -> None:
    class Args:
        model, unique_blocks, coverage_mask = "unroll", 3, False
    assert model_kwargs(Args) == {"unique_blocks": 3}
