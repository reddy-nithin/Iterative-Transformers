# T13 · A readout at every loop (H5) and "each loop must lower the loss" (H2)

**Goal.** In every run so far only the final output is trained, so loops 1-9 get no direct signal and early-exit rules
only tie fixed depth. Here each loop gets a training signal: CLS is read out at every loop with the model's own
readout modules (no new parameters), and the loss covers all 10 readouts.

**From:** advisor meeting 2026-10-27 (Dr. Uddin), due Tue 2026-11-03.
Notion task: https://notion.so/40938c73470581edb17ad2e5707e1f78

**Started / finished:** 2026-10-30 / –

## Why two layouts
T11 showed the loop behaves differently depending on what sits after it. With a tail (T11-c), the tail does most of
the class-tightening; with nothing after the loop (T11-d), the looped block does it itself. So "if every loop must give
an answer, does the loop start working?" is asked on both:
- **T11-c layout** (B0 head → B1 x10 → B11 tail): does a per-loop readout fix the lazy body and close part of the
  5.5-pt gap to T11-d?
- **T11-d layout** (unrolling n = 3: B0, B1, then B2 x10, no tail): does early exit finally beat fixed depth when the
  loop really tightens the classes?

## Checklist
- [ ] Predictions committed before any run
- [ ] Per-loop readout for both layouts (`src/per_loop_readout.py`) + `--readout-loss` in `src.train`, tests
      (`test_t13_readout.py`, incl. logits_10 = normal model output on both layouts)
- [ ] Smoke test of all four runs
- [ ] T13-a to T13-d, 30 epochs each, one GPU and one tmux session each
- [ ] Exit analysis on 10k test images for T13-a..d + T11-c + T11-d (fixed depth, confidence / KL / patience exits,
      flip rate), T12 metric (a), separation per loop (train + test)
- [ ] Plots `results/exit_curves.png`, `acc_per_loop.png`, `rel_update.png`, `loss_curves.png`, `separation_vs_loop.png`
- [ ] Results table and the four one-line answers

## Runs
Initialised from the PRETRAINED weights exactly as in T11 (not from the T11 checkpoints). Shared recipe: pretrained
timm `vit_tiny_patch16_224`, CIFAR-100 at 224 px, AdamW lr 5e-4 for all parameters, wd 0.05, 1 warm-up epoch + cosine,
label smoothing 0.1, batch 128, AMP, 30 epochs, seed 0, nothing frozen. Each run differs from its reference only in
the loss.

| Run | Layout | Loss | Reference | Checkpoint |
|---|---|---|---|---|
| T13-a | T11-c: head b0, body b1 x10, tail b11 | readout at every loop (H5) | T11-c (68.59) | `T13-a_readout_every_loop.pt` |
| T13-b | T11-c | readout at every loop + loss margin (H2) | T11-c (68.59) | `T13-b_loss_margin.pt` |
| T13-c | T11-d: unrolling n = 3 (b0, b1, b2 x10, no tail) | readout at every loop (H5) | T11-d (74.11) | `T13-c_unroll_readout_every_loop.pt` |
| T13-d | T11-d | readout at every loop + loss margin (H2) | T11-d (74.11) | `T13-d_unroll_loss_margin.pt` |

## The losses
State x_t = all 197 tokens after body loop t (t = 1..10).
- Readout, T11-c layout: logits_t = classifier(final_norm(tail(x_t))[:, 0]), the same tail, final norm and classifier for
  every t.
- Readout, T11-d layout: logits_t = classifier(final_norm(x_t)[:, 0]); x_t = output of the t-th repeat of block 2.
- In both, logits_10 is exactly the normal model output.
- CE_t = per-sample cross-entropy of logits_t, label smoothing 0.1.
- **Readout at every loop (T13-a, T13-c):** loss = Σ_t w_t · mean(CE_t) / Σ_t w_t, w_t = 0.85^(10−t) (loop 10 weight
  1.0, loop 1 ≈ 0.23). Discounted because strong early-loop pressure can push the shared block toward shallow shortcuts.
- **Loss margin (T13-b, T13-d):** the loss above + λ · (1/9) Σ_{t=2..10} mean over batch of
  [max(0, CE_t − CE_{t−1}.detach() + m) · 1(CE_{t−1}.detach() > floor)], m = 0.02, λ = 1.0, floor = 1.0. Progress is
  measured in the loss (bigger CLS steps could be gamed), the floor switches the rule off for images already solved
  (minimum CE with smoothing 0.1 over 100 classes ≈ 0.78), and CE_{t−1} is detached so loop t is pushed to beat loop
  t−1, not loop t−1 pushed to get worse.

## Predictions (written before running)
T13-a final top-1 within ~1 pt of the T11 winner. T13-a's confidence or KL exit
beats its own fixed-depth curve: same accuracy with ≥20% fewer mean loops. T13-b vs T13-a: tie on final accuracy,
T13-b has larger rel_cls in loops 1-5. Flip rate lower than the T11 winner's.
Added 2026-10-30 after T11: (i) if the tail makes the body lazy, T13-a's lowest body separation (test) is below
T11-c's 1.61 even if accuracy ties. (ii) T13-c final top-1 within ~1 pt of T11-d (74.11). (iii) The early-exit gain
is larger on the T11-d layout than on the T11-c layout (T13-c needs fewer mean loops than T13-a at the same
accuracy drop). Low confidence on (i) and (iii).

## Results
_Pending._
