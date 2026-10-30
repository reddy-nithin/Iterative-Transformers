# T13 · A readout at every loop (H5) and "each loop must lower the loss" (H2)

**Goal.** In every run so far only the final output is trained, so loops 1-9 get no direct signal and early-exit rules
only tie fixed depth. Here each loop gets a training signal: CLS is read out at every loop with the model's own
readout modules (no new parameters), and the loss covers all 10 readouts.

**From:** advisor meeting 2026-10-27 (Dr. Uddin), due Tue 2026-11-03.
Notion task: https://notion.so/40938c73470581edb17ad2e5707e1f78

**Started / finished:** 2026-10-30 / 2026-10-30

## Why two layouts
T11 showed the loop behaves differently depending on what sits after it. With a tail (T11-c), the tail does most of
the class-tightening; with nothing after the loop (T11-d), the looped block does it itself. So "if every loop must give
an answer, does the loop start working?" is asked on both:
- **T11-c layout** (B0 head → B1 x10 → B11 tail): does a per-loop readout fix the lazy body and close part of the
  5.5-pt gap to T11-d?
- **T11-d layout** (unrolling n = 3: B0, B1, then B2 x10, no tail): does early exit finally beat fixed depth when the
  loop really tightens the classes?

## Checklist
- [x] Predictions committed before any run
- [x] Per-loop readout for both layouts (`src/per_loop_readout.py`) + `--readout-loss` in `src.train`, tests
      (`test_t13_readout.py`, incl. logits_10 = normal model output on both layouts)
- [x] Smoke test of all four runs
- [x] T13-a to T13-d, 30 epochs each, one GPU and one tmux session each
- [x] Exit analysis on 10k test images for T13-a..d + T11-c + T11-d (fixed depth, confidence / KL / patience exits,
      flip rate), T12 metric (a), separation per loop (train + test)
- [x] Plots `results/exit_curves.png`, `acc_per_loop.png`, `rel_update.png`, `loss_curves.png`, `separation_vs_loop.png`
- [x] Results table and the four one-line answers

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

## Commands
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests tasks -q
# smoke (300 images, 1 epoch), then one tmux session and one GPU per run
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T13_per_loop_readout_loss/sweep.sh T13-a
tmux new -d -s itertf-T13-a "CUDA_VISIBLE_DEVICES=0 bash tasks/T13_per_loop_readout_loss/sweep.sh T13-a"
# after training (eval only)
C=/storage/INSLAB/Data/nrs3qy/checkpoints/itertf
PYTHONPATH=. .venv/bin/python tasks/T13_per_loop_readout_loss/exit_analysis.py $C/T11-c_hbt_body_b1.pt $C/T11-d_unroll_n3_30ep.pt $C/T13-*.pt
PYTHONPATH=. .venv/bin/python tasks/T12_per_loop_cls_diagnostics/loop_diagnostics.py $C/T13-*.pt --out-dir tasks/T13_per_loop_readout_loss/results/t12_diag
.venv/bin/python tasks/T13_per_loop_readout_loss/plot.py
```

## Results
All four runs: 30 epochs on GPUs 0-3 in parallel (2026-10-30, ~41 min each, ~80 s/epoch), seed 0, shared recipe
unchanged (batch 128 fit: peak 3.7 GB on the T11-c layout, 2.2 GB on the T11-d layout; full tail, no token-0
shortcut). Exit numbers on all 10,000 test images in fp32; loop 10 matches the logged test acc within 0.03 for all six
models (one or two images, AMP vs fp32). Separation = tr(within) / tr(between) of CLS, 4 classes (apple, bicycle, fox,
cloud), test, T5 `cls_svd.py` as in T11. "Mean rel_cls loops 1-5" = T12 metric (a) on the 1,000 T8 images.

| Run | Layout | Loss | Params (M) | Final top-1 | Best top-1 | Loop-5 acc | Flip 9→10 (%) | Fewest loops within 1 pt of loop 10 | Mean rel_cls loops 1-5 | Sep. last body loop (test) | Min body sep. (test, call) | Sep. last block (test) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T11-c | T11-c | ref: final loss only | 1.540 | 68.59 | 68.63 | 24.54 | 9.05 | confidence 0.7: 8.78 loops (67.99%) vs fixed 9 | 0.297 | 1.61 | 1.61 (10) | 1.38 |
| T13-a | T11-c | readout every loop | 1.540 | 68.85 | 68.92 | 69.34 | 2.54 | patience 2: 3.54 loops (68.64%) vs fixed 4 | 0.328 | 1.89 | 1.84 (7) | 1.29 |
| T13-b | T11-c | every loop + margin | 1.540 | 68.85 | 68.95 | 67.56 | 2.71 | kl 0.01: 4.99 loops (68.15%) vs fixed 6 | 0.250 | 2.14 | 2.00 (6) | 1.54 |
| T11-d | T11-d | ref: final loss only | 1.540 | 74.11 | 74.13 | 49.67 | 5.23 | confidence 0.5: 7.55 loops (73.47%) vs fixed 8 | 0.299 | 1.15 | 1.06 (9) | 1.15 |
| T13-c | T11-d | readout every loop | 1.540 | 72.82 | 72.84 | 73.27 | 1.49 | kl 0.1: 3.44 loops (72.63%) vs fixed 4 | 0.595 | 1.27 | 1.23 (8) | 1.27 |
| T13-d | T11-d | every loop + margin | 1.540 | 73.17 | 73.17 | 73.30 | 1.58 | kl 0.0316: 4.07 loops (73.13%) vs fixed 4 | 0.512 | 1.40 | 1.38 (9) | 1.40 |

Readout accuracy at loop t (fixed depth, %):

| loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| T11-c | 1.6 | 3.2 | 6.2 | 11.8 | 24.5 | 41.6 | 56.5 | 64.6 | 68.0 | 68.6 |
| T13-a | 56.9 | 64.5 | 67.7 | 68.8 | 69.3 | **69.8** | 69.6 | 69.4 | 69.2 | 68.8 |
| T13-b | 53.5 | 61.5 | 64.8 | 66.6 | 67.6 | 68.2 | 68.4 | 68.6 | 68.8 | 68.9 |
| T11-d | 1.2 | 3.8 | 11.3 | 29.6 | 49.7 | 62.6 | 69.6 | 73.1 | 74.2 | 74.1 |
| T13-c | 49.0 | 66.0 | 71.4 | 72.9 | 73.3 | **73.5** | 73.4 | 73.1 | 73.0 | 72.8 |
| T13-d | 45.1 | 63.9 | 70.2 | 72.5 | 73.3 | **73.6** | 73.5 | 73.4 | 73.3 | 73.2 |

Exit rules vs fixed depth (fewest mean loops whose top-1 stays within 1 pt of loop 10, vs the first fixed depth that
does; loops saved as a share of that fixed depth):

| Model | fixed depth | confidence | KL | patience |
|---|---|---|---|---|
| T11-c | 9 | 8.78 (2%) | 9.25 (none) | none |
| T13-a | 4 | 4.89 (none) | 3.72 (7%) | 3.54 (12%) |
| T13-b | 6 | 5.28 (12%) | 4.99 (17%) | none |
| T11-d | 8 | 7.55 (6%) | 8.71 (none) | none |
| T13-c | 4 | 4.81 (none) | 3.44 (14%) | 3.74 (7%) |
| T13-d | 4 | 5.22 (none) | 4.07 (none) | 4.88 (none) |

Final-output train CE at epoch 30 (comparable with T11's train loss): T13-a 1.83, T13-b 1.83 vs T11-c 1.81; T13-c
1.70, T13-d 1.72 vs T11-d 1.68. T12 metric (b) on the T13 checkpoints: late-loop cos(A_t, A_t+1) 0.994-0.997 (references
0.956-0.958), so with deep supervision CLS re-reads the same patches even more; the T12 rule still says "coverage mask"
for all four.

Plots: `results/exit_curves.png`, `results/acc_per_loop.png`, `results/rel_update.png`, `results/loss_curves.png`,
`results/separation_vs_loop.png`. Raw: `results/exit_analysis.json`, `results/t12_diag/`, `results/*.json`,
`results/*.log`, table in `results/summary.md`. Checkpoints: `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/T13-*.pt`.

**Against the predictions** (one seed; the predictions section above is unchanged):
- T13-a within ~1 pt of the T11 winner: right, 68.85 vs 68.59 (+0.3).
- T13-a's confidence or KL exit gives the same accuracy with ≥20% fewer mean loops than its own fixed depth: wrong. KL
  saves 7% (3.72 vs 4 loops), confidence saves nothing. The saving is in fixed depth itself: 4 loops are enough
  instead of 9 for T11-c.
- T13-b vs T13-a tie on final accuracy: right (68.85 / 68.85). T13-b has larger rel_cls in loops 1-5: wrong, smaller
  (0.250 vs 0.328).
- Flip rate lower than the T11 winner's: right, 2.5% vs 9.1%.
- (i) T13-a's lowest body separation below T11-c's 1.61: wrong, it is higher, 1.84 (call 7); the body stops tightening
  after about loop 4 and the tail still takes it down (1.29).
- (ii) T13-c within ~1 pt of T11-d: just missed, 72.82 (−1.3). T13-d is within it (73.17, −0.9).
- (iii) Early-exit gain larger on the T11-d layout: not supported. At ≤1 pt below loop 10, T13-c needs 3.44 mean loops
  and T13-a 3.54; both layouts get there at fixed depth 4.

**One line each:**
- **Does deep supervision change final accuracy (per layout)?** T11-c layout: no, a tie (68.85 both vs 68.59). T11-d
  layout: slightly lower, −1.3 (T13-c) and −0.9 (T13-d) vs 74.11. On both layouts the readout peaks at loop 6
  (T13-a 69.8, T13-c 73.5), above its own loop 10.
- **Does any exit rule beat fixed depth? which, on which layout, by how much?** Barely. Best: KL on the T11-d layout
  (T13-c, 3.44 vs 4 loops, 14% fewer) and patience-2 on the T11-c layout (T13-a, 3.54 vs 4, 12% fewer), both within
  1 pt of loop 10. The real gain is that fixed depth 4 is now enough (T11-c needed 9, T11-d 8).
- **Does T13-b / T13-d make early loops move CLS more than T13-a / T13-c?** No, less. Mean rel_cls in loops 1-5 is
  0.250 vs 0.328 (T13-b vs T13-a) and 0.512 vs 0.595 (T13-d vs T13-c). Both deep-supervision losses do move CLS more
  early on than the references (0.30).
- **Does deep supervision tighten the T11-c body's separation (the lazy-tail guess)?** No. The body's lowest test
  separation gets worse (T13-a 1.84, T13-b 2.00 vs T11-c 1.61) and flattens after about loop 4, while the tail still
  does the last tightening (T13-a 1.29 after the tail vs 1.38). The same happens on the T11-d layout (1.23 / 1.38 vs
  1.06). Per-loop readouts make each loop answer early; they don't make later loops refine.

**Takeaways:**
- A readout at every loop moves almost all the work into loops 1-4. Loop 1 alone is now 45-57% (references ~1-2%), and
  loops 5-10 add nothing or slightly hurt. The loop becomes "answer early, then idle". It doesn't refine.
- Final accuracy is unchanged on the T11-c layout and ~1 pt lower on the T11-d layout. The gap to T11-d narrows only
  because T11-d's own layout drops, not because H-B-T improves (T13-a 68.85 vs T13-c 72.82, still 4 pts behind).
- The H2 margin loss (T13-b, T13-d) does its job on CE: loop-to-loop accuracy rises monotonically to loop 10 for T13-b. But
  it makes CLS move less, not more, and it does not help accuracy.
- For T14 (base rule): T13-a and T13-b tie (within 2 pts) → T13-a would be the T11-c-layout base, but T13-d (73.17)
  beats both T13-a and T13-b by ≥2 pts, so the exception applies: T14's base would be T13-d, with the add-on from
  T12's T11-d decision line (coverage mask). T14 not started.
