# T17 · Large, near-orthogonal CLS updates (Dr. Uddin's H2, original form) [H2]

**Goal.** Test Dr. Uddin's idea as he stated it: each loop should push CLS in a NEW direction, not repeat the last
push. T12 (c) found the opposite: successive late CLS updates have cos 0.94 to 0.98.

**From:** planning session 2026-11-02 (Nithin), after the T11 to T14 results. Due Thu 2026-11-05.
Notion task: https://notion.so/40c38c73470581628f0ace33162596fa

## Base, reference and the one change
- Base and reference: **T13-c** (T11-d layout: B0, B1, then B2 x10, no tail; readout every loop, no margin; 72.82).
  The per-loop readout makes every loop's CLS answer to the classifier, so the block cannot satisfy the new term by
  moving CLS in directions the classifier ignores. Same pretrained init, recipe (lr 5e-4, 30 epochs, seed 0) and
  T13-c loss; the only change is the added term.
- CLS_t = x_t[:, 0] (raw state, before final norm), t = 0..10, CLS_0 = output of block 1; d_t = CLS_t − CLS_{t−1}.
- **T17-a (orthogonal only):** loss = T13-c loss + λ_o · mean over batch and t = 2..10 of cos(d_t, d_{t−1})², λ_o = 0.5
  (cos per sample, eps 1e-8 added to each norm).
- **T17-b (orthogonal + large):** T17-a loss + λ_s · mean over batch and t = 1..10 of
  max(0, r_min − ‖d_t‖ / (‖CLS_{t−1}‖ + 1e-8)), λ_s = 1.0, r_min = 0.25.
- Code: `src/cls_update_loss.py`, flag `--cls-update-loss {none,orth,orth_large}` (default none = as before).
  Test: `test_t17_cls_update_loss.py`. Pre-check: `precheck.py` (rel_cls on T13-c vs T12).

| Run | Flags | Checkpoint |
|---|---|---|
| T13-c (reference) | `--model unroll --unique-blocks 3 --readout-loss every_loop` | `T13-c_unroll_readout_every_loop.pt` |
| T17-a | + `--cls-update-loss orth` | `T17-a_orth_cls.pt` |
| T17-b | + `--cls-update-loss orth_large` | `T17-b_orth_large_cls.pt` |

Run (repo root, tmux, one GPU): `CUDA_VISIBLE_DEVICES=N bash tasks/T17_orthogonal_cls_updates/sweep.sh T17-a`

## Predictions (written before running)
Late successive-update cos drops below 0.5 in both runs. Final top-1 within
±1.5 pts of T13-c (72.82). T17-b moves CLS more in late loops (rel_cls ≥ 0.25 by construction, near it in practice).
Per-loop accuracy keeps rising later than T13-c's loop-6 peak. Low confidence on accuracy.
Competing prediction (Gemini Round 3): once the answer is found (~loop 6), forcing large orthogonal steps kicks CLS
out of the class basin, so T17-b has WORSE late-loop drift and lower top-1 than T17-a and T13-c. T17-b is kept
because it is Dr. Uddin's idea as stated; the result decides between the two predictions.

## Results (CIFAR-100 test, seed 0; all tables in `results/tables.md`)

| model | final top-1 | peak per-loop (loop) | late update cos (loops 6-10) | late rel_cls (loops 6-10) | best exit-rule gain | sep. last call (test / train) |
|---|---|---|---|---|---|---|
| T13-c (reference) | 72.82 | 73.53 (6) | 0.989 | 0.111 | +0.79 (patience, 2.5 loops) | 1.27 / 1.13 |
| **T17-a** (orth) | **72.18** | 72.91 (6) | −0.005 | 0.190 | +0.57 (KL, 3.6 loops) | 1.42 / 1.25 |
| **T17-b** (orth + large) | **72.35** | 73.25 (6) | −0.012 | 0.301 | +0.81 (patience, 2.6 loops) | 1.50 / 1.35 |

Seed spread for this layout is 0.7 to 0.8 pt (T15: T11-d 0.77, T13-d 0.68), so −0.64 (T17-a) and −0.47 (T17-b) vs
T13-c are within noise.

Successive CLS-update cos, T12 (c) (`results/update_cos_per_loop.png`):

| loop | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| T13-c | 0.029 | 0.479 | 0.828 | 0.946 | 0.978 | 0.988 | 0.991 | 0.994 | 0.995 |
| T17-a | 0.007 | 0.023 | −0.002 | 0.012 | −0.004 | −0.009 | −0.001 | −0.014 | 0.003 |
| T17-b | 0.010 | 0.024 | −0.002 | 0.005 | −0.002 | −0.022 | −0.022 | −0.007 | −0.006 |

rel_cls per loop, T12 (a) (`results/rel_update.png`):

| loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| T13-c | 0.824 | 0.932 | 0.636 | 0.358 | 0.226 | 0.161 | 0.125 | 0.103 | 0.089 | 0.079 |
| T17-a | 0.822 | 0.996 | 0.815 | 0.538 | 0.332 | 0.282 | 0.198 | 0.184 | 0.147 | 0.140 |
| T17-b | 0.825 | 0.998 | 0.811 | 0.546 | 0.364 | 0.318 | 0.296 | 0.295 | 0.297 | 0.300 |

Fixed-depth top-1 per loop (`results/acc_per_loop.png`):

| loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| T13-c | 49.00 | 66.02 | 71.38 | 72.93 | 73.27 | 73.53 | 73.36 | 73.12 | 73.04 | 72.82 |
| T17-a | 48.62 | 65.39 | 70.72 | 72.52 | 72.81 | 72.91 | 72.71 | 72.52 | 72.44 | 72.17 |
| T17-b | 48.04 | 65.16 | 71.24 | 72.56 | 72.99 | 73.25 | 73.07 | 72.79 | 72.66 | 72.32 |

Class separation per block call, test (`results/separation_vs_loop.png`; calls 3-12 = B2 loops 1-10):

| call | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|
| T13-c | 1.367 | 1.270 | 1.240 | 1.233 | 1.236 | 1.244 | 1.254 | 1.267 |
| T17-a | 1.430 | 1.316 | 1.327 | 1.318 | 1.344 | 1.361 | 1.393 | 1.421 |
| T17-b | 1.407 | 1.283 | 1.288 | 1.289 | 1.336 | 1.369 | 1.443 | 1.502 |

- The loss did exactly what it asks: successive updates are orthogonal at every loop (|cos| ≤ 0.02), and nearly all
  of each update is a new direction (T12 (b) 0.99). T17-b holds rel_cls at the 0.30 floor from loop 7 on; T17-a
  also moves CLS ~2× more late than T13-c.
- Accuracy does not move: same top-1 within seed noise, same per-loop shape (peak at loop 6, then a ~1-pt slide to
  loop 10), same exit-rule gains.
- The extra late movement costs class structure: separation is best at call 6 to 8 (B2 loops 4-6) in all three, but
  after that it degrades faster with the new loss (T17-b 1.28 → 1.50 from loop 4 to loop 10, T13-c 1.27 → 1.27).
  The late orthogonal steps move CLS around the class basin instead of further into it.

**Did the loss make updates orthogonal?** Yes, completely: late successive-update cos −0.01 vs 0.99 for T13-c, with
the size floor held in T17-b.
**Did orthogonal updates help accuracy or the per-loop curve?** No: top-1 72.18 / 72.35 vs 72.82 (inside seed noise),
the per-loop curve still peaks at loop 6, and late-loop class separation gets worse, not better. Collinear late
updates are not what stops the looped model from improving after loop 6.

### Prediction check
- Late successive-update cos below 0.5 in both runs: **right** (−0.005, −0.012).
- Final top-1 within ±1.5 pts of T13-c: **right** (−0.64, −0.47).
- T17-b moves CLS more in late loops, rel_cls near 0.25: **right** (0.30 from loop 7; T17-a 0.19, T13-c 0.11).
- Per-loop accuracy keeps rising later than loop 6: **wrong** (all three peak at loop 6).
- Gemini, T17-b drifts more late and has lower top-1 than T17-a and T13-c: **drift part right** (separation 1.28 →
  1.50 late, the worst of the three), **top-1 part wrong** (T17-b 72.35 ≥ T17-a 72.18; all within noise).
