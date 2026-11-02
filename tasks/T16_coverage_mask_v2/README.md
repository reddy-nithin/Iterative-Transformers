# T16 · Coverage mask v2: a mask that actually blocks re-reads [H1]

**Goal.** Give H1 its first real test. T14's mask could not bite: A_t sums to 1 over 196 patches and CLS attention is
nearly flat, so each loop drained only ~η/196 supply per patch (mean supply 0.968 after loop 10, late attention cos
0.995). v2 drains only patches that CLS read MORE than its fair share, by an amount that removes a heavily-read patch
in about one loop.

**From:** planning session 2026-11-02 (Nithin), after the T11 to T14 results. Due Thu 2026-11-05.
Notion task: https://notion.so/40c38c73470581f1a2b1d44c7bb33465

## Base, reference and the one change
- Base and reference: **T13-d** (T11-d layout: B0, B1, then B2 x10, no tail; readout every loop + loss margin; 73.17),
  same pretrained init, shared recipe (lr 5e-4, 30 epochs, seed 0) and loss.
- One change: the v2 coverage mask in B2's attention, CLS query row only (as in T14):
  - supply c ∈ R^196 per image, c = 1 before loop 1;
  - in loop t, add log(c_i + 1e-6) to the CLS → patch-i logits (all heads); CLS → CLS and all patch query rows unchanged;
  - after loop t: c ← clamp(c − η · max(0, 196 · A_t − 1), 0, 1), A_t = CLS → patch attention averaged over heads,
    renormalised over the 196 patches (the attention CLS actually used, after the mask, as in T14);
  - **η fixed** (a buffer, not a parameter): no new parameters.
- Code: `src/coverage_mask.py` rule `"v2"` (`update_supply_v2`), flag `--coverage-mask-v2-eta <η>` for
  `--model unroll`. T14's `--coverage-mask` (rule `"v1"`, learnable η) is unchanged; its tests still pass.

| Run | η | Flags | Checkpoint |
|---|---|---|---|
| T13-d (reference) | – | `--model unroll --unique-blocks 3 --readout-loss loss_margin` | `T13-d_unroll_loss_margin.pt` |
| T16-a | 0.5 | + `--coverage-mask-v2-eta 0.5` | `T16-a_covmask_v2_eta05.pt` |
| T16-b | 1.0 | + `--coverage-mask-v2-eta 1.0` | `T16-b_covmask_v2_eta1.pt` |

## Pre-flight check (no training, before the runs)
`preflight.py`: load `T13-d_unroll_loss_margin.pt`, turn on the v2 mask zero-shot at η = 0.5 and 1.0, full test set.
Reports mean supply after loop 10, share of patches with c < 0.1 after loop 10, late-loop (loops 6 to 10) CLS attention
cos, CLS attention entropy per loop (masked vs unmasked), the output change ‖o_mask − o_nomask‖ / ‖o_nomask‖ averaged
over loops 6 to 10 (o = CLS row of the attention output after `attn.proj`; at each loop both are computed from the
masked run's input state x_{t−1}, with the current supply and with no mask), and test top-1.
**Gate:** the mask bites if (mean supply after loop 10 < 0.8 OR late-loop cos < 0.9) AND output change ≥ 0.10, for at
least one η. Otherwise no training (gotcha with the numbers).

## Predictions (written before running)
Late-loop attention cos drops below 0.8 in both runs. Top-1 within ±1.5 pts
of T13-d (73.17); if H1 is right, at least one run is ≥ +1 pt AND its per-loop accuracy keeps rising after loop 6
(T13-d peaks at loop 6). Low confidence on the accuracy part.
Competing prediction (Gemini Round 3): on flat attention the v2 mask shaves the few peaks and pushes CLS attention
entropy UP toward uniform (5.28), with no accuracy gain. Report entropy per loop so the two can be told apart.

## Pre-flight results (zero-shot on `T13-d_unroll_loss_margin.pt`, all 10,000 test images; `results/preflight.json`)
The script's masked loop equals the training code's `CoverageMaskLoop` (rule v2) exactly (max logit diff 0).

| | η = 0.5 | η = 1.0 | unmasked |
|---|---|---|---|
| mean supply after loop 10 | 0.009 | 0.000 | 1 |
| share of patches with c < 0.1 after loop 10 | 96.7% | 100% | 0 |
| late-loop CLS attention cos (pairs 6-7 .. 9-10) | 0.497 | 0.754 | 0.995 (T14 / T12) |
| output change, loops 6 to 10 | 1.25 | 1.39 | 0 |
| test top-1 | 28.53 | 4.99 | 73.19 |

CLS attention entropy per loop (nats, max 5.28):

| loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| unmasked | 4.48 | 4.36 | 4.54 | 4.70 | 4.80 | 4.87 | 4.91 | 4.94 | 4.97 | 4.99 |
| η = 0.5 | 4.48 | 4.38 | 4.64 | 4.74 | 4.74 | 4.63 | 4.07 | 3.69 | 3.96 | 4.29 |
| η = 1.0 | 4.48 | 4.29 | 4.48 | 4.37 | 3.56 | 1.58 | 4.69 | 4.96 | 5.00 | 5.02 |

**Gate: passes at both η** (supply < 0.8 and output change ≥ 0.10 at both; late cos < 0.9 at both). Top-1 is below 5%
only at η = 1.0 (4.99), not at both, so the collapse note does not apply. Zero-shot, the v2 mask empties almost the
whole supply by loop 10: once the most-read patches are drained, the next ones are read above the fair share and get
drained too. At η = 1.0 supply is gone by loop 6, every patch then gets the same −13.8 bias, so the relative
attention over patches goes back to near uniform (entropy 5.0 in loops 8 to 10).

## Results (CIFAR-100 test, seed 0; all tables in `results/tables.md`)

| model | final top-1 | peak per-loop (loop) | late attn cos (6-7..9-10) | mean supply after loop 10 | share c < 0.1 after loop 10 |
|---|---|---|---|---|---|
| T13-d (reference) | 73.17 | 73.58 (6) | 0.998 | – | – |
| **T16-a** (η 0.5) | **9.98** | 9.95 (10) | 0.440 | 0.018 | 96.0% |
| **T16-b** (η 1.0) | **6.41** | 6.42 (10) | 0.812 | 0.433 | 51.0% |

**Neither run learned.** Train accuracy never passed 8% (T16-a) / 6% (T16-b) and train loss stayed at 4.2 to 4.6
(chance is ln 100 = 4.61); T13-d is at 23% test top-1 after epoch 1. The pre-flight showed the mask bites on trained
weights; in training it bites from the first step and the model does not recover in 30 epochs.

Fixed-depth top-1 per loop (`results/acc_per_loop.png`, exit rules in `results/exit_curves.png`):

| loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| T13-d | 45.08 | 63.94 | 70.15 | 72.51 | 73.30 | 73.58 | 73.54 | 73.42 | 73.33 | 73.19 |
| T16-a | 5.37 | 6.66 | 7.70 | 8.70 | 9.18 | 9.46 | 9.65 | 9.67 | 9.81 | 9.95 |
| T16-b | 3.11 | 4.35 | 4.89 | 5.10 | 5.22 | 5.58 | 5.86 | 6.00 | 6.31 | 6.42 |

Exit rules give nothing on the masked models (best gains +0.25 / +0.08 for KL, ≈ 0 for confidence and patience).

CLS attention (`results/attn_cos_entropy.png`) and supply (`results/supply_per_loop.png`):

| loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| entropy T13-d | 4.48 | 4.35 | 4.55 | 4.71 | 4.81 | 4.87 | 4.91 | 4.94 | 4.97 | 4.98 |
| entropy T16-a | 2.17 | 2.51 | 2.87 | 3.73 | 4.23 | 4.24 | 4.12 | 3.57 | 2.94 | 3.26 |
| entropy T16-b | 2.51 | 2.65 | 3.25 | 3.31 | 2.90 | 3.09 | 3.54 | 3.74 | 3.87 | 3.85 |
| supply T16-a | 0.906 | 0.809 | 0.701 | 0.571 | 0.434 | 0.283 | 0.141 | 0.057 | 0.027 | 0.018 |
| supply T16-b | 0.899 | 0.809 | 0.679 | 0.549 | 0.503 | 0.482 | 0.470 | 0.458 | 0.445 | 0.433 |

- The trained masked models attend much more sharply than T13-d (entropy 2.2 to 4.2 vs 4.4 to 5.0) and attention
  moves between loops (cos 0.2 to 0.6 for T16-a). At η 0.5 the supply empties steadily (96% of patches below 0.1 by
  loop 10). At η 1.0 it stops at ~0.45 from loop 5: half the patches are drained in one read, the other half are
  never read above the fair share again.
- T12 (a)-(c): rel_cls is smaller than T13-d's early (0.33 vs 0.77 at loop 1) and similar late (0.07 / 0.11 vs 0.08);
  successive CLS updates stay collinear (cos(Δ_t, Δ_t−1) 0.94 to 0.99 at every loop, T13-d 0.97 to 0.996 from loop 5).
  So the attention pattern changes loop to loop, but the CLS update does not: the update direction is set elsewhere
  (T19: the MLP part and the bias).
- Class separation (test, 4 classes) ends at 2.13 (T16-a) and 3.82 (T16-b) vs 1.40 for T13-d. At 6 to 10% top-1 these
  geometry numbers say little beyond "no class structure"; they are listed in `results/tables.md` for completeness.

**Did the mask bite?** Yes, hard: supply drained (0.02 / 0.43 after loop 10), attention changed loop to loop (late cos
0.44 / 0.81 vs 0.998) and became sharper, not flatter.
**Did blocking re-reads change accuracy or the per-loop curve?** It destroyed both: training fails (9.98 / 6.41 vs
73.17). The per-loop curve does rise to loop 10 instead of peaking at loop 6, but at a level near chance, so it says
nothing about H1. This version of the mask is not a usable test of H1; a softer variant (e.g. supply detached, a
smaller η, or the mask only after the first few loops) would be a new task, not run here (rule 1).

### Prediction check
- Late-loop attention cos below 0.8 in both runs: **half right** (T16-a 0.44; T16-b 0.81, just above).
- Top-1 within ±1.5 pts of T13-d: **wrong** (−63 and −67 pts; training failed).
- H1 test (≥ +1 pt and per-loop accuracy still rising after loop 6): **not met** (curve rises, accuracy collapsed).
- Gemini, entropy pushed UP toward uniform with no accuracy gain: entropy part **wrong** (entropy went down, 2.2 to
  4.2 vs T13-d's 4.4 to 5.0); "no accuracy gain" right, far worse than no gain.
