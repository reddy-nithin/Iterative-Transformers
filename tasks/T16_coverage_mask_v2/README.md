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

## Results
_Pending (training)._
