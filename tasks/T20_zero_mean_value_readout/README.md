# T20 · Zero-mean value readout for CLS in B2 (gated by T19)

**Goal (Gemini Round 3, Brainstorm 1).** Make it impossible for CLS to gain anything from evenly spread attention. In
the looped block B2 only, CLS reads each patch value minus the average patch value, so a uniform read adds zero and
the only way to change CLS through attention is to attend selectively.

**From:** planning session 2026-11-02 (Nithin), after the T11 to T14 results (T20 added after Gemini Round 3). Due Thu 2026-11-05.
Notion task: https://notion.so/40c38c7347058126b10bf7ba151b1362

**Gate:** T19's decision line on T11-d was **"run T20"** (late mean share 0.539 ≥ 0.5, late ‖a_t‖/‖d_t‖ 0.917 ≥ 0.3;
`tasks/T19_cls_update_decomposition/`).

## Layout, reference and the one change
- Layout and reference: **T11-d** (B0, B1, then B2 x10, no tail), final-output loss only, exactly the T11-d config;
  reference T11-d (74.11, plus the T15 seeds).
- One change: inside B2's attention, for the CLS query row only, each patch value V_{h,i} (i = 1..196) is replaced by
  V_{h,i} − V̄_h (V̄_h = mean over the 196 patches, per head, per image, per loop). CLS→CLS value, all patch query
  rows, and B0 / B1 unchanged. No new parameters.
- Code: `src/zero_mean_value.py`, flag `--zero-mean-value` for `--model unroll`. Test: `test_t20_zero_mean.py`.

| Run | Flags | Checkpoint |
|---|---|---|
| T11-d (reference) | `--model unroll --unique-blocks 3` | `T11-d_unroll_n3_30ep.pt` |
| T20-a | + `--zero-mean-value` | `T20-a_zero_mean_value.pt` |

Pretrained init, shared recipe (lr 5e-4, 30 epochs, seed 0).
Run (repo root, tmux, one GPU): `CUDA_VISIBLE_DEVICES=N bash tasks/T20_zero_mean_value_readout/run.sh`

## Predictions (written before running)
Gemini predicts CLS attention entropy < 3.5, late successive-update cos < 0.5 and
top-1 +1.5 to +3 over 74.11. Mac Claude predicts entropy drops (below 4.5) but late successive-update cos stays ≥ 0.8
if T19 shows the MLP part dominates, and top-1 within ±1.5 of T11-d.

## Results (CIFAR-100 test, seed 0)

| model | final top-1 | peak per-loop (loop) | late update cos (loops 6-10) | CLS attention entropy, loops 1-10 | sep. last call (test / train) |
|---|---|---|---|---|---|
| T11-d (reference) | 74.11 (seeds 0/1/2 mean 73.79, spread 0.77; T15) | 74.25 (9) | 0.881 | 4.42 to 4.80 (mean 4.64) | 1.151 / 1.022 |
| **T20-a** (zero-mean value) | **71.51** | 71.51 (10) | 0.912 | 4.49 to 4.83 (mean 4.63) | 1.175 / 0.999 |

T20-a is −2.60 vs T11-d's seed 0 and −2.28 vs its 3-seed mean, about three times the seed spread: a small but real
loss (one seed for T20-a).

Fixed-depth top-1 per loop (`results/acc_per_loop.png`; exit rules in `results/exit_analysis.json`):

| loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| T11-d | 1.22 | 3.79 | 11.30 | 29.61 | 49.67 | 62.56 | 69.60 | 73.12 | 74.25 | 74.12 |
| T20-a | 1.36 | 2.99 | 8.85 | 23.17 | 41.81 | 57.73 | 66.19 | 70.13 | 71.12 | 71.51 |

Same shape, 3 to 8 pts lower at loops 3 to 8. Best exit-rule gain +2.55 (confidence, 6.5 loops) vs +3.74 for T11-d.

CLS attention and T12 (`results/attn_cos_entropy.png`, `results/update_cos_per_loop.png`; T12 output in
`results/t12_diag/`, the recompute check matches the zero-mean block exactly):

| loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| entropy T11-d | 4.80 | 4.68 | 4.52 | 4.42 | 4.45 | 4.57 | 4.65 | 4.71 | 4.76 | 4.80 |
| entropy T20-a | 4.83 | 4.78 | 4.70 | 4.69 | 4.62 | 4.53 | 4.49 | 4.51 | 4.56 | 4.61 |
| T12 (b) new-direction share T11-d | 0.969 | 0.988 | 0.993 | 0.997 | 0.995 | 0.987 | 0.982 | 0.979 | 0.980 | 0.983 |
| T12 (b) new-direction share T20-a | 0.939 | 0.963 | 0.979 | 0.995 | 0.996 | 0.987 | 0.973 | 0.964 | 0.962 | 0.964 |
| T12 (c) cos(Δ_t, Δ_t−1) T11-d | – | 0.387 | 0.531 | 0.685 | 0.788 | 0.784 | 0.850 | 0.906 | 0.926 | 0.939 |
| T12 (c) cos(Δ_t, Δ_t−1) T20-a | – | 0.467 | 0.679 | 0.808 | 0.822 | 0.851 | 0.893 | 0.924 | 0.941 | 0.953 |

T19 split of the CLS update, adapted to the zero-mean block (mean part is 0 by construction; `results/t19/`):

| late loops 6-10 (mean) | ‖a‖/‖d‖ | ‖m‖/‖d‖ | mean share | self share | cos a (succ.) | cos m (succ.) |
|---|---|---|---|---|---|---|
| T11-d | 0.917 | 1.048 | 0.539 | 0.050 | 0.957 | 0.942 |
| T20-a | 0.939 | 1.063 | 0 | 0.097 | 0.982 | 0.958 |

(`decompose.py` also prints its T20 decision line for this checkpoint, "skip T20"; that line is only meaningful for the
T11-d gate and is ignored here.)

- Removing the mean read did not sharpen attention: entropy is the same on average (4.63 vs 4.64), a little lower in
  loops 6 to 9, higher early. The attention part of the update keeps its size (‖a‖/‖d‖ 0.94 vs 0.92): with the mean
  gone, the deviation part (and CLS reading itself, self share 0.10 vs 0.05) fills the same role, and the late push is
  just as collinear (attention-part cos 0.98, full update 0.91 vs 0.88).
- Class separation is unchanged (test 1.175 vs 1.151; train 0.999 vs 1.022).

**Did removing the mean read sharpen attention, and did it change the late-loop push?** No and no: entropy stays at
4.5 to 4.8 (T11-d 4.4 to 4.8), the late updates stay collinear (cos 0.91 vs 0.88), and top-1 drops 2.3 pts below the
T11-d seed mean.

### Prediction check
- Gemini, entropy < 3.5: **wrong** (minimum 4.49). Late successive-update cos < 0.5: **wrong** (0.91). Top-1 +1.5 to
  +3 over 74.11: **wrong** (−2.60).
- Mac Claude, entropy below 4.5: **wrong** (only loop 7 at 4.49; mean 4.63). Late successive-update cos ≥ 0.8:
  **right** (0.91). Top-1 within ±1.5 of T11-d: **wrong** (−2.60; −2.28 vs the seed mean).
