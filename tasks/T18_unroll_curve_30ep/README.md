# T18 · Progressive unrolling curve at 30 epochs

**Goal.** The unrolling curve used in the write-ups comes from 10-epoch runs, which understate every size. Redo the
small end at 30 epochs, where it matters for "how few distinct blocks do we need?".

**From:** planning session 2026-11-02 (Nithin), after the T11 to T14 results. Due Thu 2026-11-05.
Notion task: https://notion.so/40c38c73470581e7bbaed98f2f01ecdd

## Runs
Existing unrolling code (`--model unroll --unique-blocks n`: blocks 0..n−2 kept, block n−1 repeated so the total is 12
applications), unchanged; only epochs 10 → 30. lr 5e-4, seed 0, pretrained init, shared recipe.

| Run | n | Checkpoint |
|---|---|---|
| T18-n2 | 2 | `T18-n2_unroll_30ep.pt` |
| T18-n4 | 4 | `T18-n4_unroll_30ep.pt` |
| T18-n5 | 5 | `T18-n5_unroll_30ep.pt` |
| T18-n6 | 6 | `T18-n6_unroll_30ep.pt` |

n = 1 (iter b0, 58.06) and n = 3 (T11-d, 74.11) already exist at 30 epochs and are reused, not rerun.
10-epoch numbers (lr 5e-4): n = 1..4 from T3 (`tasks/T3_progressive_unrolling/results/lr5e-4/ft/`), n = 5, 6 from T6.

Run (repo root, tmux, one GPU): `CUDA_VISIBLE_DEVICES=N bash tasks/T18_unroll_curve_30ep/sweep.sh T18-n2`

## Predictions (written before running)
Every n gains 3 to 8 pts from 10 → 30 ep; the gain shrinks as n grows. n=4 at 30 ep ≥ 78.

## Results (CIFAR-100 test top-1, lr 5e-4, seed 0; `results/table.md`, `results/acc_vs_n_10_vs_30ep.png`)

| n | params | top-1 10 ep | top-1 30 ep | gain 10 → 30 ep | 30-ep run |
|---|---|---|---|---|---|
| 1 | 0.65M | 46.02 | 58.06 | +12.04 | T4 iter b0 (reused) |
| 2 | 1.10M | 57.65 | **66.95** | +9.30 | T18-n2 |
| 3 | 1.54M | 67.12 | 74.11 | +6.99 | T11-d (reused) |
| 4 | 1.98M | 74.03 | **77.45** | +3.42 | T18-n4 |
| 5 | 2.43M | 77.67 | **80.04** | +2.37 | T18-n5 |
| 6 | 2.87M | 79.34 | **81.96** | +2.62 | T18-n6 |

Reference: all 12 blocks distinct (5.54M) 86.77 at 30 epochs. One seed per point; T15 puts the seed spread at
~0.3 to 0.8 pt, so the n = 5 vs 6 gain difference (2.37 vs 2.62) is noise.

- Longer training helps the small end most: +12 (n = 1), +9.3 (n = 2), +7.0 (n = 3), then only +2.4 to +3.4 for
  n = 4 to 6. The 10-epoch curve understated n ≤ 3 badly and n ≥ 4 only a little.
- At 30 epochs the curve flattens after n = 3: +7.2 from n = 2 to 3, then +3.3, +2.6, +1.9 per extra distinct block.
  n = 3 (1.54M, 28% of the baseline's params) keeps 85% of the baseline's accuracy (74.11 / 86.77); n = 6 (52% of the
  params) keeps 94%.

### Prediction check
- Every n gains 3 to 8 pts from 10 → 30 ep: **wrong**, in both directions: n = 1, 2 gain more (+12.0, +9.3), n = 5, 6
  gain less (+2.4, +2.6); only n = 3, 4 are inside 3 to 8.
- The gain shrinks as n grows: **right** in trend (12.0 → 9.3 → 7.0 → 3.4 → 2.4), with n = 6 (+2.6) level with n = 5
  within seed noise.
- n = 4 at 30 ep ≥ 78: **wrong**, just (77.45).
