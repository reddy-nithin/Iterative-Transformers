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

## Results
_Pending._
