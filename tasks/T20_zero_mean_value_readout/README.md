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

## Results
_Pending._
