# T21 · Contiguous tail: B0 → B1 x10 → B2 (one change from T11-c)

**Goal.** Gemini says B11 fails as a tail because B11 was pretrained on block-10 features, far from what the B1 loop
produces ("interface shock"). Test it with ONE change: the tail is pretrained block 2 instead of block 11.

**From:** planning session 2026-11-02 (Nithin), after the T11 to T14 results (T21 added after Gemini Round 3). Due Thu 2026-11-05.
Notion task: https://notion.so/40c38c7347058136b0d0ec3ee2e82d62

## Run
| Run | Layout | Flags | Checkpoint |
|---|---|---|---|
| T11-c (reference) | head B0, body B1 x10, tail B11 | `--model hbt --hbt-pretrained --head-idx 0 --body-idx 1 --tail-idx 11` | `T11-c_hbt_body_b1.pt` |
| T21-a | head B0, body B1 x10, tail **B2** | same with `--tail-idx 2` | `T21-a_hbt_tail_b2.pt` |

Same 1.54M params, pretrained init, shared recipe (lr 5e-4, 30 epochs, seed 0) and final-only loss as T11-c. The tail
block was already a flag in the T11 code (`--tail-idx`, default 11), so no code changed and T11 reproduces as before.
After training, in the same job: class separation per block call (T5 `cls_svd.py`, T11 settings, train + test) and the
per-loop readout through the tail (T12 `loop_diagnostics.py`, part (d)), into `results/t12_diag/`.

Run (repo root, tmux, one GPU): `CUDA_VISIBLE_DEVICES=N bash tasks/T21_contiguous_tail_b2/run.sh`

## Predictions (written before running)
Gemini predicts T21-a beats T11-c clearly. Mac Claude predicts within ±2 pts of
T11-c: after 10 loops of B1 the state is no longer "B1's output", so B2 is not on its home ground either. Low confidence.

## Results
_Pending._
