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

## Results (CIFAR-100 test, seed 0)

| Run | Tail | Params | Final top-1 | Sep. last body loop (call 11, test) | Sep. after tail (call 12, test) | Min body sep. (call) |
|---|---|---|---|---|---|---|
| T11-c | B11 | 1.54M | 68.59 (seeds 0/1/2 mean 68.64, spread 0.34; T15) | 1.61 | 1.38 | 1.61 (10) |
| **T21-a** | **B2** | 1.54M | **69.04** | 1.73 | **1.20** | 1.62 (9) |

Separation = tr(within) / tr(between) of CLS, 4 classes (apple, bicycle, fox, cloud), 100 test images each, after each
block call (T5 `cls_svd.py`, T11 settings; lower = better separated). Train set (500 images per class) gives the same
picture: after the tail 1.17 (T21-a) vs 1.30 (T11-c), last body loop 1.78 vs 1.62.

Separation per block call (test; call 1 = head B0, calls 2-11 = body loops 1-10, call 12 = tail):

| call | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 (tail) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T11-c | 3.65 | 1.97 | 2.20 | 2.28 | 2.29 | 2.11 | 1.90 | 1.74 | 1.65 | 1.61 | 1.61 | 1.38 |
| T21-a | 3.12 | 2.08 | 2.38 | 2.45 | 2.24 | 1.92 | 1.72 | 1.64 | 1.62 | 1.66 | 1.73 | 1.20 |

Per-loop readout through the tail (T12 (d), full test set, top-1 % after body loop t):

| loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| T11-c | 1.6 | 3.2 | 6.2 | 11.8 | 24.5 | 41.6 | 56.5 | 64.6 | 68.0 | 68.6 |
| T21-a | 1.8 | 3.0 | 6.0 | 14.3 | 28.9 | 46.3 | 58.6 | 65.5 | 68.4 | 69.0 |

T12 metrics are close to T11-c's as well (late-loop attention cos 0.961 vs 0.958, successive-update cos at loop 10
0.949 vs 0.953,
attention entropy 4.66 to 4.78 vs 4.59 to 4.79); full table `results/t12_diag/table_T21-a_hbt_tail_b2.md`.

- T21-a is +0.45 over T11-c's seed-0 run and +0.40 over its 3-seed mean, about the size of the seed spread: a tie on
  accuracy (one seed for T21-a).
- The B2 tail separates the classes a little better at the end (1.20 vs 1.38) although the body's last loop is
  slightly worse (1.73 vs 1.61): the tail still does the final tightening, and the pretrained B2 does it at least as
  well as B11. The readout through the tail is 2.1 to 4.7 pts higher at loops 4 to 7, then the curves meet.

**Does the tail's pretrained position matter?** Hardly: a tail taken from right after the body's block (B2) and the
original last block (B11) give the same accuracy within seed noise (69.04 vs 68.64 ± 0.34) and similar per-loop
behaviour; no sign of an "interface shock" that B2 fixes.

### Prediction check
- Gemini, T21-a beats T11-c clearly: **wrong** (+0.4, inside the seed spread).
- Mac Claude, within ±2 pts of T11-c: **right**.
