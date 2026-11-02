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

## Results
_Pending._
