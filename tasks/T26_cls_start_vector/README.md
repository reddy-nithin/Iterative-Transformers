# T26 · Does the CLS starting vector matter? Learned vs zero vs random (no training)

**From:** advisor meeting 2026-11-05.
Notion task: https://notion.so/40f38c73470581479959fb7544373987

**Goal.** Dr. Uddin asked why every image starts from one learned CLS vector (a dataset prior) instead of zero. Huginn
starts from random noise and finds the end state does not depend on the start ("path independence"). Test that on our
models without training.

## Setup

- **Checkpoints:** `T11-d_unroll_n3_30ep.pt` (74.11) and `T13-c_unroll_readout_every_loop.pt` (72.82), in
  `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`. The learned start must reproduce each model's logged top-1.
- **Starts.** Only the learned `cls_token` (before the position embedding is added) is replaced; position embedding,
  B0, B1, B2 untouched:
  1. learned (no change, reference);
  2. zeros;
  3. Gaussian noise N(0, σ²) per image and dimension, σ = std of the learned token's 192 entries, seeds 0..4;
  4. learned + the same noise as 3 (seeds 0..4);
  5. the other model's learned token (T13-c's token in T11-d and vice versa).
- **How (no model code changed).** timm adds the position embedding right after concatenating the CLS token, so token 0
  after `_pos_embed` is cls_token + pos_embed[0]. A context manager wraps `model._pos_embed` and adds an offset to that
  row: zeros = −cls, noise = n − cls, learned + noise = n, swap = cls_other − cls. Tests check this equals actually
  replacing `cls_token`, and that every parameter is bit-identical afterwards.
- **Measured** on the 10,000 TEST images (test transform, `model.eval()`, fp32): final top-1 and per-loop readout
  top-1 (mean and spread over seeds for 3 and 4), and the gap
  g_s = mean over images of ‖CLS_s(alt) − CLS_s(learned)‖ / ‖CLS_s(learned)‖ after B0, after B1 and after loops 1..10
  (raw CLS, before the final norm).

## Checklist

- [x] README + predictions committed before the first run
- [x] code + tests (`test_t26_cls_start.py`) pass
- [x] smoke run (`SMOKE=1`)
- [x] full run, learned start reproduces the logged top-1 for both models
- [x] table, plot `results/cls_start_gap_per_stage.png`, line per model, answer line, prediction check, inbox entry

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T26_cls_start_vector -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T26_cls_start_vector/run.sh
CUDA_VISIBLE_DEVICES=N bash tasks/T26_cls_start_vector/run.sh
```

## Predictions (written before running)

Copied verbatim from the task spec:

- zero start costs < 1 pt (Dr. Shuhao expects zero to be bad: competing prediction); random noise costs ≤ 2 pts; g_t
  shrinks steadily through the loops; the swapped token (5) is within 1 pt. Low confidence.

## Results

Run 2026-11-11, one GPU (26 passes over the 10,000 test images). Sanity check: the learned start reproduces the logged
top-1, T11-d 74.12 (logged 74.11), T13-c 72.82 (logged 72.82). σ of the learned token: 0.480 (T11-d), 0.474 (T13-c).

### Final top-1 and gap to the learned start

g = ‖CLS(alt) − CLS(learned)‖ / ‖CLS(learned)‖, mean over images, raw CLS. Noise rows: mean over seeds 0..4
(spread = max − min of the final top-1).

| Model | Start | final top-1 | Δ vs learned | g after B0 | g after B1 | g loop 1 | g loop 5 | g loop 10 |
|---|---|---|---|---|---|---|---|---|
| T11-d | learned | 74.12 | 0 | 0 | 0 | 0 | 0 | 0 |
| T11-d | zeros | 59.27 | **−14.85** | 0.420 | 0.407 | 0.447 | 0.589 | 0.638 |
| T11-d | noise (spread 1.42) | 42.33 | **−31.79** | 0.647 | 0.624 | 0.645 | 0.796 | 0.841 |
| T11-d | learned + noise (spread 0.53) | 71.74 | −2.38 | 0.374 | 0.328 | 0.302 | 0.290 | 0.280 |
| T11-d | swap (T13-c's token) | 74.06 | −0.06 | 0.009 | 0.008 | 0.008 | 0.009 | 0.009 |
| T13-c | learned | 72.82 | 0 | 0 | 0 | 0 | 0 | 0 |
| T13-c | zeros | 53.68 | **−19.14** | 0.444 | 0.531 | 0.972 | 0.642 | 0.592 |
| T13-c | noise (spread 0.68) | 50.32 | **−22.50** | 0.699 | 0.818 | 1.263 | 0.702 | 0.641 |
| T13-c | learned + noise (spread 0.46) | 70.02 | −2.80 | 0.436 | 0.459 | 0.662 | 0.319 | 0.289 |
| T13-c | swap (T11-d's token) | 72.89 | +0.07 | 0.011 | 0.012 | 0.021 | 0.012 | 0.011 |

Per-loop top-1 (loops 1..10):

| Model | learned | zeros | noise (mean of 5) |
|---|---|---|---|
| T11-d | 1.2 3.8 11.3 29.6 49.7 62.6 69.6 73.1 74.3 74.1 | 2.2 4.2 10.5 28.4 46.8 57.0 61.7 63.0 61.9 59.3 | 2.0 3.7 7.9 17.6 30.4 40.0 45.2 46.6 45.3 42.3 |
| T13-c | 49.0 66.0 71.4 72.9 73.3 73.5 73.4 73.1 73.0 72.8 | 24.2 28.6 39.2 46.1 50.5 52.8 53.9 54.1 54.0 53.7 | 13.9 23.6 35.1 42.9 47.0 49.0 50.0 50.4 50.5 50.3 |

- **The gap does not close.** In T11-d it GROWS through the loops (zeros 0.42 → 0.64, noise 0.65 → 0.84). In T13-c it
  jumps at loop 1 (0.97 / 1.26), shrinks to ~0.6 by loop 5, and then stays there. No alternative start comes within
  0.1 of the learned-start CLS.
- The loss is not just a slow start: zeros and noise plateau far below the learned start (T13-c ~54 and ~50 from loop
  7 on; T11-d peaks at loop 8 and then falls).
- Learned + noise (the same noise added on top of the learned token) costs only 2.4 to 2.8 pt, and its gap shrinks a
  little through the loops (T13-c 0.66 → 0.29): the learned token is a working starting point; moving away from it is
  what costs.
- **The swap is not an informative test here:** the two models' learned tokens are almost identical (g = 0.009 after
  B0), presumably because both start from the same pretrained token and fine-tuning barely moved it. So "swap within
  1 pt" says nothing about whether different learned tokens are interchangeable.

**Line (T11-d):** start matters (zeros cost 14.85 pt, noise cost 31.79 pt (mean of 5 seeds), g_10 zeros 0.638, noise
0.841).

**Line (T13-c):** start matters (zeros cost 19.14 pt, noise cost 22.50 pt (mean of 5 seeds), g_10 zeros 0.592, noise
0.641).

**Does the learned CLS start carry information the loop needs?** Yes. Replacing it with zeros or noise costs 15 to 32
pt, and the loop does not pull the alternative start back to where the learned start ends: the gap stays at 0.6 to
0.8 (T11-d even grows). Our models are not path independent in Huginn's sense. One caveat: Huginn TRAINS with a random
start, and our models never saw any start but the learned one, so this test shows path independence does not appear
for free; it does not show it could not be trained in.

### Prediction check

| Prediction | Result | Verdict |
|---|---|---|
| Zero start costs < 1 pt (Dr. Shuhao expected zero to be bad) | −14.85 (T11-d), −19.14 (T13-c) | wrong; Dr. Shuhao right |
| Random noise costs ≤ 2 pts | −31.79, −22.50 | wrong |
| g_t shrinks steadily through the loops | T11-d grows; T13-c shrinks from loop 1 to 5, then flat at ~0.6 | wrong |
| Swapped token within 1 pt | −0.06 / +0.07 | right, but uninformative (the two tokens are almost the same vector) |

### Plot and files

`results/cls_start_gap_per_stage.png` (g per stage, one line per start, 2 panels). All numbers in
`results/cls_start.json`. Made by `run.sh` → `cls_start.py`, `plot.py`.
