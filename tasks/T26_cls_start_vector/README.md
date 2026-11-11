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

- [ ] README + predictions committed before the first run
- [ ] code + tests (`test_t26_cls_start.py`) pass
- [ ] smoke run (`SMOKE=1`)
- [ ] full run, learned start reproduces the logged top-1 for both models
- [ ] table, plot `results/cls_start_gap_per_stage.png`, line per model, answer line, prediction check, inbox entry

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

_Pending._
