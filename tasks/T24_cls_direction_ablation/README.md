# T24 · Is there one repeated CLS direction in late loops? (no training) [H11]

**From:** paper-read session 2026-11-03 (Nithin), alphaXiv Itertf folder.
Notion task: https://notion.so/40d38c73470581cfa0e3eb096812fd91

**Goal.** T12 (c) found that late CLS updates point almost the same way every loop (successive-update cos 0.94 to
0.98). The Weight-Space Programming post traced a language model's repetition loop to ONE direction in its hidden state
and removed it. Test the same here: is there one direction that late loops keep writing into CLS? Does it carry class
information, or is it a "keep going" push that can be cut at no cost? The answer shapes T17 (orthogonal updates) and
the halting rules.

## Setup

- **Checkpoints** (analysed separately), in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`:
  `T11-d_unroll_n3_30ep.pt` (74.11) and `T13-c_unroll_readout_every_loop.pt` (72.82). Both use the T11-d layout
  (B0, B1, then B2 x10, no tail). Each model's no-edit top-1 on the full test set must match the logged value.
- **Definitions.** Loop t = the t-th application of B2, t = 1..10. CLS_t = x_t[:, 0], the raw residual state after
  loop t (before the final norm); CLS_0 = output of B1. d_t = CLS_t − CLS_{t−1}. Test transform, 224 px,
  `model.eval()`, fp32.
- **(a) Find the direction.** First 10,000 CIFAR-100 TRAIN images in stored order. D = d_t for t = 6..10, shape
  [50,000, 192]. SVD WITHOUT centering; u1 = top right singular vector, sign chosen so mean(D · u1) > 0. Report
  σ1² / Σσ², top-5 cumulative shares, cos(u1, mean row of D), and the energy share of centered D (context). Save
  `results/u1_<model>.npy` (and `u2_<model>.npy` for control 5).
- **(b) Class information.** 10,000 TEST images. p = CLS_10 · u1. Fisher ratio = between-class variance of p (variance
  of the 100 class means) / mean within-class variance of p. Compared with 200 random unit directions (seed 0,
  Gaussian then normalised): median, 95th percentile, and u1's percentile. Also mean ‖d_t‖ and mean |d_t · u1| per
  loop t = 1..10.
- **(c) Cut it out at test time.** In each loop t of the edit range, after B2 computes x_t, its CLS row becomes
  CLS_{t−1} + d_t − (d_t · u) u; patch tokens unchanged; later loops run on the edited state. Test top-1 (final
  output, i.e. final norm + classifier on the last loop's CLS, as both models are evaluated) for:
  1. no edit; 2. u1 removed in loops 6..10; 3. u1 removed in loops 1..10;
  4. a random unit direction removed in loops 6..10 (seeds 0..4: `torch.Generator().manual_seed(s)`, Gaussian,
     normalised; mean and spread); 5. u2 removed in loops 6..10.
  For T13-c also the per-loop readout curve for runs 1 and 2.
- **Supported line** (exact, per model): "H11 supported" if energy share ≥ 0.5 AND top-1 with u1 removed in loops
  6..10 ≥ no-edit top-1 − 0.5. "H11 refuted" if energy share < 0.3. Otherwise "H11 mixed", with the two numbers.

## Checklist

- [x] README + predictions committed before the first run
- [x] code + tests (`test_t24_projection.py`) pass
- [x] smoke run (`SMOKE=1`)
- [x] full run, no-edit top-1 matches the logged value for both models
- [x] tables (a) to (c), plots `results/{singular_spectrum,push_along_u1_per_loop,fisher_u1_vs_random}.png`
- [x] supported line per model, answer line ("Is the late-loop push one class-free direction we can cut?"),
      prediction check, inbox entry

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T24_cls_direction_ablation -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T24_cls_direction_ablation/run.sh
CUDA_VISIBLE_DEVICES=N bash tasks/T24_cls_direction_ablation/run.sh
```

## Predictions (written before running)

Copied verbatim from the task spec:

- Energy share ≥ 0.6 on both models (late updates have cos ≥ 0.94, so one direction should hold most of them).
- u1's Fisher ratio is below the random-direction median: the push is mostly class-free.
- Removing u1 in loops 6..10 costs ≤ 0.5 pt on T11-d. On T13-c it costs less than on T11-d, because T13-c's answer
  is ready by loop 6.
- Removing it in loops 1..10 costs more than 0.5 pt.
- Low confidence on the accuracy parts.

## Results

Run 2026-11-11, one GPU (a few minutes). Sanity check: the no-edit run reproduces the logged top-1, T11-d 74.12
(logged 74.11), T13-c 72.82 (logged 72.82). Directions: `results/u1_{T11-d,T13-c}.npy`, `results/u2_*.npy`.

### (a) Is there one direction?

| model | energy share σ1² / Σσ² | top-5 cumulative | cos(u1, mean row of D) | centred energy share |
|---|---|---|---|---|
| T11-d | **0.132** | 0.132 / 0.180 / 0.219 / 0.246 / 0.271 | 0.995 | 0.055 |
| T13-c | **0.375** | 0.375 / 0.399 / 0.419 / 0.436 / 0.452 | 0.998 | 0.074 |

- **No.** u1 is simply the AVERAGE late update (cos 0.995 to 0.998 with the mean row), and that average holds only
  13% (T11-d) and 38% (T13-c) of the late-update energy. Once the average is removed, no direction holds more than
  6 to 7%: the rest is spread over many directions.
- How this fits T12 (c) (successive-update cos 0.94 to 0.98): that cos is measured **per image across loops**.
  Each image keeps pushing in a steady direction, but **each image has its own direction**, so across images
  there is no single shared one. This matches T27's straight per-class rays that fan out from a shared start.

### (b) Does u1 carry class information?

| model | Fisher ratio of CLS_10 · u1 | random median | random 95th pct | u1's percentile among 200 random |
|---|---|---|---|---|
| T11-d | 1.569 | 0.705 | 0.971 | 100 |
| T13-c | 1.019 | 0.638 | 0.823 | 99.5 |

u1 separates classes **better** than 199 to 200 of the 200 random directions: it is not a class-free push.

Mean ‖d_t‖ and mean |d_t · u1| per loop (test images):

| model | | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T11-d | ‖d_t‖ | 3.90 | 3.50 | 3.40 | 3.98 | 4.50 | 4.39 | 4.38 | 4.18 | 3.87 | 3.58 |
| T11-d | \|d_t · u1\| | 0.42 | 0.34 | 0.26 | 0.40 | 0.57 | 1.38 | 1.78 | 1.63 | 1.31 | 1.02 |
| T13-c | ‖d_t‖ | 7.44 | 5.71 | 5.41 | 4.33 | 3.50 | 2.95 | 2.60 | 2.37 | 2.21 | 2.10 |
| T13-c | \|d_t · u1\| | 3.35 | 1.12 | 2.66 | 2.48 | 2.13 | 1.80 | 1.57 | 1.42 | 1.30 | 1.22 |

In T11-d the push along u1 switches on at loop 6 (0.57 → 1.38) and peaks at loop 7, i.e. u1 is a late-loop
direction there. In T13-c it shrinks steadily with the updates after loop 3.

### (c) Cutting it out at test time (test top-1, %, change vs no edit)

| setting | T11-d | T13-c |
|---|---|---|
| 1. no edit | 74.12 | 72.82 |
| 2. u1 removed, loops 6-10 | 72.24 (**−1.88**) | 72.10 (−0.72) |
| 3. u1 removed, loops 1-10 | 71.53 (−2.59) | 68.22 (**−4.60**) |
| 4. random direction removed, loops 6-10 (5 seeds) | 73.97 (−0.15, spread 0.26) | 72.74 (−0.08, spread 0.08) |
| 5. u2 removed, loops 6-10 | 73.72 (−0.40) | 72.77 (−0.05) |

Per-loop readout (T13-c), runs 1 and 2:

| run | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|
| no edit | 49.00 | 66.02 | 71.38 | 72.93 | 73.27 | 73.53 | 73.36 | 73.12 | 73.04 | 72.82 |
| u1 removed, loops 6-10 | 49.00 | 66.02 | 71.38 | 72.93 | 73.27 | 73.48 | 73.23 | 73.06 | 72.71 | 72.10 |

- On T11-d, removing u1 in the late loops costs 1.88 pt, 12x the random-direction control: the model needs that push.
- On T13-c the late cut costs 0.72 pt (9x the random control, but under our 0.8 pt bar, so not called an effect).
  The cost builds up only in loops 9 and 10. Cutting u1 from loop 1 on costs 4.6 pt.
- The edit is deterministic on a fixed model and the same images (no seed noise), but it is one checkpoint each.

**Supported line (T11-d):** H11 refuted (energy share 0.132, top-1 72.24 with u1 removed in loops 6-10 vs 74.12).

**Supported line (T13-c):** H11 mixed (energy share 0.375, top-1 72.10 with u1 removed in loops 6-10 vs 72.82).

**Is the late-loop push one class-free direction we can cut?** No. There is no single dominant direction: the
average push holds 13% (T11-d) to 38% (T13-c) of the late-update energy, and each image moves along its own steady
direction. The average direction is also not class-free: it separates classes better than any of 200 random
directions, and cutting it costs 1.9 pt on T11-d. For T17 (orthogonal updates) this means there is no single
"keep going" direction to project out. Any repetition fix has to work per image, not on one global direction.

### Prediction check

| Prediction | Result | Verdict |
|---|---|---|
| Energy share ≥ 0.6 on both models | 0.132 (T11-d), 0.375 (T13-c) | wrong |
| u1's Fisher ratio below the random median (class-free push) | above every / all but one of 200 random directions | wrong |
| Removing u1 in loops 6..10 costs ≤ 0.5 pt on T11-d; less on T13-c | T11-d −1.88 (wrong); T13-c −0.72, less than T11-d (right) | mostly wrong |
| Removing it in loops 1..10 costs more than 0.5 pt | −2.59 / −4.60 | right |

### Plots and files

`results/singular_spectrum.png`, `results/push_along_u1_per_loop.png`, `results/fisher_u1_vs_random.png`; all numbers
in `results/cls_direction.json`. Made by `run.sh` → `cls_direction.py` (uses `projection.py`), `plot.py`.
