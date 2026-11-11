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

- [ ] README + predictions committed before the first run
- [ ] code + tests (`test_t24_projection.py`) pass
- [ ] smoke run (`SMOKE=1`)
- [ ] full run, no-edit top-1 matches the logged value for both models
- [ ] tables (a) to (c), plots `results/{singular_spectrum,push_along_u1_per_loop,fisher_u1_vs_random}.png`
- [ ] supported line per model, answer line ("Is the late-loop push one class-free direction we can cut?"),
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

_Pending._
