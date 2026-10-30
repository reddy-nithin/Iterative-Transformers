# T12 · Per-loop diagnostics: does CLS move, and does it re-read the same patches?

**Goal.** Measure H1 (CLS re-reads the same patches every loop) and H3 (the body spends its update on the patches,
not on CLS) directly, and pick next week's add-on (T14) from data, not guesses. No training.

**From:** advisor meeting 2026-10-27 (Dr. Uddin), due Tue 2026-11-03.
Notion task: https://notion.so/40938c734705816f8a6ae96c0319a71c

**Started:** 2026-10-30

## Checklist
- [ ] `diagnostics.py` + tests (`test_t12_diagnostics.py`): recomputed block output equals the real one (< 1e-4, fp32)
- [ ] iter b0 (`T5-b_iter_b0.pt`, block 0 x12, loops 1..12, x_0 = patch embedding + pos + CLS)
- [ ] The four T11 checkpoints (loops 1..10 of the body, x_0 = head output; T11-d: block 2's 10 repeats)
- [ ] Results table, 4 plots, decision line

## What is measured
Per loop t, averaged over images (per-image values also saved in `results/per_image_<model>.pt`):
- (a) [H3] Δ_t = x_t − x_{t−1}. rel_cls_t = ‖Δ_t[:,0]‖ / ‖x_{t−1}[:,0]‖; rel_patch_t = mean over the 196 patches of
  ‖Δ_t[:,i]‖ / ‖x_{t−1}[:,i]‖; ratio = rel_cls / rel_patch.
- (b) [H1] A_t = the CLS query's attention over the 196 patch keys (CLS→CLS dropped, renormalised), per head and
  averaged over the 3 heads. cos(A_t, A_{t+1}) and entropy H(A_t) (max log 196 = 5.28).
- (c) o_t = attention sublayer output for token 0 (after proj, before the residual add), c = x_{t−1}[:,0],
  o_perp = o_t − (o_t·c / ‖c‖²) c: ‖o_perp‖ / ‖o_t‖. Also cos(Δ_t[:,0], Δ_{t−1}[:,0]) for t ≥ 2.
- (d) Exit curve, all 10k test images: T11 models read out through the tail, classifier(final_norm(tail(x_t))[:,0]);
  iter b0 (no tail) uses final norm + classifier on x_t[:,0] (check: loops 11, 12 = 58.1, as in T8).

Images: the same fixed 1,000 CIFAR-100 test images as T8 (its `test_loaders`, seed 0) for (a)-(c), all 10,000 for (d).
Eval mode, no augmentation, fp32. The forward pass is not changed: the body's inputs/outputs come from a forward hook,
attention weights and o_t are recomputed from the block's own modules, and the recomputed block output is asserted
equal to the hooked one.

## Predictions (written before running)
(a) for iter b0, rel_cls / rel_patch < 0.5 at most loops (CLS moves less
than patches). (b) for iter b0, cos(A_t, A_{t+1}) > 0.8 from loop 4 on (CLS keeps reading the same patches);
confidence only ~60%. (c) no prediction (exploratory). (d) for T11 models, tail-readout accuracy rises with t and
is already within 5 pts of loop 10 by loop 6 or 7.

## Decision rule
Computed with exactly this rule, for iter b0 AND for the T11 winner: "T14 pick: coverage mask" if late-loop
(loops ≥ 4) mean cos(A_t, A_{t+1}) > 0.8; "T14 pick: CLS adapters" if mean rel_cls / rel_patch < 0.3; both true →
"coverage mask first, then CLS adapters"; neither → "per-loop LayerNorm". If iter b0 and the T11 winner disagree,
the T11 winner decides.

How the averages are taken (my reading, also logged as a gotcha): cos(A_t, A_{t+1}) and H(A_t) are computed per head,
averaged over the 3 heads, then over images (the cosine of the head-averaged A is saved too); "late-loop mean cos" =
mean over t ≥ 4 of the per-loop values; "mean rel_cls / rel_patch" = mean over all loops of the per-loop ratio.

## Commands
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests tasks -q
C=/storage/INSLAB/Data/nrs3qy/checkpoints/itertf
CUDA_VISIBLE_DEVICES=N PYTHONPATH=. .venv/bin/python tasks/T12_per_loop_cls_diagnostics/diagnostics.py $C/T5-b_iter_b0.pt
```

## Results
_(after the run)_
