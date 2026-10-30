# T12 · Per-loop diagnostics: does CLS move, and does it re-read the same patches?

**Goal.** Measure H1 (CLS re-reads the same patches every loop) and H3 (the body spends its update on the patches,
not on CLS) directly, and pick next week's add-on (T14) from data, not guesses. No training.

**From:** advisor meeting 2026-10-27 (Dr. Uddin), due Tue 2026-11-03.
Notion task: https://notion.so/40938c734705816f8a6ae96c0319a71c

**Started:** 2026-10-30

## Checklist
- [x] `loop_diagnostics.py` + tests (`test_t12_diagnostics.py`): recomputed block output equals the real one (< 1e-4, fp32)
- [x] iter b0 (`T5-b_iter_b0.pt`, block 0 x12, loops 1..12, x_0 = patch embedding + pos + CLS)
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
CUDA_VISIBLE_DEVICES=N PYTHONPATH=. .venv/bin/python tasks/T12_per_loop_cls_diagnostics/loop_diagnostics.py $C/T5-b_iter_b0.pt
```

## Results

### iter b0 (`T5-b_iter_b0.pt`, block 0 x12, 2026-10-30)
Own top-1 on the full test set 58.06 = logged 58.06. Recomputed block output vs real: max |diff| 8.3e-6 (fp32).
(a)-(c) on the 1,000 T8 images, (d) on all 10,000; readout = final norm + head on x_t[:, 0] (no tail). Loops 11 and 12
give 58.06 / 58.06, matching T8's 58.1 / 58.1. Pair metrics are listed at loop t for (t, t+1) and (t, t-1).

| loop t | rel_cls | rel_patch | ratio | cos(A_t, A_t+1) | H(A_t) | new-direction share | cos(ΔCLS_t, ΔCLS_t-1) | readout acc |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6842 | 0.8201 | 0.834 | 0.652 | 5.021 | 0.998 | – | 1.00 |
| 2 | 0.5221 | 0.6149 | 0.849 | 0.788 | 4.717 | 0.995 | 0.273 | 1.05 |
| 3 | 0.4505 | 0.5098 | 0.884 | 0.817 | 4.535 | 0.995 | 0.337 | 1.59 |
| 4 | 0.3881 | 0.3828 | 1.014 | 0.895 | 4.493 | 0.986 | 0.628 | 4.67 |
| 5 | 0.3203 | 0.2966 | 1.080 | 0.920 | 4.604 | 0.983 | 0.755 | 13.55 |
| 6 | 0.2620 | 0.2438 | 1.075 | 0.940 | 4.703 | 0.991 | 0.814 | 24.81 |
| 7 | 0.2312 | 0.2063 | 1.121 | 0.962 | 4.736 | 0.994 | 0.848 | 38.34 |
| 8 | 0.2137 | 0.1787 | 1.196 | 0.977 | 4.725 | 0.991 | 0.889 | 48.10 |
| 9 | 0.1958 | 0.1577 | 1.242 | 0.986 | 4.713 | 0.988 | 0.922 | 53.90 |
| 10 | 0.1755 | 0.1412 | 1.243 | 0.990 | 4.716 | 0.986 | 0.943 | 56.97 |
| 11 | 0.1561 | 0.1279 | 1.220 | 0.992 | 4.729 | 0.985 | 0.956 | 58.06 |
| 12 | 0.1395 | 0.1169 | 1.193 | – | 4.747 | 0.985 | 0.964 | 58.06 |

Cosine of the head-averaged attention (saved as a cross-check, see the gotcha): 0.709, 0.835, 0.872, 0.924, 0.943,
0.958, 0.971, 0.982, 0.989, 0.993, 0.994 for t = 1..11. Same picture.

Plots: `results/rel_update.png`, `results/attn_cos_entropy.png`, `results/new_direction.png`,
`results/exit_tail_readout.png` (T11 lines get added when their checkpoints exist).

**Against the predictions (iter b0):**
- (a) wrong. rel_cls / rel_patch is never below 0.5: 0.83-0.88 in loops 1-3, then 1.0-1.24 from loop 4 on. CLS moves
  about as much as the patches, relative to its own size, and slightly more in late loops. Both shrink every loop,
  from ~0.7-0.8 at loop 1 to ~0.12-0.14 at loop 12.
- (b) right. cos(A_t, A_t+1) is above 0.8 from loop 3 on (0.895 at loop 4, 0.99 by loop 10). Attention stays broad
  (entropy 4.5-4.7 of a possible 5.28) and nearly the same set of patches is read in every late loop.
- (c) exploratory. The CLS attention update is almost entirely off the CLS direction (share 0.98-1.0 at every loop),
  so it is not just rescaling CLS. But successive CLS updates line up more and more: cos(ΔCLS_t, ΔCLS_t-1) rises from 0.27
  (loop 2) to 0.96 (loop 12). Late loops keep pushing CLS the same way with smaller and smaller steps.
- (d) iter b0 is at chance until loop 4 and rises steeply over loops 5-10 (13.6 → 57.0), flat from loop 11.

**Decision line (iter b0):** late-loop (t ≥ 4) mean cos(A_t, A_t+1) = 0.958 > 0.8 → yes; mean rel_cls / rel_patch =
1.08, not < 0.3 → no. **iter b0: "T14 pick: coverage mask".** (The head-averaged cosine gives 0.969, same pick.)

**T11 winner:** _(T11 checkpoints still training on 2026-10-30; the T11 winner decides if it disagrees)_
