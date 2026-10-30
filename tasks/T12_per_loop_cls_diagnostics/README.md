# T12 · Per-loop diagnostics: does CLS move, and does it re-read the same patches?

**Goal.** Measure H1 (CLS re-reads the same patches every loop) and H3 (the body spends its update on the patches,
not on CLS) directly, and pick next week's add-on (T14) from data, not guesses. No training.

**From:** advisor meeting 2026-10-27 (Dr. Uddin), due Tue 2026-11-03.
Notion task: https://notion.so/40938c734705816f8a6ae96c0319a71c

**Started / finished:** 2026-10-30 / 2026-10-30

## Checklist
- [x] `loop_diagnostics.py` + tests (`test_t12_diagnostics.py`): recomputed block output equals the real one (< 1e-4, fp32)
- [x] iter b0 (`T5-b_iter_b0.pt`, block 0 x12, loops 1..12, x_0 = patch embedding + pos + CLS)
- [x] The four T11 checkpoints (loops 1..10 of the body, x_0 = head output; T11-d: block 2's 10 repeats)
- [x] Results table, 4 plots, decision line

## What is measured
Per loop t, averaged over images (per-image values also saved in `results/per_image_<model>.pt`):
- (a) [H3] Δ_t = x_t − x_{t−1}. rel_cls_t = ‖Δ_t[:,0]‖ / ‖x_{t−1}[:,0]‖; rel_patch_t = mean over the 196 patches of
  ‖Δ_t[:,i]‖ / ‖x_{t−1}[:,i]‖; ratio = rel_cls / rel_patch.
- (b) [H1] A_t = the CLS query's attention over the 196 patch keys (CLS→CLS dropped, renormalised), per head and
  averaged over the 3 heads. cos(A_t, A_{t+1}) and entropy H(A_t) (max log 196 = 5.28).
- (c) o_t = attention sublayer output for token 0 (after proj, before the residual add), c = x_{t−1}[:,0],
  o_perp = o_t − (o_t·c / ‖c‖²) c: ‖o_perp‖ / ‖o_t‖. Also cos(Δ_t[:,0], Δ_{t−1}[:,0]) for t ≥ 2.
- (d) Exit curve, all 10k test images: T11 models read out through the tail, classifier(final_norm(tail(x_t))[:,0]);
  iter b0 and T11-d (no tail) use final norm + classifier on x_t[:,0] (decided 2026-10-30; checks: iter b0 loops 11,
  12 = 58.1, as in T8; T11-d loop 10 = its own top-1).

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

### T11 checkpoints (2026-10-30)
Loops t = 1..10 of the body, x_0 = head output (T11-d: block 2's 10 repeats, x_0 = block 1 output). (d) is read out
through the tail, classifier(final_norm(tail(x_t))[:, 0]), on all 10,000 test images; T11-d has no tail, so its (d) uses
final norm + classifier on x_t (decided 2026-10-30, added the same evening). Own top-1 vs logged:
T11-a 62.06 / 62.05, T11-b 63.47 / 63.47, T11-c 68.60 / 68.59, T11-d 74.12 / 74.11. The one-image gap is fp32 here vs
AMP when the number was logged during training; T11-d's loop-10 readout is identical to its own full-depth top-1 (74.12).
Recompute check: max |diff| 4.8e-6 to 8.6e-6. Logs: `results/diagnostics_T11.log`,
`results/diagnostics_T11-d_readout.log`.

**T11-c (winner): head b0, body b1 x10, tail b11**

| loop t | rel_cls | rel_patch | ratio | cos(A_t, A_t+1) | H(A_t) | new-direction share | cos(ΔCLS_t, ΔCLS_t-1) | readout acc |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3519 | 0.5009 | 0.703 | 0.869 | 4.733 | 0.953 | – | 1.57 |
| 2 | 0.3327 | 0.4237 | 0.785 | 0.908 | 4.789 | 0.998 | 0.569 | 3.15 |
| 3 | 0.2584 | 0.3494 | 0.740 | 0.948 | 4.776 | 0.998 | 0.529 | 6.16 |
| 4 | 0.2749 | 0.3162 | 0.869 | 0.938 | 4.695 | 0.994 | 0.692 | 11.82 |
| 5 | 0.2650 | 0.2743 | 0.966 | 0.928 | 4.672 | 0.994 | 0.810 | 24.54 |
| 6 | 0.2383 | 0.2238 | 1.065 | 0.946 | 4.684 | 0.993 | 0.794 | 41.63 |
| 7 | 0.2199 | 0.1857 | 1.184 | 0.968 | 4.670 | 0.992 | 0.849 | 56.46 |
| 8 | 0.2041 | 0.1601 | 1.275 | 0.981 | 4.633 | 0.995 | 0.909 | 64.59 |
| 9 | 0.1854 | 0.1414 | 1.311 | 0.988 | 4.601 | 0.996 | 0.942 | 68.00 |
| 10 | 0.1656 | 0.1273 | 1.301 | – | 4.587 | 0.995 | 0.953 | 68.60 |

**T11-a: body b0**

| loop t | rel_cls | rel_patch | ratio | cos(A_t, A_t+1) | H(A_t) | new-direction share | cos(ΔCLS_t, ΔCLS_t-1) | readout acc |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4962 | 0.6602 | 0.752 | 0.776 | 4.552 | 0.989 | – | 1.54 |
| 2 | 0.3534 | 0.3955 | 0.894 | 0.877 | 4.486 | 0.987 | 0.226 | 2.66 |
| 3 | 0.2852 | 0.2866 | 0.995 | 0.934 | 4.456 | 0.975 | 0.653 | 6.26 |
| 4 | 0.2455 | 0.2314 | 1.061 | 0.934 | 4.588 | 0.984 | 0.788 | 16.91 |
| 5 | 0.2119 | 0.1940 | 1.092 | 0.952 | 4.714 | 0.995 | 0.832 | 32.49 |
| 6 | 0.1844 | 0.1658 | 1.112 | 0.974 | 4.763 | 0.995 | 0.885 | 46.75 |
| 7 | 0.1569 | 0.1440 | 1.090 | 0.986 | 4.769 | 0.989 | 0.930 | 55.36 |
| 8 | 0.1340 | 0.1273 | 1.053 | 0.991 | 4.764 | 0.982 | 0.952 | 59.71 |
| 9 | 0.1171 | 0.1142 | 1.025 | 0.994 | 4.756 | 0.974 | 0.964 | 61.91 |
| 10 | 0.1049 | 0.1039 | 1.010 | – | 4.751 | 0.968 | 0.971 | 62.06 |

**T11-b: body b6**

| loop t | rel_cls | rel_patch | ratio | cos(A_t, A_t+1) | H(A_t) | new-direction share | cos(ΔCLS_t, ΔCLS_t-1) | readout acc |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4745 | 0.5724 | 0.829 | 0.897 | 4.743 | 0.999 | – | 3.11 |
| 2 | 0.2805 | 0.3944 | 0.711 | 0.934 | 4.722 | 0.991 | 0.380 | 6.47 |
| 3 | 0.2465 | 0.3355 | 0.735 | 0.958 | 4.681 | 0.989 | 0.738 | 13.87 |
| 4 | 0.2180 | 0.2758 | 0.790 | 0.969 | 4.668 | 0.994 | 0.811 | 27.85 |
| 5 | 0.1967 | 0.2296 | 0.857 | 0.978 | 4.670 | 0.996 | 0.861 | 41.67 |
| 6 | 0.1795 | 0.1950 | 0.920 | 0.985 | 4.670 | 0.996 | 0.909 | 51.85 |
| 7 | 0.1629 | 0.1675 | 0.973 | 0.990 | 4.673 | 0.996 | 0.939 | 58.37 |
| 8 | 0.1472 | 0.1458 | 1.010 | 0.993 | 4.684 | 0.995 | 0.956 | 61.81 |
| 9 | 0.1334 | 0.1288 | 1.036 | 0.994 | 4.700 | 0.994 | 0.967 | 63.37 |
| 10 | 0.1218 | 0.1155 | 1.054 | – | 4.718 | 0.993 | 0.975 | 63.47 |

**T11-d: unrolling n = 3, no tail, (d) read out with final norm + classifier on x_t**

| loop t | rel_cls | rel_patch | ratio | cos(A_t, A_t+1) | H(A_t) | new-direction share | cos(ΔCLS_t, ΔCLS_t-1) | readout acc |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2778 | 0.6481 | 0.429 | 0.808 | 4.797 | 0.969 | – | 1.22 |
| 2 | 0.2703 | 0.4619 | 0.585 | 0.906 | 4.684 | 0.988 | 0.387 | 3.79 |
| 3 | 0.2729 | 0.3588 | 0.761 | 0.925 | 4.516 | 0.993 | 0.530 | 11.30 |
| 4 | 0.3232 | 0.3241 | 0.997 | 0.926 | 4.419 | 0.997 | 0.685 | 29.61 |
| 5 | 0.3488 | 0.2781 | 1.254 | 0.933 | 4.449 | 0.995 | 0.788 | 49.67 |
| 6 | 0.3071 | 0.2245 | 1.368 | 0.949 | 4.571 | 0.987 | 0.783 | 62.56 |
| 7 | 0.2797 | 0.1949 | 1.435 | 0.967 | 4.653 | 0.982 | 0.850 | 69.60 |
| 8 | 0.2448 | 0.1737 | 1.409 | 0.978 | 4.708 | 0.979 | 0.906 | 73.12 |
| 9 | 0.2083 | 0.1573 | 1.324 | 0.983 | 4.759 | 0.980 | 0.926 | 74.25 |
| 10 | 0.1788 | 0.1448 | 1.235 | – | 4.796 | 0.983 | 0.939 | 74.12 |

Summary per model (late = loops t ≥ 4; ratio = mean over loops of rel_cls / rel_patch):

| Model | late mean cos(A_t, A_t+1) | (head-averaged A) | mean rel_cls / rel_patch | pick by the rule |
|---|---|---|---|---|
| iter b0 | 0.958 | 0.969 | 1.08 | coverage mask |
| T11-a (body b0) | 0.972 | 0.980 | 1.01 | coverage mask |
| T11-b (body b6) | 0.985 | 0.991 | 0.89 | coverage mask |
| **T11-c (body b1, winner)** | **0.958** | 0.976 | **1.02** | **coverage mask** |
| T11-d (unroll n = 3) | 0.956 | 0.971 | 1.08 | coverage mask |

**Against the predictions, T11 models:**
- (d) partly wrong. Tail-readout accuracy does rise with t in all three, but it is within 5 pts of loop 10 only from
  loop 8, not loop 6 or 7: T11-a 55.4 at loop 7 (−6.7), T11-b 58.4 (−5.1), T11-c 56.5 (−12.1). The first 3-4 loops are near
  chance even through the tail (1.5-14%), so the tail cannot "clean up" an early state. Most of the accuracy appears
  in loops 5-8.
  T11-d (final norm + classifier readout, added later): 1.2 / 3.8 / 11.3 / 29.6 / 49.7 / 62.6 / 69.6 / 73.1 / 74.2 / 74.1. It
  rises earlier than the H-B-T models and is within 5 pts of loop 10 from loop 7 (69.6, −4.5); loop 9 (74.25) is even a
  hair above loop 10 (74.12).
- (a) The iter b0 prediction (ratio < 0.5) also fails for every T11 model. Only T11-d at loop 1 (0.43) is below 0.5.
  The ratio starts at 0.7-0.8 in loops 1-3 and rises to 1.0-1.4 by the late loops. CLS does not move less than the patches.
- (b) holds for every model. cos(A_t, A_t+1) > 0.8 from loop 1 or 2 (T11-a from loop 2), 0.93-0.99 from loop 4. All the
  pretrained bodies re-read nearly the same patches each loop, more so than iter b0 in early loops.
- (c) Same pattern as iter b0. The CLS attention update is 95-100% off the CLS direction, and successive CLS updates line
  up more and more (cos 0.2-0.6 at loop 2 → 0.94-0.98 at loop 10).

**Takeaways:** H1 (CLS re-reads the same patches) holds in every looped model measured, including the best body (b1)
and the unrolled model. H3 (the body updates patches, not CLS) is not supported: relative to its own size, CLS moves
as much as or more than the patches in the late loops. The late loops look like the same step repeated with a shrinking
size (aligned updates, near-identical attention), not new evidence being gathered.

**Decision lines** (rule applied to iter b0, the T11 winner T11-c, and T11-d, as the 2026-10-30 spec update asks):
- iter b0: late cos 0.958 > 0.8 → yes; mean ratio 1.08, not < 0.3 → no. **"T14 pick: coverage mask".**
- T11-c (T11 winner): late cos 0.958 > 0.8 → yes; mean ratio 1.02, not < 0.3 → no. **"T14 pick: coverage mask".**
- T11-d: late cos 0.956 > 0.8 → yes; mean ratio 1.08, not < 0.3 → no. **"T14 pick: coverage mask".**

All three agree, so the T11 winner does not have to break a tie (T11-a and T11-b give the same pick, with either cosine
definition). T11-d's line matters only if T14 switches to the T11-d base (T14 exception rule); it would take the same
add-on.

**T14 pick: coverage mask**
