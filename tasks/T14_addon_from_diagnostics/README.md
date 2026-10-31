# T14 · One add-on chosen by T12's decision line: coverage mask

**Goal.** Add exactly one thing to the best looped model, picked by T12's measurements rather than by guessing. T12
showed that CLS keeps reading the same patches loop after loop (late-loop cos(A_t, A_t+1) 0.956 on T11-d, 0.995 on
T13-d). The coverage mask gives CLS a memory of which patches it has already read (H1).

**From:** advisor meeting 2026-10-27 (Dr. Uddin), due Tue 2026-11-03.
Notion task: https://notion.so/40938c7347058111b31ef3ba75360dd9

**Started / finished:** 2026-10-30 / 2026-10-30

## Base and add-on (why these)
- Base rule: the T13 winner on the T11-c layout, unless T13-c or T13-d beats both T13-a and T13-b by ≥2 pts. T13-d
  (73.17) beats T13-a / T13-b (68.85 each) by 4.3 pts, so the base is **T13-d**: unrolling n = 3 (B0, B1 distinct,
  then B2 x10, no tail), readout at every loop + loss margin (m 0.02, λ 1.0, floor 1.0, w_t = 0.85^(10−t)).
- Add-on from T12's decision line for T11-d: late cos 0.956 > 0.8, mean rel_cls / rel_patch 1.08 (not < 0.3) →
  **"T14 pick: coverage mask"**.
- Pretrained init exactly as in T11 / T13 (NOT from the T13-d checkpoint), shared recipe, 30 epochs, so the only
  difference from T13-d is the mask (+1 parameter).

## The coverage mask (run T14-a)
Only in the looped block's attention (B2), only on the CLS query row:
- per-image supply vector c ∈ R^196, c = 1 before loop 1;
- in loop t, add log(c_i + 1e-6) to the CLS→patch-i attention logits (all heads; the CLS→CLS logit and all patch
  query rows unchanged);
- after loop t: c ← clamp(c − η · A_t, min 0), A_t = CLS→patch attention averaged over heads (CLS→CLS dropped,
  renormalised, as in T12 (b));
- η = one learnable scalar, initialised at 0.5. Readout unchanged (final norm + classifier on x_t).

Code: `src/coverage_mask.py`, flag `--coverage-mask` for `--model unroll` in `src.train`.

| Run | Model | Loss | Reference | Checkpoint |
|---|---|---|---|---|
| T14-a | T13-d + coverage mask (B0, B1, masked B2 x10) | readout every loop + loss margin | T13-d (73.17) | `T14-a_coverage_mask.pt` |

## Checklist
- [x] Predictions committed before any run
- [x] `src/coverage_mask.py`, `--coverage-mask`, tests `test_t14_coverage_mask.py` (13 tests; full suite 132 pass)
- [x] Smoke test
- [x] T14-a, 30 epochs, one GPU, one tmux session
- [x] T12 metrics (a)-(c), T13 exit curves, fixed-depth curve, flip rate, separation per loop; T13-d as reference
- [x] Results row, plots, learned η, the line "did the add-on do what T12 predicted?"

## Predictions (written before running)
Coverage mask → T12 metric (b) drops below 0.5 and top-1 +1 to +3 over base. CLS adapters →
rel_cls/rel_patch doubles, top-1 +1 to +3. Per-loop LayerNorm → small change (≤1 pt). Low confidence on all.

## Commands
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests tasks -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T14_addon_from_diagnostics/run.sh
tmux new -d -s itertf-T14-a "CUDA_VISIBLE_DEVICES=N bash tasks/T14_addon_from_diagnostics/run.sh"
.venv/bin/python tasks/T14_addon_from_diagnostics/plot.py
```

## Results
T14-a: 30 epochs on GPU 0 alone (2026-10-30, 19:41-20:13, ~63 s/epoch, peak 3.0 GB at batch 128), seed 0, shared
recipe unchanged. Train/eval scripts are the T13 ones: exit numbers on all 10,000 test images in fp32 (loop 10 = 72.88,
logged 72.89 with AMP), T12 metrics (a)-(c) on the 1,000 T8 images, separation = T5 `cls_svd.py`, 4 classes, test.
The T12 recompute with the mask replayed matches the real block output exactly (max diff 0). T13-d numbers are read
from `tasks/T13_per_loop_readout_loss/results/` (same scripts, same images).

| Run | Model | Params | Learned η | Final top-1 | Best top-1 | Loop-5 acc | Flip 9→10 (%) | Fewest loops within 1 pt of loop 10 | Mean rel_cls loops 1-5 | Late-loop cos(A_t, A_t+1) | Mean H(A_t) | Sep. last loop (test) | Min loop sep. (test, call) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T13-d | base: every loop + margin | 1,539,940 | – | 73.17 | 73.17 | 73.30 | 1.58 | kl 0.0316: 4.07 loops (73.13%) vs fixed 4 | 0.512 | 0.995 | 4.76 | 1.40 | 1.38 (9) |
| T14-a | base + coverage mask | 1,539,941 | 0.6375 | 72.89 | 72.89 | 73.20 | 1.66 | kl 0.0316: 4.08 loops (72.83%) vs fixed 4 | 0.517 | 0.995 | 4.75 | 1.38 | 1.34 (8) |

Readout accuracy at loop t (fixed depth, %):

| loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| T13-d | 45.1 | 63.9 | 70.2 | 72.5 | 73.3 | **73.6** | 73.5 | 73.4 | 73.3 | 73.2 |
| T14-a | 44.7 | 64.2 | 70.2 | 72.5 | 73.2 | 73.4 | **73.4** | 73.3 | 73.1 | 72.9 |

T12 metric (b) per loop, cos(A_t, A_t+1) (t = 1..9): T13-d 0.664 0.863 0.960 0.986 0.993 0.996 0.998 0.998 0.999;
T14-a 0.680 0.857 0.959 0.985 0.993 0.996 0.997 0.998 0.999. Entropy of A_t, mean ratio rel_cls / rel_patch (1.16
vs 1.17) and the new-direction share are the same to 2 decimals. Learned η: 0.5 → 0.66 by epoch 10 → 0.64 at
epoch 30 (stayed positive, so c stayed in [0, 1]; see the G2 gotcha). Final-output train CE 1.715 (T13-d 1.717).

**Why the mask did nothing (checked on 200 test images, final checkpoint):** CLS attention over patches is spread out
(entropy ~4.8 of max 5.28; the most-attended patch gets ~3% at loop 1 and ~2% at loop 10). A_t sums to 1, so each
loop removes only η ≈ 0.64 of supply in total, spread over 196 patches. Mean supply after loop 10 is 0.968; even each
image's most-read patch keeps c ≈ 0.85 (lowest anywhere 0.64, no patch below 0.5), so the strongest bias the mask ever
adds is log 0.85 ≈ −0.17 on a logit. For the mask to block a patch, η would need to be ~30x larger. Training raised η only
from 0.5 to 0.64 (weight decay also pulls it toward 0). As specified, the add-on can't change which patches CLS
reads.

Plots: `results/exit_curves.png`, `results/acc_per_loop.png`, `results/rel_update.png`,
`results/attn_cos_entropy.png` (incl. supply left per loop), `results/new_direction.png`,
`results/separation_vs_loop.png`, `results/loss_curves.png` (incl. η per epoch). Raw: `results/unroll_T14-a.json`,
`results/exit_analysis.json`, `results/t12_diag/`, `results/cls_svd_T14-a_*.json`, logs; table in
`results/summary.md`. Checkpoint: `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/T14-a_coverage_mask.pt`.

**Against the predictions** (one seed; the predictions section above is unchanged):
- Coverage mask → T12 metric (b) drops below 0.5: wrong. Late-loop cos(A_t, A_t+1) is 0.995, identical to T13-d.
- Top-1 +1 to +3 over base: wrong. 72.89 vs 73.17 (−0.28, a tie at one seed).

**Did the add-on do what T12 predicted?** No. Late-loop attention cos stayed at 0.995 (not < 0.5) and top-1 was 72.89 vs
73.17 (−0.3, not +1 to +3). The mask is too weak to act: with spread-out CLS attention, η · A_t uses up at most ~15%
of any patch's supply in 10 loops, and training moved η only from 0.5 to 0.64.

**Takeaways:**
- With CLS attention this flat, a supply that drains by η · A_t (A_t summing to 1) can't build a memory. Every other
  number (exit curves, flip rate, separation, rel_cls) matches T13-d within noise.
- So H1 is not tested yet: we know CLS re-reads the same patches, but this mask never forced it not to. A test that
  actually bites would need a much larger or differently scaled depletion. That is a new design decision for the
  meeting, not something to try here.
