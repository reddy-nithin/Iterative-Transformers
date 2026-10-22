# T8 · Per-loop diagnostics on iter b0: token collapse and KL exit (no training)

**Goal.** Two cheap checks on models we already have.
1. Do the tokens collapse (all become alike) faster when one block is repeated than with 12 distinct blocks? That
   would explain why iter b0 stalls at ~58%.
2. Can a KL-based stopping rule (the zero-shot exit of the Huginn / Recurrent Depth paper, arXiv 2502.05171) beat
   "always run 11 loops" (58.1)?

**From:** advisor meeting 2026-10-20 (Dr. Uddin), meeting task T4, due Tue 2026-10-27.
Notion task: https://notion.so/40138c73470581ad8b1ecb99346e3a84

**Started / finished:** 2026-10-22 / 2026-10-22 (T7 checkpoint follow-up still open)

## Checklist
- [x] Collapse check on 1,000 fixed test images: token spectra per loop (`results/token_spectra.png`), effective rank
      and mean token cosine vs loop (`results/collapse_vs_loop.png`), iter b0 vs the fine-tuned 12-block baseline
- [x] Exit signals on all 10k test images: KL(p_l ‖ p_l-1) and ‖CLS_l − CLS_l-1‖, threshold sweep, accuracy vs mean
      loops next to fixed depth and confidence (`results/exit_curves.png`)
- [x] One line each: "does iter b0 collapse faster than the baseline?" and "does KL exit beat fixed 11 loops?"
- [ ] If a T7 high-lr run stalls or diverges, rerun the collapse check on its checkpoint

## Models
- iter b0: block 0 ×12, 30 ep, lr 5e-4, 58.06% (`/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/iter_b0.pt`, from T5)
- baseline: 12 distinct blocks, fine-tuned 10 ep, lr 1e-4, 86.77% (`.../itertf/baseline_ft.pt`, from T5)

Before any geometry number is read, the script recomputes each model's own top-1 on the full test set and stops if it
differs from the logged test acc by more than 0.5 points (wrong weights or label map would give plausible but
meaningless curves).

## Method
**Collapse** (`collect_token_stats`). "Collapse" here means the tokens *within one image* becoming alike, which is
rank collapse in the sense of the CSP paper (arXiv 2410.10914, Fig. 3) and Huginn's collapse signal. For each image,
the full token matrix (197 tokens × 192 dims) is taken at the block input (loop 0) and after every loop. Per image:
- singular values of the token-centred matrix, divided by the largest (spectrum shape; a fast drop = few directions);
- their entropy effective rank (same definition as T5);
- mean pairwise cosine similarity between distinct tokens (uncentred; 1 = all tokens point the same way).

These are averaged over the 1,000 images (fixed random subset, seed 0). Stats are summed on the GPU on the fly, so no
token tensors are stored. This differs from T5, which looked at the CLS token *across images*.

**Exit rules** (`exit_analysis`). The model's own final norm + head is applied to the CLS after every loop (T5's
`exit_logits`). Signals per image and loop l:
- KL(p_l ‖ p_l-1) between successive predictions (undefined at loop 1, so no exit there);
- ‖CLS_l − CLS_l-1‖₂, the size of the CLS change (Dr. Uddin's CLS-delta rule).

Each image stops at the first loop whose signal is below a threshold (last loop otherwise). Thresholds are 13
quantiles of each signal's own values, so both signals get a sweep at their own scale. Confidence (top softmax ≥ τ,
T5's rule) is recomputed with 10 values of τ for the same plot.

Two numbers answer the question for every rule: (a) the best accuracy using ≤ 11 loops on average, compared with fixed
11 loops; (b) the fewest mean loops that stay within 1 point of full depth, compared with fixed depth.

**Caveat:** iter b0 and the baseline were trained only at the last loop, so every early-exit number is a lower bound
on what a model trained with exits at several loops could do.

## Commands
```bash
C=/storage/INSLAB/Data/nrs3qy/checkpoints/itertf
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python tasks/T8_collapse_and_kl_exit/diagnostics.py \
  $C/iter_b0.pt $C/baseline_ft.pt                        # ~10 min, fine next to the T6 run on GPU 0
PYTHONPATH=. .venv/bin/python -m pytest tests tasks -q   # tests: test_t8_diagnostics.py
```
Smoke test that was run: `--limit 500 --collapse-images 200` (about 2 min). Top-1 on those 500 images was 58.8 / 86.6,
against 58.06 / 86.77 logged on 10k.

## Results (2026-10-22, `results/diagnostics.json`, log `results/diagnostics.log`)
Own top-1 recomputed on the 10k test images: iter b0 58.06, baseline 86.77, both identical to the logged values.

**Collapse** (1,000 test images, mean per image; loop 0 = input to the first block):

| Loop | 0 | 1 | 2 | 4 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iter b0 effective rank | 33.4 | 62.4 | 79.2 | 93.6 | 100.0 | 102.3 | 104.2 | 106.0 | 107.5 | 108.8 | 110.0 |
| baseline effective rank | 31.6 | 66.9 | 85.2 | 103.3 | 102.3 | 99.8 | **20.6** | 25.7 | 30.2 | 33.4 | 34.8 |
| iter b0 token cosine | 0.31 | 0.36 | 0.31 | 0.30 | 0.30 | 0.30 | 0.29 | 0.29 | 0.28 | 0.28 | 0.28 |
| baseline token cosine | 0.29 | 0.60 | 0.46 | 0.28 | 0.26 | 0.25 | 0.30 | 0.36 | 0.38 | 0.40 | 0.43 |

Plots: `results/collapse_vs_loop.png`, `results/token_spectra.png`.

**Exit rules** (10k test images; fixed depth: iter b0 loop 10 57.0, 11 58.06, 12 58.06; baseline block 11 78.0,
12 86.8):

| Model | Rule | Best with ≤ 11 loops (acc @ mean loops) | Beats fixed 11? | Fewest loops within 1 pt of full depth |
|---|---|---|---|---|
| iter b0 | KL | 57.86 @ 10.57 | no (58.06) | 10.57 (fixed: 11) |
| iter b0 | confidence | 57.74 @ 10.72 | no | 10.72 |
| iter b0 | CLS change | 55.59 @ 10.64 | no | 11.65 |
| baseline | KL | 69.84 @ 10.46 | no (77.98) | never |
| baseline | confidence | 63.05 @ 9.20 | no | 11.44 (86.17) |
| baseline | CLS change | 66.70 @ 10.14 | no | never |

Plot: `results/exit_curves.png`.

## Takeaways
- **Does iter b0 collapse faster than the baseline? No, the opposite.** iter b0's tokens keep spreading out: effective
  rank rises every loop (33 → 110 of 192) and token cosine drifts down (0.31 → 0.28). The 12-block baseline is the
  one that concentrates: after block 8 its effective rank drops from ~100 to ~21 and cosine climbs to 0.43 by block 12.
  The spectrum shows this is one direction suddenly dominating (s₂/s₁ ≈ 0.1), not full collapse (cosine stays far
  from 1). The drop lines up with T5's finding that the baseline's classes separate from block 9. So token collapse
  does not explain the 58% stall. If anything, iter b0 lacks the late "compress to a few directions" step that the
  trained 12-block model performs.
- **Does KL exit beat fixed 11 loops (58.06)? No, it ties at slightly lower cost.** KL gives 57.86% with 10.57 loops
  on average: −0.2 points for ~4% fewer loops. That is the same trade as the confidence rule, and the L2 CLS change is
  worse. On the baseline, only confidence stays within 1 point of full depth (86.17 @ 11.44); KL and CLS change never
  do. Same conclusion as T5: with heads trained only at the last loop, no zero-shot exit signal beats simply running
  11 loops on iter b0. Exit losses at several loops (deep supervision) remain the next step.
- Caveat: one seed, and heads trained only at the last loop, so early exits are lower bounds.

## Changes outside this folder
None. Imports `cls_svd.py` and `early_exit.py` from `tasks/T5_cls_svd_per_block/` read-only.
