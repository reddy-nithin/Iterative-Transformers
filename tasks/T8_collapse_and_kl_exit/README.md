# T8 · Per-loop diagnostics on iter b0: token collapse and KL exit (no training)

**Goal.** Two cheap checks on models we already have.
1. Do the tokens collapse (all become alike) faster when one block is repeated than with 12 distinct blocks? That
   would explain why iter b0 stalls at ~58%.
2. Can a KL-based stopping rule (the zero-shot exit of the Huginn / Recurrent Depth paper, arXiv 2502.05171) beat
   "always run 11 loops" (58.1)?

**From:** advisor meeting 2026-10-20 (Dr. Uddin), meeting task T4, due Tue 2026-10-27.
Notion task: https://notion.so/40138c73470581ad8b1ecb99346e3a84

**Started / finished:** 2026-10-22 / –

## Checklist
- [ ] Collapse check on 1,000 fixed test images: token spectra per loop (`results/token_spectra.png`), effective rank
      and mean token cosine vs loop (`results/collapse_vs_loop.png`), iter b0 vs the fine-tuned 12-block baseline
- [ ] Exit signals on all 10k test images: KL(p_l ‖ p_l-1) and ‖CLS_l − CLS_l-1‖, threshold sweep, accuracy vs mean
      loops next to fixed depth and confidence (`results/exit_curves.png`)
- [ ] One line each: "does iter b0 collapse faster than the baseline?" and "does KL exit beat fixed 11 loops?"
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

## Results
_Pending._

## Takeaways
_Pending._

## Changes outside this folder
None. Imports `cls_svd.py` and `early_exit.py` from `tasks/T5_cls_svd_per_block/` read-only.
