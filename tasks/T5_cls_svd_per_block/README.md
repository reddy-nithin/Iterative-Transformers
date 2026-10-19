# T5 · CLS SVD per class, per block

**Goal.** Test the "stopping rule" hypothesis: as depth grows, do the CLS vectors of images from one class collapse
toward a low-rank set? If they did, a looped model could stop iterating once per-class CLS rank stays low.

**From:** advisor meeting 2026-10-15 (Dr. Uddin + Dr. Shua), meeting task T4 (repo task T5), due Tue 2026-10-20.
Notion task: https://app.notion.com/p/3fa38c734705818ab4a0dff947c81913

**Started / finished:** 2026-10-19 / 2026-10-19 (ImageNet-pretrained model; CIFAR fine-tuned model still to do)

## Definition
For each group of images, take the CLS vector (192 numbers) at every layer and stack them into an N x 192 matrix:
```
layer 0  = CLS entering block 1 (learned token + position 0; identical for every image)
layer k  = CLS after block k (residual stream, before the final LayerNorm), k = 1..12
groups   = apple, bicycle, fox, cloud (4 CIFAR-100 superclasses, N = 500 train images each)
           + control: 500 images drawn from all 100 classes (seed 0)
```
Model: `vit_tiny_patch16_224` with its ImageNet weights as-is (no CIFAR training). Images resized to 224 with the
model's eval transform. Two versions of every SVD:
- **raw**: the matrix as is. Its top singular value mostly measures the class mean vector.
- **centred**: the class mean subtracted first. This measures how spread out one class is around its mean, which
  is the actual "collapse" question.

Rank measures: **effective rank** = exp(entropy of s / sum(s)) (Roy & Vetterli 2007; a smooth count of how many
directions are used, max 192), and **90%-energy rank** = how many singular values hold 90% of sum(s²).

Where the CLS token lives in timm 1.0.30 (`timm/models/vision_transformer.py`): created at `:840`
(`self.cls_token`), prepended to the patch tokens in `_pos_embed` at `:1075` (so it is token 0), blocks run at
`:1258`, final norm at `:1260`, and the head reads `x[:, 0]` in `pool()` at `:699`.

## Status
- [x] `cls_svd.py`: forward hooks (call order, so a shared block running 12 times also works), raw + centred SVD
- [x] `plot.py`: decay plots, effective rank vs block, summary table
- [x] Tests: `test_t5_cls_svd.py` (9 tests: shapes, hook output equals `forward_features` before the final norm,
      shared block, hooks removed, layer 0 centred rank 0, rank measures, unknown class name)
- [x] Smoke test (32 images per group) and full run (500 per group, GPU 3, 15 s), 2026-10-19
- [ ] Same analysis on the CIFAR fine-tuned baseline and on iter b0 (needs saved checkpoints; see "Next")

## Commands
Run from the repo root. Inference only, about 15 s on one A4000 (a few minutes on CPU).
```
nvidia-smi                                       # pick an idle GPU
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python tasks/T5_cls_svd_per_block/cls_svd.py \
    --save-cls /storage/INSLAB/Data/nrs3qy/itertf/t5_cls_pretrained.pt
.venv/bin/python tasks/T5_cls_svd_per_block/plot.py
PYTHONPATH=. .venv/bin/python -m pytest tests tasks/T5_cls_svd_per_block -q
```
Options: `--classes apple fox ...`, `--per-class`, `--control-size 0` (off), `--model-tag`. The raw CLS tensors
(24 MB, for a later t-SNE) are in `/storage/INSLAB/Data/nrs3qy/itertf/t5_cls_pretrained.pt`, not in the repo.

## Results (`results/`, one model, 4 classes: read as observations, not proof)
Centred effective rank (of max 192) after selected blocks; full table in `results/summary.md`.

| layer | apple | bicycle | fox | cloud | control (all classes) |
|---|---|---|---|---|---|
| 1 | 13.1 | 15.4 | 15.2 | 12.5 | 15.1 |
| 3 | 56.5 | 59.5 | 59.4 | 57.8 | 62.5 |
| 5 | 94.7 | 100.0 | 99.0 | 98.9 | 103.5 |
| 8 | 110.5 | 116.3 | 117.8 | 110.7 | 122.9 |
| 10 | 126.1 | 135.9 | 134.9 | 124.4 | 147.6 |
| 12 | 134.2 | 146.5 | 145.2 | 133.6 | 160.2 |

Plots: `results/effective_rank_vs_block.png` (main figure), `results/sv_decay_centred.png`, `results/sv_decay_raw.png`.

## Takeaways
- **Answer: no, rank does not drop with depth in the pretrained model; it rises at every block.** Centred effective
  rank goes from about 13 after block 1 to 134-147 after block 12, for all four classes. The steepest rise is blocks
  1-5 (+80), then a slower climb (+35-45 over blocks 6-12). The decay plots show the same: deeper layers have
  flatter spectra (the yellow lines sit on top).
- **Some class structure does appear, but only relative to the mix.** Classes and the control are almost equal up
  to block 4, and then the control pulls ahead: at block 12 the classes are 14-27 effective-rank points below it
  (90%-energy rank 64-81 vs 102). So images of one class use fewer directions than a random mix, and that gap
  grows with depth. It is a weak "class-shaped" signal, not a collapse.
- **Raw SVD is dominated by one shared direction.** Up to block 9, one singular value holds 90% of the energy for
  every group (the large mean CLS vector that all images share). Only blocks 10-12 spread energy out (block 12:
  16-50 directions). This is why the raw version alone would have been misleading.
- **Caveats.** These are ImageNet features never trained on CIFAR classes (and 32 px images upsampled to 224),
  so the stopping-rule idea isn't really tested yet. Class collapse is mainly expected after training with a
  classification loss (neural collapse, mostly in the last layers). Measured on the residual stream before the
  final LayerNorm.

## Next
- Run the same analysis on the CIFAR fine-tuned baseline (86.8%) and on iter b0 lr 5e-4: add a checkpoint-save
  flag to `src/train.py`, then a `--checkpoint` option here. The hooks already handle the shared block.
- Optional: repeat on the CLS after the final norm (what the head actually reads), and a t-SNE from the saved tensors.

## Changes outside this folder
None (one row in `tasks/README.md`).
