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
- [x] Follow-up (2026-10-19): `--save-checkpoint` in `src/train.py`, CIFAR fine-tuned baseline (train + test),
      original model on ImageNet, CLS after the final norm, class separation, top-1 check (see "Follow-up")
- [ ] Same analysis on iter b0 (shared block; the hooks already handle it, needs a checkpointed run)

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

## Follow-up: fine-tuned on CIFAR, and the original model on ImageNet (2026-10-19)
**Question.** The run above used a model that never learned CIFAR classes. Does per-class CLS collapse show up
(a) after fine-tuning on CIFAR-100, and (b) for the original model on its own training data, ImageNet?

**Setup.**
- Fine-tuned baseline: retrained with the T1 recipe (10 ep, lr 1e-4, seed 0) plus `--save-checkpoint`; final test
  acc **86.77%** (identical to T1). SVD on CIFAR **train** (500/class) and **test** (100/class, control 100).
- ImageNet: `vit_tiny_patch16_224` = `augreg_in21k_ft_in1k`, trained on these exact images. Read in place from the lab
  copy `/storage/INSLAB/Data/ILSVRC2012_img_train` (a labmate's; train split only, no val on the server). 500 images
  per class: 4 matched to the CIFAR set (Granny Smith, mountain bike, red fox, alp as a stand-in for "cloud") and two
  look-alike pairs (malamute / Siberian husky, tabby / tiger cat), plus a 500-image control from 500 random classes.
- New: layer **N** = CLS after the final LayerNorm (what the head reads). **Separation** = within-class spread /
  between-class spread, `tr(Σ_within) / tr(Σ_between)`, classes weighted equally (NC1-style, Papyan et al. 2020).
  Rank ignores scale; separation says whether classes get tight *relative to their distances*. **Top-1** of the
  model's own head per group as a check that weights and labels are right.

**Sanity checks.** The pretrained CIFAR rerun reproduces the table above exactly. Fine-tuned top-1: train 87-99%, test
80-98% (control 89%, in line with 86.8%). ImageNet top-1: 41-91% (look-alikes lowest: tiger cat 41, husky 47).

**Results.** Separation (lower = tighter classes), 4 classes per row:

| layer | 1 | 4 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | N |
|---|---|---|---|---|---|---|---|---|---|---|
| pretrained, CIFAR train | 2.46 | 3.02 | 2.84 | 2.45 | 2.29 | 1.71 | 1.91 | 1.66 | 1.88 | 1.86 |
| **fine-tuned, CIFAR train** | 2.57 | 2.55 | 1.95 | 1.72 | 1.34 | 0.93 | 0.77 | 0.54 | **0.49** | 0.54 |
| **fine-tuned, CIFAR test** | 2.51 | 2.42 | 1.86 | 1.65 | 1.36 | 0.93 | 0.80 | 0.58 | **0.55** | 0.62 |
| pretrained, ImageNet (matched 4) | 2.58 | 3.09 | 2.79 | 1.94 | 1.64 | 1.36 | 1.44 | 1.21 | 1.52 | 1.39 |
| ImageNet malamute vs husky | 133 | 197 | 131 | 107 | 96 | 81 | 66 | 58 | 51 | 55 |
| ImageNet tabby vs tiger cat | 50 | 109 | 84 | 72 | 39 | 38 | 36 | 22 | 29 | 29 |

Centred effective rank (class mean, train, max 192), block 8 → 12: pretrained CIFAR 114 → 140, **fine-tuned CIFAR
116 → 127** (flat from block 10: apple 123.4 / 122.1 / 123.2), ImageNet 118 → 148. Test rank is capped near 99
(N = 100) and is not comparable with train. Full tables: `results/summary_<tag>.md`.

Plots: **`results/compare.png`** (headline: rank and separation for all four runs), `separation_vs_block_<tag>.png`,
`effective_rank_vs_block_<tag>.png`, `sv_decay_*_<tag>.png`. Tags: `pretrained`, `ft_baseline_train`,
`ft_baseline_test`, `imagenet_pretrained`.

**Takeaways.**
- **Training on the classes is what makes them tighten.** After fine-tuning, separation falls from ~2.5 to 0.49 by
  block 12, below 1 (class spread smaller than the gaps between classes) from **block 9**. The same model before
  fine-tuning stays at ~1.9. The fall happens in blocks 6-11, the second half of the network.
- **It holds on unseen images.** Test (0.55) is almost the same as train (0.49), so this is not memorisation: a
  signal like this could be measured at inference time.
- **Rank alone stays a weak signal.** Even fine-tuned, per-class rank never drops; it only stops growing after
  block 10 (127 vs 146 for the control). Classes don't collapse to a few directions; they become small compared with
  the distances between them. For a stopping rule, separation (or how much the CLS still moves) looks more useful
  than rank.
- **ImageNet, the model's own data:** in between. Separation falls from ~3 to 1.2 at block 11 (starting at block 7),
  so the pretrained model does organise its own classes in the late blocks, but less tightly than CIFAR after
  fine-tuning (ImageNet has 1000 classes and heavy-augmentation training). Look-alike pairs stay heavily overlapped
  to the end (husky/malamute 55, tabby/tiger cat 29), matching their low top-1.
- **Block 12 and the final norm slightly undo block 11** in the pretrained runs (1.21 → 1.52 on ImageNet).
- Caveats: one seed, 4 classes per set, train-split ImageNet only; separation over 4 classes is not comparable with
  other class counts.

**Commands** (run from the repo root; the fine-tune runs in tmux, ~9 min; the rest is inference, seconds to 1 min):
```
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.train --model baseline --epochs 10 --lr 1e-4 \
  --out-dir tasks/T5_cls_svd_per_block/results/finetune --tag _t5 \
  --save-checkpoint /storage/INSLAB/Data/nrs3qy/checkpoints/itertf/baseline_ft.pt
C=/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/baseline_ft.pt; S=tasks/T5_cls_svd_per_block
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python $S/cls_svd.py --checkpoint $C --model-tag ft_baseline_train
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python $S/cls_svd.py --checkpoint $C --split test --per-class 100 \
  --control-size 100 --model-tag ft_baseline_test
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python $S/cls_svd.py --dataset imagenet --model-tag imagenet_pretrained
for t in pretrained ft_baseline_train ft_baseline_test imagenet_pretrained; do .venv/bin/python $S/plot.py --model-tag $t; done
.venv/bin/python $S/plot.py --compare pretrained ft_baseline_train ft_baseline_test imagenet_pretrained
```
Raw CLS tensors (for a later t-SNE): `/storage/INSLAB/Data/nrs3qy/itertf/t5_cls_{pretrained,ft_baseline_train,imagenet_pretrained}.pt`.

## Next
- iter b0 (lr 5e-4, 30 ep) with `--save-checkpoint`, then the same analysis: does the shared block tighten classes
  the same way, and from which loop?
- A stopping-rule proxy: per image, how much the CLS changes between blocks, and whether it settles where
  separation drops.
- Optional: t-SNE from the saved tensors.

## Changes outside this folder
- `src/train.py`: new `--save-checkpoint PATH` (after every epoch: model name, kwargs, state_dict, epoch, test acc).
  Without the flag nothing changes.
- One row in `tasks/README.md`.
