# T22 · Block phases from layer similarity + max-cut (no training) [H8]

**From:** paper-read session 2026-11-03 (Nithin), alphaXiv Itertf folder.
Notion task: https://notion.so/40d38c734705812a9e1fdc2a95db9187

**Goal.** Our head / body / tail cuts (B0 | B1..B10 | B11 in T11-c, B0 | B1 | B2..B11 in T11-d) were picked by hand.
Raptor (arXiv 2512.19941) shows that a ViT's 12 blocks fall into a few contiguous **phases**, where blocks inside a phase
transform tokens in a similar way, found from a block-to-block similarity matrix and a **max-cut** (a split into
contiguous groups that keeps similar blocks together). If the data picks different cuts from ours, H8 step 2 (one
training run with the data's layout) is worth a GPU. This task only measures.

## Setup

- **Models** (two separate analyses):
  - (A) pretrained timm `vit_tiny_patch16_224`, exactly the weights T11 starts from (ImageNet head unused);
  - (B) the **baseline** checkpoint `T5-a_baseline_ft.pt` (12 distinct pretrained blocks fine-tuned 10 ep, lr 1e-4,
    86.77), in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`. Its top-1 on the full test set is checked against
    the logged value first.
- **Data:** the 10,000 CIFAR-100 test images, 224 px, test transform, `model.eval()`, fp32.
- **Similarity matrices.** h_l = residual state after block l, l = 0..11, [N, 197, 192].
  S_tok[l, m] = mean over images and all 197 tokens of cos(h_l[n, j], h_m[n, j]); S_cls[l, m] = the same on token 0.
  Running sums over batches; no states stored.
- **Contiguous max-cut, k = 3:** cut points 0 < c1 < c2 < 12 (55 options).
  cut score = Σ over pairs (l < m) in DIFFERENT groups of (1 − S[l, m]);
  within score = mean over groups of the mean S[l, m] over pairs l < m inside the group (one-block groups left out).
  Brute force for S_tok and S_cls, models (A) and (B); top 5 by cut score with within score; best cut for k = 2 (11
  options) and k = 4 (165 options). The cut score is our own definition, not necessarily Raptor's.
- **Rank** (1 = best cut score) of the T11-c cut {0}, {1..10}, {11} and the T11-d cut {0}, {1}, {2..11} among the 55.
- **Decision line:** "run H8 step 2 with groups <best groups>" if, on (A) S_tok, the best k = 3 cut differs from the
  T11-c cut AND the T11-c cut ranks 6th or worse; otherwise "skip H8 step 2", with both ranks.

## Checklist

- [ ] README + predictions committed before the first run
- [ ] code + tests (`test_t22_maxcut.py`) pass
- [ ] smoke run (`SMOKE=1`)
- [ ] full run, baseline top-1 check passes
- [ ] `results/S_*.npy`, ranking tables, plots `results/layer_similarity_{pretrained,baseline,cls}.png`
- [ ] decision line, answer line, prediction check, inbox entry

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T22_block_phase_maxcut -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T22_block_phase_maxcut/run.sh
CUDA_VISIBLE_DEVICES=N bash tasks/T22_block_phase_maxcut/run.sh
```

## Predictions (written before running)

Copied verbatim from the task spec:

- The best k = 3 cut on (A) S_tok puts blocks 0 to 1 or 0 to 2 in the first group and the last 2 to 4 blocks in the
  last group, so it differs from the T11-c cut.
- The T11-c cut ranks 6th or worse.
- The T11-d cut ranks better than the T11-c cut.
- The baseline (B) has sharper phases (larger gap between within-group and across-group S) than (A), because
  fine-tuning specialises the late blocks.
- Low confidence on all.

## Results

_Pending._
