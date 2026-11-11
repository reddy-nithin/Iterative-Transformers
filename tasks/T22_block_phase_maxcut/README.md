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

- [x] README + predictions committed before the first run
- [x] code + tests (`test_t22_maxcut.py`) pass
- [x] smoke run (`SMOKE=1`)
- [x] full run, baseline top-1 check passes
- [x] `results/S_*.npy`, ranking tables, plots `results/layer_similarity_{pretrained,baseline,cls}.png`
- [x] decision line, answer line, prediction check, inbox entry

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

Run 2026-11-11, one GPU (~2 min). Sanity check: the baseline's top-1 on the full test set is 86.77 (logged 86.77).
The pretrained model has no CIFAR head, so no check applies to it. Matrices: `results/S_{tok,cls}_{pretrained,baseline}.npy`.

### k = 3 ranking (top 5 + our two hand-picked cuts, out of 55)

| Matrix | rank | groups | cut score | within | across mean |
|---|---|---|---|---|---|
| (A) pretrained S_tok | 1 | {0..3} {4..7} {8..11} | 30.744 | 0.678 | 0.359 |
| | 2 | {0..2} {3..6} {7..11} | 30.741 | 0.685 | 0.346 |
| | 3 | {0..3} {4..6} {7..11} | 30.527 | 0.684 | 0.350 |
| | 4 | {0..2} {3..7} {8..11} | 30.384 | 0.680 | 0.354 |
| | 5 | {0..4} {5..7} {8..11} | 30.080 | 0.687 | 0.360 |
| | 54 (T11-c) | {0} {1..10} {11} | 15.000 | 0.521 | 0.286 |
| | 53 (T11-d) | {0} {1} {2..11} | 15.004 | 0.521 | 0.286 |
| (A) pretrained S_cls | 1 | {0..2} {3..7} {8..11} | 16.392 | 0.845 | 0.651 |
| | 2 | {0..3} {4..7} {8..11} | 16.385 | 0.847 | 0.659 |
| | 3 | {0..3} {4..8} {9..11} | 16.362 | 0.831 | 0.652 |
| | 4 | {0..4} {5..8} {9..11} | 16.342 | 0.835 | 0.652 |
| | 5 | {0..2} {3..6} {7..11} | 16.137 | 0.859 | 0.657 |
| | 51 (T11-c) | {0} {1..10} {11} | 10.253 | 0.802 | 0.512 |
| | 55 (T11-d) | {0} {1} {2..11} | 7.569 | 0.743 | 0.640 |
| (B) baseline S_tok | 1 | {0..3} {4..7} {8..11} | 32.648 | 0.671 | 0.320 |
| | 2 | {0..2} {3..6} {7..11} | 32.394 | 0.675 | 0.311 |
| | 3 | {0..3} {4..6} {7..11} | 32.299 | 0.672 | 0.313 |
| | 4 | {0..4} {5..7} {8..11} | 32.158 | 0.683 | 0.316 |
| | 5 | {0..2} {3..7} {8..11} | 32.128 | 0.674 | 0.316 |
| | 53 (T11-c) | {0} {1..10} {11} | 15.426 | 0.486 | 0.265 |
| | 54 (T11-d) | {0} {1} {2..11} | 15.326 | 0.483 | 0.270 |
| (B) baseline S_cls | 1 | {0..4} {5..8} {9..11} | 21.432 | 0.769 | 0.544 |
| | 2 | {0..5} {6..9} {10..11} | 21.353 | 0.760 | 0.515 |
| | 3 | {0..3} {4..8} {9..11} | 21.320 | 0.769 | 0.546 |
| | 4 | {0..5} {6..8} {9..11} | 21.229 | 0.770 | 0.528 |
| | 5 | {0..4} {5..9} {10..11} | 21.178 | 0.757 | 0.529 |
| | 52 (T11-c) | {0} {1..10} {11} | 13.088 | 0.738 | 0.377 |
| | 55 (T11-d) | {0} {1} {2..11} | 8.049 | 0.626 | 0.617 |

Best cut for k = 2 and k = 4:

| Matrix | best k = 2 | best k = 4 |
|---|---|---|
| (A) S_tok | {0..5} {6..11} | {0..2} {3..5} {6..8} {9..11} |
| (A) S_cls | {0..6} {7..11} | {0..2} {3..6} {7..9} {10..11} |
| (B) S_tok | {0..5} {6..11} | {0..2} {3..5} {6..8} {9..11} |
| (B) S_cls | {0..6} {7..11} | {0..2} {3..6} {7..9} {10..11} |

### What the matrices look like (`results/layer_similarity_pretrained.png`)

- **There are no sharp phases. Similarity simply fades with distance:** each block is ~0.75 to 0.9 similar to its
  neighbours, and that falls off smoothly further away (block 0 vs block 6: 0.15). The best cuts sit where the
  balanced split falls, and the top 5 are almost tied (30.74, 30.74, 30.53, ...).
- Two local features stand out: blocks **8 to 10** form the tightest cluster (0.80 to 0.90 with each other), and
  **block 11** differs from everything except 9 and 10 (−0.11 against block 0).
- **Caveat on the score (written down as a gotcha).** The cut score adds up (1 − S) over every pair across groups, so
  it rewards splits with many cross-group pairs. A 4/4/4 split has 48 such pairs; our {0} {1..10} {11} has only 21.
  So a cut with one-block groups ranks near the bottom almost by construction, whatever S is. The low ranks of
  T11-c and T11-d therefore mostly reflect group sizes, not a measured mismatch with the data. The within scores
  agree on the direction (0.52 for the hand cuts vs 0.68 for the best on (A) S_tok), but they also favour groups
  of similar size.

**Decision line:** run H8 step 2 with groups {0..3}, {4..7}, {8..11} (T11-c cut rank 54/55, T11-d cut rank 53/55).
Read it with the caveat above: rank 1 and rank 2 ({0..2} {3..6} {7..11}) differ by 0.003, and the score favours
equal sizes. A balanced 4/4/4 layout is a fair thing to train once, but the data does not single it out.

**Do the data's phases match our hand-picked cuts?** No. On every matrix, both hand-picked cuts rank 51st to 55th of
55, and the best data cuts are balanced groups of 3 to 5 blocks. But the pretrained ViT-Tiny has no sharp phases,
only a smooth band. The strongest structure in the data is a tight 8..10 group and a distinct block 11, which is
closer to a "tail" than to a separate phase.

### Prediction check

| Prediction | Result | Verdict |
|---|---|---|
| Best (A) S_tok cut has first group 0..1 or 0..2, last group the last 2 to 4 blocks, differs from T11-c | {0..3} {4..7} {8..11}: last group 4 blocks, differs from T11-c, but first group 0..3 (rank 2, {0..2} {3..6} {7..11}, is 0.003 behind) | mostly right |
| T11-c ranks 6th or worse | 54th (S_tok A), 51st to 53rd elsewhere | right (but see the score caveat) |
| T11-d ranks better than T11-c | (A) S_tok 53 vs 54, a tie (15.004 vs 15.000); worse on (B) S_tok and both S_cls | wrong (tie at best) |
| Baseline has sharper phases than pretrained | best-cut within − across: S_tok 0.351 (B) vs 0.319 (A); S_cls 0.225 vs 0.194 | right, but small |

### Plots and files

`results/layer_similarity_pretrained.png`, `results/layer_similarity_baseline.png`, `results/layer_similarity_cls.png`;
all rankings in `results/block_phases.json`. Made by `run.sh` → `block_phases.py` (uses `maxcut.py`), `plot.py`.
