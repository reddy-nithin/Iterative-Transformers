# T28 · Frozen-patch CLS-only loop (Dr. Uddin's reading of Huginn)

**From:** advisor meeting 2026-11-05.
Notion task: https://notion.so/40f38c73470581538ee5edce6084fe77

**Goal.** Dr. Uddin's reading of Huginn: process the image once, then freeze it and let only CLS loop, collecting
evidence from the fixed patches. In our model the loop also keeps changing the patches. Test his version with one
change. Side benefit: a loop that updates one token instead of 197 costs far less.

## Setup

- Layout and reference: **T11-d** (B0, B1 on all tokens, then B2 x10, no tail, final-only loss, T11-d config).
  Reference T11-d 73.79 (3-seed spread 0.77; seed 0: 74.11), checkpoint `T11-d_unroll_n3_30ep.pt`.
- **One change:** after B1 the patch tokens P = x[:, 1:] are frozen (no detach: gradients still flow into B0/B1
  through P). In each of the 10 loops B2 updates ONLY the CLS row: the CLS query attends to keys and values from
  norm1([CLS; P]); CLS gets the attention output and then the MLP on its own row, exactly as in a normal B2 pass; the
  patch rows stay P. Readout as T11-d. No new parameters.
- Code: `src/cls_only_loop.py` (`CLSOnlyBlock`), flag `--cls-only-loop` for `--model unroll`.
  The CLS query is the only query computed; the patch keys/values are recomputed each loop (same numbers as caching
  them once, which would be cheaper still).

| Run | Flags | Checkpoint |
|---|---|---|
| T11-d (reference) | `--model unroll --unique-blocks 3` | `T11-d_unroll_n3_30ep.pt` |
| T28-a | + `--cls-only-loop` | `T28-a_cls_only_loop.pt` |

Shared recipe: pretrained init, lr 5e-4, wd 0.05, 1 warmup epoch + cosine, label smoothing 0.1, batch 128, AMP,
30 epochs, seed 0. Checkpoints in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`.

## Checklist

- [x] README + predictions committed before the first run
- [x] code + tests (`test_t28_cls_only.py`) pass
- [x] smoke run (`SMOKE=1`)
- [x] full run T28-a (tmux), per-loop readout, separation per loop
- [x] FLOPs of one loop (both models) and wall time per epoch
- [x] results row + reference, FLOPs and time table, answer line, prediction check, inbox entry

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T28_frozen_patch_cls_loop -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T28_frozen_patch_cls_loop/run.sh
tmux new -s t28-train 'CUDA_VISIBLE_DEVICES=N bash tasks/T28_frozen_patch_cls_loop/run.sh'
```

## Predictions (written before running)

Copied verbatim from the task spec:

top-1 3 to 8 pts below T11-d (the patches stop refining); one CLS-only loop costs under 2% of a full loop. If it
lands within 2 pts of T11-d, that is a strong efficiency result. Low confidence.

## Results

Run 2026-11-11 on GPU 0, at the same time as T29-a and T30-a (one GPU each), seed 0. Sanity check: the per-loop
readout at loop 10 gives 67.94, logged 67.93.

### Accuracy, cost and separation (CIFAR-100 test)

| model | params | GFLOPs / image | throughput (img/s) | median epoch (s) | final top-1 | separation after loop 10 (test / train) |
|---|---|---|---|---|---|---|
| T11-d (reference) | 1,539,940 | 2.51 | 5,788 | 78.7* | 74.11 (3 seeds: 73.79, spread 0.77) | 1.151 / 1.022 |
| **T28-a** CLS-only loop | 1,539,940 | **0.77** | **12,697** | 62.8* | **67.93** (−5.86 vs the 3-seed mean) | 1.477 / 1.345 |

\* Wall time per epoch is set by the CPU data loading (224-px augmentation), not by the model: T28-a ran next to two
other trainings, and T11-d ran on another day under a different load. The epoch times do not show the compute saving;
throughput (inference, same GPU, 256 images per batch) does: 2.2x faster.

### FLOPs of one loop (input 1 x 197 x 192; `results/flops.json`)

| loop | FLOPs | share of a full loop |
|---|---|---|
| full B2 loop (T11-d) | 204.1 M | 100% |
| CLS-only loop **as coded** (patch keys/values recomputed each loop) | 29.9 M | **14.7%** |
| CLS-only loop with patch keys/values cached once (same numbers, not coded) | 0.89 M | 0.44% |

Counted with `FlopCounterMode` and checked against the formula. On CPU the counter does not see PyTorch's fused
attention kernel, so the attention matmuls (4·N·N·C for the full loop, 4·N·C for CLS only) are added back; the
script stops if the two do not match. Most of the coded CLS-only loop (29.0 of 29.9 M) is computing keys and values
for 197 tokens that never change.

### Per loop (test)

| model | | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T11-d | readout top-1 | 1.22 | 3.79 | 11.30 | 29.61 | 49.67 | 62.56 | 69.60 | 73.12 | 74.25 | 74.12 |
| T28-a | readout top-1 | 9.82 | 25.70 | 39.97 | 50.95 | 59.05 | 64.07 | 66.76 | 68.10 | 68.24 | 67.94 |
| T11-d | separation | 2.278 | 2.167 | 1.733 | 1.346 | 1.126 | 1.062 | 1.061 | 1.084 | 1.116 | 1.151 |
| T28-a | separation | 2.127 | 2.051 | 1.937 | 1.872 | 1.802 | 1.718 | 1.635 | 1.566 | 1.514 | 1.477 |

(separation = within-class spread / between-class spread of CLS, lower = tighter classes; T5 code, 100 test images
per class)

- T28-a is ahead of T11-d in the early loops (25.7 vs 3.8 at loop 2): with patches fixed, CLS reads a stable picture
  from the start. But it climbs slowly and levels off at about 68 by loop 8, while T11-d keeps rising to 74.
- Classes get tighter at every loop in T28-a, but end much looser than in T11-d (1.48 vs 1.15). T11-d is tightest
  around loops 6 to 7 (1.06).
- One seed, but the gap (5.9 pt) is about 7 times the seed spread, so it counts as an effect.

Plots: `results/acc_per_loop.png`, `results/separation_per_loop.png`. Made by `run.sh` and `plot.py`.

**Does CLS need the patches to keep evolving?** Yes, for accuracy: freezing the patches after B1 costs 5.9 pt
(67.9 vs 73.8) and leaves the classes looser. CLS alone, re-reading the same patches 10 times, levels off at 68.
It is a real cost-accuracy trade, not a free lunch: 0.77 instead of 2.51 GFLOPs per image (3.3x fewer) and 2.2x the
throughput, and a loop that caches the patch keys/values would cost under 0.5% of a full loop.

### Prediction check

| Prediction | Result | Verdict |
|---|---|---|
| top-1 3 to 8 pts below T11-d | −5.86 (67.93 vs 73.79) | right |
| one CLS-only loop costs under 2% of a full loop | 14.7% as coded; 0.44% with cached patch keys/values | right only with caching |
| within 2 pts of T11-d would be a strong efficiency result | 5.9 pts below: not that result | n/a |
