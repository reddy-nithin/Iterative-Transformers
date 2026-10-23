# T7 · Accuracy ceiling of the single shared block (lr sweep)

**Goal.** How far can lr alone push iter b0 (pretrained block 0 reused 12 times, 0.65M params)? Raising lr from
1e-4 to 5e-4 and training 30 epochs took it from 34.8 to 58.1 (T4), still rising. Dr. Uddin doubts it can reach 80.

**From:** advisor meeting 2026-10-20 (Dr. Uddin), meeting task T2, due Tue 2026-10-27.
Notion task: https://notion.so/3ff38c73470581f48d82c181ef38e399

**Started / finished:** 2026-10-22 / 2026-10-22

## Checklist
- [x] lr 1e-3, 2e-3, 5e-3 at 30 epochs, seed 0, checkpoints saved (meeting T4 may reuse them)
- [x] Stop raising lr once accuracy drops or a loss goes NaN (built into `sweep.sh`)
- [x] Combined plot, test acc vs epoch, one line per lr (`results/acc_vs_epoch_by_lr.png`)
- [x] Results table below, plus one line stating the best lr and its accuracy

## Setup
`--model iterative --block-idx 0`, otherwise the T4 recipe (AdamW wd 0.05, 1 warm-up epoch + cosine, label smoothing
0.1, batch 128, AMP, 30 epochs, seed 0). Only the lr changes. No gradient clipping is added: if a high lr diverges,
that is the answer, not a bug to fix.

**Stop rule** (`sweep.sh`): lrs run in increasing order. After each run, if its final test acc is below the previous
lr's (starting from T4's lr 5e-4 = 58.06), or any epoch has a NaN loss, the higher lrs are skipped.

Earlier points shown in the plot: lr 1e-4 (10 epochs only, T1, 34.76) and lr 5e-4 (30 epochs, T4, 58.06).

## Commands
```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "import torch;f,t=torch.cuda.mem_get_info();print((t-f)/2**30,'GB used')"
tmux new -s itertf-t7-lr-1022
CUDA_VISIBLE_DEVICES=1 bash tasks/T7_shared_block_lr_sweep/sweep.sh      # ~27 min per lr, up to ~80 min
.venv/bin/python tasks/T7_shared_block_lr_sweep/plot.py                  # table -> results/summary.md
```
Checkpoints: `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/iter_b0_lr<lr>.pt`.
Smoke test that was run: `OUT_ROOT=<scratch> CKPT_DIR=<scratch> EPOCHS=1 EXTRA_ARGS="--subset 512" ... sweep.sh 1e-3 2e-3`
(1-epoch acc 0.98 < 58.06, so the stop rule fired and skipped 2e-3, as designed; the NaN check was tested on a
JSON with a NaN loss).

## Results (one seed)
iter b0 (block 0 ×12, 0.65M params), CIFAR-100 test. Table also in `results/summary.md`, plot in
`results/acc_vs_epoch_by_lr.png`.

| lr | Epochs | Final test acc (%) | Best test acc (%) | Final train acc (%) | NaN loss | Source |
|---|---|---|---|---|---|---|
| 1e-4 | 10 | 34.76 | 34.76 | – | no | T1 |
| 5e-4 | 30 | 58.06 | 58.22 | 55.33 | no | T4 |
| **1e-3** | 30 | **60.46** | 60.46 | 59.50 | no | this task |
| 2e-3 | 30 | 60.04 | 60.11 | 59.02 | no | this task |
| 5e-3 | – | skipped by the stop rule (2e-3 < 1e-3) | | | | |

**Best lr: 1e-3, 60.46% test accuracy.**

## Takeaways
- lr alone tops out at about 60%: +2.4 points over lr 5e-4, and nowhere near 80, as Dr. Uddin expected. lr 1e-3 and
  2e-3 are a tie at one seed. The accuracy is flat across lr, not still climbing.
- No divergence or NaN up to 2e-3, so the ceiling is not a training-stability problem.
- Train accuracy (59.5, a running average over augmented training batches) is no higher than test accuracy (60.5). The
  model is not overfitting: it can't fit the training set any better. That points to **capacity** (one block's
  0.65M parameters) as the limit, not optimisation. Next levers are more distinct parameters (T6/T9) or per-loop
  adapters (T2 LoRA), not more lr.
- The curves flatten near epoch 30 partly because the cosine schedule takes lr to zero. Training longer might add
  a point or two, but train ≈ test says it won't close a 20-point gap.
- Checkpoints for any follow-up (e.g. the T8 collapse check):
  `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/iter_b0_lr1e-3.pt` and `iter_b0_lr2e-3.pt`.

## Changes outside this folder
None.
