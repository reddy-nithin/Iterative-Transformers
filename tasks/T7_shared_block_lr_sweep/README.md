# T7 · Accuracy ceiling of the single shared block (lr sweep)

**Goal.** How far can lr alone push iter b0 (pretrained block 0 reused 12 times, 0.65M params)? Raising lr from
1e-4 to 5e-4 and training 30 epochs took it from 34.8 to 58.1 (T4), still rising. Dr. Uddin doubts it can reach 80.

**From:** advisor meeting 2026-10-20 (Dr. Uddin), meeting task T2, due Tue 2026-10-27.
Notion task: https://notion.so/3ff38c73470581f48d82c181ef38e399

**Started / finished:** 2026-10-22 / –

## Checklist
- [ ] lr 1e-3, 2e-3, 5e-3 at 30 epochs, seed 0, checkpoints saved (meeting T4 may reuse them)
- [ ] Stop raising lr once accuracy drops or a loss goes NaN (built into `sweep.sh`)
- [ ] Combined plot, test acc vs epoch, one line per lr (`results/acc_vs_epoch_by_lr.png`)
- [ ] Results table below, plus one line stating the best lr and its accuracy

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
_Pending._

## Takeaways
_Pending._

## Changes outside this folder
None.
