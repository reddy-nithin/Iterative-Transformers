# T6 · Progressive unrolling at lr 5e-4, full curve (n = 5..12)

**Goal.** T3 reran progressive unrolling at lr 5e-4 for n = 1..4 only (the "green curve"), where it beat lr 1e-4 by
7 to 13 points. Does lr 5e-4 stay above the lr 1e-4 curve all the way to n = 12, or does it flatten and merge with it?
**Prediction (Dr. Uddin):** it flattens, meets the lr 1e-4 curve late, and does not cross the 86.77 baseline.

**From:** advisor meeting 2026-10-20 (Dr. Uddin), meeting task T1, due Tue 2026-10-27.
Notion task: https://notion.so/3ff38c73470581f3bf00de4e639c2c2f

**Started / finished:** 2026-10-22 / –

## Checklist
- [ ] Runs n = 5..12 at lr 5e-4, 10 epochs, seed 0 (same recipe as the T3 n = 1..4 lr 5e-4 points)
- [ ] Plot: test acc vs n and vs params, both lrs, baseline line (`results/acc_vs_n_and_params.png`)
- [ ] Results table below, plus one line on where (which n) the two curves meet

## Setup
n = number of distinct blocks: pretrained blocks 0..n-2 keep their own weights and block n-1 fills the remaining
12 - n + 1 slots (one shared object). n = 12 is the normal ViT-Tiny, so **n = 12 at lr 5e-4 is the full baseline
fine-tuned at the higher lr**, not the 86.77 run (which is lr 1e-4). Recipe: AdamW wd 0.05, 1 warm-up epoch + cosine,
label smoothing 0.1, batch 128, AMP, 10 epochs, seed 0.

No new training code: the T3 sweep script is reused with `--lr 5e-4` (a later `--lr` overrides the pass default).
lr 1e-4 n = 1..12 and lr 5e-4 n = 1..4 are read from `tasks/T3_progressive_unrolling/results/`.

## Commands
```bash
# GPU check (nvidia-smi is broken on 2026-10-22, driver/library mismatch); expect ~0.2 GB used when idle
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c "import torch;f,t=torch.cuda.mem_get_info();print((t-f)/2**30,'GB used')"
tmux new -s itertf-t6-unroll-1022
CUDA_VISIBLE_DEVICES=0 OUT_ROOT=tasks/T6_unroll_lr5e-4_full_curve/results EXTRA_ARGS="--lr 5e-4" \
  bash tasks/T3_progressive_unrolling/sweep.sh ft 5 6 7 8 9 10 11 12      # ~9 min per n, ~75 min total
.venv/bin/python tasks/T6_unroll_lr5e-4_full_curve/plot.py              # table -> results/summary.md
```
Smoke test that was run: `OUT_ROOT=<scratch> EXTRA_ARGS="--lr 5e-4 --subset 512 --epochs 1" ... sweep.sh ft 5 12`
(params n=5 2.43M, n=12 5.54M, both finished).

## Results (`results/ft/`, one seed: gaps under ~2 points are ties)
_Pending._

## Takeaways
_Pending._

## Changes outside this folder
None.
