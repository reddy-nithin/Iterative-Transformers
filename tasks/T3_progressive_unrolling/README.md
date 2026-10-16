# T3 · Progressive unrolling

**Goal (Dr. Shua's idea).** Keep the first few pretrained blocks as they are, and fill the remaining slots by
repeating the next block. Does accuracy jump sharply after the first few distinct blocks, or climb steadily?

**From:** advisor meeting 2026-10-15 (Dr. Uddin + Dr. Shua), meeting task T3, due Tue 2026-10-20.
Notion task: https://app.notion.com/p/3fa38c734705811b9164c14ad428037e

**Started / finished:** 2026-10-16 / 2026-10-16

## Definition
n = number of distinct blocks (1..12). Slots 1..n-1 hold pretrained blocks 0..n-2, and pretrained block n-1 fills
the remaining 13-n slots (one shared object, like T1).

```
n = 1   [0][0][0][0][0][0][0][0][0][0][0][0]   = T1's "block 0 x12"     0.650 M params
n = 3   [0][1][2][2][2][2][2][2][2][2][2][2]                            1.540 M
n = 12  [0][1][2][3][4][5][6][7][8][9][10][11] = normal ViT-Tiny        5.544 M
```
Params = 0.650 M + (n-1) x 0.445 M.

**Two passes**, n = 1..12 each, 10 epochs, seed 0, otherwise the T1 recipe:
- **probe**: body lr 1e-6, head lr 1e-3. The blocks stay almost exactly as pretrained, and only the new
  100-class head really learns. This measures how good the unrolled pretrained blocks are *as they are*.
- **ft**: everything at lr 1e-4. Blocks are free to re-adapt. Comparable with T1 (iter b0 34.8, baseline 86.8).
The gap between the two curves shows how much training makes up for the repetition.

## Status
- [x] `build_unrolled` model + `--model unroll --unique-blocks n` (shared code, see below)
- [x] `--head-lr` flag (separate lr for the classifier head)
- [x] Tests: `test_t3_unroll.py` (14 tests: slot layout, sharing, param counts, bad n, lr groups)
- [x] Smoke test 2026-10-16: n=1 and n=12, both passes, 512 images, 1 epoch. Params 650,212 / 5,543,716 as
      expected. Extra check: probe n=12 on 5,000 images for 2 epochs reached 52.1%, so the head learns with the
      body near-frozen.
- [x] Full probe sweep (n = 1..12), GPU 0, 2026-10-16
- [x] Full ft sweep (n = 1..12), GPU 2, 2026-10-16
- [x] Plot + table (`plot.py`), takeaways below
- [x] Sanity check: ft n=1 = 34.76 and ft n=12 = 86.77, identical to T1's block-0 and baseline runs

## Commands
Run from the repo root. Each run is ~9 min on one A4000; 12 runs per pass ≈ 1.8 h.
```
nvidia-smi                                   # pick two idle GPUs (only your own python lines matter)
cd ~/itertf
tmux new -s t3-probe
CUDA_VISIBLE_DEVICES=0 bash tasks/T3_progressive_unrolling/sweep.sh probe
#   detach: Ctrl+b then d
tmux new -s t3-ft
CUDA_VISIBLE_DEVICES=2 bash tasks/T3_progressive_unrolling/sweep.sh ft
#   detach: Ctrl+b then d
```
- Watch: `tmux attach -t t3-probe`. Quick progress: `tail -n 3 tasks/T3_progressive_unrolling/results/probe/*.log`
- If a sweep dies, run the same command again: finished runs are skipped.
- Only one GPU free? Run both in one session: `CUDA_VISIBLE_DEVICES=0 bash ... probe && CUDA_VISIBLE_DEVICES=0 bash ... ft`
- When both finish: `.venv/bin/python tasks/T3_progressive_unrolling/plot.py` → `results/acc_vs_n.png`,
  `results/summary.md`. Then `tmux kill-session -t t3-probe` (and `t3-ft`).

Smoke test that was run: `CUDA_VISIBLE_DEVICES=0 OUT_ROOT=<scratch> EXTRA_ARGS="--subset 512 --epochs 1" bash tasks/T3_progressive_unrolling/sweep.sh probe 1 12`
(and the same with `ft`).

## Results (`results/probe/`, `results/ft/`, one seed: gaps under ~2 points are ties)
| n | Params (M) | % of baseline params | Probe acc (%) | Fine-tune acc (%) |
|---|---|---|---|---|
| 1 | 0.650 | 12% | 17.49 | 34.76 |
| 2 | 1.095 | 20% | 18.01 | 44.73 |
| 3 | 1.540 | 28% | 28.54 | 56.26 |
| 4 | 1.985 | 36% | 45.19 | 67.21 |
| 5 | 2.430 | 44% | 53.53 | 72.45 |
| 6 | 2.875 | 52% | 59.59 | 76.37 |
| 7 | 3.319 | 60% | 65.91 | 80.05 |
| 8 | 3.764 | 68% | 72.12 | 82.55 |
| 9 | 4.209 | 76% | 76.70 | 84.43 |
| 10 | 4.654 | 84% | 78.69 | 85.37 |
| 11 | 5.099 | 92% | 82.03 | 85.95 |
| 12 | 5.544 | 100% | 83.67 | **86.77** |

Plot: `results/acc_vs_n.png`. Table: `results/summary.md`.

## Takeaways
- **No sharp jump; a steady climb with diminishing returns.** Fine-tune gains ~11 points per extra block for
  n=1..4 (34.8 → 67.2), then ~4 per block for n=5..8, then under 1 per block for n=10..12. So every block carries
  its own knowledge, but the early ones matter most.
- **Good trade-off zone around n=8-9:** n=9 keeps 76% of the params and loses 2.3 points (84.4 vs 86.8); n=8 keeps
  68% and loses 4.2. Compute (FLOPs, speed) does not change, since all 12 slots still run.
- **Probe (blocks almost frozen):** n=1 and n=2 are the same (17.5 vs 18.0), then the big jump is n=3 → n=4
  (+16.7). Repeating an early block as is gives poor features; training has to fix them.
- **Training makes up for repetition most when n is small:** the fine-tune minus probe gap is ~27 points at n=2-3
  and shrinks to 3 points at n=12.
- **Caveat:** fine-tune used lr 1e-4. T2 showed the shared block gains +11 points at lr 5e-4, so the low-n
  fine-tune numbers are probably underestimates. A rerun of n=1..4 at 5e-4 would tell.

## Changes outside this folder
- `src/models.py`: new `build_unrolled(unique_blocks, ...)`, registered as `"unroll"` in `build()`.
- `src/train.py`: `--model unroll`, `--unique-blocks`, `--head-lr`; new `make_param_groups()`. The cosine schedule now
  uses each param group's own `base_lr`. Without `--head-lr` there is one group, exactly as before.
- Run tests with `PYTHONPATH=. python -m pytest tests tasks -q`.
