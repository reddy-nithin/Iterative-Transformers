# T3 · Progressive unrolling

**Goal (Dr. Shua's idea).** Keep the first few pretrained blocks as they are, and fill the remaining slots by
repeating the next block. Does accuracy jump sharply after the first few distinct blocks, or climb steadily?

**From:** advisor meeting 2026-10-15 (Dr. Uddin + Dr. Shua), meeting task T3, due Tue 2026-10-20.
Notion task: https://app.notion.com/p/3fa38c734705811b9164c14ad428037e

**Started:** 2026-10-16

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
- [ ] Full probe sweep (n = 1..12)
- [ ] Full ft sweep (n = 1..12)
- [ ] Plot + table (`plot.py`), takeaways below
- [ ] Sanity check: ft n=1 ≈ 34.8 and ft n=12 ≈ 86.8 (same seed as T1)

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

## Results (`results/probe/`, `results/ft/`)
_Pending._

## Takeaways
_Pending._ What I expect: n=12 is high in both passes and n=1 is low. If n=2-4 is already close to n=12, only the
first few blocks are really different (good news for sharing). A steady climb means every block carries its own
knowledge.

## Changes outside this folder
- `src/models.py`: new `build_unrolled(unique_blocks, ...)`, registered as `"unroll"` in `build()`.
- `src/train.py`: `--model unroll`, `--unique-blocks`, `--head-lr`; new `make_param_groups()`. The cosine schedule now
  uses each param group's own `base_lr`. Without `--head-lr` there is one group, exactly as before.
- Run tests with `PYTHONPATH=. python -m pytest tests tasks -q`.
