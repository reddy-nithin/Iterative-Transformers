# T15 · Seed repeats for the headline runs

**Goal.** Turn single-seed numbers into mean ± spread so Friday's comparisons are real.

**From:** planning session 2026-11-02 (Nithin), after the T11 to T14 results. Due Thu 2026-11-05.
Notion task: https://notion.so/40c38c73470581f7b84fcb2340d3b90c

## Runs
The existing configs of T11-c, T11-d and T13-d, unchanged except `--seed` (seed 0 already exists). Pretrained init,
shared recipe (lr 5e-4, 30 epochs, otherwise `src.train` defaults). `set_seed(args.seed)` in `src/train.py` runs before
the model and the loaders are built, so the seed sets both the new 100-class head's init and the training data order.

| Run | Config | Flags | Seed | Checkpoint |
|---|---|---|---|---|
| T15-a | T11-c | `--model hbt --hbt-pretrained --head-idx 0 --body-idx 1 --tail-idx 11` | 1 | `T15-a_t11c_seed1.pt` |
| T15-b | T11-c | same | 2 | `T15-b_t11c_seed2.pt` |
| T15-c | T11-d | `--model unroll --unique-blocks 3` | 1 | `T15-c_t11d_seed1.pt` |
| T15-d | T11-d | same | 2 | `T15-d_t11d_seed2.pt` |
| T15-e | T13-d | `--model unroll --unique-blocks 3 --readout-loss loss_margin` | 1 | `T15-e_t13d_seed1.pt` |
| T15-f | T13-d | same | 2 | `T15-f_t13d_seed2.pt` |

Checkpoints in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`. T15-e/f also get the fixed-depth curve (top-1 at
loop 1..10) from T13's `exit_analysis.py`.

Run (repo root, tmux, one GPU): `CUDA_VISIBLE_DEVICES=N bash tasks/T15_seed_repeats/sweep.sh T15-a`

## Predictions (written before running)
Spread ≤ 1 pt for each config. T11-d stays above T11-c on every seed. T13-d mean within 1.5 pts of T11-d mean.

## Results (CIFAR-100 test top-1, 30 epochs, lr 5e-4; seed 0 = the existing run; `results/table.md`)

| config | seed 0 | seed 1 | seed 2 | mean | spread (max − min) |
|---|---|---|---|---|---|
| T11-c (B0 → B1 x10 → B11) | 68.59 | 68.83 (T15-a) | 68.49 (T15-b) | 68.64 | 0.34 |
| T11-d (B0, B1, B2 x10) | 74.11 | 73.34 (T15-c) | 73.91 (T15-d) | 73.79 | 0.77 |
| T13-d (T11-d + readout every loop + margin) | 73.17 | 72.86 (T15-e) | 72.49 (T15-f) | 72.84 | 0.68 |

T13-d fixed-depth curve (readout at loop t for every image, fp32, full test set):

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 45.08 | 63.94 | 70.15 | 72.51 | 73.30 | 73.58 | 73.54 | 73.42 | 73.33 | 73.19 |
| 1 | 43.88 | 63.51 | 70.02 | 72.30 | 72.93 | 73.05 | 73.04 | 72.95 | 73.04 | 72.86 |
| 2 | 44.70 | 64.22 | 69.83 | 71.73 | 72.32 | 72.64 | 72.65 | 72.64 | 72.50 | 72.51 |

The shape repeats on every seed: within 1 pt of loop 10 by loop 4, peak at loop 6 or 7, then a slight decline
(0.3 to 0.4 pt). (Loop-10 fp32 numbers can differ from the logged AMP top-1 by 0.02, e.g. 72.51 vs 72.49.)

**Which last-round gaps are larger than the seed spread?**
- T11-d vs T11-c: +5.15 on average (+4.5 to +5.5 per seed), 7 to 15 times the spread: **real**.
- T13-d vs T11-d: −0.95 on average, negative on all three seeds (−0.48 to −1.42), a bit larger than either spread
  (0.68 / 0.77): **real but small** (the per-loop readout + margin costs about 1 pt on this layout).
- T13-c vs T11-d (−1.29, T13-c has one seed): larger than T11-d's spread, so probably real.
- T14-a vs T13-d (−0.28) and T13-a vs T11-c (+0.26): inside the spread (0.68 / 0.34): **noise**, i.e. the T14
  coverage mask and the T11-c-layout readout loss changed nothing measurable.

One line: only the layout gap (T11-d over T11-c, ~5 pts) and the ~1-pt cost of the per-loop readout on the T11-d
layout survive the seeds; the 0.3-pt gaps from last round are noise.

### Prediction check
- Spread ≤ 1 pt for each config: **right** (0.34, 0.77, 0.68).
- T11-d above T11-c on every seed: **right** (by 4.5 to 5.5).
- T13-d mean within 1.5 pts of T11-d mean: **right** (−0.95).
