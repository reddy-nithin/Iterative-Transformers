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

## Results
_Pending._
