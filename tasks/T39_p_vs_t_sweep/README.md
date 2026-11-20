# T39 · Does more loop time help when memory is fixed? Loop-count sweep at 3 and 4 blocks

From: advisor meeting 2026-11-19
Notion: https://notion.so/41d38c734705811bb475d6a76de9d6c3

## Goal
Dr. Uddin split a model's cost into two axes: **P** = parameters (how many DISTINCT blocks are stored, i.e. memory) and
**T** = time (how many block passes run per image). In a normal ViT they grow together (12 blocks = 12 passes); looping
separates them, because extra loops add T but not P. This task holds P fixed and varies only the loop count r, at two
parameter budgets (3 blocks = "3W", 4 blocks = "4W"), to measure what extra time buys.
- If 2+1 is close to 2+10, looping adds little. If accuracy rises then flattens, the flat point says how many loops
  are worth paying for.
- Comparing the 3W and 4W lines at equal T gives the "exchange rate" between one extra block and extra loops.

**Notation "h+r":** h distinct pretrained blocks B0..B(h-1), then pretrained block B(h) applied r times, no tail,
readout = classifier(final_norm(x)[:, 0]). P = h+1 blocks, T = h+r passes. T11-d is 2+10 (3W, T = 12).

## Runs
Recipe identical to T11-d: pretrained timm `vit_tiny_patch16_224`, CIFAR-100 224 px, AdamW lr 5e-4 all params,
wd 0.05, 1 warmup epoch + cosine over 30 epochs, label smoothing 0.1, batch 128, AMP, seed 0, loss at the last loop
only, nothing frozen. Each run differs from T11-d ONLY in h and r, during training and at test.

h+r is built with the existing flags `--unique-blocks (h+1) --body-loops r` (`--unique-blocks n` keeps pretrained
blocks 0..n-2 distinct and loops block n-1; `--body-loops` is T36's fixed loop count). No new code in `src/`.

| Run | Layout | Flags (`--model unroll` plus) | P | Params | T | Checkpoint |
|---|---|---|---|---|---|---|
| T39-a | 2+2 | `--unique-blocks 3 --body-loops 2` | 3W | 1,539,940 | 4 | `T39-a_2p2_30ep.pt` |
| T39-b | 2+8 | `--unique-blocks 3 --body-loops 8` | 3W | 1,539,940 | 10 | `T39-b_2p8_30ep.pt` |
| T39-c | 2+16 | `--unique-blocks 3 --body-loops 16` | 3W | 1,539,940 | 18 | `T39-c_2p16_30ep.pt` |
| T39-d | 3+1 | `--unique-blocks 4 --body-loops 1` | 4W | 1,984,804 | 4 | `T39-d_3p1_30ep.pt` |
| T39-e | 3+2 | `--unique-blocks 4 --body-loops 2` | 4W | 1,984,804 | 5 | `T39-e_3p2_30ep.pt` |
| T39-f | 3+4 | `--unique-blocks 4 --body-loops 4` | 4W | 1,984,804 | 7 | `T39-f_3p4_30ep.pt` |
| T39-g | 3+8 | `--unique-blocks 4 --body-loops 8` | 4W | 1,984,804 | 11 | `T39-g_3p8_30ep.pt` |

Checkpoints go to `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`. Reused, NOT rerun here: 2+1 and 2+4 (T36-a,
T36-b), 2+10 (T11-d, 74.11 seed 0), 3+9 (T18 n = 4, 77.45; `--unique-blocks 4` with the default fill of 9 loops).

Params follow the formula 205,348 + P × 444,864 (one ViT-Tiny block = 444,864; everything else = patch embedding
147,648 + CLS 192 + position embedding 37,824 + final_norm 384 + classifier 19,300).

    PYTHONPATH=. .venv/bin/python -m pytest tasks/T39_p_vs_t_sweep -q -s                  # layout test
    CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T39_p_vs_t_sweep/run.sh T39-c               # 1 epoch, --subset 512
    CUDA_VISIBLE_DEVICES=N bash tasks/T39_p_vs_t_sweep/run.sh T39-c                       # one full run
    tmux new-session -d -s itertf-T39-queue "bash tasks/T39_p_vs_t_sweep/queue.sh"        # all 7, GPUs 0 and 1

`queue.sh` runs the 7 trainings on GPUs 0 and 1 only (GPUs 2 and 3 are kept for T40), one tmux session per run
(`itertf-T39-x-<date>`), longest first (c, g, b, f, e, a, d). A GPU takes the next run only when its previous run has
ended and it shows under 1 GB in use. Log: `results/queue.log`; exit codes: `results/T39-x.exit`.

**Test** (`test_t39_layouts.py`, 21 checks, random weights, no download): for each of the 7 layouts the param count
equals the formula (3W = 1,539,940, 4W = 1,984,804; printed), the looped block is called exactly r times (forward
hook), and the command-line flags combine as `run.sh` uses them.

**Smoke tests (2026-11-20, 1 epoch, `--subset 512`, batch 128):** T39-c (2+16) on GPU 0: 1,539,940 params, 3.73 GFLOPs,
peak GPU memory about 3.8 GB of 16 GB, so batch 128 fits and no gradient accumulation is needed. T39-d (3+1) on GPU 1:
1,984,804 params, 0.87 GFLOPs. Both exit 0.

**GFLOPs** are forward FLOPs per 224-px image from `src/metrics.py:count_gflops` (torch `FlopCounterMode`), logged by
every training run on CUDA on this server, the same counter and device as T36. Reused values: 2+1 0.67, 2+4 1.28,
2+10 2.51, 3+9 2.51.

## Decision rules (fixed in advance)
- "Looping buys X pt at 3W" with X = top-1(2+10) − top-1(2+1); same for 4W with 3+9 − 3+1.
- "Saturation loop count": for each line, the smallest r whose top-1 is within 0.8 pt of that line's best.
- "Exchange rate at T = 4": top-1(3+1) − top-1(2+2) (same time, one more block). Also report 3+8 (T = 11) vs 2+8
  (T = 10) and say the T values differ by 1.
- If X < 2 pt at 3W, write: "Looping adds little at a fixed parameter budget on this model."
- Seed spread is 0.3 to 0.8 pt (T15): a one-seed difference under 0.8 pt is not an effect.

## Predictions (written before running)
2+2 64 to 69; 2+8 73 to 74.5; 2+16 73 to 74.5 (no gain over 2+10); 3+1 68 to 74; 3+2 72 to 76; 3+4 75 to 77.5; 3+8
76.5 to 78. Saturation at r = 4 to 8 on both lines. One extra block at T = 4 is worth 3 pt or more.

## Results
(pending: runs queued 2026-11-20)
