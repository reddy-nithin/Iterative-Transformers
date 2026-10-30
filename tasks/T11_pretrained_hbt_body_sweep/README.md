# T11 · Pretrained head + body x10 + tail: which block is the body?

**Goal.** First pretrained head / body / tail runs (from now on everything starts from pretrained weights). T9 tried this
layout only from scratch (44.87). Here we test which pretrained block makes the best looped body, and whether a tail
is worth its block compared with spending it on a bigger head (T11-d, progressive unrolling n = 3).

**From:** advisor meeting 2026-10-27 (Dr. Uddin), due Tue 2026-11-03.
Notion task: https://notion.so/40938c734705817e8a8adc7980fed0cd

**Started / finished:** 2026-10-30 / 2026-10-30

## Checklist
- [x] Pretrained-init option for `--model hbt` (`--hbt-pretrained --head-idx H --body-idx B --tail-idx T`, default off so
      T9 is unchanged) + tests (`test_t11_pretrained_hbt.py`)
- [x] Smoke test of all four configs (300 images, 1 epoch): all four 1,539,940 params
- [x] T11-a to T11-d, 30 epochs each, one GPU each, in parallel
- [x] Class separation per loop (T5 `cls_svd.py`, 4 classes, train + test) for every run
- [x] Plots `results/loss_curves.png`, `results/acc_vs_epoch.png`, `results/separation_vs_loop.png`
- [x] Results table, winner line, H-B-T vs T11-d line

## Runs
Shared recipe: pretrained timm `vit_tiny_patch16_224`, CIFAR-100 at 224 px, AdamW lr 5e-4 for all parameters,
wd 0.05, 1 warm-up epoch + cosine, label smoothing 0.1, batch 128, AMP, 30 epochs, seed 0, nothing frozen.
Head, body and tail are `copy.deepcopy` of the pretrained blocks, so they are three separate modules even when the
indices repeat (T11-a: head and body both start from block 0 and train independently).

| Run | Head | Body x10 | Tail | Checkpoint |
|---|---|---|---|---|
| T11-a | blocks[0] | blocks[0] | blocks[11] | `T11-a_hbt_body_b0.pt` |
| T11-b | blocks[0] | blocks[6] | blocks[11] | `T11-b_hbt_body_b6.pt` |
| T11-c | blocks[0] | blocks[1] | blocks[11] | `T11-c_hbt_body_b1.pt` |
| T11-d | progressive unrolling n = 3 (blocks 0, 1 distinct, block 2 x10, no tail), existing code, only epochs 10 → 30 | | | `T11-d_unroll_n3_30ep.pt` |

Why these: T11-a = our best looping block (b0 x12 58.06 beat b6 x12 54.51). T11-b = mid-network block as body
(Gemini A8). T11-c = block 1 naturally follows block 0, so the head → body seam is smallest. T11-d = the same
parameter count spent on a 2-block head instead of a tail (67.12 at 10 epochs) = the reference to beat.

## Predictions (written before running)
T11-d reaches ~69 to 72. T11-b (block 6 body) is lowest of T11-a to T11-c.
T11-a and T11-c within 2 pts of each other. Unsure whether any looped body gets separation below 1.

## Commands
```bash
# smoke test (any idle GPU, ~1 min each)
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T11_pretrained_hbt_body_sweep/sweep.sh T11-a T11-b T11-c T11-d
# real runs: one tmux session and one GPU per run
tmux new -d -s itertf-T11-a "CUDA_VISIBLE_DEVICES=0 bash tasks/T11_pretrained_hbt_body_sweep/sweep.sh T11-a"
```
Checkpoints: `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/T11-<run>_<slug>.pt`.
Tests: `PYTHONPATH=. .venv/bin/python -m pytest tests tasks -q`.

## Results
All four runs: 30 epochs, GPUs 0-3 in parallel (2026-10-30, ~38 min each, ~75 s/epoch while sharing the CPU), seed 0,
no change to the shared recipe. "Train acc" is the running accuracy over the augmented batches of the last epoch.
Separation = tr(within) / tr(between) of the CLS token, 4 classes (apple, bicycle, fox, cloud), 100 test images each,
after each block call. "Last body loop" = call 11 for H-B-T (before the tail) and call 12 for T11-d (it has no tail).

| Run | Layout | Params (M) | Final top-1 | Best top-1 | Train acc | Sep. last body loop (test) | Sep. last block (test) | Min sep. test (call) |
|---|---|---|---|---|---|---|---|---|
| T11-a | head b0, body b0 x10, tail b11 | 1.540 | 62.05 | 62.05 | 62.19 | 1.79 | 1.80 | 1.52 (6) |
| T11-b | head b0, body b6 x10, tail b11 | 1.540 | 63.47 | 63.62 | 66.79 | 2.28 | 2.04 | 1.56 (2) |
| **T11-c** | head b0, body b1 x10, tail b11 | 1.540 | **68.59** | 68.63 | 68.87 | 1.61 | 1.38 | 1.38 (12) |
| T11-d | unrolling n=3: b0, b1, b2 x10, no tail | 1.540 | **74.11** | 74.13 | 73.02 | 1.15 | 1.15 | 1.06 (9) |
| _ref_ | baseline, 12 distinct blocks, fine-tuned (T5-a) | 5.544 | 86.77 | | | | 0.55 | |
| _ref_ | iter b0, block 0 x12, 30 ep (T4 / T5-b) | 0.650 | 58.06 | | | | 1.60 | 1.45 (8) |
| _ref_ | per-loop LoRA r8 on b0 (T4) | 0.945 | 61.33 | | | | | |
| _ref_ | head + body x10 + tail from scratch (T9-a) | 1.540 | 44.87 | | | | | |
| _ref_ | unrolling n=3, lr 5e-4, 10 ep (T3) | 1.540 | 67.12 | | | | | |

Train-set separation (same classes, 500 train images each) gives the same order: T11-d reaches 1.004 at call 10, the
closest any looped model has come to 1, but none goes below it.

Plots: `results/loss_curves.png`, `results/acc_vs_epoch.png`, `results/separation_vs_loop.png`. Raw: `results/*.json`,
`results/*.log`, table in `results/summary.md`.
Checkpoints: `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/T11-{a_hbt_body_b0, b_hbt_body_b6, c_hbt_body_b1, d_unroll_n3_30ep}.pt`.

Winner rule: T11 winner = highest final test top-1 among T11-a, T11-b, T11-c; if the top two are within 2 pts (one
seed = a tie), pick the one with lower final separation; if still tied, T11-a.

- **T11 winner: T11-c** (head b0, body b1 x10, tail b11), 68.59. The runner-up T11-b (63.47) is 5.1 pts behind, so the
  separation tie-break is not needed.
- **Best H-B-T vs T11-d: loses by ≥2** (68.59 vs 74.11, −5.5 pts).

**Against the predictions:**
- T11-d ~69 to 72: wrong, it went higher, 74.11 (+7.0 over its 10-epoch run).
- T11-b lowest of T11-a to T11-c: wrong. T11-a (block 0 body) is lowest at 62.05, T11-b 63.47.
- T11-a and T11-c within 2 pts: wrong, 6.5 pts apart.
- Separation below 1 for any looped body: no. Best is T11-d, 1.06 test / 1.00 train.

**Takeaways:**
- Pretrained init is worth 17-24 pts for this layout (44.87 from scratch → 62-69). All three H-B-T runs beat iter b0
  (58.06) and the per-loop LoRA model (61.33).
- The block that loops best alone (b0, 58.06 as b0 x12) is the worst body here. Block 1, which normally follows block 0,
  is best, which fits the "smallest head → body seam" reason. One seed, so the a-vs-b order (1.4 pts) is a tie.
- Spending the third block on a second head block (T11-d) beats spending it on a tail by 5.5 pts at the same params and
  FLOPs.
- Separation tells the same story as T9 for the H-B-T runs: the body does not steadily tighten classes. T11-a is flat
  around 1.5-1.8 and drifts up in the late loops, and T11-b gets worse every loop (1.56 → 2.28), with the tail pulling
  it back to 2.04. T11-c is the exception: its body loops tighten from 2.29 (call 5) to 1.61, then the tail takes it to
  1.38. In T11-d the repeated block 2 does most of the tightening itself (2.17 → 1.06 over calls 4-9).
- Train and test accuracy are close for T11-a, T11-c and T11-d (no overfitting; capacity-limited, as with iter b0).
  T11-b shows a small gap (66.8 train vs 63.5 test).
- Loss curves are still falling slowly at epoch 30 for all four runs.

## Why T13 now uses both layouts
These results changed the T13 plan (spec box "Changed after the T11 results (2026-10-30 evening): what and why" in the
next-meeting tasks). The loop is not always lazy. With a tail after it (T11-a/b/c) the tail does most of the class
tightening, but with nothing after it (T11-d) the repeated block 2 tightens classes itself (2.3 → 1.06). Working guess:
**the tail lets the body slack off.** This isn't proven, because T11-c and T11-d also differ in which block loops (b1 vs b2).
So T13's question "if every loop must give an answer, does the loop start working?" means different things per layout,
and T13 runs on both. On the T11-c layout (has a tail): does a per-loop readout make the body tighten classes and
close part of the 5.5-pt gap to T11-d? On the T11-d layout (no tail, loop already works): does early exit finally beat
fixed depth? T11-d's per-loop readout is final norm + classifier on x_t, since it has no tail. Caveat for every
write-up: unrolling n = 3 went 67.12 → 74.11 from 10 → 30 epochs, so the 10-epoch T6 size curve understates each size.
