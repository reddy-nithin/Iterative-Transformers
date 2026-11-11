# T29 · Huginn-style input injection: concat + linear adapter on T11-d

**From:** advisor meeting 2026-11-05.
Notion task: https://notion.so/40f38c734705813da8c0f0365ecf6efc

**Goal.** Huginn re-injects the processed input e into every loop through a linear adapter on [state; e]. Our T9 tried
plain addition (`--inject`): the state grew by e every loop and separation jumped to ~6. Test the paper's form on the
better T11-d layout.

## Setup

- Layout and reference: **T11-d**, final-only loss, T11-d config. Reference T11-d 73.79 (3-seed spread 0.77; seed 0:
  74.11), checkpoint `T11-d_unroll_n3_30ep.pt`. T9's plain-add run for the norm comparison: `T9-c_hbt_inject.pt`.
- **One change:** e = x after B1 (all 197 tokens, kept fixed, no detach). Before each B2 application:
  x_in = A([x_{t−1}; e]), A = Linear(384 → 192) applied per token, then B2 as usual. A starts at weight [I, 0] and
  bias 0, so at initialisation x_in = x_{t−1} and the model equals T11-d at step 0. New parameters: 73,920.
- Code: `src/concat_inject.py` (`ConcatInjectBody`), new flag `--concat-inject` for `--model unroll`. The existing
  `--inject` (hbt, T9 plain addition) is unchanged, so T9 and T11 still reproduce.

| Run | Flags | Checkpoint |
|---|---|---|
| T11-d (reference) | `--model unroll --unique-blocks 3` | `T11-d_unroll_n3_30ep.pt` |
| T29-a | + `--concat-inject` | `T29-a_concat_inject.pt` |

Shared recipe: pretrained init, lr 5e-4, wd 0.05, 1 warmup epoch + cosine, label smoothing 0.1, batch 128, AMP,
30 epochs, seed 0. Checkpoints in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`.

## Checklist

- [x] README + predictions committed before the first run
- [x] code + tests (`test_t29_concat_inject.py`) pass
- [x] smoke run (`SMOKE=1`)
- [x] full run T29-a (tmux), per-loop readout, separation per loop
- [x] mean ‖x_t‖ per loop (T29-a, T11-d, T9 plain add), ‖W_state‖ vs ‖W_e‖
- [x] results row + reference, norm and separation tables, answer line, prediction check, inbox entry

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T29_concat_input_injection -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T29_concat_input_injection/run.sh
tmux new -s t29-train 'CUDA_VISIBLE_DEVICES=N bash tasks/T29_concat_input_injection/run.sh'
# after training: norms and adapter split (GPU), then tables and plots
tmux new -s t29-norms 'CUDA_VISIBLE_DEVICES=N PYTHONPATH=. .venv/bin/python tasks/T29_concat_input_injection/state_norms.py'
PYTHONPATH=. .venv/bin/python tasks/T29_concat_input_injection/plot.py
```

## Predictions (written before running)

Copied verbatim from the task spec:

top-1 within ±1.5 of T11-d; ‖x_t‖ grows less than in T9's plain-add run; separation does not jump; ‖W_e‖ ends small
(the patches already carry e). Low confidence.

## Results

Run 2026-11-11 on GPU 2, at the same time as T28-a and T30-a (one GPU each), seed 0. Sanity checks (full test set):
T29-a 73.37 (logged 73.32), T11-d 74.12 (74.11), T9-c 44.52 (44.53).

### Accuracy and cost (CIFAR-100 test)

| model | params | GFLOPs / image | throughput (img/s) | final top-1 |
|---|---|---|---|---|
| T11-d (reference) | 1,539,940 | 2.51 | 5,788 | 74.11 (3 seeds: 73.79, spread 0.77) |
| **T29-a** concat + adapter | 1,613,860 (+73,920) | 2.80 | 4,807 | **73.32** (−0.47 vs the 3-seed mean) |

−0.47 is inside the seed spread (0.77) and under the 0.8 pt bar: **no effect on accuracy**. The adapter costs 12%
more FLOPs (it runs on all 197 tokens in every loop) and 17% less throughput.

### Per loop (test)

| model | | x_0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T11-d | readout top-1 | | 1.22 | 3.79 | 11.30 | 29.61 | 49.67 | 62.56 | 69.60 | 73.12 | 74.25 | 74.12 |
| T29-a | readout top-1 | | 1.34 | 1.82 | 5.23 | 18.21 | 40.15 | 57.77 | 66.98 | 71.07 | 72.99 | 73.37 |
| T11-d | mean ‖x_t‖ | 12.9 | 12.1 | 12.9 | 13.9 | 14.9 | 16.1 | 17.1 | 18.1 | 19.0 | 20.0 | 21.0 |
| T29-a | mean ‖x_t‖ | 9.9 | 10.4 | 12.2 | 14.2 | 16.0 | 17.8 | 19.3 | 20.6 | 21.7 | 22.7 | 23.6 |
| T9-c (plain add) | mean ‖x_t‖ | 11.6 | 21.3 | 29.3 | 37.7 | 46.2 | 54.8 | 63.5 | 72.2 | 81.0 | 89.8 | 98.6 |
| T11-d | separation | | 2.278 | 2.167 | 1.733 | 1.346 | 1.126 | 1.062 | 1.061 | 1.084 | 1.116 | 1.151 |
| T29-a | separation | | 2.432 | 2.156 | 1.778 | 1.419 | 1.210 | 1.108 | 1.050 | 1.011 | 0.980 | 0.956 |
| T9-c (plain add) | separation | | 5.866 | 4.834 | 4.388 | 4.128 | 3.974 | 3.879 | 3.818 | 3.779 | 3.754 | 3.737 |

‖x_t‖ = L2 norm of a token, averaged over all 197 tokens and the 10,000 test images (x_0 = output of the head blocks).
Separation = within-class / between-class spread of CLS (lower = tighter; T5 code, 100 test images per class).
T9-c is T9's plain-add run (head 1 / body x10 / tail 1, trained from scratch), so its numbers show the T9 problem, not
a like-for-like model.

- **Norm:** T29-a's state grows by 13.8 over 10 loops (x2.4), a bit faster than T11-d's (+8.0, x1.6) and far from
  T9's plain addition (+87, x8.5), which adds e in every loop.
- **Separation:** no jump. T29-a's classes get tighter at every loop and end tighter than T11-d's (0.956 vs 1.151
  test, 0.845 vs 1.022 train), where T11-d loosens again after loop 7. One seed, so a direction to check, not a
  measured effect.
- **Readout:** T29-a reaches its answer later (loop 5: 40.2 vs 49.7) and is still rising at loop 10.

### Adapter weights (A = Linear(384 → 192), weight [W_state, W_e])

| | ‖W_state‖ | ‖W_e‖ | ‖bias‖ |
|---|---|---|---|
| initialisation | 13.86 (= √192, identity) | 0 | 0 |
| after training | 12.65 | **2.90** | 0.15 |

The model does use e: ‖W_e‖ grew from 0 to 2.90, 23% of ‖W_state‖. The state path still dominates. ‖W_state‖ shrank a
little, which fits the state not growing much faster than in T11-d.

Plots: `results/acc_per_loop.png`, `results/state_norm_per_loop.png`, `results/separation_per_loop.png`. Made by
`run.sh`, `state_norms.py` (norms, adapter split, run on GPU in tmux `t29-norms`) and `plot.py`.

**Does re-injecting the input help when it can't blow up the state?** It does no harm and no clear good: top-1 73.32,
inside the seed spread of T11-d (−0.47). The adapter keeps the state in check (×2.4 over 10 loops, vs ×8.5 for T9's
plain addition), learns a modest weight on e (‖W_e‖ 2.9 vs ‖W_state‖ 12.6), and the classes end tighter than in
T11-d. It costs 12% more FLOPs for no accuracy gain on this task.

### Prediction check

| Prediction | Result | Verdict |
|---|---|---|
| top-1 within ±1.5 of T11-d | −0.47 | right |
| ‖x_t‖ grows less than in T9's plain-add run | +13.8 vs +87 | right |
| separation does not jump | falls every loop, ends at 0.956 (T11-d 1.151) | right |
| ‖W_e‖ ends small | 2.90 vs ‖W_state‖ 12.65 (23%); clearly used, not near 0 | partly right |
