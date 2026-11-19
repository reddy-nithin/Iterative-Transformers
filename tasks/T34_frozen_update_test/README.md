# T34 · Do the late loops just repeat the same change? Replace them with copies and check (no training)

From: planning session 2026-11-16 (Fable review)

## Goal
Test causally whether, after loop k, the looped block only adds a repeated update. If it does, loops k+1..10 can be
replaced by a plain vector addition (no block calls) with almost no loss in accuracy.

## Method
Layout 2+10: B0, B1 (distinct pretrained blocks), then B2 applied 10 times, no tail. x_0 = tokens after B1,
x_t = B2(x_{t-1}). Readout at loop t = classifier(final_norm(x_t)[:, 0]) with the checkpoint's own modules. Full
CIFAR-100 test set (10,000 images, 224 px, eval transform), fp32.

Checkpoints: `T11-d_unroll_n3_30ep.pt` (loss at loop 10 only), `T13-c_unroll_readout_every_loop.pt` (loss at every
loop), `T31-a_random_loops.pt` (random loop count per batch).

- **frozen(k)**, k = 3..9: run B2 to loop k, d_k = x_k − x_{k−1}; for t = k+1..10 set x_t = x_{t−1} + d_k without
  calling B2; read out at loop 10. Also "loop-k readout" (normal top-1 at loop k) and "true-10" (normal top-1 at loop 10).
- **Angular speed**, normal run to loop 20: for t = 1..20, mean over images of 1 − cos(final_norm(x_t),
  final_norm(x_{t−1})) for CLS and for the patches (per-token, averaged over tokens 1..196); mean raw CLS L2 norm
  ‖x_t[:, 0]‖; top-1 at every t.
- **Sanity check:** loop-10 top-1 must match the logged number within 0.05 pt (T11-d 74.11, T13-c 72.82, T31-a 71.77),
  otherwise that checkpoint is not analysed. The per-loop top-1 is also compared with the logged depth curves
  (T31 `depth_sweep.json` for T11-d and T31-a, loops 1..20; T13 `exit_analysis.json` for T13-c, loops 1..10).

**Verdict rule (fixed in advance):**
- **Story confirmed** if frozen(k) is within 1 pt of true-10 for some k ≤ 4 on BOTH T13-c and T31-a, AND within 2 pt
  for some k ≤ 6 on T11-d.
- **Story killed** if on T11-d at k = 5 frozen(5) is closer to the loop-5 readout (49.7) than to true-10 (74.11), i.e.
  below 61.9.
- Anything else: "mixed", and say which checkpoint broke the rule.

## Run
    PYTHONPATH=. .venv/bin/python -m pytest tasks/T34_frozen_update_test -q
    CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T34_frozen_update_test/run.sh   # first 500 test images
    CUDA_VISIBLE_DEVICES=N bash tasks/T34_frozen_update_test/run.sh           # full run

Outputs: `results/frozen_update.json`, `results/frozen_update.log`, `results/frozen_vs_k.png`,
`results/angular_speed.png`, `results/cls_norm.png`.

## Predictions (written before running)
(low to medium confidence)
- T13-c k = 4: frozen 72.0 to 73.0 (true-10 72.82).
- T31-a k = 4: frozen 71.0 to 72.0 (true-10 71.77).
- T11-d k = 6: 71 to 74; k = 5: 66 to 72; k = 4: below 60.
- Verdict: "story confirmed". The weakest point is T11-d.

## Results
Run 2026-11-19 on GPU 2 (about 1.5 min for all three checkpoints), commit 49e2e7b. Numbers from
`results/frozen_update.json`.

**Sanity check: passed for all three.** Loop-10 top-1: T11-d 74.12 (logged 74.11), T13-c 72.82 (72.82), T31-a 71.77
(71.77). The per-loop top-1 matches the logged depth curves exactly (max difference 0.00 over loops 1..20 for T11-d
and T31-a, loops 1..10 for T13-c).

**frozen(k) / loop-k readout** (CIFAR-100 test top-1, %):

| Checkpoint | k = 3 | k = 4 | k = 5 | k = 6 | k = 7 | k = 8 | k = 9 | true-10 |
|---|---|---|---|---|---|---|---|---|
| T11-d | 20.59 / 11.30 | 42.54 / 29.61 | 58.32 / 49.67 | 68.89 / 62.56 | 73.08 / 69.60 | 73.92 / 73.12 | 74.07 / 74.25 | 74.12 |
| T13-c | 69.63 / 71.38 | 72.51 / 72.93 | 72.87 / 73.27 | 72.97 / 73.53 | 72.81 / 73.36 | 72.80 / 73.12 | 72.76 / 73.04 | 72.82 |
| T31-a | 68.18 / 68.29 | 70.88 / 70.91 | 71.43 / 71.54 | 71.73 / 71.78 | 71.78 / 72.10 | 71.80 / 72.08 | 71.76 / 71.82 | 71.77 |

**Verdict: story killed.** On T11-d, frozen(5) = 58.32, below 61.9 (closer to the loop-5 readout 49.67 than to
true-10 74.12). The confirm rule also fails on T11-d alone: no k ≤ 6 within 2 pt of true-10 (best is k = 6, 68.89,
5.2 pt short; k = 7 is the first within 2 pt). T13-c (k = 4: 72.51, 0.31 below true-10) and T31-a (k = 4: 70.88,
0.89 below) both pass their part of the confirm rule.

What the table shows besides the verdict:
- On T11-d the frozen copies do recover part of the gap (frozen(k) is 6 to 13 pt above the loop-k readout for
  k = 3..6), so the loop-k update points the right way, but repeating it is not enough until k = 7 or 8. On T11-d the
  late loops still do work that a repeated update cannot replace.
- On T13-c and T31-a, frozen(k) is equal to or slightly below the loop-k readout for every k. These two models are
  already finished by loop 4 to 5, so adding copies of the update neither helps nor hurts much: their late loops are
  close to wasted, but because the model has already settled, not because it keeps adding a useful repeated nudge.

**Angular speed** (`results/angular_speed.png`, mean 1 − cos of the normalised state between loops):
- T13-c and T31-a: large steps at loops 1 to 3 (CLS 0.25 to 0.45 at loop 1), then a fast fall: CLS step 2.2e-3 and
  1.7e-3 at loop 8, and 5.4e-4 and 2.5e-4 at loop 20. Patches behave the same way.
- T11-d: the CLS step stays around 0.05 to 0.06 for loops 1 to 5 and only then falls (1.35e-2 at loop 10, 3.3e-3 at
  loop 20), 6 to 14 times larger than the other two at loops 10 and 20. Its normalised state is still turning
  through loop 10, which fits the frozen result.

**Raw CLS norm** (`results/cls_norm.png`): grows at every loop in all three models and does not level off by loop 20
(T11-d 13.0 → 21.9 at loop 10 → 32.6 at loop 20; T13-c 6.2 → 28.5 → 43.1; T31-a 8.2 → 34.0 → 52.6), as in T27.

Plots: `results/frozen_vs_k.png`, `results/angular_speed.png`, `results/cls_norm.png`.

## Prediction check
| Prediction | Result | |
|---|---|---|
| T13-c k = 4: frozen 72.0 to 73.0 | 72.51 | hit |
| T31-a k = 4: frozen 71.0 to 72.0 | 70.88 | miss (0.12 below the range) |
| T11-d k = 6: 71 to 74 | 68.89 | miss |
| T11-d k = 5: 66 to 72 | 58.32 | miss |
| T11-d k = 4: below 60 | 42.54 | hit |
| Verdict "story confirmed", weakest point T11-d | story killed, by T11-d | verdict wrong; weakest point right |
