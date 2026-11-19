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
(to come)
