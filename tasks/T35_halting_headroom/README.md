# T35 · How much could early stopping ever save? Best-case test (no training)

From: planning session 2026-11-16 (Fable review)

## Goal
Measure the most that ANY per-image halting rule could gain on our model, and how well simple signals predict "this
loop's answer is already right", before spending GPU time on training a halting gate. ("Halting" = stopping the loop
early for each image separately.)

## Method
Checkpoint `T31-a_random_loops.pt` only (it works at any depth), loops t = 1..10, full CIFAR-100 test set, fp32.
Readout at loop t = classifier(final_norm(x_t)[:, 0]).

1. **Store per-loop outputs** (logits fp16, z_t = final_norm(x_t)[:, 0] fp16, raw CLS norm, label) to
   `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/T35_T31-a_per_loop_outputs.pt`. Sanity: top-1 per loop from the
   stored logits must match T31-a's logged values within 0.05 pt (exact values from T31 `depth_sweep.json`). Every
   later step reads the stored file.
2. **Per-loop table:** top-1, mean cross-entropy in nats without label smoothing, drop in mean CE from t−1 to t.
3. **Settling loop:** smallest t such that the prediction at every loop from t to 10 equals the loop-10 prediction.
   Histogram, mean, standard deviation (population), mean for images right / wrong at loop 10.
4. **Oracle:** e = earliest loop whose prediction is correct (none if never). Greedy curve: every image starts at
   loop 1 (wrong anyway if e = none, correct for free if e = 1); rescuable images are added in ascending e, each
   costing e − 1 extra loops; after each addition record (mean loops over all 10,000 images, accuracy). Accuracy at
   mean loops 2, 3, 4 (last point within budget); mean loops to reach 68.3 (fixed depth 3) and 70.9 (fixed depth 4);
   **headroom at 3 = oracle accuracy at mean loops 3 − 68.3**.
5. **Signal AUROCs** (`sklearn.metrics.roc_auc_score`), target "loop-t prediction is correct", t = 2..10 and pooled
   over t = 2..10: margin (top-1 − top-2 logit), max softmax, −KL(softmax_t ‖ softmax_{t−1}),
   −(1 − cos(z_t, z_{t−1})), log raw CLS norm. Values reported as computed, even below 0.5.

**Decision rule (fixed in advance):** if headroom at 3 is under 3 pt: "Halting: no room to train a gate on this
model; keep as an explained negative." If it is 3 pt or more: "Halting: worth a gate (next: fit a small gate on the
stored outputs)." No gate is built or trained in this task.

## Run
    PYTHONPATH=. .venv/bin/python -m pytest tasks/T35_halting_headroom -q
    CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T35_halting_headroom/run.sh   # first 500 test images
    CUDA_VISIBLE_DEVICES=N bash tasks/T35_halting_headroom/run.sh           # full run

Outputs: `results/halting_headroom.json`, `results/halting.log`, `results/oracle_curve.png`,
`results/settling_hist.png`, `results/auroc_per_loop.png`; stored outputs on `/storage` (above).
Needs scikit-learn (added to `requirements.txt` for this task).

## Predictions (written before running)
(low to medium confidence)
- The oracle reaches 70.9 at 2.2 to 2.8 mean loops.
- Settling loop standard deviation 1.0 to 1.5 loops.
- Pooled AUROC: margin 0.80 to 0.88; angular step 0.55 to 0.70; log CLS norm 0.55 to 0.65.

## Results
Run 2026-11-19 on GPU 2 (about 20 s), commit 49e2e7b. Numbers from `results/halting_headroom.json`; stored outputs
in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/T35_T31-a_per_loop_outputs.pt`.

**Sanity check: passed.** Top-1 per loop from the stored (fp16) logits vs the logged T31-a values: largest difference
0.03 pt (loop 2: 57.99 vs 57.96). Loop 10: 71.76 vs 71.77.

**Per-loop table**

| Loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Top-1 (%) | 23.40 | 57.99 | 68.30 | 70.93 | 71.54 | 71.76 | 72.11 | 72.08 | 71.81 | 71.76 |
| Mean CE (nats) | 3.434 | 1.702 | 1.188 | 1.067 | 1.033 | 1.023 | 1.022 | 1.024 | 1.029 | 1.034 |
| CE drop from t−1 | n/a | 1.732 | 0.514 | 0.121 | 0.035 | 0.010 | 0.001 | −0.003 | −0.005 | −0.006 |

Mean CE stops falling after loop 7 and rises slightly at loops 8 to 10.

**Settling loop** (smallest t from which the prediction never changes again)

| Settling loop | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Images | 2190 | 4045 | 1716 | 662 | 440 | 281 | 232 | 161 | 144 | 129 |

Mean 2.75, standard deviation 1.91. Images right at loop 10 settle at 2.26 on average, images wrong at loop 10 at 3.97.

**Oracle** (a perfect halting rule that knows the label)

| | Value |
|---|---|
| Oracle accuracy (right at some loop 1..10) | 79.55 |
| Oracle accuracy at mean loops ≤ 2 / ≤ 3 / ≤ 4 | 79.55 / 79.55 / 79.55 |
| Mean loops to reach 68.3 (fixed depth 3) | 1.52 |
| Mean loops to reach 70.9 (fixed depth 4) | 1.58 |
| Mean loops for the full oracle (all 7955 rescuable images) | 1.90 |
| **Headroom at 3** (79.55 − 68.3) | **+11.25 pt** |

First correct loop e: 2340 images at loop 1, 3744 at 2, 1156 at 3, 353 at 4, 362 at loops 5 to 10, and 2045 never.

**Decision: Halting: worth a gate (next: fit a small gate on the stored outputs).**

Reading this number: the oracle beats the best fixed depth (72.11) by 7.4 pt, so part of its accuracy comes from
images that are right at some early loop and wrong at loop 10 (the 2045 never-right images cap it at 79.55). A real
gate has to find those images without the label. The AUROCs below show how far simple signals get.

**Signal AUROCs** (target: the loop-t prediction is correct; 0.5 = useless)

| Signal | t = 2 | t = 3 | t = 4 | t = 5 | t = 6 | t = 7 | t = 8 | t = 9 | t = 10 | Pooled 2..10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Margin (top-1 − top-2 logit) | 0.791 | 0.825 | 0.835 | 0.843 | 0.845 | 0.841 | 0.842 | 0.845 | 0.844 | 0.837 |
| Max softmax | 0.817 | 0.844 | 0.854 | 0.860 | 0.862 | 0.858 | 0.858 | 0.860 | 0.859 | 0.851 |
| −KL(p_t ‖ p_{t−1}) | 0.247 | 0.550 | 0.692 | 0.748 | 0.767 | 0.766 | 0.765 | 0.764 | 0.761 | 0.641 |
| −(1 − cos(z_t, z_{t−1})) | 0.426 | 0.555 | 0.601 | 0.605 | 0.600 | 0.592 | 0.591 | 0.591 | 0.594 | 0.553 |
| log raw CLS norm | 0.649 | 0.659 | 0.665 | 0.670 | 0.674 | 0.673 | 0.676 | 0.677 | 0.679 | 0.596 |

Confidence signals (max softmax, margin) are the best predictors at every loop. The change-based signals are weak and
below 0.5 at t = 2 (a big change between loops 1 and 2 goes with being right at loop 2). The CLS norm is better per
loop (0.65 to 0.68) than pooled (0.60), because the norm grows with t at the same time as accuracy does.

Plots: `results/oracle_curve.png`, `results/settling_hist.png`, `results/auroc_per_loop.png`.

## Prediction check
| Prediction | Result | |
|---|---|---|
| Oracle reaches 70.9 at 2.2 to 2.8 mean loops | 1.58 | miss (much earlier) |
| Settling loop standard deviation 1.0 to 1.5 | 1.91 | miss (wider) |
| Pooled AUROC margin 0.80 to 0.88 | 0.837 | hit |
| Pooled AUROC angular step 0.55 to 0.70 | 0.553 | hit (just inside) |
| Pooled AUROC log CLS norm 0.55 to 0.65 | 0.596 | hit |
