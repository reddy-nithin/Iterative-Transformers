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
(to come)
