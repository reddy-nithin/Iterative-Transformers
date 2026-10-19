# T4 · Longer runs at lr 5e-4, with loss curves

**Goal (Dr. Uddin's main concern).** Are the shared-block models under-trained? Raising lr from 1e-4 to 5e-4 took
iter b0 from 34.8 to 46.0 in 10 epochs (T2). Here every shared-block variant, plus a fair from-scratch control, is
trained for 30 epochs at lr 5e-4, with train and test loss and accuracy logged every epoch, so we can see whether
each run levels off or is still climbing.

**From:** advisor meeting 2026-10-15 (Dr. Uddin + Dr. Shua), meeting tasks T1 (rerun all variants longer at lr 5e-4)
and T2 (fair from-scratch baseline at lr 5e-4), due Tue 2026-10-20.
Notion tasks: https://app.notion.com/p/3fa38c734705817b81a5e06699516b4a (T1),
https://app.notion.com/p/3fa38c73470581149354f81bbe36e9d7 (T2)

**Started / finished:** 2026-10-19 / 2026-10-19

## Runs
All at lr 5e-4, 30 epochs, seed 0, otherwise the T1 recipe (AdamW wd 0.05, 1 warmup epoch + cosine, label smoothing
0.1, batch 128, AMP).

| Run | Model | Params (M) | Notes |
|---|---|---|---|
| iter_b0 | block 0 x12 | 0.650 | |
| iter_b6 | block 6 x12 | 0.650 | |
| iter_avg | mean of the 12 blocks, x12 | 0.650 | |
| lora_r2 | block 0 x12 + rank-2 LoRA per loop | 0.724 | LoRA factors at lr 5e-3 (10x), SVD init. **Base is block 0** (T2's LoRA runs used the avg block) |
| lora_r8 | same, rank 8 | 0.945 | same |
| scratch | normal 12-block ViT-Tiny, random init | 5.544 | the fair control for iter b0 (meeting T2) |

**What is logged per epoch** (`history` in each run's JSON): `train_loss`, `test_loss`, `train_acc`, `test_acc`.
Test loss uses the same label-smoothed loss as training, so the two are comparable. Train acc is a *running*
accuracy over the epoch's training batches (augmented images, train mode, weights changing during the epoch); it
needs no extra pass, but it reads a little lower than a clean re-evaluation would.

**Plateau rule:** a run has plateaued if test acc rose by less than 0.5 points over the last 5 epochs.
Caveat: the cosine schedule takes the lr to ~0 at the end, so the last epochs flatten partly *by construction*. So the
table also shows the gain over the 5 epochs before that, and the train - test gap. A small gap with train acc still
rising means under-trained: more epochs should help. A large gap with flat test acc means the model is fitting the
training set, so more epochs won't help.

**Don't compare epoch 10 here with the old 10-epoch runs.** With 30 epochs the cosine keeps the lr higher for
longer, so the early epochs look different. Only the final numbers are comparable.

## Status
- [x] `--lora-lr` flag (separate lr for the LoRA factors; without it, one param group exactly as before)
- [x] Test loss + train acc logged per epoch (all models; see "Changes outside this folder")
- [x] `sweep.sh` (skips finished runs), `plot.py` (curves, overview, summary table with verdicts)
- [x] Tests: `test_t4_logging.py` (9 tests: lr groups, bad `--lora-lr`, CLI, evaluate loss, plateau rule); all 37 pass
- [x] Smoke test 2026-10-19, GPU 0: all 6 runs, 512 images, 1 epoch. Params 650,212 / 723,940 / 945,124 / 5,543,716
      as expected; LoRA runs show 2 lr groups (0.65M @ 5e-4, LoRA @ 5e-3). Rerun skips finished runs; a typo exits early.
      Plot check: iter_b0 + scratch, 2,048 images, 6 epochs.
- [x] Full runs (two queues, GPUs 0 and 2), 2026-10-19 12:12 to ~15:00, all 30/30 epochs, no errors
- [x] Plots + table (`plot.py`), verdict per run, takeaways below
- [ ] Optional follow-up: 50-epoch reruns of iter_b0, lora_r8, scratch (the three the rule calls still climbing):
      `EPOCHS=50 OUT_ROOT=tasks/T4_longer_runs_lr5e-4/results/e50 bash tasks/T4_longer_runs_lr5e-4/sweep.sh iter_b0 lora_r8 scratch`

## Commands
Run from the repo root. Each 30-epoch run is ~27 min on one A4000; 3 runs per queue ≈ 1.4 h.
```
nvidia-smi                                   # pick two idle GPUs (only your own python lines matter)
cd ~/itertf
tmux new -s t4-a
CUDA_VISIBLE_DEVICES=0 bash tasks/T4_longer_runs_lr5e-4/sweep.sh scratch lora_r8 iter_b6
#   detach: Ctrl+b then d
tmux new -s t4-b
CUDA_VISIBLE_DEVICES=2 bash tasks/T4_longer_runs_lr5e-4/sweep.sh iter_b0 lora_r2 iter_avg
#   detach: Ctrl+b then d
```
- Watch: `tmux attach -t t4-a`. Quick progress: `tail -n 2 tasks/T4_longer_runs_lr5e-4/results/*.log`
- If a queue dies, run the same command again: finished runs are skipped. A half-finished run restarts from
  epoch 1 (there are no checkpoints).
- Only one GPU free? Put all six in one queue: `CUDA_VISIBLE_DEVICES=0 bash tasks/T4_longer_runs_lr5e-4/sweep.sh` (~2.7 h).
- When both finish: `.venv/bin/python tasks/T4_longer_runs_lr5e-4/plot.py` → `results/summary.md`,
  `results/curves_<run>.png`, `results/test_acc_all.png`, `results/b0_vs_scratch.png`.
  Then `tmux kill-session -t t4-a` (and `t4-b`).

Smoke test that was run: `CUDA_VISIBLE_DEVICES=0 OUT_ROOT=<scratch> EXTRA_ARGS="--subset 512 --epochs 1" bash tasks/T4_longer_runs_lr5e-4/sweep.sh`

## Results (`results/`, one seed: gaps under ~2 points are ties)
| Run | Params (M) | Old: 10 ep, lr 1e-4 (%) | Final acc, 30 ep, lr 5e-4 (%) | Gain ep 20→25 | Gain ep 25→30 | Train - test gap | Verdict (rule) |
|---|---|---|---|---|---|---|---|
| iter_b0 | 0.650 | 34.76 (46.02 at lr 5e-4) | 58.06 | 3.15 | 0.57 | -2.7 | still climbing |
| iter_b6 | 0.650 | 30.70 | 54.51 | 2.53 | 0.46 | -3.1 | plateaued |
| iter_avg | 0.650 | 28.54 | 52.96 | 2.92 | 0.30 | -2.7 | plateaued |
| lora_r2 | 0.724 | 27.51 (avg base) | 59.52 | 4.06 | 0.48 | -2.2 | plateaued |
| **lora_r8** | 0.945 | 30.69 (avg base) | **61.33** | 3.39 | 0.79 | -0.4 | still climbing |
| scratch | 5.544 | 28.87 | 47.37 | 3.10 | 0.87 | -0.2 | still climbing |
| baseline (T1, pretrained, reference) | 5.544 | 86.77 | - | - | - | - | - |

Plots: `results/test_acc_all.png` (quickest read), `results/curves_<run>.png` (loss + acc, train and test, one per
run, old 10-epoch result marked), `results/b0_vs_scratch.png`. Table: `results/summary.md`.
Train acc is the running value on augmented batches, so a negative gap means "no sign of overfitting", not that the
model does better on unseen data.

## Takeaways
- **Yes, the shared-block models were under-trained (Dr. Uddin's concern confirmed).** Higher lr plus 3x the epochs
  lifts every shared model by ~23-24 points over the old runs: iter b0 34.8 → 58.1 (46.0 of that came from the lr
  alone at 10 epochs, the rest from the longer schedule), b6 30.7 → 54.5, avg 28.5 → 53.0.
- **Plateaued or still climbing? None has clearly saturated.** By the 0.5-point rule, b6, avg and LoRA r2 plateaued
  and b0, LoRA r8 and scratch are still climbing. But every run sits close to the threshold (0.3 to 0.9), and every
  run gained 2.5 to 4 points in epochs 20→25. The final flattening is mostly the cosine lr reaching ~0. Test loss is
  still at its lowest in the last 1-2 epochs and train acc is not above test acc, so there is no overfitting yet.
  A 50-epoch run is the real test (follow-up command above).
- **Fair scratch control (meeting T2): the 0.65M shared model beats the 5.5M scratch model by 10.7 points** (58.1 vs
  47.4), same lr, epochs and recipe. So starting from one pretrained block is worth far more than 8.5x the parameters
  trained from random. Caveat: 5e-4 may not be the best lr for scratch, and ViTs from scratch need heavy
  augmentation on small datasets.
- **Which block:** b0 > b6 by 3.6 points (more than the ~2-point noise level). b6 vs avg (1.5) is a tie. Avg init went
  from the worst model to close to b6 once the lr was right.
- **LoRA helps once the confounders are gone:** with the block-0 base and its own lr (5e-3), r2 is +1.5 over b0 (a
  tie) and r8 is +3.3 (61.3%, the best shared model, +0.3M params = 17% of ViT-Tiny). T2's flat LoRA results were
  the weak avg base + low lr, not LoRA itself.
- **Still far from the pretrained model** (86.8%): the best shared model is ~25 points behind. For scale, T3's n=4
  (four distinct blocks, 2.0M params) reached 74.0% in only 10 epochs at the same lr.
- Not a bug: epoch 10 here (b0 45.7) is not equal to the old 10-epoch lr 5e-4 run (46.0), because the 30-epoch
  cosine is still at a high lr at epoch 10.

## Changes outside this folder
- `src/train.py`: `evaluate()` now returns `(test_acc, test_loss)`; the training loop records a running `train_acc`;
  each epoch's history has `test_loss` and `train_acc` too (older JSONs simply lack them). New `--lora-lr` flag and
  `make_param_groups(..., lora_lr=...)`: the LoRA factors (`blocks.lora_A/B`) get their own group. It raises if the
  model has no LoRA factors. The run log now prints the param groups.
- Run tests with `PYTHONPATH=. python -m pytest tests tasks -q`.
