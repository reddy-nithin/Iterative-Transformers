# T4 · Longer runs at lr 5e-4, with loss curves

**Goal (Dr. Uddin's main concern).** Are the shared-block models under-trained? Raising lr from 1e-4 to 5e-4 took
iter b0 from 34.8 to 46.0 in 10 epochs (T2). Here every shared-block variant, plus a fair from-scratch control, is
trained for 30 epochs at lr 5e-4, with train and test loss and accuracy logged every epoch, so we can see whether
each run levels off or is still climbing.

**From:** advisor meeting 2026-10-15 (Dr. Uddin + Dr. Shua), meeting tasks T1 (rerun all variants longer at lr 5e-4)
and T2 (fair from-scratch baseline at lr 5e-4), due Tue 2026-10-20.
Notion tasks: https://app.notion.com/p/3fa38c734705817b81a5e06699516b4a (T1),
https://app.notion.com/p/3fa38c73470581149354f81bbe36e9d7 (T2)

**Started / finished:** 2026-10-19 / -

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
- [ ] Full runs (two queues, GPUs 0 and 2)
- [ ] Plots + table (`plot.py`), verdict per run, takeaways below
- [ ] Any run still climbing at 30 epochs: rerun at 50 epochs if time allows (`EPOCHS=50 OUT_ROOT=.../results/e50`)

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
_Pending the server runs._

| Run | Params (M) | Old: 10 ep, lr 1e-4 (%) | Final acc, 30 ep, lr 5e-4 (%) | Gain last 5 ep | Train - test gap | Verdict |
|---|---|---|---|---|---|---|
| iter_b0 | 0.650 | 34.76 (46.02 at lr 5e-4) | | | | |
| iter_b6 | 0.650 | 30.70 | | | | |
| iter_avg | 0.650 | 28.54 | | | | |
| lora_r2 | 0.724 | 27.51 (avg base) | | | | |
| lora_r8 | 0.945 | 30.69 (avg base) | | | | |
| scratch | 5.544 | 28.87 | | | | |

## Takeaways
_Pending._

## Changes outside this folder
- `src/train.py`: `evaluate()` now returns `(test_acc, test_loss)`; the training loop records a running `train_acc`;
  each epoch's history has `test_loss` and `train_acc` too (older JSONs simply lack them). New `--lora-lr` flag and
  `make_param_groups(..., lora_lr=...)`: the LoRA factors (`blocks.lora_A/B`) get their own group. It raises if the
  model has no LoRA factors. The run log now prints the param groups.
- Run tests with `PYTHONPATH=. python -m pytest tests tasks -q`.
