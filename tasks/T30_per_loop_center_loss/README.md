# T30 · Per-loop center loss: pull CLS toward its class center every loop

**From:** advisor meeting 2026-11-05.
Notion task: https://notion.so/40f38c73470581cb9dc1c74c1b849714

**Goal.** The trained version of the class-cloud idea. Make each class's cloud tight at every loop, so a class-cloud
exit (T25) has clean clouds to stop on. Recipe from DE3-BERT and the intermediate-layer center loss (SVSL).
T25 is done (`tasks/T25_class_cloud_exit/`), so this task can start.

## Setup

- Base and reference: **T13-c** (T11-d layout + readout at every loop), checkpoint
  `T13-c_unroll_readout_every_loop.pt`, top-1 72.82.
- **One change:** add L_center = α · mean over loops t = 1..10 of mean over the batch of (1 − cos(z_t, p_y)),
  z_t = final_norm(x_t)[:, 0], y = true label, α = 0.1. Prototypes p_c: one shared set, a 100 × 192 tensor (not a
  parameter, no gradient), updated after every training step as p_c ← 0.9 · p_c + 0.1 · (batch mean of z_10 for
  class c), only for classes present in the batch. p_c starts at 0; α = 0 during epoch 1 (prototypes warm up), then
  α = 0.1. The prototypes are saved in the checkpoint (key `prototypes`).
- Code: `src/center_loss.py`, flag `--center-loss-weight` (needs `--readout-loss every_loop`).

| Run | Flags | Checkpoint |
|---|---|---|
| T13-c (reference) | `--model unroll --unique-blocks 3 --readout-loss every_loop` | `T13-c_unroll_readout_every_loop.pt` |
| T30-a | + `--center-loss-weight 0.1` | `T30-a_center_loss.pt` |

Shared recipe: pretrained init, lr 5e-4, wd 0.05, 1 warmup epoch + cosine, label smoothing 0.1, batch 128, AMP,
30 epochs, seed 0. Checkpoints in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`.

After training, T25 (a) and (b) are rerun on T30-a with T25's code and T25's train-set centers (not the EMA
prototypes), so it compares like for like with T13-c.

## Checklist

- [x] README + predictions committed before the first run
- [x] code + tests (`test_t30_center_loss.py`) pass
- [x] smoke run (`SMOKE=1`)
- [x] full run T30-a (tmux), per-loop readout, separation per loop vs T13-c
- [x] T25 (a) and (b) on T30-a
- [x] results row + reference, separation table, T25 tables, answer line, prediction check, inbox entry

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T30_per_loop_center_loss -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T30_per_loop_center_loss/run.sh
tmux new -s t30-train 'CUDA_VISIBLE_DEVICES=N bash tasks/T30_per_loop_center_loss/run.sh'
```

## Predictions (written before running)

Copied verbatim from the task spec:

top-1 within ±1 of T13-c; separation lower than T13-c at every loop; the class-cloud exit gain over fixed depth
larger than on T13-c. Low confidence.

## Results

Run 2026-11-11 on GPU 3, at the same time as T28-a and T29-a (one GPU each), seed 0. Sanity check: loop-10 readout
72.57, logged 72.59. The center term (unweighted, train) fell from 0.281 (epoch 1, α = 0) to 0.144 (epoch 30), i.e.
z_t ends at cos ≈ 0.86 to its class prototype on average. All 100 prototypes are filled (norms 46 to 52) and saved in
the checkpoint (`prototypes`).

### Accuracy (CIFAR-100 test)

| model | params | final top-1 | separation after loop 10 (test / train) |
|---|---|---|---|
| T13-c (reference) | 1,539,940 | 72.82 | 1.267 / 1.128 |
| **T30-a** + center loss α = 0.1 | 1,539,940 (no new) | **72.59** (−0.23) | 1.176 / 1.078 |

−0.23 is one seed and under the 0.8 pt bar: **no effect on accuracy**.

### Per loop (test)

| model | | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T13-c | readout top-1 | 49.00 | 66.02 | 71.38 | 72.93 | 73.27 | 73.53 | 73.36 | 73.12 | 73.04 | 72.82 |
| T30-a | readout top-1 | 50.01 | 66.20 | 71.33 | 73.01 | 73.35 | 73.26 | 73.11 | 72.97 | 72.80 | 72.57 |
| T13-c | separation, test | 1.622 | 1.684 | 1.367 | 1.270 | 1.240 | 1.233 | 1.236 | 1.244 | 1.254 | 1.267 |
| T30-a | separation, test | 1.586 | 1.619 | 1.338 | 1.233 | 1.192 | 1.174 | 1.168 | 1.167 | 1.170 | 1.176 |
| T13-c | separation, train | 1.680 | 1.707 | 1.337 | 1.209 | 1.158 | 1.136 | 1.126 | 1.123 | 1.124 | 1.128 |
| T30-a | separation, train | 1.646 | 1.644 | 1.310 | 1.182 | 1.128 | 1.101 | 1.087 | 1.080 | 1.077 | 1.078 |

Separation = within-class / between-class spread of CLS (lower = tighter; T5 code). T30-a is lower at every loop on
both splits, but only by 2 to 7% (test, loop 10: 1.176 vs 1.267). The readout curves are the same within 0.3 pt at
every loop.

### T25 on T30-a (T25 code, centres from T30-a's own 50,000 train images, not the EMA prototypes)

(a) Nearest-class-centre (NCC) top-1 vs the readout, per loop (test):

| model | | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T13-c | NCC, per-loop centres | 42.49 | 60.70 | 66.92 | 68.89 | 69.55 | 69.79 | 69.71 | 69.77 | 69.64 | 69.31 |
| T30-a | NCC, per-loop centres | 42.81 | 61.45 | 67.55 | 69.43 | 70.18 | 70.53 | 70.56 | 70.48 | 70.22 | 70.01 |
| T13-c | NCC ≠ readout (%) | 37.85 | 25.19 | 19.50 | 16.95 | 16.28 | 16.03 | 15.87 | 15.89 | 15.78 | 15.79 |
| T30-a | NCC ≠ readout (%) | 37.38 | 23.44 | 17.56 | 15.72 | 14.92 | 14.71 | 14.61 | 14.66 | 14.94 | 14.98 |

NCC gets 0.3 to 0.8 pt better and agrees with the readout about 1 pt more often, but still trails the readout by 2.5
to 3 pt at every loop.

(b) Exit rules (stop at loop t ≥ 2, prediction = readout; best gain over the fixed-depth curve at the same mean loops):

| rule | T13-c best gain (mean loops) | T30-a best gain (mean loops) |
|---|---|---|
| R1 radius, per-loop centres | −0.63 (8.02) | −0.26 (8.92) |
| R2 ratio, per-loop centres | −0.26 (8.42) | −0.27 (8.75) |
| R3 agree, per-loop centres | −0.26 (8.42) | −0.27 (8.75) |
| R1 radius, shared centres | **−0.19** (8.87) | −0.22 (9.23) |
| R2 ratio, shared centres | −0.25 (8.62) | −0.23 (9.00) |
| R3 agree, shared centres | −0.25 (8.62) | −0.23 (9.00) |
| T13's best rule on T13-c (baseline) | +0.79 | |

No class-cloud rule beats fixed depth on either model; the best is −0.22 on T30-a vs −0.19 on T13-c. Only R1 with
per-loop centres improves (−0.63 → −0.26), and it gets there by stopping later (8.9 loops).

Plots: `results/acc_per_loop.png`, `results/separation_per_loop.png`, `results/class_cloud_best_gain.png`. Made by
`run.sh` (training, readout, separation, T25's `class_cloud.py`) and `plot.py`.

**Do tighter clouds make a better stopping rule?** No. The center loss makes the clouds tighter at every loop, but
only a little (2 to 7%), costs nothing measurable in accuracy, and no class-cloud exit rule beats fixed depth on T30-a
(best −0.22, as on T13-c −0.19, far from T13's +0.79). Clouds this much tighter are not enough to stop on.

### Prediction check

| Prediction | Result | Verdict |
|---|---|---|
| top-1 within ±1 of T13-c | −0.23 | right |
| separation lower than T13-c at every loop | lower at all 10 loops, test and train (by 2 to 7%) | right |
| class-cloud exit gain over fixed depth larger than on T13-c | best −0.22 vs −0.19; per rule within ±0.05 except R1 per-loop (−0.26 vs −0.63) | wrong |
