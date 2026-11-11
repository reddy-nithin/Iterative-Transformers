# T31 · Random-loop training (Huginn recipe) then re-test exit rules

**From:** advisor meeting 2026-11-05.
Notion task: https://notion.so/40f38c73470581e5a47cc163094a3d47

**Goal.** Every exit rule so far ties fixed depth, and the model breaks if run longer than trained. Huginn trains with
a random loop count per batch so the model works at any depth. Test that recipe. Lowest priority this round: run last.

## Setup

- Layout and reference: **T11-d**, final-only loss. Reference T11-d 73.79 (3-seed spread 0.77; seed 0: 74.11),
  checkpoint `T11-d_unroll_n3_30ep.pt`.
- **One change:** the number of B2 loops r is sampled per training batch (same r for the whole batch):
  τ ~ N(log(10) − 0.125, 0.5), r = 1 + Poisson(e^τ), capped at 20. Full backprop (no truncation). Evaluation uses
  the fixed r stated in each table (per-epoch test accuracy: r = 10).
- Code: `src/random_loops.py`, flag `--random-loops` (default off: fixed 10 loops). The loop counts come from their own
  `np.random.default_rng(seed)`, so data order and initialisation are the same as T11-d.
- **Spec fix (Nithin, 2026-11-11):** the spec's test asks for a mean of 10 ± 0.4 over 10,000 draws, but its own
  formula gives about 10.55 (1 + Poisson adds one loop; the cap at 20 takes a little off). The formula is kept as
  written and the test checks 10.55 ± 0.4. Logged as a gotcha for the vault.

| Run | Flags | Checkpoint |
|---|---|---|
| T11-d (reference) | `--model unroll --unique-blocks 3` | `T11-d_unroll_n3_30ep.pt` |
| T31-a | + `--random-loops` | `T31-a_random_loops.pt` |

Shared recipe: pretrained init, lr 5e-4, wd 0.05, 1 warmup epoch + cosine, label smoothing 0.1, batch 128, AMP,
30 epochs, seed 0. Checkpoints in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`.

## Checklist

- [x] README + predictions committed before the first run
- [x] code + tests (`test_t31_random_loops.py`) pass
- [x] smoke run (`SMOKE=1`)
- [x] full run T31-a (tmux)
- [x] test top-1 at fixed r = 1..20 for T31-a and T11-d, `results/acc_vs_loops.png`
- [x] T13's exit rules and the T8 KL rule on T31-a with a 20-loop budget
- [x] answer line, prediction check, inbox entry

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T31_random_loop_training -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T31_random_loop_training/run.sh
tmux new -s t31-train 'CUDA_VISIBLE_DEVICES=N bash tasks/T31_random_loop_training/run.sh'
```

## Predictions (written before running)

Copied verbatim from the task spec:

at r = 10, within 2 pts below T11-d; T11-d drops past loop 12 while T31-a stays flat or rises; at least one exit rule
beats fixed depth by > 1 pt on T31-a. Low confidence.

## Results

Run 2026-11-11 on GPU 0 (started when T28-a finished), seed 0. Sanity checks (full test set, r = 10): T31-a 71.77
(logged 71.76), T11-d 74.12 (74.11). Mean training loops per batch about 10.4 to 11.0 per epoch (formula mean 10.55);
peak training memory 3.8 GB (no memory problem at r = 20). Median epoch 49 s (ran alone for most of the run).

### Test top-1 at a fixed number of loops r (CIFAR-100 test, `results/acc_vs_loops.png`)

| model | r=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | **10** | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T11-d | 1.2 | 3.8 | 11.3 | 29.6 | 49.7 | 62.6 | 69.6 | 73.1 | 74.3 | **74.1** | 73.2 | 72.0 | 70.2 | 67.9 | 65.6 | 63.2 | 60.3 | 56.5 | 52.6 | 48.7 |
| **T31-a** | 23.4 | 58.0 | 68.3 | 70.9 | 71.5 | 71.8 | **72.1** | 72.1 | 71.8 | **71.8** | 71.5 | 71.3 | 71.3 | 71.1 | 70.9 | 70.9 | 70.7 | 70.7 | 70.5 | 70.5 |

- **At r = 10, T31-a costs 2.0 pt:** 71.77 vs T11-d 73.79 (−2.02) (3-seed mean, spread 0.77); vs seed 0 (74.11) −2.3. This
  is a real effect (2.6x the spread).
- **T31-a works at any depth.** It is within 1 pt of its own r = 10 accuracy from 4 loops on (70.9), peaks at 7 loops
  (72.1) and loses only 1.3 pt from 10 to 20 loops. T11-d is useless below 6 loops (29.6 at r = 4) and falls apart
  past 10 (48.7 at 20, −25 pt).
- So random depth buys robustness to the number of loops, paid for with 2 pt at the trained depth. The loop count
  also drops from 10 to about 4 to 5 for almost the same answer: at r = 4 or 5, T31-a matches T11-d at r = 7.5.

Separation per loop (within / between class spread of CLS, lower = tighter; T5 code):

| model | | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T11-d | test | 2.278 | 2.167 | 1.733 | 1.346 | 1.126 | 1.062 | 1.061 | 1.084 | 1.116 | 1.151 |
| T31-a | test | 1.917 | 1.546 | 1.313 | 1.203 | 1.168 | 1.156 | 1.154 | 1.156 | 1.161 | 1.167 |
| T31-a | train | 1.995 | 1.598 | 1.296 | 1.156 | 1.103 | 1.078 | 1.066 | 1.059 | 1.057 | 1.057 |

T31-a's classes separate much earlier (loop 2: 1.55 vs 2.17) and then stay put from loop 5 on; the final value is
about the same as T11-d's (1.167 vs 1.151).

### Exit rules on T31-a with a 20-loop budget (T13 rules and the T8 KL rule; prediction = readout at the exit loop)

| rule | best threshold | top-1 | mean loops | gain over fixed depth at the same mean loops | cheapest within 1 pt of r = 20 (fixed needs 4 loops) |
|---|---|---|---|---|---|
| confidence | 0.9 | 70.47 | 15.09 | −0.47 | 70.49 at 7.12 loops |
| KL (T8) | 0.0316 | 71.49 | 4.59 | **+0.21** | 70.93 at 3.95 loops |
| patience | 3 | 71.78 | 5.33 | +0.16 | 70.90 at 4.18 loops |

The fixed-depth curve of T31-a is so flat after loop 4 that there is little left for a rule to win: the best rule
gains +0.21 pt (KL), under the 0.8 pt bar. The useful saving is the one fixed depth already gives: 4 to 5 loops
instead of 10 for −0.2 to −0.9 pt.

Files: `results/depth_sweep.json` (curves, all rule points), `results/acc_vs_loops.png`, plus the usual
`exit_analysis.json` and `cls_svd_T31-a_*.json`. Made by `run.sh` (training, readout, separation, `depth_sweep.py`,
`plot.py`).

**Does training with random depth make adaptive depth pay off?** No, not as an exit rule: no rule beats fixed depth
by more than 0.21 pt. What random-depth training does give is a model that works at any depth: within 1 pt from 4
loops on and only −1.3 pt at 20 loops (T11-d: −25 pt), for 2 pt less at the trained 10 loops. The adaptive-depth
saving comes from simply running fewer loops, not from deciding per image.

### Prediction check

| Prediction | Result | Verdict |
|---|---|---|
| at r = 10, within 2 pts below T11-d | −2.02 vs the 3-seed mean 73.79 (−2.34 vs seed 0) | just missed (by 0.02 pt; borderline) |
| T11-d drops past loop 12 | 71.96 at 12 → 48.69 at 20 (already falling after 10) | right |
| T31-a stays flat or rises | falls slowly: 71.77 at 10 → 70.45 at 20 (−1.3 over 10 extra loops) | mostly right (nearly flat, not rising) |
| at least one exit rule beats fixed depth by > 1 pt | best +0.21 (KL) | wrong |
