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

- [ ] README + predictions committed before the first run
- [ ] code + tests (`test_t31_random_loops.py`) pass
- [ ] smoke run (`SMOKE=1`)
- [ ] full run T31-a (tmux)
- [ ] test top-1 at fixed r = 1..20 for T31-a and T11-d, `results/acc_vs_loops.png`
- [ ] T13's exit rules and the T8 KL rule on T31-a with a 20-loop budget
- [ ] answer line, prediction check, inbox entry

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

_Not run yet._
