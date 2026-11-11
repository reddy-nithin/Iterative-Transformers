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

- [ ] README + predictions committed before the first run
- [ ] code + tests (`test_t29_concat_inject.py`) pass
- [ ] smoke run (`SMOKE=1`)
- [ ] full run T29-a (tmux), per-loop readout, separation per loop
- [ ] mean ‖x_t‖ per loop (T29-a, T11-d, T9 plain add), ‖W_state‖ vs ‖W_e‖
- [ ] results row + reference, norm and separation tables, answer line, prediction check, inbox entry

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T29_concat_input_injection -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T29_concat_input_injection/run.sh
tmux new -s t29-train 'CUDA_VISIBLE_DEVICES=N bash tasks/T29_concat_input_injection/run.sh'
```

## Predictions (written before running)

Copied verbatim from the task spec:

top-1 within ±1.5 of T11-d; ‖x_t‖ grows less than in T9's plain-add run; separation does not jump; ‖W_e‖ ends small
(the patches already carry e). Low confidence.

## Results

_Not run yet._
