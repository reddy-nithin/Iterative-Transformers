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
- [ ] smoke run (`SMOKE=1`)
- [ ] full run T30-a (tmux), per-loop readout, separation per loop vs T13-c
- [ ] T25 (a) and (b) on T30-a
- [ ] results row + reference, separation table, T25 tables, answer line, prediction check, inbox entry

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

_Not run yet._
