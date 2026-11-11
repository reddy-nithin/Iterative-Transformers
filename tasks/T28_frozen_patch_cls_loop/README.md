# T28 · Frozen-patch CLS-only loop (Dr. Uddin's reading of Huginn)

**From:** advisor meeting 2026-11-05.
Notion task: https://notion.so/40f38c73470581538ee5edce6084fe77

**Goal.** Dr. Uddin's reading of Huginn: process the image once, then freeze it and let only CLS loop, collecting
evidence from the fixed patches. In our model the loop also keeps changing the patches. Test his version with one
change. Side benefit: a loop that updates one token instead of 197 costs far less.

## Setup

- Layout and reference: **T11-d** (B0, B1 on all tokens, then B2 x10, no tail, final-only loss, T11-d config).
  Reference T11-d 73.79 (3-seed spread 0.77; seed 0: 74.11), checkpoint `T11-d_unroll_n3_30ep.pt`.
- **One change:** after B1 the patch tokens P = x[:, 1:] are frozen (no detach: gradients still flow into B0/B1
  through P). In each of the 10 loops B2 updates ONLY the CLS row: the CLS query attends to keys and values from
  norm1([CLS; P]); CLS gets the attention output and then the MLP on its own row, exactly as in a normal B2 pass; the
  patch rows stay P. Readout as T11-d. No new parameters.
- Code: `src/cls_only_loop.py` (`CLSOnlyBlock`), flag `--cls-only-loop` for `--model unroll`.
  The CLS query is the only query computed; the patch keys/values are recomputed each loop (same numbers as caching
  them once, which would be cheaper still).

| Run | Flags | Checkpoint |
|---|---|---|
| T11-d (reference) | `--model unroll --unique-blocks 3` | `T11-d_unroll_n3_30ep.pt` |
| T28-a | + `--cls-only-loop` | `T28-a_cls_only_loop.pt` |

Shared recipe: pretrained init, lr 5e-4, wd 0.05, 1 warmup epoch + cosine, label smoothing 0.1, batch 128, AMP,
30 epochs, seed 0. Checkpoints in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`.

## Checklist

- [ ] README + predictions committed before the first run
- [ ] code + tests (`test_t28_cls_only.py`) pass
- [ ] smoke run (`SMOKE=1`)
- [ ] full run T28-a (tmux), per-loop readout, separation per loop
- [ ] FLOPs of one loop (both models) and wall time per epoch
- [ ] results row + reference, FLOPs and time table, answer line, prediction check, inbox entry

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T28_frozen_patch_cls_loop -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T28_frozen_patch_cls_loop/run.sh
tmux new -s t28-train 'CUDA_VISIBLE_DEVICES=N bash tasks/T28_frozen_patch_cls_loop/run.sh'
```

## Predictions (written before running)

Copied verbatim from the task spec:

top-1 3 to 8 pts below T11-d (the patches stop refining); one CLS-only loop costs under 2% of a full loop. If it
lands within 2 pts of T11-d, that is a strong efficiency result. Low confidence.

## Results

_Not run yet._
