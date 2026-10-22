# T9 · One head + repeated body + one tail, from scratch

**Goal.** Test Dr. Uddin's head / body / tail design: one free head block, one body block repeated for the middle
steps, one free tail block before the classifier. 12 execution steps in total, so the body repeats 10 times. Trained
from scratch: this is not a transfer-learning test. Also measure how tight the classes are (CLS separation) after
every loop.

**From:** advisor meeting 2026-10-20 (Dr. Uddin), meeting task T3, due Tue 2026-10-27.
Notion task: https://notion.so/3ff38c734705814098c8e0d4ecadeab0

**Started / finished:** 2026-10-22 / –

**Not built, on purpose:** reusing frozen pretrained late blocks (B9 to B12) as head / tail. That was Nithin's idea and
Dr. Uddin rejected it: late blocks cannot take raw patch embeddings, and this is not transfer learning.

## Checklist
- [x] Model variant `--model hbt` (`--head-blocks`, `--tail-blocks`, `--inject`, default off) + tests
- [ ] Main run: head + body ×10 + tail, from scratch, lr 5e-4, 30 epochs
- [ ] Optional (a): head-only control, head + body ×11 (makes head+tail vs head-only a fair, all-from-scratch comparison;
      the existing n = 2 unrolling point is fine-tuned from pretrained weights, so it is not)
- [ ] Optional (b): input injection, body(x + e) every loop, e = head output (Huginn, arXiv 2502.05171; plain addition
      first, the paper concatenates and maps 2d → d with a linear layer)
- [ ] Extra (asked 2026-10-22): loop-only, one random block ×12. The from-scratch counterpart of iter b0 (pretrained
      block 0 ×12), so all layouts can be compared from scratch with one recipe
- [ ] Class separation after every loop, train and test, for every run (`results/separation_vs_loop.png`)
- [ ] Accuracy curves (`results/acc_vs_epoch.png`) and results table below

## Runs
All from scratch (random init), lr 5e-4, 30 epochs, seed 0, otherwise the T4 recipe (AdamW wd 0.05, 1 warm-up epoch
+ cosine, label smoothing 0.1, batch 128, AMP). Same FLOPs for every layout (12 block calls).

| Run | Layout | Params (M) | Distinct blocks |
|---|---|---|---|
| hbt | head + body ×10 + tail | 1.540 | 3 |
| hbt_inject | same, body input = state + head output every loop | 1.540 | 3 |
| head_only | head + body ×11 | 1.095 | 2 |
| loop_only | body ×12 | 0.650 | 1 |

References at the same lr and epochs (T4): scratch 12 distinct blocks 5.54M = 47.37%; iter b0 (pretrained block 0
×12, 0.65M) = 58.06%.

**Input injection note.** With plain addition the head output is added before every body step, and the body is a
residual block, so the state grows by about e per loop. If hbt_inject trails hbt, that is a likely reason, and the
paper's concat + linear adapter is the next variant.

**Class separation** = tr(within-class covariance) / tr(between-class covariance) of the CLS token after each loop
(NC1-style, T5's `cls_svd.py`). Same 4 classes (apple, bicycle, fox, cloud) and image counts as the iter b0 numbers:
train 500 / class + 500 mixed control, test 100 / class + 100 control. Below 1 = classes tighter than the distances
between them. Reference: fine-tuned 12-block ViT below 1 from block 9 (0.55 test); iter b0 minimum 1.45 at loop 8.

## Commands
```bash
# GPU check (nvidia-smi broken 2026-10-22); ~0.2 GB used = idle
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c "import torch;f,t=torch.cuda.mem_get_info();print((t-f)/2**30,'GB used')"
tmux new -s itertf-t9-hbt-1022
CUDA_VISIBLE_DEVICES=0 bash tasks/T9_head_body_tail/sweep.sh hbt hbt_inject       # GPU 0 after T6, ~1 h
tmux new -s itertf-t9-controls-1022
CUDA_VISIBLE_DEVICES=1 bash tasks/T9_head_body_tail/sweep.sh head_only loop_only  # GPU 1 after T7, ~1 h
.venv/bin/python tasks/T9_head_body_tail/plot.py                                  # table -> results/summary.md
```
Checkpoints: `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/t9_<run>.pt`. Each run takes ~27 min of training and
~2 min of CLS analysis.
Smoke test that was run: `EPOCHS=1 EXTRA_ARGS="--subset 512" CLS_ARGS="--per-class 20 --control-size 0"` for all four
runs (train, checkpoint save, strict reload, 12-loop separation, plot), 2026-10-22.

## Results (one seed)
_Pending._

## Takeaways
_Pending._

## Changes outside this folder
- `src/models.py`: `build_head_body_tail` (registered as `"hbt"` in `build`).
- `src/head_body_tail.py` (new): `InjectedLoop`, the block container for input injection. `len()` and iteration list
  the 12 steps in call order, so T5's per-loop hooks work unchanged.
- `src/train.py`: `--model hbt` with `--head-blocks`, `--tail-blocks`, `--inject` (all other models unchanged).
- `tests/test_head_body_tail.py` (new): head, body and tail are three distinct objects and the body is reused 10 times;
  param count; gradients reach all three; inject off = hand-built stack; inject on differs with the same weights;
  checkpoint round trip. `tests/test_sharing.py` still passes.
