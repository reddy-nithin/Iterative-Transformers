# T11 · Pretrained head + body x10 + tail: which block is the body?

**Goal.** First pretrained head / body / tail runs (from now on everything starts from pretrained weights). T9 tried this
layout only from scratch (44.87). Here we test which pretrained block makes the best looped body, and whether a tail
is worth its block compared with spending it on a bigger head (T11-d, progressive unrolling n = 3).

**From:** advisor meeting 2026-10-27 (Dr. Uddin), due Tue 2026-11-03.
Notion task: https://notion.so/40938c734705817e8a8adc7980fed0cd

**Started:** 2026-10-30

## Checklist
- [x] Pretrained-init option for `--model hbt` (`--hbt-pretrained --head-idx H --body-idx B --tail-idx T`, default off so
      T9 is unchanged) + tests (`test_t11_pretrained_hbt.py`)
- [x] Smoke test of all four configs (300 images, 1 epoch): all four 1,539,940 params
- [ ] T11-a to T11-d, 30 epochs each, one GPU each, in parallel
- [ ] Class separation per loop (T5 `cls_svd.py`, 4 classes, train + test) for every run
- [ ] Plots `results/loss_curves.png`, `results/acc_vs_epoch.png`, `results/separation_vs_loop.png`
- [ ] Results table, winner line, H-B-T vs T11-d line

## Runs
Shared recipe: pretrained timm `vit_tiny_patch16_224`, CIFAR-100 at 224 px, AdamW lr 5e-4 for all parameters,
wd 0.05, 1 warm-up epoch + cosine, label smoothing 0.1, batch 128, AMP, 30 epochs, seed 0, nothing frozen.
Head, body and tail are `copy.deepcopy` of the pretrained blocks, so they are three separate modules even when the
indices repeat (T11-a: head and body both start from block 0 and train independently).

| Run | Head | Body x10 | Tail | Checkpoint |
|---|---|---|---|---|
| T11-a | blocks[0] | blocks[0] | blocks[11] | `T11-a_hbt_body_b0.pt` |
| T11-b | blocks[0] | blocks[6] | blocks[11] | `T11-b_hbt_body_b6.pt` |
| T11-c | blocks[0] | blocks[1] | blocks[11] | `T11-c_hbt_body_b1.pt` |
| T11-d | progressive unrolling n = 3 (blocks 0, 1 distinct, block 2 x10, no tail), existing code, only epochs 10 → 30 | | | `T11-d_unroll_n3_30ep.pt` |

Why these: T11-a = our best looping block (b0 x12 58.06 beat b6 x12 54.51). T11-b = mid-network block as body
(Gemini A8). T11-c = block 1 naturally follows block 0, so the head → body seam is smallest. T11-d = the same
parameter count spent on a 2-block head instead of a tail (67.12 at 10 epochs) = the reference to beat.

## Predictions (written before running)
T11-d reaches ~69 to 72. T11-b (block 6 body) is lowest of T11-a to T11-c.
T11-a and T11-c within 2 pts of each other. Unsure whether any looped body gets separation below 1.

## Commands
```bash
# smoke test (any idle GPU, ~1 min each)
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T11_pretrained_hbt_body_sweep/sweep.sh T11-a T11-b T11-c T11-d
# real runs: one tmux session and one GPU per run
tmux new -d -s itertf-T11-a "CUDA_VISIBLE_DEVICES=0 bash tasks/T11_pretrained_hbt_body_sweep/sweep.sh T11-a"
```
Checkpoints: `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/T11-<run>_<slug>.pt`.
Tests: `PYTHONPATH=. .venv/bin/python -m pytest tests tasks -q`.

## Results
_(after the runs)_

Winner rule: T11 winner = highest final test top-1 among T11-a, T11-b, T11-c; if the top two are within 2 pts (one
seed = a tie), pick the one with lower final separation; if still tied, T11-a.

- T11 winner: _(after the runs)_
- Best H-B-T vs T11-d: _(beats by ≥2 / tie / loses by ≥2)_
