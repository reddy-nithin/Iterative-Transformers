# T32 · 2+10 for 100 epochs: pretrained vs scratch

**From:** advisor meeting 2026-11-10 (Dr. Uddin, Dr. Shuhao, Nithin).
Notion task: https://notion.so/41438c73470581c883fbf39bcdc0d108

**Goal.** Two 100-epoch training curves of the same 2+10 model, one starting from pretrained weights and one from
random weights, identical in everything else. Find where (if ever) the scratch curve meets the pretrained curve.
All earlier runs stop at 30 epochs and T11-d's training loss was still falling at epoch 30; this long curve becomes the
reference for guessing where future 30-epoch runs would end up.

**Model (2+10 = T11-d layout = progressive unrolling n = 3).** B0, B1 distinct, then B2 x10, no tail.
Readout = classifier(final_norm(x)[:, 0]). 1,539,940 params.

## Runs

Seed 0. The only difference between the two runs is the initialisation.

| Run | Init | Flags | Checkpoint |
|---|---|---|---|
| T32-a | pretrained timm `vit_tiny_patch16_224`, exactly as T11-d (new 100-class classifier) | `--model unroll --unique-blocks 3` | `T32-a_2p10_pretrained_100ep.pt` |
| T32-b | fully random: blocks, patch embedding, position embedding, CLS token and classifier (timm's default init, the same as `--model scratch`) | `--model hbt --head-blocks 2 --tail-blocks 0` | `T32-b_2p10_scratch_100ep.pt` |

Checkpoints: `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`.

**Why T32-b uses `--model hbt`.** `--model unroll` always loads the pretrained weights and has no random-init switch.
`--model hbt` without `--hbt-pretrained` builds the random timm ViT-Tiny and lays its blocks out as
`nn.Sequential(B0, B1, B2 x10)` when head = 2 and tail = 0, which is the same module structure as unroll n = 3. The
smoke test checks that both models have the same parameter count and identical state_dict keys and shapes. No new
model code.

**Recipe (both runs, identical).** CIFAR-100 at 224 px, AdamW lr 5e-4 for all params, wd 0.05, 1 warmup epoch + cosine
decay over **100 epochs**, label smoothing 0.1, batch 128, AMP, nothing frozen, seed 0. Same code path as T11-d
otherwise. The scratch run does not get its own settings (no longer warmup, no other lr): both runs follow the same
rules. Note: this recipe was tuned for fine-tuning pretrained weights, so the scratch run may be slightly handicapped
by it.

**Logging.** Every epoch, both runs: test top-1, train loss, test loss and the learning rate actually used (`lr` = lr
at the epoch's last step; added to `src/train.py` for this task in 7112df3, training unchanged).

## Commands

Repo root, one GPU per run, each in its own tmux session:

```bash
# smoke test (1 epoch, 300 images, no checkpoint)
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T32_long_run_pretrained_vs_scratch/run.sh T32-a T32-b
# full runs
CUDA_VISIBLE_DEVICES=N bash tasks/T32_long_run_pretrained_vs_scratch/run.sh T32-a
CUDA_VISIBLE_DEVICES=M bash tasks/T32_long_run_pretrained_vs_scratch/run.sh T32-b
```

Outputs: `results/unroll_T32-a.json`, `results/hbt_T32-b.json` (per-epoch history), `results/T32-a.log`,
`results/T32-b.log`.

## Predictions (written before running)

Low confidence; copied verbatim from the task spec.

- T32-a ends at 76 to 79 top-1; its gain over the last 10 epochs is under 0.5 pt.
- T32-b ends at 52 to 62 top-1 and is still rising at epoch 100.
- T32-a at epoch 30 is BELOW 74.11 (learning rate still high).
- The gap never drops below 1 pt and is still over 10 pts at epoch 100.

## Results

**CAVEAT.** The learning rate follows a cosine curve stretched over all 100 epochs, so at epoch 30 it is still about
80% of its maximum. T32's epoch-30 accuracy is therefore NOT comparable with T11-d's 74.11: T11-d was a 30-epoch run
whose learning rate had already decayed to about 0 by epoch 30, so it had settled. Compare T32-a with T32-b at the same
epoch (they share the same schedule), and judge the final result at epoch 100.

Both runs finished 100 epochs on 2026-11-10 (~52 s/epoch each, one GPU each). Smoke test: both models 1,539,940
params; the random-init model started at 2.00 top-1 (300-image smoke subset), i.e. near chance. Checkpoints
`T32-a_2p10_pretrained_100ep.pt` and `T32-b_2p10_scratch_100ep.pt` saved.

### Test top-1 (%) at selected epochs

| Run | ep 10 | ep 30 | ep 50 | ep 75 | ep 100 | best (epoch) | gain ep 90 → 100 |
|---|---|---|---|---|---|---|---|
| T32-a pretrained | 62.07 | 69.85 | 72.68 | 74.49 | **75.51** | 75.61 (95) | +0.03 |
| T32-b scratch | 28.53 | 44.34 | 49.28 | 52.84 | **53.87** | 53.98 (92) | +0.19 |
| gap g(e) | 33.54 | 25.51 | 23.40 | 21.65 | **21.64** | | |

- Gap: smallest 16.93 at epoch 1, largest 36.31 at epoch 7, then shrinks slowly and stops shrinking around epoch 70
  (21.5 at ep 70, 21.6 at ep 100). First epoch with g < 1 pt: **never**.
- Both curves are flat at the end: the last 10 epochs add +0.03 (a) and +0.19 (b), both well under the seed spread.
- Final losses: T32-a test 1.598 / train 1.34; T32-b test 2.360 / train 1.87. T32-b's test loss stops falling at
  about epoch 70 (min 2.356 at ep 77) while its train loss keeps falling: the scratch model has run out of what it can
  learn from 50k images, not out of epochs.
- Throughput and FLOPs are identical (2.51 GFLOPs, ~5,800 img/s), as expected for the same architecture.

**Does scratch catch pretrained within 100 epochs? No: gap 21.6 pt at epoch 100, and it has stopped shrinking.**

### What this means for 30-epoch runs

- T32-a first passes T11-d's 74.11 at **epoch 67** and ends at 75.51, so 100 epochs buys only **+1.4 pt** over the
  30-epoch T11-d run (one seed each; about twice the seed spread). The 30-epoch runs are close to where 2+10 can go:
  read a 30-ep number as roughly 1 to 1.5 pt below its 100-ep ceiling.
- Pretraining is worth ~22 pt for this model on CIFAR-100. Scratch after 100 epochs (53.87) is about where
  pretrained is after **4 epochs** (53.36). Earlier 30-ep scratch numbers (T9 44.87, T4 47.37) are in line with
  T32-b's 44.34 at epoch 30.

### Prediction check

| Prediction | Result | Verdict |
|---|---|---|
| T32-a ends at 76 to 79 | 75.51 | wrong (just below the range) |
| T32-a gain over the last 10 epochs < 0.5 pt | +0.03 | right |
| T32-b ends at 52 to 62 | 53.87 | right (low end) |
| T32-b still rising at epoch 100 | +0.19 over the last 10 epochs, best at ep 92, test loss flat since ~ep 70 | wrong |
| T32-a at epoch 30 below 74.11 | 69.85 | right |
| Gap never below 1 pt | min 16.93 | right |
| Gap still over 10 pt at epoch 100 | 21.64 | right |

### Plots

`results/acc_vs_epoch_100ep.png`, `results/loss_vs_epoch_100ep.png`, `results/gap_vs_epoch.png`,
`results/lr_vs_epoch.png` (made by `plot.py`, which also prints the numbers above). `results/plots/` holds
presentation versions of the accuracy and loss curves (stacked, overlay, dual axis) for posting in Teams.
