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

_Pending: runs not started yet._
