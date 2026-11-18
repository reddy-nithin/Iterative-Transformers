# T37 · Fair baseline: 12-block ViT-Tiny on the T11-d recipe

From: planning session 2026-11-16 (Fable review)

## Goal
Our only plain-ViT number (86.77) was trained 10 epochs at lr 1e-4, while every looped run uses 30 epochs at lr 5e-4.
Train the normal 12-block pretrained ViT-Tiny on exactly the T11-d recipe so every "looped vs baseline" comparison
uses the same rules.

## Run
T37-a = `--model baseline`, pretrained timm `vit_tiny_patch16_224`, all 12 blocks, recipe identical to T11-d
(CIFAR-100 224 px, AdamW lr 5e-4 all params, wd 0.05, 1 warmup epoch + cosine over 30 epochs, label smoothing 0.1,
batch 128, AMP, seed 0, nothing frozen). Checkpoint `T37-a_baseline_12blk_30ep.pt`.

    CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T37_fair_baseline_30ep/run.sh   # 1 epoch, --subset 512
    CUDA_VISIBLE_DEVICES=N bash tasks/T37_fair_baseline_30ep/run.sh           # full run
    CUDA_VISIBLE_DEVICES=N .venv/bin/python -m tasks.T37_fair_baseline_30ep.eval_train   # train-set top-1

## Predictions (written before running)
T37-a 85.0 to 87.5 test top-1; training-set top-1 above 95.

## Results
(filled in after the run)
