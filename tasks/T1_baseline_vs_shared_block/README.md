# T1 · Baseline vs shared block

**Goal.** First test of the iterative-transformer idea (asked for by Dr. Uddin). Take pretrained ViT-Tiny
(`vit_tiny_patch16_224`, 12 blocks), keep **one** block and apply it 12 times. How much accuracy do we lose for the
smaller model? Fine-tune on CIFAR-100 (224 px) with the same recipe as the normal model.

**Started / finished:** 2026-10-04 / 2026-10-04

## Status
- [x] Load pretrained ViT-Tiny, inspect the 12 blocks (444,864 params each)
- [x] Build the iterative model (one shared block object) + tests proving the sharing
- [x] Train/eval script with `--model {baseline,iterative}`
- [x] Server runs: baseline, iterative block 0, iterative block 6
- [x] Compare accuracy, params, FLOPs, throughput

## Commands that were run
Recipe for all three: AdamW lr 1e-4, wd 0.05, 1 warmup epoch + cosine, label smoothing 0.1, batch 128,
10 epochs, seed 0, AMP. One A4000 per run, ~8 min each.
```
CUDA_VISIBLE_DEVICES=0 python -m src.train --model baseline 2>&1 | tee results/baseline.log
CUDA_VISIBLE_DEVICES=1 python -m src.train --model iterative 2>&1 | tee results/iterative.log
CUDA_VISIBLE_DEVICES=2 python -m src.train --model iterative --block-idx 6 --tag _b6 2>&1 | tee results/iterative_b6.log
```

## Results (`results/`)
| Model | Params | GFLOPs | img/s | Final acc | Best acc |
|---|---|---|---|---|---|
| Baseline (12 different blocks) | 5.544 M | 2.51 | 5653 | **86.77%** | 86.83% |
| Iterative, block 0 x 12 | 0.650 M | 2.51 | 5773 | 34.76% | 34.76% |
| Iterative, block 6 x 12 | 0.650 M | 2.51 | 5820 | 30.70% | 30.70% |

## Takeaways
- Params drop 88%, but FLOPs and speed don't change: the block still runs 12 times. Sharing saves memory, not time.
- Accuracy collapses (-52 / -56 points). Both shared models were still rising at epoch 10 (not converged).
- Block 6 did *worse* than block 0, against my guess that a middle block would be better. One seed, so this is not
  a ranking.
- Suspects: lr 1e-4 is a fine-tuning rate, too low for re-learning a shared block; short schedule; no
  per-iteration flexibility. These led to T2.

## Changes outside this folder
- `src/models.py` (`build_baseline`, `build_iterative`), `src/data.py`, `src/metrics.py`, `src/train.py`,
  `tests/test_sharing.py`, `scripts/compare.py`, `notebooks/colab_walkthrough.*` (the notebook fetches this
  folder's JSONs from GitHub).
