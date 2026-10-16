# T2 · Shared block with per-loop LoRA

**Goal.** Follow-ups to T1 from the 09-29 meeting. (a) A from-scratch control: same ViT-Tiny with random weights.
(b) Does starting the shared block from the *average* of the 12 pretrained blocks help? (c) Does a higher lr help?
(d) "Idea 2": one shared block plus a small low-rank change (LoRA) for each of the 12 loops, so each loop can behave
a bit differently (the Relaxed Recursive Transformers recipe, Bae et al., ICLR 2025).

**Started / finished:** 2026-10-12 / 2026-10-12

## Status
- [x] `--model scratch` (random init control)
- [x] `--init avg` (shared block = mean of the 12 blocks)
- [x] `--model lora --rank r --lora-init {svd,zero}` (`src/lora_loop.py`) + tests (`tests/test_lora_loop.py`)
- [x] 8 server runs (GPUs 0 and 2, two tmux queues, ~40 min wall time)
- [x] Table + accuracy-vs-params plot
- [ ] LoRA from block 0 at lr 5e-4 (not run; all LoRA runs here use the weakest base and the low lr)

## Commands that were run
Same recipe as T1 (lr 1e-4 unless noted, 10 epochs, seed 0). Each run ~9 min on one A4000.
```
python -m src.train --model scratch
python -m src.train --model iterative --init avg --tag _avg
python -m src.train --model iterative --lr 5e-4 --tag _lr5e-4
python -m src.train --model lora --rank 1 --tag _r1          # LoRA runs start from the avg block (default)
python -m src.train --model lora --rank 2 --tag _r2
python -m src.train --model lora --rank 2 --lora-init zero --tag _r2zero
python -m src.train --model lora --rank 4 --tag _r4
python -m src.train --model lora --rank 8 --tag _r8
# table + plots (they include T1's runs for reference):
python -m scripts.compare --results-dir tasks/T1_baseline_vs_shared_block/results \
    tasks/T2_shared_block_with_lora/results --out-dir tasks/T2_shared_block_with_lora/results
```

## Results (`results/`, one seed: gaps under ~2 points are ties)
| Run | Params (M) | Final acc (%) |
|---|---|---|
| baseline (T1, reference) | 5.544 | 86.77 |
| scratch (random init, 12 blocks) | 5.544 | 28.87 |
| iterative, block 0 (T1) | 0.650 | 34.76 |
| iterative, block 6 (T1) | 0.650 | 30.70 |
| iterative, avg of 12 blocks | 0.650 | 28.54 |
| **iterative, block 0, lr 5e-4** | 0.650 | **46.02** |
| LoRA r1 (avg init, SVD) | 0.687 | 27.66 |
| LoRA r2 | 0.724 | 27.51 |
| LoRA r2, zero init | 0.724 | 28.74 |
| LoRA r4 | 0.798 | 27.22 |
| LoRA r8 | 0.945 | 30.69 |

Plots: `results/final_acc_bars.png` (quickest read), `results/accuracy.png` (acc per epoch), `results/acc_vs_params.png`.

## Takeaways
- **Learning rate is the biggest lever:** block 0 goes from 34.8 (lr 1e-4) to 46.0 (lr 5e-4), +11.3 points.
- Sharing is not worse than training from scratch: every shared model ties or beats scratch (28.9), and the
  0.65M block-0 model beats the 5.5M scratch model. Caveat: scratch only ran at lr 1e-4 for 10 epochs.
- Avg init does not help (28.5, about scratch level).
- LoRA r1/r2/r4 are flat (~27.5); only r8 moves (+3). SVD vs zero init is a tie. Confounded: all LoRA runs used
  lr 1e-4 and the weak avg base. Biases, LayerNorm and layer-scale stay averaged, so even full-rank SVD can't rebuild
  the original network.
- Next (became the 10-15 meeting tasks): rerun at lr 5e-4 with longer schedules and loss curves; fair scratch
  baseline at lr 5e-4; progressive unrolling (T3).

## Changes outside this folder
- `src/models.py` (`build_scratch`, `_average_block`, `build_lora_loop`), `src/lora_loop.py`, `src/train.py`
  (`--init`, `--rank`, `--lora-init`), `tests/test_lora_loop.py`.
