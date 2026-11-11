# T23 · Per-loop gradient norms of the shared block (no training) [H10]

**From:** paper-read session 2026-11-03 (Nithin), alphaXiv Itertf folder.
Notion task: https://notion.so/40d38c73470581068c58d1ce9f5763e6

**Goal.** MiniViT (arXiv 2204.07154) found that when blocks share weights, sharing the LayerNorms too gives large,
uneven gradients across the repeated uses and unstable training; un-sharing the LayerNorms fixed it. Our looped block
shares everything, LayerNorms included. Its gradient is the SUM of the gradients from its 10 or 12 uses: if one or two
loops dominate that sum, the block is effectively trained for those loops only, which could explain the ~60% ceiling.
This task measures each loop's share before spending a run on per-loop LayerNorms (H10 step 2).

## Setup

- **Per-loop gradient.** The loop is unrolled so that loop t uses its own copy W_t of the shared block's parameters
  (leaf tensors with identical values, run through `torch.func.functional_call`). Then g_t = ∂L/∂W_t is loop t's share,
  and Σ_t g_t must equal the normal shared gradient.
  - **Gate:** on one batch per model, ‖Σ_t g_t − g_shared‖ / ‖g_shared‖ ≤ 1e-4, and the unrolled logits equal the
    normal logits. If either fails: `gotcha` entry, T23 stops.
  - No weight values are changed. No optimizer step.
- **Parameter groups:** whole looped block, plus norm1, attn.qkv, attn.proj, norm2, mlp.fc1, mlp.fc2 (each weight +
  bias). These 6 groups are all 12 parameters of the block. **There is no LayerScale** (`ls1` / `ls2` are `Identity`
  in `vit_tiny_patch16_224`), so there is no 7th group; the code checks the groups cover the block exactly.
- **Loss:** cross-entropy with label smoothing 0.1 on the final output only, for every model (also models trained with
  extra losses). fp32, no AMP, `model.eval()` (ViT-Tiny has no dropout or drop-path, so eval only fixes the mode).
- **Data:** 20 batches of 128 CIFAR-100 TRAIN images, test transform. "Fixed order with seed 0" is implemented as:
  `torch.randperm(50000, generator=torch.Generator().manual_seed(0))[:2560]`, in that order. g_t is computed per
  batch; the reported ‖g_t‖ is the mean of the per-batch norms over the 20 batches.
- **Models** (checkpoints in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`):

  | Name | Layout | Weights | Loops |
  |---|---|---|---|
  | b0-init | B0 x12 (`build("iterative", block_idx=0, init="block")`) | pretrained; new classifier from `set_seed(0)` right before `build`, as `src/train.py` does (seed default 0) | 12 |
  | b0-trained | B0 x12 | `T5-b_iter_b0.pt` (58.06) | 12 |
  | d-init | T11-d layout (`build("unroll", unique_blocks=3)`: B0, B1, B2 x10) | pretrained; classifier as above | 10 (B2) |
  | d-trained | T11-d layout | `T11-d_unroll_n3_30ep.pt` (74.11) | 10 (B2) |

  The trained checkpoints' top-1 on the full test set is checked against the logged value first.
- **Reported** per model, group and loop: ‖g_t‖. Per model and group:
  - spread = max_t ‖g_t‖ / min_t ‖g_t‖;
  - dominance = share of Σ_t ‖g_t‖ carried by the 2 largest loops;
  - agreement = mean over t of cos(g_t, Σ_s g_s);
  - mean pairwise cos between loops (cosines computed per batch, then averaged over batches);
  - ‖x_t‖ per loop = mean norm of the residual state after loop t, over tokens and images.
- **Decision line** (exact): "run H10 step 2" if, on b0-init OR d-init, the whole-block spread ≥ 3, OR a LayerNorm
  group's (norm1 or norm2) spread ≥ 5. Otherwise "skip H10 step 2", with the four numbers (whole-block spread and
  largest LayerNorm spread, for b0-init and d-init).

## Checklist

- [x] README + predictions committed before the first run
- [x] code + tests (`test_t23_per_loop_grads.py`) pass
- [x] smoke run (`SMOKE=1`)
- [x] full run: gate passes on every model, trained top-1 checks pass
- [x] per-loop tables, summary table, plots `results/{grad_norm_per_loop,grad_norm_per_group,residual_norm_per_loop}.png`
- [x] decision line, answer line ("Is one part of the loop training the shared block for everyone?"), prediction
      check, inbox entry

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T23_per_loop_grad_norms -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T23_per_loop_grad_norms/run.sh
CUDA_VISIBLE_DEVICES=N bash tasks/T23_per_loop_grad_norms/run.sh
```

## Predictions (written before running)

Copied verbatim from the task spec:

- In both init models, the gradient share falls with loop index (early loops get the largest share), with
  whole-block spread between 2 and 5.
- norm1 and norm2 spreads are larger than the whole-block spread.
- ‖x_t‖ grows by at least 2x from loop 1 to the last loop.
- Trained models are more even than init models.
- Low confidence on all.

## Results

Run 2026-11-11, one GPU (a few minutes). Sanity checks:
- trained top-1 on the full test set: b0-trained 58.06 (logged 58.06), d-trained 74.12 (logged 74.11);
- gate on every model: ‖Σ_t g_t − g_shared‖ / ‖g_shared‖ = 8.0e-8 to 9.0e-8 (limit 1e-4); unrolled logits identical
  to the normal ones (max diff 0);
- the init classifier is bit-identical across two `set_seed(0)` + `build` calls.

### Per-loop ‖g_t‖ (mean over 20 batches of 128)

Flat (every loop equal) would give dominance 2/12 = 0.167 for b0 and 2/10 = 0.20 for d.

**b0-init** (12 loops)

| group | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block | 4.00 | 4.58 | 3.49 | 2.38 | 1.66 | 1.22 | 0.968 | 0.827 | 0.754 | 0.732 | 0.761 | 0.866 |
| norm1 | 0.595 | 0.254 | 0.157 | 0.107 | 0.0636 | 0.0431 | 0.0333 | 0.0283 | 0.0263 | 0.0268 | 0.0295 | 0.0347 |
| attn.qkv | 1.80 | 1.68 | 1.32 | 0.910 | 0.581 | 0.403 | 0.315 | 0.270 | 0.250 | 0.250 | 0.265 | 0.299 |
| attn.proj | 2.33 | 3.25 | 2.33 | 1.48 | 0.904 | 0.619 | 0.478 | 0.401 | 0.358 | 0.340 | 0.342 | 0.371 |
| norm2 | 0.182 | 0.161 | 0.0956 | 0.0504 | 0.0340 | 0.0242 | 0.0189 | 0.0161 | 0.0145 | 0.0144 | 0.0161 | 0.0208 |
| mlp.fc1 | 1.53 | 1.43 | 1.14 | 0.851 | 0.675 | 0.533 | 0.440 | 0.379 | 0.341 | 0.327 | 0.341 | 0.404 |
| mlp.fc2 | 2.12 | 2.33 | 1.90 | 1.37 | 1.07 | 0.814 | 0.642 | 0.552 | 0.510 | 0.500 | 0.524 | 0.598 |
| ‖x_t‖ | 11.2 | 12.8 | 15.2 | 18.8 | 23.2 | 28.2 | 33.7 | 39.5 | 45.6 | 51.9 | 58.3 | 64.8 |

**b0-trained** (12 loops)

| group | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 | t11 | t12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block | 1.62 | 1.77 | 1.43 | 1.09 | 0.856 | 0.705 | 0.615 | 0.555 | 0.504 | 0.458 | 0.429 | 0.441 |
| norm1 | 0.213 | 0.167 | 0.123 | 0.0879 | 0.0640 | 0.0499 | 0.0412 | 0.0353 | 0.0316 | 0.0301 | 0.0311 | 0.0366 |
| attn.qkv | 0.601 | 0.690 | 0.572 | 0.452 | 0.340 | 0.260 | 0.209 | 0.178 | 0.158 | 0.144 | 0.138 | 0.146 |
| attn.proj | 0.532 | 0.715 | 0.544 | 0.415 | 0.297 | 0.224 | 0.185 | 0.161 | 0.140 | 0.122 | 0.111 | 0.109 |
| norm2 | 0.116 | 0.104 | 0.0738 | 0.0520 | 0.0389 | 0.0315 | 0.0269 | 0.0234 | 0.0205 | 0.0185 | 0.0177 | 0.0198 |
| mlp.fc1 | 0.833 | 0.934 | 0.789 | 0.596 | 0.477 | 0.409 | 0.367 | 0.340 | 0.317 | 0.296 | 0.285 | 0.303 |
| mlp.fc2 | 1.11 | 1.11 | 0.888 | 0.667 | 0.543 | 0.457 | 0.403 | 0.365 | 0.328 | 0.292 | 0.264 | 0.261 |
| ‖x_t‖ | 13.3 | 14.9 | 17.5 | 20.4 | 23.4 | 26.2 | 28.9 | 31.5 | 34.1 | 36.6 | 39.1 | 41.6 |

**d-init** (10 loops of B2, after B0 and B1)

| group | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|
| block | 0.663 | 0.834 | 1.12 | 1.40 | 1.45 | 1.30 | 1.24 | 1.25 | 1.47 | 2.26 |
| norm1 | 0.0518 | 0.0544 | 0.0740 | 0.0819 | 0.0769 | 0.0595 | 0.0468 | 0.0436 | 0.0498 | 0.100 |
| attn.qkv | 0.287 | 0.366 | 0.484 | 0.554 | 0.568 | 0.531 | 0.489 | 0.488 | 0.560 | 1.02 |
| attn.proj | 0.330 | 0.415 | 0.510 | 0.595 | 0.668 | 0.590 | 0.527 | 0.523 | 0.579 | 0.820 |
| norm2 | 0.0313 | 0.0483 | 0.0579 | 0.0504 | 0.0484 | 0.0444 | 0.0461 | 0.0534 | 0.0620 | 0.113 |
| mlp.fc1 | 0.363 | 0.463 | 0.614 | 0.738 | 0.758 | 0.694 | 0.701 | 0.747 | 0.925 | 1.45 |
| mlp.fc2 | 0.336 | 0.413 | 0.611 | 0.869 | 0.869 | 0.760 | 0.722 | 0.699 | 0.804 | 1.12 |
| ‖x_t‖ | 7.8 | 6.9 | 6.7 | 7.5 | 8.3 | 9.1 | 10.2 | 11.7 | 13.4 | 15.2 |

**d-trained** (10 loops of B2)

| group | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|
| block | 1.10 | 0.820 | 0.728 | 0.659 | 0.569 | 0.467 | 0.406 | 0.365 | 0.343 | 0.374 |
| norm1 | 0.0829 | 0.0793 | 0.0732 | 0.0568 | 0.0430 | 0.0337 | 0.0280 | 0.0251 | 0.0245 | 0.0322 |
| attn.qkv | 0.530 | 0.381 | 0.334 | 0.288 | 0.250 | 0.212 | 0.184 | 0.162 | 0.148 | 0.165 |
| attn.proj | 0.603 | 0.377 | 0.295 | 0.256 | 0.225 | 0.182 | 0.149 | 0.123 | 0.103 | 0.102 |
| norm2 | 0.0599 | 0.0529 | 0.0474 | 0.0393 | 0.0308 | 0.0231 | 0.0188 | 0.0169 | 0.0165 | 0.0196 |
| mlp.fc1 | 0.528 | 0.456 | 0.424 | 0.388 | 0.331 | 0.272 | 0.241 | 0.226 | 0.225 | 0.258 |
| mlp.fc2 | 0.527 | 0.411 | 0.379 | 0.360 | 0.313 | 0.254 | 0.222 | 0.200 | 0.183 | 0.184 |
| ‖x_t‖ | 12.1 | 12.9 | 13.9 | 14.9 | 16.1 | 17.2 | 18.1 | 19.0 | 20.0 | 21.0 |

### Summary per model and group

| model | group | spread | dominance | agreement | pairwise cos |
|---|---|---|---|---|---|
| b0-init | **block** | **6.26** | 0.386 | 0.505 | 0.305 |
| b0-init | norm1 | **22.65** | 0.607 | 0.362 | 0.312 |
| b0-init | attn.qkv | 7.19 | 0.417 | 0.507 | 0.332 |
| b0-init | attn.proj | 9.57 | 0.423 | 0.479 | 0.329 |
| b0-init | norm2 | **12.67** | 0.529 | 0.412 | 0.282 |
| b0-init | mlp.fc1 | 4.69 | 0.353 | 0.525 | 0.290 |
| b0-init | mlp.fc2 | 4.65 | 0.344 | 0.524 | 0.293 |
| b0-trained | block | 4.13 | 0.323 | 0.509 | 0.255 |
| b0-trained | norm1 | 7.08 | 0.418 | 0.505 | 0.309 |
| b0-trained | attn.qkv | 5.00 | 0.332 | 0.514 | 0.284 |
| b0-trained | attn.proj | 6.54 | 0.354 | 0.495 | 0.278 |
| b0-trained | norm2 | 6.53 | 0.405 | 0.513 | 0.309 |
| b0-trained | mlp.fc1 | 3.27 | 0.297 | 0.526 | 0.256 |
| b0-trained | mlp.fc2 | 4.25 | 0.331 | 0.497 | 0.243 |
| d-init | **block** | **3.41** | 0.287 | 0.635 | 0.350 |
| d-init | norm1 | 2.30 | 0.285 | 0.608 | 0.307 |
| d-init | attn.qkv | 3.56 | 0.297 | 0.632 | 0.347 |
| d-init | attn.proj | 2.49 | 0.268 | 0.663 | 0.383 |
| d-init | norm2 | 3.62 | 0.316 | 0.638 | 0.359 |
| d-init | mlp.fc1 | 4.00 | 0.319 | 0.627 | 0.347 |
| d-init | mlp.fc2 | 3.33 | 0.276 | 0.631 | 0.343 |
| d-trained | block | 3.21 | 0.329 | 0.566 | 0.284 |
| d-trained | norm1 | 3.38 | 0.339 | 0.596 | 0.338 |
| d-trained | attn.qkv | 3.58 | 0.343 | 0.558 | 0.279 |
| d-trained | attn.proj | 5.88 | 0.406 | 0.521 | 0.268 |
| d-trained | norm2 | 3.63 | 0.347 | 0.600 | 0.347 |
| d-trained | mlp.fc1 | 2.35 | 0.294 | 0.583 | 0.293 |
| d-trained | mlp.fc2 | 2.88 | 0.309 | 0.576 | 0.291 |

### What it shows

- **b0-init is MiniViT's picture.** The first 3 loops carry the biggest pushes (whole block 4.0, 4.6, 3.5 vs ~0.75
  from loop 8 on), and the LayerNorms are the most uneven: norm1's gradient at loop 1 is 23x its smallest one
  (loops 9-10), and the top 2 loops carry 61% of norm1's total (flat would be 17%). Meanwhile the residual state
  grows 5.8x (11 → 65), and a LayerNorm's gradient shrinks roughly as 1/‖x‖, so the late loops barely move the norms.
- **b0-trained is more even, but the same shape:** whole block 4.1, norm1 7.1, norm2 6.5; still falling with loop.
- **The T11-d layout (two free blocks in front) is much more even.** d-init: whole block 3.4, LayerNorms only 2.3
  and 3.6, and its gradient actually GROWS toward the last loop (loop 10 is the largest: 2.26 vs 0.66 at loop 1).
  Its residual state starts small (7.8) and only doubles. d-trained: 3.2, falling with loop, LayerNorms 3.4 / 3.6.
- **Loops do not fight each other.** Pairwise cos between loops is +0.24 to +0.38 everywhere, and each loop's
  gradient points partly along the total (agreement 0.36 to 0.66). The pushes are partly aligned, not cancelling.

**Decision line:** run H10 step 2 (b0-init: whole-block spread 6.26, largest LayerNorm spread 22.65 (norm1); d-init:
whole-block spread 3.41, largest LayerNorm spread 3.62 (norm2)).
Both init models cross a threshold on their own (b0-init on both; d-init on the whole block only, 3.41 ≥ 3). The
LayerNorm signal MiniViT warned about is strong only in the b0 x12 layout.

**Is one part of the loop training the shared block for everyone?** In b0 x12, yes, especially for the LayerNorms:
the first 2 to 3 loops carry most of the gradient (top-2 loops 39% of the whole block, 61% of norm1, vs 17% if flat),
because the residual grows 6x through the loop and shrinks the later loops' gradients. In the T11-d layout, no: the
loops share the work much more evenly (top-2 29 to 33% vs 20% flat), LayerNorms included. That fits per-loop
LayerNorms as a fix for b0 x12, and it suggests T11-d's free head blocks already remove most of the imbalance.

### Prediction check

| Prediction | Result | Verdict |
|---|---|---|
| In both init models the share falls with loop index, whole-block spread 2 to 5 | b0-init falls (after a peak at loop 2), spread 6.26; d-init RISES toward the last loop, spread 3.41 | wrong |
| norm1 and norm2 spreads larger than the whole-block spread | b0-init 22.7 / 12.7 vs 6.3, b0-trained 7.1 / 6.5 vs 4.1, d-trained 3.4 / 3.6 vs 3.2: yes; d-init 2.3 / 3.6 vs 3.4: no | mostly right (not d-init) |
| ‖x_t‖ grows ≥ 2x from loop 1 to the last loop | b0-init 5.8x, b0-trained 3.1x, d-init 1.95x, d-trained 1.7x | right for b0, wrong for d |
| Trained models are more even than init models | b0: 4.1 vs 6.3 (LayerNorms 7.1 vs 22.7) yes; d: 3.2 vs 3.4 whole, but norm1 3.4 vs 2.3 and attn.proj 5.9 vs 2.5 less even | right for b0, not for d |

### Plots and files

`results/grad_norm_per_loop.png`, `results/grad_norm_per_group.png`, `results/residual_norm_per_loop.png`; all numbers
in `results/grad_norms.json`. Made by `run.sh` → `grad_norms.py` (uses `unroll.py`), `plot.py`.
