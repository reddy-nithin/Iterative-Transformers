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

- [ ] README + predictions committed before the first run
- [ ] code + tests (`test_t23_per_loop_grads.py`) pass
- [ ] smoke run (`SMOKE=1`)
- [ ] full run: gate passes on every model, trained top-1 checks pass
- [ ] per-loop tables, summary table, plots `results/{grad_norm_per_loop,grad_norm_per_group,residual_norm_per_loop}.png`
- [ ] decision line, answer line ("Is one part of the loop training the shared block for everyone?"), prediction
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

_Pending._
