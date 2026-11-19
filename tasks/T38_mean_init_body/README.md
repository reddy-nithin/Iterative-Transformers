# T38 · Start the looped block as the average of blocks 3 to 12, not block 3

From: advisor meeting 2026-11-19
Notion: https://notion.so/41d38c734705810fb0a5d78ba8c4df01

## Goal
In 2+10 (T11-d: B0, B1, then one block applied 10 times) the looped block has to stand in for the ten pretrained
blocks it replaces. Today it starts as a copy of pretrained block 2 (0-based; "block 3" in 1-based counting).
Dr. Uddin's idea: start it as the element-wise average of those ten blocks (0-based 2 to 11) instead. One run,
compared with T11-d.

"Element-wise mean" means: every tensor of the looped block (attention qkv and proj weights and biases, MLP fc1/fc2
weights and biases, both LayerNorms' weight and bias) is set to the average of the same tensor over blocks 2..11.
Everything else (B0, B1, patch/position embedding, CLS token, final_norm, classifier) starts exactly as in T11-d.

## Runs
Recipe identical to T11-d: pretrained timm `vit_tiny_patch16_224`, CIFAR-100 224 px, AdamW lr 5e-4 all params,
wd 0.05, 1 warmup epoch + cosine over 30 epochs, label smoothing 0.1, batch 128, AMP, seed 0, loss at the last loop
only, nothing frozen.

| Run | Looped block starts as | Flags | Checkpoint |
|---|---|---|---|
| T11-d (reference) | pretrained block 2 | `--model unroll --unique-blocks 3` | `T11-d_unroll_n3_30ep.pt` |
| T38-a | mean of pretrained blocks 2..11 | `--model unroll --unique-blocks 3 --body-init-range 2 11` | `T38-a_2p10_meaninit_30ep.pt` |

Checkpoints in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`. T11-d's per-epoch history (for the comparison and
the loss-curve plot): `tasks/T11_pretrained_hbt_body_sweep/results/unroll_T11-d.json`.

    CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T38_mean_init_body/run.sh T38-a   # 1 epoch, --subset 512
    CUDA_VISIBLE_DEVICES=N bash tasks/T38_mean_init_body/run.sh T38-a           # full run

New flag `--body-init-range START END` (src.train, src.models.build_unrolled): the repeated block starts as the
element-wise mean of pretrained blocks START..END (inclusive, 0-based). Default (not set) = pretrained block
`unique_blocks - 1`, exactly as before; old checkpoints rebuild unchanged. Plain unrolled layout only (not with the
T14/T16/T20/T28/T29 options). The existing `--init avg` is different: it averages all 12 blocks and only for the old
fully shared `iterative` / `lora` models. Test:
`PYTHONPATH=. .venv/bin/python -m pytest tasks/T38_mean_init_body -q` (2..2 equals block 2 exactly; 2..11 matches a
hand-computed mean within 1e-6 for `attn.qkv.weight` and `norm1.bias`; params stay 1,539,940; B0 and B1 unchanged; the
flag is saved in the checkpoint kwargs only when set). The test uses the ViT-Tiny architecture with random weights
(no download); the checks hold for any weights.

Smoke test (1 epoch, `--subset 512`, GPU 3): runs, params 1,539,940, `body_init_range: [2, 11]` in the run json.

## Decision rule (fixed in advance)
Compared with T11-d's 3-seed mean 73.79 (seed 0: 74.11):
- T38-a ≥ 74.6 (0.8 above 73.79): "mean start helps, run seeds 1 and 2 before claiming it".
- Within 0.8 of 73.79: "tie".
- Below 73.0: "mean start hurts".
Seed spread is 0.3 to 0.8 pt (T15): a one-seed difference under 0.8 pt is not an effect.

## Predictions (written before running)
T38-a 72.0 to 74.5, so a tie or slightly worse. Reason: in the old fully shared model (one block x12) an
all-12-block mean start scored 52.96 vs block 0's 58.06. Epoch-1 top-1 lower than T11-d's.

## Results
(pending)
