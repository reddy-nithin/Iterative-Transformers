# T36 · Is looping worth it? Train 2+1 (no loop) and 2+4 (4 loops), compare with 2+10

From: planning session 2026-11-16 (Fable review)
Notion: https://notion.so/41a38c734705815a98d6d8cd7645cc2e

## Goal
Two training runs that differ from T11-d (2+10: B0, B1, then B2 x10) ONLY in how many times B2 is applied, during
training and at test. Same parameters (1,539,940) in all three.
- T36-a (2+1) says what loops 2..10 buy over a single pass with the same parameters.
- T36-b (2+4) says whether 4 loops trained as 4 match 10 loops (T13 and T31 suggest the model answers by loop 4).

## Runs
Recipe identical to T11-d: pretrained timm `vit_tiny_patch16_224`, CIFAR-100 224 px, AdamW lr 5e-4 all params,
wd 0.05, 1 warmup epoch + cosine over 30 epochs, label smoothing 0.1, batch 128, AMP, seed 0, loss at the last loop
only, nothing frozen.

| Run | Layout | Flags | Checkpoint |
|---|---|---|---|
| T11-d (reference) | 2+10 | `--model unroll --unique-blocks 3` | `T11-d_unroll_n3_30ep.pt` |
| T36-a | 2+1 (no loop) | `--model unroll --unique-blocks 3 --body-loops 1` | `T36-a_2p1_noloop_30ep.pt` |
| T36-b | 2+4 | `--model unroll --unique-blocks 3 --body-loops 4` | `T36-b_2p4_30ep.pt` |

    CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T36_truncation_controls/run.sh T36-a   # 1 epoch, --subset 512
    CUDA_VISIBLE_DEVICES=N bash tasks/T36_truncation_controls/run.sh T36-a           # full run

New flag `--body-loops N` (src.train, src.models.build_unrolled): apply the repeated block N times. Default (not set)
= 12 - unique_blocks + 1, i.e. exactly the old behaviour for every `--unique-blocks` (10 for n = 3). The task text says
"default 10"; that is only true for n = 3, so the default is "as before" instead. Test:
`PYTHONPATH=. .venv/bin/python -m pytest tasks/T36_truncation_controls -q` (default = old layout logits within 1e-5;
N = 4 calls B2 exactly 4 times; params 1,539,940 for N = 1, 4, 10).

Note: 2+1 has no repeated block, so per-loop tools (`src.per_loop_readout.split_loop`, `--readout-loss`,
`--random-loops`, T34/T35-style analyses) do not apply to T36-a; `src.train` rejects those combinations.

## Decision rules (fixed in advance)
- T36-b within 0.8 pt of 73.79 (T11-d 3-seed mean): "2+4 matches 2+10 at about half the FLOPs (needs seeds before we
  switch the default; Nithin decides with Dr. Uddin)". More than 2 pt below: "late loops matter when trained for".
  Between: "unclear, run seeds".
- "Looping adds X pt over a single pass at equal parameters" with X = 73.79 - T36-a.
- Seed spread is 0.3 to 0.8 pt (T15): a one-seed difference under 0.8 pt is not an effect.

## Predictions (written before running)
T36-a 55 to 63. T36-b 72.3 to 74.0.

## Results
(pending)
