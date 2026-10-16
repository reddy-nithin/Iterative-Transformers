# Tasks

One folder per task. Each folder's README has the goal, a done / not-done checklist, the exact commands, the
results and the takeaways. Shared code lives in `src/`; each README lists what it changed there.

| Task | What it asks | Status |
|---|---|---|
| [T1 · Baseline vs shared block](T1_baseline_vs_shared_block/) | Pretrained ViT-Tiny vs one block reused 12 times | done (2026-10-04) |
| [T2 · Shared block with LoRA](T2_shared_block_with_lora/) | Scratch control, avg init, higher lr, per-loop LoRA | done (2026-10-12) |
| [T3 · Progressive unrolling](T3_progressive_unrolling/) | Keep the first n-1 blocks, repeat the next one; accuracy vs n | done (2026-10-16) |
