# Tasks

One folder per task. Each folder's README has the goal, a done / not-done checklist, the exact commands, the
results and the takeaways. Shared code lives in `src/`; each README lists what it changed there.

| Task | What it asks | Status |
|---|---|---|
| [T1 · Baseline vs shared block](T1_baseline_vs_shared_block/) | Pretrained ViT-Tiny vs one block reused 12 times | done (2026-10-04) |
| [T2 · Shared block with LoRA](T2_shared_block_with_lora/) | Scratch control, avg init, higher lr, per-loop LoRA | done (2026-10-12) |
| [T3 · Progressive unrolling](T3_progressive_unrolling/) | Keep the first n-1 blocks, repeat the next one; accuracy vs n | done (2026-10-16) |
| [T4 · Longer runs at lr 5e-4](T4_longer_runs_lr5e-4/) | All shared-block variants + scratch for 30 epochs at lr 5e-4, with loss curves: under-trained or plateaued? | done (2026-10-19) |
| [T5 · CLS SVD per block](T5_cls_svd_per_block/) | Per-class CLS rank after each block (raw + centred SVD): does it collapse with depth? | pretrained done (2026-10-19); fine-tuned pending |
| [T6 · Unrolling at lr 5e-4, full curve](T6_unroll_lr5e-4_full_curve/) | Extend the lr 5e-4 unrolling curve to n=5..12: does it meet lr 1e-4? | runs pending (2026-10-22) |
| [T7 · Shared-block lr sweep](T7_shared_block_lr_sweep/) | iter b0 at lr 1e-3 / 2e-3 / 5e-3, 30 epochs: accuracy ceiling from lr alone | runs pending (2026-10-22) |
| [T8 · Collapse check and KL exit](T8_collapse_and_kl_exit/) | iter b0 vs baseline: do tokens collapse per loop? Do KL / CLS-change exits beat fixed 11 loops? | code ready, run pending (2026-10-22) |
