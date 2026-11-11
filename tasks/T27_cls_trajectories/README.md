# T27 · CLS trajectories per loop: PCA paths and distance to final CLS (no training)

**From:** advisor meeting 2026-11-05 (priority raised again on 2026-11-10: Dr. Uddin wants to see that CLS stops
changing after a few loops).
Notion task: https://notion.so/40f38c73470581e79fe0d71c52fa1f9d

**Goal.** See how CLS evolves through the loops and find where classes split apart ("bifurcation", Dr. Shuhao). Draw it
the way Huginn did (its Fig. 11 and 12), on our models.

## Setup

- **Checkpoints:** `T11-d_unroll_n3_30ep.pt` (74.11) and `T13-c_unroll_readout_every_loop.pt` (72.82), in
  `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`. Before any geometry number, the script checks each model's own
  top-1 on the full test set against these recorded values.
- **Images:** 50 TEST images per class (the first 50 of that class in the CIFAR-100 test set), 10 classes, 224 px,
  test transform, `model.eval()`:
  - the four T5 classes: apple (0), bicycle (8), cloud (23), fox (34);
  - six more drawn with seed 0 from the remaining 96 labels (`random.Random(0).sample(sorted(remaining), 6)`):
    bee (6), house (37), orange (53), pear (57), raccoon (66), rocket (69).
- **Stages:** s = 0 the input CLS (token 0 after the position embedding), 1 after B0, 2 after B1, 3..12 after loops
  1..10. Raw CLS, before the final norm (as Huginn used the raw state). Collected with T5's
  `collect_cls_per_layer(include_final_norm=False)`.
- **(a) PCA paths.** PCA on all 500 images × 13 stages. PC1-PC2, PC3-PC4, PC5-PC6: one line per image, coloured by
  class, light to dark along the stages, class mean at the last stage marked. Also the same with each class's mean path
  only. Variance share of PC1 to PC6.
- **(b) Distance-to-final heatmap.** For 100 images (10 per class), ‖CLS_s − CLS_12‖ / ‖CLS_12‖ per stage.
- **(c) Split step.** For each of the 45 class pairs and stage s = 1..12:
  sep_ab(s) = ‖μ_a(s) − μ_b(s)‖ / (ρ_a(s) + ρ_b(s)), μ = class mean, ρ = RMS distance of the class's images to its mean.
  Split step = first stage with sep_ab ≥ 1, else "none". Stage 0 is skipped (ρ = 0, identical means).

## Checklist

- [ ] README + predictions committed before the first run
- [ ] code + tests (`test_t27_trajectories.py`) pass
- [ ] smoke run (`SMOKE=1`)
- [ ] full run, top-1 sanity check passes for both models
- [ ] plots `results/pca_paths_<model>.png`, `results/pca_mean_paths_<model>.png`, `results/dist_to_final_<model>.png`,
      `results/split_steps.png`
- [ ] split-step table, answer line, prediction check, inbox entry

## Commands

Repo root, one idle GPU (no training: a few minutes):

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T27_cls_trajectories -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T27_cls_trajectories/run.sh
CUDA_VISIBLE_DEVICES=N bash tasks/T27_cls_trajectories/run.sh
```

## Predictions (written before running)

Copied verbatim from the task spec:

- most pairs split by stage 3 (loop 1), the rest by loop 4; look-alike pairs split last; mean paths look like straight
  converging lines with no orbits; T13-c settles (distance to final < 0.1) by loop 6, T11-d later. Low confidence.

## Results

_Pending._
