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

- [x] README + predictions committed before the first run
- [x] code + tests (`test_t27_trajectories.py`) pass
- [x] smoke run (`SMOKE=1`)
- [x] full run, top-1 sanity check passes for both models
- [x] plots `results/pca_paths_<model>.png`, `results/pca_mean_paths_<model>.png`, `results/dist_to_final_<model>.png`,
      `results/split_steps.png`
- [x] split-step table, answer line, prediction check, inbox entry

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

Run 2026-11-11 on one GPU (a few minutes). Top-1 sanity check on the full test set: T11-d 74.12 (logged 74.11), T13-c
72.82 (logged 72.82). Top-1 on the 500 chosen images: T11-d 83.6, T13-c 81.6.

### Stage by stage (500 images)

Stages: in = input, then B0, B1, loops 1..10.

| | in | B0 | B1 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **T11-d** median dist to final | 1.10 | 1.05 | 1.05 | 1.02 | 0.99 | 0.95 | 0.88 | 0.78 | 0.65 | 0.49 | 0.32 | 0.16 | 0 |
| T11-d mean ‖CLS‖ | 13.3 | 13.8 | 14.1 | 13.0 | 12.6 | 12.5 | 13.2 | 14.7 | 16.3 | 18.0 | 19.8 | 21.6 | 23.2 |
| T11-d sep, median over 45 pairs | | 0.36 | 0.46 | 0.44 | 0.48 | 0.51 | 0.58 | 0.65 | 0.69 | 0.69 | 0.68 | 0.67 | 0.67 |
| **T13-c** median dist to final | 1.03 | 0.93 | 0.91 | 0.92 | 0.82 | 0.67 | 0.53 | 0.42 | 0.32 | 0.23 | 0.15 | 0.07 | 0 |
| T13-c mean ‖CLS‖ | 13.1 | 11.3 | 9.1 | 6.4 | 9.0 | 12.9 | 16.6 | 19.7 | 22.4 | 24.8 | 27.0 | 29.0 | 30.9 |
| T13-c sep, median over 45 pairs | | 0.32 | 0.44 | 0.55 | 0.54 | 0.60 | 0.64 | 0.65 | 0.66 | 0.66 | 0.66 | 0.65 | 0.65 |

- **PCA variance share PC1..PC6:** T11-d 14.9 / 8.9 / 6.2 / 5.4 / 4.4 / 4.0 %; T13-c 22.9 / 9.0 / 5.8 / 4.4 / 4.0 / 3.5 %.
- **Split step: "none" for all 45 pairs in both models.** The largest separation any pair reaches is 0.90 (T11-d) and
  0.87 (T13-c), so under this definition (sep ≥ 1) no pair ever splits. The closest pairs are the look-alikes:
  apple-pear, orange-pear, bee-raccoon, fox-raccoon (both models), cloud-rocket (T13-c).
- **Separation stops growing early.** The median sep rises through the loops and then plateaus: from loop 6 on in
  T11-d, and from about loop 4 to 5 in T13-c (0.64 at L4, 0.66 from L6). This matches T13's finding that T13-c's
  answer is ready by about loop 4 to 6.
- **The raw CLS never settles.** Distance to the final CLS falls steadily, roughly linearly, all the way to loop 10.
  Only at loop 9 is the median under 0.1 (T13-c 0.07; T11-d 0.16). Meanwhile ‖CLS‖ grows every loop from loop 3 on
  (T13-c 6.4 → 30.9).
- **PCA paths** (`pca_mean_paths_*.png`): every class's mean path is a nearly straight ray out from a shared start. The
  class picks its direction early (by about loop 2 to 3) and then keeps moving along it. There are no orbits and no
  turning back. The per-image plots (`pca_paths_*.png`) show the same fan.
- **Heatmaps** (`dist_to_final_*.png`): the same smooth gradient for every class and image; no class settles earlier
  than the others.

**Where do the classes split, and does CLS settle or keep moving?** No class pair reaches sep ≥ 1, so there is no
split step under this definition. Class separation stops improving after about loop 6 (T11-d) or loop 4 to 5
(T13-c). The raw CLS does not settle: after picking a class direction early, it keeps moving outward along that same
direction, with its norm still growing at loop 10. What changes late is the length of CLS, not which class it
points to. The final norm, which the classifier reads, removes most of that length. So "CLS keeps moving" (raw state)
and "the answer is ready by loop 4 to 6" (T13) are both true.

### Prediction check

| Prediction | Result | Verdict |
|---|---|---|
| Most pairs split by stage 3 (loop 1), the rest by loop 4 | no pair reaches sep ≥ 1 at any stage | wrong |
| Look-alike pairs split last | look-alikes are the least separated pairs (apple-pear, orange-pear, fox-raccoon, bee-raccoon) | right in spirit (none split) |
| Mean paths are straight converging lines with no orbits | straight, no orbits, but they DIVERGE outward from a shared start | half right |
| T13-c settles (dist to final < 0.1) by loop 6, T11-d later | T13-c under 0.1 only at loop 9; T11-d never before loop 10 (0.16 at loop 9); T11-d later than T13-c is right | wrong on timing |

### Plots

`results/pca_paths_{T11-d,T13-c}.png`, `results/pca_mean_paths_{T11-d,T13-c}.png`,
`results/dist_to_final_{T11-d,T13-c}.png`, `results/split_steps.png`. Numbers in `results/trajectories_*.json`, CLS
states in `results/cls_*.npz` (float16), console summary in `results/plot.log`. Made by `run.sh` → `trajectories.py`,
`plot.py`.
