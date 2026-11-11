# T25 · Class-cloud exit: stop looping when CLS reaches a class center (no training)

**From:** advisor meeting 2026-11-05.
Notion task: https://notion.so/40f38c73470581ac96aace2ffa9f534b

**Goal.** Test Dr. Uddin's class-cloud stopping rule on models we already have. Each class's CLS vectors should sit in a
cloud around a center; if an image's CLS is already inside a cloud at loop t, the remaining loops are wasted. Also
answer: can ONE set of centers serve every loop (only possible because our loop reuses one block), or does each loop
need its own (as in DE3-BERT, where every layer has its own prototypes)?

## Setup

- **Checkpoints:** `T13-c_unroll_readout_every_loop.pt` (main: every loop was trained to answer; 72.82) and
  `T11-d_unroll_n3_30ep.pt` (74.11), in `/storage/INSLAB/Data/nrs3qy/checkpoints/itertf/`. Each model's own loop-10
  top-1 on the full test set is checked against its logged value first.
- **Feature:** z_t = final_norm(x_t)[:, 0] at loop t = 1..10 (the vector the classifier reads). Distance: cosine
  distance dist(a, b) = 1 − cos(a, b). Images at 224 px, test transform, `model.eval()`, fp32.
- **Centers** from all 50,000 TRAIN images (test transform, fixed order), true labels:
  - per-loop centers μ_c^(t) = mean of z_t over class-c training images, t = 1..10;
  - shared centers μ_c^(10), used at every loop;
  - radius r_c(q) = q-quantile of dist(z_10, μ_c^(10)) over class-c training images, q ∈ {0.25, 0.5, 0.75}.
  - Saved as `results/centers_<model>.pt` (T30 reuses it).
- **(a) NCC per loop** on the 10,000 TEST images: NCC top-1 with per-loop centers, with shared centers, the readout's
  top-1, and the mismatch (share of images where per-loop NCC and the readout disagree).
- **(b) Exit rules** on TEST, at loops t ≥ 2 (d1, d2 = distance to nearest / second-nearest center, c1 = nearest
  class; the prediction at exit is the readout at that loop; images that never stop use loop 10):
  - R1 radius: d1 ≤ r_{c1}(q), q ∈ {0.25, 0.5, 0.75};
  - R2 ratio: d1 / d2 ≤ τ, τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9};
  - R3 agree: R2 AND c1 = the readout's top-1 class.
  - Every rule with per-loop and with shared centers. Gain over fixed depth at matched mean loops computed as in T13 /
    T19 (`with_gain` / `compare_with_fixed` in `tasks/T13_per_loop_readout_loss/exit_analysis.py`).
- **Reference:** T13's best rule on T13-c = patience m = 1, gain +0.79 (`tasks/T13_per_loop_readout_loss/results/exit_analysis.json`).

## Checklist

- [x] README + predictions committed before the first run
- [x] code + tests (`test_t25_ncc.py`) pass
- [x] smoke run (`SMOKE=1`)
- [x] full run, top-1 sanity check passes for both models
- [x] tables (a) and (b), plots `results/ncc_vs_readout_per_loop.png`, `results/exit_curves_class_cloud.png`
- [x] both decision lines, answer line, prediction check, inbox entry

## Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest tasks/T25_class_cloud_exit -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T25_class_cloud_exit/run.sh
CUDA_VISIBLE_DEVICES=N bash tasks/T25_class_cloud_exit/run.sh
```

## Predictions (written before running)

Copied verbatim from the task spec:

- NCC top-1 at loop 10 is within 1 pt of the readout (Dr. Shuhao: kNN on this feature is slightly better than the head).
- Per-loop centers beat shared centers by > 5 pts at loops 1 to 3 and are within 1 pt at loops 6 to 10 on T13-c.
- R2 or R3 beats R1 (DE3-BERT: the ratio is more reliable than raw distance).
- On T13-c the best class-cloud gain over fixed depth is ≤ +0.79 (does not beat T13's best rule). Low confidence.

## Results

Run 2026-11-11, one GPU (~3 min). Sanity check: loop-10 top-1 T13-c 72.82 (logged 72.82), T11-d 74.12 (logged
74.11). Centers from all 50,000 train images, evaluated on all 10,000 test images. Note: the radii r_c(q) are
measured at loop 10 (as specified) and used with both center modes.

### (a) Nearest class center vs readout, per loop (test top-1, %)

| loop | T13-c NCC per-loop | T13-c NCC shared | T13-c readout | T13-c mismatch | T11-d NCC per-loop | T11-d NCC shared | T11-d readout | T11-d mismatch |
|---|---|---|---|---|---|---|---|---|
| 1 | 42.49 | 32.44 | 49.00 | 37.85 | 21.92 | 2.04 | 1.22 | 98.61 |
| 2 | 60.70 | 60.18 | 66.02 | 25.19 | 29.34 | 4.11 | 3.79 | 96.25 |
| 3 | 66.92 | 66.65 | 71.38 | 19.50 | 39.46 | 11.01 | 11.30 | 88.53 |
| 4 | 68.89 | 68.65 | 72.93 | 16.95 | 50.96 | 25.23 | 29.61 | 68.28 |
| 5 | 69.55 | 69.59 | 73.27 | 16.28 | 59.65 | 43.39 | 49.67 | 46.54 |
| 6 | 69.79 | 69.67 | 73.53 | 16.03 | 65.19 | 57.37 | 62.56 | 31.56 |
| 7 | 69.71 | 69.75 | 73.36 | 15.87 | 68.20 | 65.82 | 69.60 | 23.05 |
| 8 | 69.77 | 69.62 | 73.12 | 15.89 | 69.81 | 68.84 | 73.12 | 18.65 |
| 9 | 69.64 | 69.48 | 73.04 | 15.78 | 70.08 | 69.90 | 74.25 | 17.23 |
| 10 | 69.31 | 69.31 | 72.82 | 15.79 | 70.03 | 70.03 | 74.12 | 17.12 |

- NCC is **3.5 to 4 pt below the classifier** at loop 10 on both models (69.31 vs 72.82; 70.03 vs 74.12), and NCC and
  the classifier disagree on ~16 to 17% of test images.
- **T13-c: shared and per-loop centers agree from loop 2 on** (within 0.6 pt at every loop 2..10). Only loop 1 differs
  (42.5 vs 32.4). T13-c's CLS features at every loop live near the same class centers.
- **T11-d needs per-loop centers early:** at loops 1 to 6, per-loop centers beat shared ones by 8 to 25 pt, and per-loop
  NCC is far better than the readout there (e.g. loop 4: 50.96 vs 29.61). The class information is already in the
  early-loop features; T11-d's classifier just cannot read it before loop 7 (it was only trained on loop 10).

### (b) Exit rules: best point of each rule family (gain over fixed depth at the same mean loops)

| Rule (centers) | T13-c best point | acc | mean loops | gain | T11-d best point | acc | mean loops | gain |
|---|---|---|---|---|---|---|---|---|
| R1 radius (per-loop) | q 0.25 | 72.49 | 8.02 | −0.63 | q 0.25 | 3.85 | 2.00 | +0.06 |
| R2 ratio (per-loop) | τ 0.5 | 72.83 | 8.42 | −0.26 | τ 0.5 | 72.58 | 9.19 | −1.65 |
| R3 agree (per-loop) | τ 0.5 | 72.83 | 8.42 | −0.26 | τ 0.9 | 71.71 | 6.45 | **+5.98** |
| R1 radius (shared) | q 0.25 | 72.86 | 8.87 | −0.19 | q 0.25 | 74.07 | 9.43 | −0.12 |
| R2 ratio (shared) | τ 0.5 | 72.82 | 8.62 | −0.25 | τ 0.8 | 72.76 | 7.80 | +0.34 |
| R3 agree (shared) | τ 0.5 | 72.82 | 8.62 | −0.25 | τ 0.8 | 73.27 | 7.85 | +0.68 |
| *T13 reference* | *patience m=1* | *69.60* | *2.52* | *+0.79* | *confidence 0.3* | *69.19* | *6.41* | *+3.74* |

- **On T13-c, every class-cloud rule is at or below fixed depth** (best −0.19). The rules either stop late (8 to 9 mean
  loops, where the fixed curve has already peaked at loop 6 and is slowly falling) or, when they stop early, they lose
  accuracy faster than fixed depth does (plot).
- On T11-d, R3 with per-loop centers (τ 0.9) gains +5.98 at 6.45 mean loops, more than T13's best rule on T11-d
  (confidence, +3.74). T11-d's fixed-depth curve is very steep around loop 6, so gains there are easy to get; one seed.

**Decision line 1:** class-cloud exit does not beat T13's best rule (best class-cloud gain on T13-c −0.19, R1 radius
with shared centers, q = 0.25, vs T13's patience rule +0.79).

**Decision line 2:** one center set is enough (on T13-c, NCC with shared centers is within 1 pt of per-loop centers at
every loop 4..10; largest difference 0.24 pt).

**Is reaching a class cloud a better stopping signal than confidence?** No, not on T13-c, the model trained to answer
at every loop: no class-cloud rule beats fixed depth, and T13's simple patience rule (+0.79) stays the best. The class
clouds are real and stable (one shared set of centers serves loops 2 to 10), but the nearest center is a weaker
classifier than the trained head (−3.5 pt), so "inside a cloud" is a noisier signal than the head's own agreement.
On T11-d (trained on loop 10 only) the clouds are informative earlier than the head, which is why R3 helps there.

### Prediction check

| Prediction | Result | Verdict |
|---|---|---|
| NCC at loop 10 within 1 pt of the readout | NCC 3.5 (T13-c) and 4.1 (T11-d) pt BELOW the readout | wrong |
| Per-loop centers beat shared by > 5 pt at loops 1-3 on T13-c | only at loop 1 (+10.1); loops 2-3 within 0.6 | wrong (partly) |
| ... and are within 1 pt at loops 6-10 on T13-c | max difference 0.16 | right |
| R2 or R3 beats R1 | T13-c: R1 (shared) is best, all within 0.45 pt of each other; T11-d: R3 best by far | mixed |
| On T13-c the best class-cloud gain is ≤ +0.79 | −0.19 | right |

### Plots and files

`results/ncc_vs_readout_per_loop.png`, `results/exit_curves_class_cloud.png`. All numbers and every rule point in
`results/class_cloud_{T13-c,T11-d}.json`; centers and radii (for T30) in `results/centers_{T13-c,T11-d}.pt`
(`per_loop` (10, 100, 192), `shared` (100, 192), `radii` (100, 3), `quantiles`). Made by `run.sh` → `class_cloud.py`,
`plot.py`.
