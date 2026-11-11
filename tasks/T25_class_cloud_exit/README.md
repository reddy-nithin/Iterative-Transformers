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

- [ ] README + predictions committed before the first run
- [ ] code + tests (`test_t25_ncc.py`) pass
- [ ] smoke run (`SMOKE=1`)
- [ ] full run, top-1 sanity check passes for both models
- [ ] tables (a) and (b), plots `results/ncc_vs_readout_per_loop.png`, `results/exit_curves_class_cloud.png`
- [ ] both decision lines, answer line, prediction check, inbox entry

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

_Pending._
