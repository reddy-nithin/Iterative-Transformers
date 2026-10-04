# Learning Log

## Build phase (2026-10-04) — code written by Claude, to be toured after results
- Mode switched to build-first; a code walkthrough comes after the runs.
- ViT-Tiny: 12 identical-shaped blocks, 444,864 params each. Baseline with 100-class head = 5,543,716 params.
- Iterative (block 0 x12, one shared object): 650,212 params = 5,543,716 - 11 x 444,864. FLOPs identical (2.15 G) because the compute still happens 12 times; only storage shrinks.
- Gotcha: `[block] * 12` repeats the SAME object (shared). `copy.deepcopy` would silently make 12 independent blocks. tests/test_sharing.py guards this.
- Gotcha: both runs use the same recipe, seed and script; only `--model` differs.

## To tour later
models.py -> data.py -> metrics.py -> train.py -> tests/test_sharing.py
