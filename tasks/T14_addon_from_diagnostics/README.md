# T14 · One add-on chosen by T12's decision line: coverage mask

**Goal.** Add exactly one thing to the best looped model, picked by T12's measurements rather than by guessing. T12
showed that CLS keeps reading the same patches loop after loop (late-loop cos(A_t, A_t+1) 0.956 on T11-d, 0.995 on
T13-d). The coverage mask gives CLS a memory of which patches it has already read (H1).

**From:** advisor meeting 2026-10-27 (Dr. Uddin), due Tue 2026-11-03.
Notion task: https://notion.so/40938c7347058111b31ef3ba75360dd9

**Started:** 2026-10-30

## Base and add-on (why these)
- Base rule: the T13 winner on the T11-c layout, unless T13-c or T13-d beats both T13-a and T13-b by ≥2 pts. T13-d
  (73.17) beats T13-a / T13-b (68.85 each) by 4.3 pts, so the base is **T13-d**: unrolling n = 3 (B0, B1 distinct,
  then B2 x10, no tail), readout at every loop + loss margin (m 0.02, λ 1.0, floor 1.0, w_t = 0.85^(10−t)).
- Add-on from T12's decision line for T11-d: late cos 0.956 > 0.8, mean rel_cls / rel_patch 1.08 (not < 0.3) →
  **"T14 pick: coverage mask"**.
- Pretrained init exactly as in T11 / T13 (NOT from the T13-d checkpoint), shared recipe, 30 epochs, so the only
  difference from T13-d is the mask (+1 parameter).

## The coverage mask (run T14-a)
Only in the looped block's attention (B2), only on the CLS query row:
- per-image supply vector c ∈ R^196, c = 1 before loop 1;
- in loop t, add log(c_i + 1e-6) to the CLS→patch-i attention logits (all heads; the CLS→CLS logit and all patch
  query rows unchanged);
- after loop t: c ← clamp(c − η · A_t, min 0), A_t = CLS→patch attention averaged over heads (CLS→CLS dropped,
  renormalised, as in T12 (b));
- η = one learnable scalar, initialised at 0.5. Readout unchanged (final norm + classifier on x_t).

Code: `src/coverage_mask.py`, flag `--coverage-mask` for `--model unroll` in `src.train`.

| Run | Model | Loss | Reference | Checkpoint |
|---|---|---|---|---|
| T14-a | T13-d + coverage mask (B0, B1, masked B2 x10) | readout every loop + loss margin | T13-d (73.17) | `T14-a_coverage_mask.pt` |

## Checklist
- [x] Predictions committed before any run
- [ ] `src/coverage_mask.py`, `--coverage-mask`, tests `test_t14_coverage_mask.py`
- [ ] Smoke test
- [ ] T14-a, 30 epochs, one GPU, one tmux session
- [ ] T12 metrics (a)-(c), T13 exit curves, fixed-depth curve, flip rate, separation per loop; T13-d as reference
- [ ] Results row, plots, learned η, the line "did the add-on do what T12 predicted?"

## Predictions (written before running)
Coverage mask → T12 metric (b) drops below 0.5 and top-1 +1 to +3 over base. CLS adapters →
rel_cls/rel_patch doubles, top-1 +1 to +3. Per-loop LayerNorm → small change (≤1 pt). Low confidence on all.

## Commands
```bash
PYTHONPATH=. .venv/bin/python -m pytest tests tasks -q
CUDA_VISIBLE_DEVICES=N SMOKE=1 bash tasks/T14_addon_from_diagnostics/run.sh
tmux new -d -s itertf-T14-a "CUDA_VISIBLE_DEVICES=N bash tasks/T14_addon_from_diagnostics/run.sh"
.venv/bin/python tasks/T14_addon_from_diagnostics/plot.py
```

## Results
(after the run)
