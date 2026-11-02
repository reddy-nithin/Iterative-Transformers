# T19 · Where does each CLS update come from? Attention vs MLP, mean vs deviation; collinear halting rule

**Goal.** Gemini's Round 3 claim: because CLS attention is nearly flat, the attention part of every CLS update is
mostly the plain average of the patch values, so every loop pushes CLS the same way. Nobody has measured it, and it
ignores the MLP part of the update. Measure both before spending GPU time on a fix (T20), and test a free halting rule
on checkpoints we already have. No training.

**From:** planning session 2026-11-02 (Nithin), after the T11 to T14 results (T19 added after Gemini Round 3). Due Thu 2026-11-05.
Notion task: https://notion.so/40c38c734705811d9949f696c4c66500

## Setup
Checkpoints (all T11-d layout: B0, B1, then B2 x10, no tail): `T11-d_unroll_n3_30ep.pt`,
`T13-c_unroll_readout_every_loop.pt`, `T13-d_unroll_loss_margin.pt`. CIFAR-100 test set, all 10,000 images, eval mode,
fp32. Loops t = 1..10 are the 10 applications of B2; x_0 = the output of B1.

- **(a)** One B2 application: x' = x + attn(norm1(x)), x'' = x' + mlp(norm2(x')). CLS update d_t = a_t + m_t,
  a_t = attention output row 0 (after `attn.proj`), m_t = MLP output row 0. B2 in `vit_tiny_patch16_224` has no
  LayerScale (`ls1`, `ls2` are Identity) and drop-path is Identity in eval; the script asserts both.
- **(b)** Per head h, CLS-row weights A_{h,j} over the 197 keys and values V_{h,j}: self part A_{h,0} V_{h,0}, mean part
  s_h · V̄_h, deviation part Σ_i (A_{h,i} − s_h/196) V_{h,i}, with s_h = Σ_{i≥1} A_{h,i} and V̄_h = mean patch value.
  Heads are concatenated and the proj weight (no bias) is applied to each part; the bias is reported on its own. The
  three parts + bias must equal a_t to 1e-4 (asserted on every batch).
- **(c)** Halting rule on the per-loop readout (final norm + classifier on x_t[:, 0], as in T13): stop at the first
  t ≥ 3 with cos(d_t, d_{t−1}) ≥ θ AND ‖d_t‖ / ‖x_{t−1}[:, 0]‖ ≤ ρ; θ ∈ {0.80, 0.85, 0.90, 0.95}, ρ ∈ {0.10, 0.20,
  0.30, no limit}; images that never stop use loop 10. Compared with T13's fixed-depth curve and its three rules.

Per-loop numbers are means over images of the per-image values. Code: `decompose.py` (analysis), `plot.py` (plots),
`test_t19_decomposition.py`.

Run (repo root): `PYTHONPATH=. .venv/bin/python tasks/T19_cls_update_decomposition/decompose.py CKPT ...`

## Predictions (written before running)
Gemini predicts mean share ≥ 0.9. Mac Claude predicts the MLP part is at least
half of ‖d_t‖ in late loops and is the most collinear part (cos ≥ 0.95). The halting rule beats loop 10 on T13-c
(because T13-c peaks at loop 6: 73.5 vs 72.82) by ≤ 0.7 pts and does not beat fixed depth by more than T13's best rule.
Low confidence on all.

## Decision line rule (from the spec)
"run T20" if, on T11-d, the mean share averaged over loops 6 to 10 is ≥ 0.5 AND ‖a_t‖ / ‖d_t‖ averaged over loops 6 to
10 is ≥ 0.3; otherwise "skip T20" with the two numbers.

## Results (all 10,000 test images, fp32; full tables in `results/tables.md`)
Checks: parts + bias = a_t to ≤ 1.3e-6 and a_t + m_t = the real CLS update to ≤ 2e-6 on every batch; each model's
loop-10 readout equals its own top-1 (74.12 / 72.82 / 73.19 vs logged 74.11 / 72.82 / 73.17, fp32 vs AMP); our
cos(d_t, d_{t−1}) matches T12 (c) within 0.0028 / 0.0054 / 0.0016 (tolerance 0.02). B2 has no LayerScale and no
drop-path in eval (asserted), so nothing extra had to be folded into a part. proj bias norm ≈ 2.64 in all three.

**(a) + (b), averaged over loops 6 to 10**

| model | ‖a‖/‖d‖ | ‖m‖/‖d‖ | mean share | self share | cos a | cos m | cos d | cos mean | cos dev |
|---|---|---|---|---|---|---|---|---|---|
| T11-d | 0.917 | 1.048 | 0.539 | 0.050 | 0.935 | 0.922 | 0.880 | 0.928 | 0.946 |
| T13-c | 1.226 | 1.318 | 0.678 | 0.036 | 0.991 | 0.995 | 0.989 | 0.992 | 0.992 |
| T13-d | 1.139 | 1.282 | 0.615 | 0.040 | 0.994 | 0.996 | 0.992 | 0.994 | 0.994 |

(self share = ‖W·self‖ / (‖W·self‖ + ‖W·mean‖ + ‖W·dev‖); cos columns are successive-loop cos, mean over t = 6..10.)

- Attention and MLP parts are each about as large as the whole update or larger (‖a‖/‖d‖ and ‖m‖/‖d‖ ≈ 0.9 to 1.5),
  so they partly cancel: from the per-loop means, cos(a_t, m_t) ≈ −0.5 at loop 10 on T11-d and ≈ −0.75 on T13-c
  (approximate, computed from the mean ratios, not measured per image).
- Inside the attention part the mean part is only slightly larger than the deviation part (mean share 0.54 to 0.68),
  far from Gemini's ≥ 0.9. The CLS-self part is small after loop 3 (≤ 5%).
- Late loops repeat the same push in every part: on T13-c / T13-d every part has successive cos ≥ 0.99; on T11-d
  0.92 to 0.95, with the full update (0.88) a bit less collinear than its parts.

**(c) Collinear halting rule** (`results/exit_curves_collinear_rule.png`): best gain over the fixed-depth curve is +0.04
(T11-d), +0.14 (T13-c), +0.18 (T13-d), versus T13's best rules +3.74 / +0.79 / +0.67 (T11-d's large confidence gain
comes from its steep fixed curve). Best top-1 on T13-c: 73.57 at 5.97 loops (θ 0.95, ρ 0.2), +0.75 over loop 10 (72.82),
i.e. it recovers the loop-6 peak (73.53). ρ = 0.1 almost never fires on T11-d (its updates stay > 10% of ‖CLS‖).

Plots: `results/update_split_per_loop.png`, `results/successive_cos_parts.png`, `results/exit_curves_collinear_rule.png`.

**Decision line:** run T20
(T11-d: mean share averaged over loops 6 to 10 = 0.539 ≥ 0.5 AND ‖a_t‖/‖d_t‖ averaged over loops 6 to 10 = 0.917 ≥ 0.3.)

**Is the late-loop push mostly attention-mean, attention-deviation, or MLP?** None of them alone: the MLP part is the
largest (‖m‖ ≈ 1.0 to 1.3 × ‖d‖) but the attention part is almost as large and points partly against it; within
attention the mean and deviation parts are roughly equal (mean share 0.54 to 0.68), and all parts repeat the same
direction loop after loop (cos ≥ 0.92 on T11-d, ≥ 0.99 on T13-c/d).

### Prediction check
- Gemini, mean share ≥ 0.9: **wrong** (0.54 / 0.68 / 0.62).
- Mac Claude, MLP part ≥ half of ‖d_t‖ in late loops: **right** (≥ 0.97 everywhere). MLP the most collinear part with
  cos ≥ 0.95: **right on T13-c/d** (0.995, 0.996, highest), **wrong on T11-d** (0.922, the deviation part 0.946 and the
  attention part 0.935 are more collinear).
- Halting rule beats loop 10 on T13-c by ≤ 0.7 pts: **just missed** (+0.75 at the best grid point; the sign is right).
- Does not beat fixed depth by more than T13's best rule: **right** (+0.14 vs +0.79 on T13-c; same on the others).
