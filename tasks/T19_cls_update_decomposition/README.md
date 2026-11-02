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

## Results
_Pending._
