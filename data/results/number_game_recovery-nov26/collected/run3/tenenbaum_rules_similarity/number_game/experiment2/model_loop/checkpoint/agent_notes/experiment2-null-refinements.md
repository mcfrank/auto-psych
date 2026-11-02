---
name: experiment2-null-refinements
description: experiment2 iter_0 quick-NUTS (2x300) tests — lapse, recall, evidence exponent, parity kind all collapse to the base; calibration near perfect
metadata:
  type: project
---
experiment2 iter_0 (2026-10-30), candidate_5. Quick 2x300 NUTS lppd on the 4500 rows: incumbent gamma_width_exception_family_concepts -2269.2, gamma_width_parity_range_concepts -2270.0.
- Lapse+guess grafted on gamma_width_parity: lapse→0.011, -2270.6.
- example_recall_limited_concepts: recall 0.987, -2286.
- Evidence exponent on the likelihood (LogNormal(0,0.5)): β≈0.97, -2269.2 → would be a near-duplicate.
- Parity-within-range kind weight ≈0.004 on exp2 (unused); calibration by p-bin, set size and participant is near perfect; worst sets look like binomial noise (n=5 cells).
- Submitted doubling_chain_concepts + gamma width (gamma_width_doubling_chain_concepts): -2269.3, chain weight 0.004, novelty RMSE vs gamma_width_parity ≈0.0036 (barely above the 0.002 gate).
- Siblings 3 and 4 both submitted an inferred noise ALPHA this round.
Speed trick: a concept's likelihood depends only on (member count m, examples inside k), so tables count[set,k,group] and per-cell member counts make an evidence exponent cost 8 ms/grad in linear space (exp(β k (gain_m − max_gain)) per (k, group)).
**Why:** exp2 data cannot tell these refinements apart; gains must come from new stimuli.
**How to apply:** don't retry these on exp2; see [[experiment2-data-structure]], [[round4-null-refinements]].
