---
name: exp3-iter4-interior-example-weight
description: Exp3 iter-4 cand_5 — on set_size_scaled_evidence_weight every tweak null (parity prior, kind shift by n, fit-dependent lapse, LLO swap, interior ω); WOR/LLO swaps are dups; one-stray truncation -77, outside-only strays -24; interior-weight model submitted
metadata:
  type: project
---
Exp3 iter 4, candidate_5 (2026-10-31), refinement slot. Base is set_size_scaled_evidence_weight, MAP -3188.13 (kappa -0.03, lapse 0.45%, guess_yes→1).
- Parity-rule prior weight: +0.02. Rule-vs-range prior shift per example: 0. Lapse driven by the best concept's fit: lapse→0.
- Log-odds response map in place of the lapse: -1.6, RMSE 0.002, a dup. Without-replacement swap: RMSE 0.0006 vs heading_weber_with_upward_counting, a dup.
- At most one stray per set: -77. Strays drawn only from outside the concept: -24.
- Interior (non-extreme) examples at weight ω on interval evidence only: ω→1, null.
- Prefix pair 31,3,1→31,3,1,15: the model gives evens 0.24→0.42, observed about 0.2→0.6. Deviance is about 650 over 1307 cells, which is the binomial ceiling.
- Submitted `interior_example_range_weight` (heading_weber_upward_with_replacement + ω), smoke -3193.5, 17 ms/grad. Likely a novelty dup.
- PITFALL: the contract check evaluates at prior draws, where the Taylor axis can go negative. Guard with pt.maximum(evidence, 1e-300) and clip the member ratio.
See [[exp3-iter3-trend-origin-room]], [[exp3-iter4-critique]].
