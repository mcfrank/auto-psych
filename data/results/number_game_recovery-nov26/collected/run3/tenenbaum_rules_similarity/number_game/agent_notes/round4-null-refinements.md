---
name: round4-null-refinements
description: Round-4 quick-NUTS tests on parity_within_range_concepts (lppd -1065.0) — sampling shape, multiples prior, fuzzy edges all fail
metadata:
  type: project
---
Round 4 (2026-10-30), iter_4/candidate_0, quick NUTS (2x300) lppd; best model parity_within_range_concepts = -1065.0. Calibration by set size, in/out of span, parity, distance, p-bin is near perfect.
- Center-weighted ("representative") sampling within ranges, exp(-k u^2): k=2 -1075, k=5 -1112, k=-2 (edge-weighted) -1080. Uniform is best.
- Multiples-of-k prior ∝ k^beta: beta≈0, -1065.0 (no gain).
- Fuzzy range edges at readout exp(-d/λ): λ=1 -1065.4, λ=3 -1067.1.
- Data = 5 participants × 450 identical trials (rows in blocks of 450; trial_index resets), but no participant_id column and design rows lack trial_index — person-level models not safe.
- find_MAP on these Dirichlet models gave bogus ll (-1390); use quick NUTS instead.
**Why:** saves re-testing. **How to apply:** see [[incumbent-residual-diagnostics]], [[concept-space-additions-map]].
- Doubling/tripling chains (a·2^k, a odd ≥3; a·3^k), own family: -1066.1 (tie, weight ≈0.04). Submitted as doubling_chain_concepts (iter_4/candidate_0); current data have no chain-like sets, so the disagreement is on untested sets like 5,10,20.
