---
name: digit-run-sharpened-exemplar
description: iter_4 c4 — digit-run boosts final-digit attention on shared-factor incumbent, -15.4 nats MLE; same gain as criterion shift
metadata:
  type: project
---
iter_4 candidate_4 (2026-11-02), `digit_run_sharpened_exemplar`, refines `shared_factor_parity_exemplar`. When >=3 examples all share a final digit, c_unit is multiplied by exp(run_sharpening). MLE NLL: base 1554.5; run-criterion shift 1539.1; run digit-attention boost 1539.1 (the same, because the base fit is degenerate with beta≈0.13); multiple-of-5 mismatch dim 1554.5 (all sets) / 1551.6 (only all-mult-5 sets); boost by modal digit count >=3 without all sharing ~0. check OK, smoke LL -1543.6.
**Why:** aims at the digit_run_nonmatch_rate critique (q=0.016). Candidate_3 (also an incumbent slot) may have used the criterion-shift version. **How to apply:** see [[digit-run-strict-criterion]], [[iter4-critique-stats]].
