---
name: digit-parity-concepts
description: exp3 iter_0 c1: "only odd digits"/"only even digits" rule family on beta_share; weight ≈0.001 at mode, RMSE 0.0074 vs incumbent
metadata:
  type: project
---
Exp3 iter_0 candidate_1 (2026-10-31): added a 9th rule family (all-odd-digit numbers, 30 of them; all-even-digit numbers, 24) to beta_share_width_exception_concepts. At an L-BFGS mode the family weight is ≈0.001, so the data essentially reject it. p_yes RMSE vs the incumbent is 0.0074 (above the 0.002 novelty threshold only because of the Dirichlet(2) prior). Smoke check OK in 16 s.
Exp3 residuals are noise-dominated (≈5 answers per set×target cell), so there are no clear concept-space misfits.
**Why:** so the digit-parity kind isn't proposed again; it's another concept addition that gains ≈0, like [[concept-space-additions-map]].
**How to apply:** for a quick residual readout, use the mode-fit script pattern at experiment3/model_loop/iter_0/candidate_1/scratch/w/resid.py (scipy L-BFGS on compile_logp; use replace_rvs_by_values for p_yes).
