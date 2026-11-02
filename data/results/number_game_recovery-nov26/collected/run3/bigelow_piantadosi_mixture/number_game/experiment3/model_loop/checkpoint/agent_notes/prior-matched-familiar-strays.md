---
name: prior-matched-familiar-strays
description: exp2 iter_4 c4: tying the familiar-stray mix to lambda_0 costs 0.03 MAP ll and drops one parameter (RMSE .0045); count-familiarity and binding graft tie
metadata:
  type: project
---
exp2 iter_4/candidate_4 (2026-11-01), refining familiar_strays_equal_rule_prior. At MAP the incumbent has ll -2032.28, with lambda_0 .19, mix .28, alpha .92 and kappa 18.
Each screen below is one change, with its MAP ll and its p_yes RMSE against the incumbent:
- mix = lambda_0 (stray generator taken from the learner's own program prior, uniform for the non-rule share): -2032.31, RMSE .0045, 3 params. **Submitted** as prior_matched_familiar_strays. Check OK, smoke 10 s.
- Familiarity as a rule count (q ∝ the number of rules containing x): -2032.44, RMSE .0075, mix→.56.
- Final-digit binding graft: eps .008, -2032.04, RMSE .0021. That is a near-dup, and sibling c3 was exploring it.
Harness: iter_4/candidate_4/scratch/map.py. It recovers p_yes from per-obs logp, because compile_fn on p_yes gives garbage.
**How to apply:** check this slot's ELPD outcome to see whether parsimony helps again. See [[equal-rule-prior-parsimony]], [[exp2-iter2-incumbent-saturation]].
