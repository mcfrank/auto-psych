---
name: anchored-without-replacement
description: exp2 iter_2 c5: without-replacement sampling on first_example_anchored_reliability fits ~7 ll worse (distinct, RMSE 0.011); rule-size prior exponent pins at 0
metadata:
  type: project
---
exp2 iter_2/candidate_5 (2026-11-01). Submitted: without_replacement_anchored_concepts = first_example_anchored_reliability with genuine examples drawn without replacement. The table splits into T_f with f = whether the first example is genuine. It uses the fast flat table + CSR layout, the check passed in 10 s per 200 iterations, and smoke loglik was -2042.
Short NUTS fits (best draw): base -2032.1, new -2038.9. Prediction RMSE vs base is 0.011. The worst cases are tiny rules (8,64→16 or 4; 81,9→27): people commit to powers of 8/9 and cubes LESS than the base model predicts.
Also tried: a rule prior ∝ size^rho within family. rho came out at -0.002 and RMSE was 0.0015, a near-duplicate.
Gotcha: scipy/pm.find_MAP on these models lands on saddles (ll -2300..-2700). Use a short NUTS run and take the best-draw ll instead.
**Why:** exp1 also found without-replacement slightly worse. The data push away from the size principle for tiny rules, yet a global size-principle exponent stays near 1.
**How to apply:** don't retry without-replacement or size-based rule priors. See [[without-replacement-and-small-number-bias]], [[first-example-anchored-reliability]].
