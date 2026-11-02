---
name: normative-distortions-collapse
description: exp2 iter_3 c2: exponent distortions of the normative concept learner all collapse; target-only final-digit binding error eps≈0.04 survives
metadata:
  type: project
---
Tested in exp2 iter_3/candidate_2 (2026-11-01) as MAP screens on the uniform-stray, family-only, Beta-reliability learner (MAP ll -2031.2). Each one collapsed to the normative value:
- set log-lik × n^-gamma: gamma 0.04
- size exponent beta: 0.94
- interval width prior rho: -0.08
- range-only size exponent: 0.96 (+0.6 ll)
- Gaussian target noise for range membership: sigma≈0.5
- final-digit binding error applied to the examples: eps→0

What was submitted: final_digit_binding_errors. Under "ends in d", only the target check accepts a number with d in the tens place, at rate eps. MAP eps is 0.037 (+0.1 ll), with posterior 90% CI [.008, .08]. Its p_yes differs from the normative fit by RMSE 0.009 on the data trials. It runs at 3.3 ms/grad with about 27 steps at 0.99.
Speed trick: with uniform strays, rules and intervals both use one (n,k,size) table. Cells read their set's posterior through a one-hot (cells,sets) pt.dot, and interval sums are pt.dot(counts, lik.T) masked by a one-hot on n. This was 3x faster than gathers.
**How to apply:** don't spend normative+distortion slots on exponents. Cell-level residuals (e.g. 16,26,96→67 at 5/5) are a better place to look. See [[noisy-criterion-decision-rule]], [[fast-incumbent-sampling]].
