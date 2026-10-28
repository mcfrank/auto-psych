---
name: incumbent-saturation-screens
description: iter_3 MAP screens on serial_position_ends_inflated_size — what was null, and that freeing the rule-weight prior gained
metadata:
  type: project
---
Experiment3 iter_3 candidate_4 (2026-10-21). MAP screens on the incumbent serial_position_ends_inflated_size (its MAP log-lik is -3189.1):
- Decade-aligned range prior boost (decade_boundary critique, q=0.21). Null under both decade definitions (tens digit, and (x-1)//10): the boost goes to about 0 and log-lik does not change.
- Uniform check lapse for arithmetic rules: -0.3 nats.
- Per-cell excess deviance is 661 over 1312 cells, about the chi-square expectation, so the model is at the noise ceiling with only 5 responses per cell.
- Replacing Dirichlet(2) rule_weights with spread*ZeroSumNormal + a HalfNormal(1.5) spread gained +3.4 nats at MAP (spread about 4.4, p_yes RMSE 0.0115). The Dirichlet(2) prior over-shrinks the rule weights. Submitted as pooled_rule_familiarity_spread.

**Why:** avoids re-screening dead ideas; the rule-weight prior is the live lever.
**How to apply:** see [[map-screening-refinements]]; scripts are in iter_3/candidate_4/scratch (mapfit.py, resid.py, bench.py).

Repair (iter_3 candidate_4_repair_1, final attempt): pooled_rule_familiarity_spread timed out at target_accept 0.99, a HalfNormal-spread funnel. The fix, uneven_rule_familiarity_ends:
- Fixed spread 3.0 × ZeroSumNormal z, with z projected orthogonal to the centred log-size direction so it cannot trade off with size_exponent.
- MAP log-lik -3186.4, which is +2.7 nats; p_yes RMSE vs the incumbent 0.0089.
- First benchmark: 31 steps per iteration, 0 divergences, but size_inflation (HalfNormal(10)) had ESS 8.
- Switching size_inflation to LogNormal(log 1.5, 0.75) raised its ESS to 119, with a minimum of 79 over 250 draws, in 435 s.
**How to apply:** on this base, put a LogNormal prior on size_inflation, never HalfNormal; it is the slowest-mixing parameter.
