---
name: map-screening-refinements
description: Fast MAP screen (~1 min) to compare refinements of the inflated_size_familiar_rule_prior incumbent before the slow check
metadata:
  type: project
---
On 2026-10-21 (experiment3, iter_2, candidate_4), pm.find_MAP on the model_loop responses took about 45 s for incumbent variants. The incumbent's MAP log-likelihood was -3190.1. Its posterior means (log_gain 0.21, size_inflation 1.3, anchor_weight 0.93, size_exponent 0.74) differ from the design fit cache in experiment3/design/_fit_cache.

One-parameter tweaks each gained at most about 1 nat:
- interval width-power prior
- anchoring on the last example
- belief floor
- set-size-dependent gain
- redundancy exponent

Replacing the 42 per-rule Dirichlet weights with 8 equal-within-family weights lost about 1260 nats, so individual rule weights are essential.

Sampling examples without replacement (no_repeat_example_sampling) gained 1.45 nats with no new parameter.

Residuals left after MAP: generalization to unseen members of powers of 2 and multiples of 7 is overpredicted.

**Why:** cheap screening avoids spending 12-minute checks on dead ideas.
**How to apply:** script pattern is at /tmp/c4/mapfit.py (may be gone); rebuild it with pm.set_data(prepare_observed(rows)) and find_MAP, and compare p_yes RMSE against the incumbent (>0.002 is needed for novelty). See [[volatile-model-speed]].

Pitfall: read p_yes from the find_MAP result dict (mp["p_yes"]). compile_fn on a Deterministic with value-var inputs silently samples the RVs. no_repeat_example_sampling at MAP: loglik -3188.7 vs -3190.1, p_yes RMSE vs incumbent 0.0041.

Further nulls (2026-10-21, experiment3 iter_3 candidate_3; serial_position_ends_inflated_size base, MAP loglik -3189.1):
- Decade categories as rule-like concepts with a prior share, using either the (x-1)//10 or the x//10 convention: share goes to about 0.005, gain 0.1 nats, RMSE vs incumbent 0.0014-0.0021. The critique's decade_boundary_effect (q=0.21) is not capturable this way.
- Inferred example-noise rate: estimate 0.106 (fixed value is 0.1), RMSE 0.0011, a null.
- Residual checks: parity and example-target calibration are fine. The incumbent is well calibrated; per-cell residuals look like noise (5 responses per cell).
