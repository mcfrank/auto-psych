---
name: experiment2-decision-rule-nulls
description: exp2 iter_1 cand1 — every decision-rule transform on beta_share incumbent is null (power, Prelec, no/yes lapse, session drift, entropy-shifted criterion)
metadata:
  type: project
---
experiment2 iter_1/candidate_1 (2026-10-30), quick 2x300 NUTS LOO on beta_share_width_exception_concepts (base -2273.6):
P^kappa: kappa 1.02, -2274.3. Prelec a 0.99, -2274.0. (1-eps)P: eps 0.007, -2273.9. eps+(1-eps)P: eps 0.005, -2274.1. 1-(1-P)^kappa: 0.985, -2274.0. Logit drift with trial_index: 0.13±0.13, -2274.1. Logit shift ∝ posterior entropy: 0.012±0.038 per nat, -2274.2.
Submitted the entropy shift as uncertainty_shifted_criterion; it may fail novelty (RMSE gate 0.002, posterior-mean p_yes).
Novelty pool rows carry only number_set/target (no trial_index), so session/participant terms can't be compared there.
Pitfall: parallel arviz runs crash on ~/.cache/arviz/daily_warning; set HOME to a tmp dir.
**Why:** the exp2 readout is calibrated; decision-level mechanisms can't be told apart here.
**How to apply:** don't spend more decision-rule slots on exp2 data; see [[experiment2-round0-readout-and-sampling]], [[experiment2-null-refinements]].
