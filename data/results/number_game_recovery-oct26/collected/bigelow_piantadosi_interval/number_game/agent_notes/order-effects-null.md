---
name: order-effects-null
description: Order/sequential mechanisms screened at MAP in experiment3 iter_3 — data strongly favour order-blind accumulation
metadata:
  type: project
---
2026-10-21, experiment3 iter_3 candidate_0 (exploratory slot, order brief), on the inflated_size_familiar_rule_prior base (MAP loglik -3190.1):
- Surprise-weighted evidence (each later example's weight = exp(-eta * log(100 * predictive prob)), soft-capped): MAP eta = -0.008. The profile is sharp: eta = -0.5 costs 44 nats, +0.5 costs 146. Submitted as surprise_weighted_sequential_evidence. Predictions are sensitive to eta (±0.05 gives RMSE 0.02), so novelty should be OK.
- First example alone sets the rule-vs-range family belief, with later examples counting by a fraction lambda at the family level: MAP lambda = 0.97, loglik -3190.5. Null.
**Why:** order/position ideas keep collapsing to the order-blind model. Only one set ("8,16"/"16,8") appears in two orders.
**How to apply:** for order briefs, expect the null; prefer mechanisms whose predictions move strongly with the parameter (to pass novelty) or accept a falsification result. Pitfall: evaluate p_yes at fixed values with pm.do(model, {all free RVs: values})["p_yes"].eval(). See [[map-screening-refinements]], [[volatile-model-speed]].
