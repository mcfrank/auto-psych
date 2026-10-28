---
name: uneven-familiarity-graft
description: iter_4 cand_5 — uneven rule familiarity grafted onto conservative_updating_recency_boost; conservatism collapses (update_rate ~0.975)
metadata:
  type: project
---
2026-10-21, experiment3 iter_4 candidate_5 (free refinement slot): uneven_familiarity_conservative_updating = conservative_updating_recency_boost with Dirichlet(2) rule_weights replaced by the incumbent's fixed-spread (3.0) ZeroSumNormal projected off the size direction.
MAP loglik -3186.8 vs incumbent uneven_rule_familiarity_ends -3186.3; update_rate 0.975 (was the mechanism's point), p_yes RMSE vs incumbent 0.0045.
**Why:** the incumbent's ~39-nat ELPD jump came from the rule-familiarity prior, so grafting it is the obvious lever for other live bases; but with it, conservative updating is ~null.
**How to apply:** don't expect the familiarity graft to rescue order/sequential mechanisms; see [[incumbent-saturation-screens]], [[order-effects-null]].
