---
name: prime-plus-marked-odd
description: exp2 iter_2 cand_5 repair, prime_salient_hundred_chart plus odd-only cost; first attempt rejected as RMSE-0 duplicate of prime_salient_hundred_chart
metadata:
  type: project
---
2026-10-30, exp2 iter_2/candidate_5. The first attempt re-implemented prime bonus + chart rows and was rejected as a duplicate of `prime_salient_hundred_chart` (RMSE 0). It was the same graft in the other direction. The repair, `prime_salient_marked_odd`, adds the odd_cost (rule index 0, Normal(0,3)) from [[marked-odd-cost-finding]] to [[prime-bonus-on-hundred-chart]].

Smoke-fit loglik was -2129.8 (prime_salient_hundred_chart: -2141.8) and the check passed.

**Why:** grafting A onto B and B onto A gives an identical model, and the novelty gate catches it.
**How to apply:** before you graft, check that the composed model is not already live under another name.
