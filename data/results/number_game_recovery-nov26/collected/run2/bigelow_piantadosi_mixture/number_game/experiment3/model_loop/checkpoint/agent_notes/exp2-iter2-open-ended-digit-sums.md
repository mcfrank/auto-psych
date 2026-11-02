---
name: exp2-iter2-open-ended-digit-sums
description: Exp2 iter-2 cand_5 — digit-sum family grafted onto open_ended_range_discount; MAP 1923.8 (parent 1925.1, incumbent ~1928); submitted
metadata:
  type: project
---
Exp2 iter 2, candidate_5 (2026-10-30), a refinement slot. I added the digit-sum family (log_sum_share ~ N(-3, 1.5)) to `open_ended_range_discount`. MAP log-likelihoods:
- parent: -1925.15 (k 0.14, a -0.70)
- graft: -1923.77 (k 0.09, a -0.69, log_sum_share -2.6)
- incumbent digit_sum_concepts: about -1928

The two components look additive. The smoke check gave loglik -1926.6 in about 60 s. I submitted it as `open_ended_ranges_with_digit_sums`. The MAP script (/tmp/c5i2/map.py) uses pm.find_MAP, then compile_fn(observedlogp) restricted to value_vars.
**How to apply:** if this wins, the range-prior and concept-family slots are both used up. The remaining lead is the recency residual (see [[exp2-iter2-critique]]).
