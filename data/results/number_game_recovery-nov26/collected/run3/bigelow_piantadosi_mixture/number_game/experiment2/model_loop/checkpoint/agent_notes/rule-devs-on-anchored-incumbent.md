---
name: rule-devs-on-anchored-incumbent
description: exp2 iter_1 c4: adding per-rule deviations (fixed sd) to first_example_anchored_reliability gives +17..21 MAP ll
metadata:
  type: project
---
exp2 iter_1/candidate_4 (2026-11-01): the incumbent first_example_anchored_reliability (family-only weights) has MAP ll -2031.5. Adding rule_logw ~ N(family_level, sd fixed) gives MAP ll -2010.7 at sd 1.6, -2012.2 at 0.8, -2013.6 at 0.6 and -2014.9 at 0.5. FPUR (no anchor, inferred tau) is at -2012.3, so the anchor itself is worth only about 1.5 ll once rules have their own weights.
The largest MAP deviations are a "5" theme (powers of 5 +1.3, ends in 5 +.7, contains 5 +.7), ends in 3 -.8 and powers of 10 -.75.
Prediction RMSE of the sd-0.6 model: 0.019 vs FPUR, 0.028 vs incumbent. Submitted at sd 0.6 as rule_specific_anchored_reliability.
**How to apply:** check this candidate's ELPD before retrying it. A digit-5 salience concept could be a parsimonious alternative. See [[first-example-anchored-reliability]], [[tau-funnel-speed-fix]].
Speed: the incumbent's own graph runs at 15.7 ms/grad × 32 steps, about an hour for admission, so it is too slow. Fix: a flat (m,first,k,s) table with rule_flat_idx gathers, plus CSR structured_dot for the interval/cell counts and the cell→set one-hot (copied from iter_0/candidate_3_repair_1). That brings it to 2.1 ms/grad with the same loglik.
