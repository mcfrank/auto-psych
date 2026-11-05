---
name: single-exception-tolerance
description: exp3 iter_2 c2 — all-but-one-example properties (n>=3) at a tolerance fraction on rule_ambiguity incumbent: +2.5 loglik MAP, tolerance ~.07, smoke -4673.5 vs -4675.4; n=1 arithmetic discount ~0
metadata:
  type: project
---
exp3 iter_2 candidate_2 (2026-11-02), built on rule_ambiguity_single_doubt. Properties held by n-1 of n>=3 examples enter the share at salience*tolerance. MAP loglik -4662.1 vs -4664.6, tolerance ~0.07; recency and primacy shrink a little because they overlap with it.
Also tried, ~0: on n=1 sets, discount the arithmetic families (keep ~0.83).
The incumbent is well calibrated: residual variance at the pair level is only ~0.004 above binomial. The biggest n=1 misses (11, 100, 85 low; 43, 35, 33 high) come from sets with 9-27 observations each.

**Why:** shows the exp3 incumbent family is close to saturated.
**How to apply:** expect small gains from tweaks to the share; a large gain needs a different structure. Related: [[rule-ambiguity-single-doubt]] [[graded-consensus-properties]] [[exp3-n1-data-facts]]
