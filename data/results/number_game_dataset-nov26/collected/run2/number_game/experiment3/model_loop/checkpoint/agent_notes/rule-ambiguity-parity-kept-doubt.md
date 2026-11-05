---
name: rule-ambiguity-parity-kept-doubt
description: exp3 iter_2 c4 — parity-keeping doubt grafted on rule_ambiguity_single_doubt: smoke -4674.5 vs incumbent -4675.4, ~+1 only
metadata:
  type: project
---
exp3 iter_2 candidate_4 (2026-11-02): took the incumbent rule_ambiguity_single_doubt and multiplied its n=1 flat yes floor by (1 - parity_keep) for opposite-parity targets, the graft from parity_preserving_random_doubt. The smoke fit gave -4674.5, against the incumbent's -4675.4 smoke, so the gain is negligible. Many low-mean n=1 sets (100, 11, 30, 50, 31) have no opposite-parity targets, so most of the remaining n=1 set variance is likely binomial noise, since each set has only 9-99 rows.

**Why:** the n1_set_mean_variance critique has q=0.128, so it is not FDR-significant, and the parity graft barely moves the fit.
**How to apply:** stop stacking more n=1 doubt tweaks; look elsewhere. Related: [[rule-ambiguity-single-doubt]] [[parity-preserving-random-doubt]] [[exp3-n1-data-facts]]
