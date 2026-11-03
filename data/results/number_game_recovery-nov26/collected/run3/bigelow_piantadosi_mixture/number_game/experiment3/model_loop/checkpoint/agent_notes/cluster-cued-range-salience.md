---
name: cluster-cued-range-salience
description: exp3 iter_1 c2: range prior × exp(beta·(1-span/99)) on range_limited incumbent, beta≈.47, +1.8 MAP; Erlang/width priors fail on the narrow-span critique
metadata:
  type: project
---
exp3 iter_1/candidate_2 (2026-11-01), cluster_cued_range_salience. The critique narrow_rule_inhull_nonmember_span_contrast is driven by narrow odd-only sets (31,3,1,15: obs .93 vs pred .53; 63,23,43; 73,35,53,59; 37,57). There people accept the evens inside the span. Wide-span sets were calibrated.
- Erlang range-size prior: ll −4 (sigma→39). Width power s^-γ: γ→.03, a tie. Range-size priors cannot fix this.
- Data-dependent range salience, log range prior + beta·tight: tight = 1−span/99 with n=1 → 0 gave +1.8 MAP (−3318.97); the narrow-span pred rose .41→.44. A log-span version gave +0.8; counting n=1 as tight gave +0.7.
- Training RMSE vs the incumbent is .011. Check OK; smoke fit 24 s.
**How to apply:** clustering cues help a little. A remaining gap could come from restricted rules ("odd within range") penalising in-span evens. See [[range-limited-rule-familiar-strays]] and [[normative-distortions-collapse]].
