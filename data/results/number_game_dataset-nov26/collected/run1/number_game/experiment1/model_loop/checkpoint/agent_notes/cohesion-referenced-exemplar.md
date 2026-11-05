---
name: cohesion-referenced-exemplar
description: iter_3 cand_1 exploratory: exemplar sim vs set's own mean pairwise similarity as criterion; smoke log-lik -1594.3
metadata:
  type: project
---
iter_3 candidate_1 (exploratory, exemplar brief): target mean similarity (mag, final digit, fixed parity) minus w * log mean pairwise similarity among examples; singleton sets get a free cohesion value. Aimed at critique set_level_yes_rate_sd (observed > model). Fixed attention, no variability tuning. Smoke fit log-lik -1594.3 (7 params, ~28s per 100/100 chain). Related: [[iter3-critique-stats]].

Repair (candidate_1_repair_1): original rejected for R-hat 1.055 / ESS 47 — cause was beta*cohesion_weight*singleton_cohesion multiplicative coupling. Fix: put cohesion weight and singleton offset directly on the logit (logit = alpha + beta*log_fit - w*log_coh, singleton gets free offset). 4x500 fit: R-hat ≤1.01, ESS ≥330, 0 divergences; smoke log-lik -1592.7.
**How to apply:** keep criterion-type parameters additive on the logit, never multiplied by the similarity slope.
