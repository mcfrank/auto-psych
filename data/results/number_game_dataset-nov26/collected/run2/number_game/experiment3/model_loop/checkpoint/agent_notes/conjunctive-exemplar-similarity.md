---
name: conjunctive-exemplar-similarity
description: Exp3 iter_4 c1 exemplar slot: geometric-mean (product-rule) exemplar sim over per-example profile-mismatch fraction + magnitude: smoke -4919.7
metadata:
  type: project
---
Exp3 iter_4 candidate_1 (exemplar brief): per-example distance = feature_scale x (salience-weighted fraction of that example's property families the target lacks) + mag_scale x |t-e|/100; set similarity = exp(-mean distance) (geometric mean of per-example sims); logit = intercept + slope x sim. prepare_observed, ~17 s per 200-iter chain. Check OK, smoke loglik -4919.7 (incumbent smoke ~-4658).

**Why:** product rule (resemble all examples) barely beats mean-similarity exemplar (-4929); exemplar accounts stay ~260 behind on exp3.
**How to apply:** don't expect exemplar slot fit gains; see [[evidence-tuned-attention-exemplar]], [[property-exemplar-gcm]].
