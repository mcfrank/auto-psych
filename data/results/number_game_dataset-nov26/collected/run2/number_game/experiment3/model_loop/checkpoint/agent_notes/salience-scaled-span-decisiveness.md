---
name: salience-scaled-span-decisiveness
description: exp3 — slope × (shared-profile salience)^gamma on span_interval_similarity: smoke -4709.7, ~0 gain vs incumbent
metadata:
  type: project
---
Tested on 2026-11-02 in experiment3 iter_0/candidate_3_retry_1 (salience_scaled_span_decisiveness). The logit was intercept + slope*exp(gamma*(log den - mean))*share. Smoke loglik -4709.7, versus incumbent MAP -4709.4, so it is probably no better. The item_mean_variance critique is not closed by temperature scaling.
**Why:** this matches [[evidence-scaled-decisiveness]]; decision-rule tweaks to the slope give ~0 on exp3.
**How to apply:** don't retry slope-scaling mechanisms; the single-example mixtures in [[single-example-random-doubt]] were the ones that gained.
