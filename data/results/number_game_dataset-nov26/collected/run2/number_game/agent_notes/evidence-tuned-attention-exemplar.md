---
name: evidence-tuned-attention-exemplar
description: Exp3 iter_0 c2 exemplar slot: GCM with attention = example-count/(n+kappa) per property; smoke loglik -4929, ~150 behind best
metadata:
  type: project
---
Exp3 iter_0 candidate_2 (exemplar brief): mean exp-similarity to stored examples, distance = family salience x (count of examples having the property)/(n+kappa) summed over the example's properties the target lacks, + n/(n+kappa) x magnitude |t-e|/10. prepare_observed precomputes (trial, slot, family) arrays; ~18 s per 200-iter chain. Passed check; smoke loglik -4929.1 vs best span_interval_similarity ELPD -4778.

**Why:** consensus-tuned attention still doesn't rescue exemplar accounts on exp3.
**How to apply:** exemplar slots keep losing ~150 nats; see [[property-exemplar-gcm]].
