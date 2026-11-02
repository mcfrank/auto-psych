---
name: exemplar-concept-similarity-pooling
description: exp2 iter_2 c0 exemplar brief: averaging one-shot Bayesian generalisation from each example separately gives smoke ll -3002 (~730 nats worse)
metadata:
  type: project
---
experiment2 iter_2/candidate_0 (2026-10-30), exemplar_concept_similarity_pooling: similarity(t,x) = one-shot posterior membership from single example x (incumbent concept space, beta-share ranges), p_yes = (mean over examples)^sensitivity. Smoke ll -3002 vs incumbent ≈ -2273; ~45 s per 200 draws.
**Why:** pooling examples separately throws away the conjunction/size-principle narrowing people clearly show.
**How to apply:** exemplar/prototype brief slots will lose big unless aggregation is conjunctive (product/geometric mean ≈ Bayesian); see [[graded-neighbourhood-magnitude]].
