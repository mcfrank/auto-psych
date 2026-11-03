---
name: conjunctive-concept-similarity-exemplars
description: exp3 iter_4 c1 exemplar brief: geometric mean of one-shot concept-mediated similarities, smoke ll -4538 (~-875); fast 100x100 similarity-matrix trick
metadata:
  type: project
---
Exp3 iter_4 candidate_1 (2026-10-31), conjunctive_concept_similarity_exemplars: S[x,t] = P(t in concept | exemplar x alone) under the incumbent prior (rules by family + beta-share intervals), logit p = criterion + gain·mean_i log S[x_i,t]. Smoke ll -4538, 5 s per 200 draws.
Trick: interval counts BOTH[L,x,t] (100³) let the one-shot similarity matrix be rebuilt per gradient step with a tensordot.
**Why:** confirms [[conjunctive-shared-feature-exemplars]] and [[exemplar-concept-similarity-pooling]]: concept-mediated similarity with conjunctive aggregation still loses by hundreds of nats.
**How to apply:** for exemplar briefs, just make the candidate admissible and move on.
