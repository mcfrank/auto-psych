---
name: conjunctive-shared-feature-exemplars
description: exp3 iter_0 c2 exemplar brief: additive-clustering similarity + geometric mean over examples, smoke ll -4522 (~-860 vs concept models)
metadata:
  type: project
---
Exp3 iter_0 candidate_2 (2026-10-31), conjunctive_shared_feature_exemplars: s(t,x) = summed weight of shared properties (incumbent rule sets, no exception rules; weight = kind salience × size^-d) + magnitude exp(-|t-x|/h), normalised by self-similarity; logit = a + b·mean log s. Passed check in ~1 min (9 s per 200 draws); smoke ll -4522 vs best LOO ≈ -3664.
**Why:** even conjunctive (geometric-mean) aggregation doesn't rescue similarity models; it confirms [[exemplar-concept-similarity-pooling]] and [[tversky-prototype-similarity]].
**How to apply:** exemplar briefs will lose by hundreds of nats regardless of aggregation; spend minimal effort, just make them admissible.
