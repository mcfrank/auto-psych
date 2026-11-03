---
name: ratio-scale-exemplar-retrieval
description: exp3 iter_3 c1 similarity slot: graded_exemplar_equal_rule_prior with Weber metric D=20|ln x-ln y|; in-graph r^D (100x100) matmul replaces distance counts
metadata:
  type: project
---
exp3 iter_3/candidate_1 (2026-11-02) is ratio_scale_exemplar_retrieval. It is the exp3 graded_exemplar_equal_rule_prior with similarity r^D, where D = 20|ln x − ln y| (unit fixed, so r is identifiable). The grid is logit r ∈ [−8, −0.5] with 100 points.
- Rule weights are computed in the graph: one-hot(examples) @ (r^D @ RULE_MEMBERSHIP.T / size). Integer distance counts are not needed.
- Range tables use row cumsums of r^D per grid point. dmin is the distance to the nearer range edge.
- A related idea lost before: log compression of range *sampling* (exp2, [[compressed-number-line-ranges]], −18 nats). This model instead uses log compression for the similarity metric.
See [[graded-exemplar-equal-rule-prior]], [[digit-position-exemplar-retrieval]].
- Check result: OK. Smoke ll is −3457, far below the ~−3325 of the concept models, so the ratio metric probably loses by about 100 nats or more. 200 NUTS iterations take about 35 s, and the full check takes 4.8 min.
**How to apply:** do not re-propose log or Weber similarity on the number line. Linear distance is far better.
