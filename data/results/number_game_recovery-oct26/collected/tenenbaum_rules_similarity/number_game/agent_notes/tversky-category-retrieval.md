---
name: tversky-category-retrieval
description: exp3 iter_4 c1 (exemplar/prototype brief): Tversky retrieval of stored rules+ranges; first try too slow (alpha unidentified); repair smoke -3875, ~20s/run
metadata:
  type: project
---
On 2026-10-22, exp3 iter_4 candidate_1 submitted `tversky_category_retrieval`: 42 rules plus 5050 intervals as stored categories, retrieval = fam·sim^gamma, p_yes = retrieval-weighted membership.
The first attempt was REJECTED as too slow (still running after 30 min at target_accept 0.99). The cause was the "examples lacked" weight alpha drifting to about 31 at MAP, a weakly identified posterior, combined with a 27 ms gradient.
The repair (candidate_1_repair_1):
- Categories lacking any example are excluded outright, so alpha is gone.
- The decade boost is dropped, so intervals are pooled by width only (100 groups).
- A lapse parameter is added. Without it, p for the examples is exactly 1 and LL is about -20000; the fitted lapse is about 0.43.
- The per-cell gather is replaced by rule matmul + batched_dot into a (set x target) table.
Result: 9.5 ms gradient, tree depth 4-5, smoke LL -3875.7 (about 780 nats worse than the incumbent), 17-21 s per smoke run.
**Why:** retrieval accounts lose by hundreds of nats, like [[exemplar-softmin-poor-fit]] and [[prototype-matching-poor-fit]]. Unbounded similarity weights wreck NUTS geometry.
**How to apply:** in similarity models, use hard consistency instead of a free "lacked" weight, and add a lapse. Check the admission outcome. See [[pymc-model-speed-patterns]].
