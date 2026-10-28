---
name: two-cluster-candidate
description: Exp2 iter_2 cand_2 submitted two_cluster_range_concepts (union of two equal-width blocks); no-conj base samples in 26s
metadata:
  type: project
---
2026-10-19, exp2 iter_2 candidate_2 (exploratory): submitted `two_cluster_range_concepts`. It is the incumbent without conjunctions, keeping upper-edge and prime fix, plus two-block union concepts (width 1-25, Erlang width prior that shares the width scale). Smoke LL was -2058.0 and each sampling run took 26 s. It was written first, with no probing (see [[write-candidate-first]]).

**Why:** This records which concept families were tried in exploratory slots.
**How to apply:** If it ties or loses, don't retry union-of-ranges concepts. See also [[concept-space-saturated]].
