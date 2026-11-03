---
name: limited-retrieval-generation
description: exp3 iter_1 cand_0 HyGene-style retrieval 1-(1-prior)^K on fives incumbent: only ~+2 loglik at K≈e; large K very bad
metadata:
  type: project
---
2026-10-31, exp3 iter_1/candidate_0 `limited_retrieval_hypothesis_generation`. Concept weight = log(1-(1-prior)^K) instead of log prior, on fives_privileged_decoupled_prime_odd (nested: K→0 is the incumbent). MAP loglik -3435.6 vs incumbent -3435.8. Profile over log K (others at MAP): -8→-3438.9, 1→-3435.6, 2→-3443, 4→-3699, 8→-4901. Data want prior differences kept (no flattening of familiar rules).
**Why:** hypothesis-generation limits on retrieval are ruled out as a big gain; the prior's full range matters.
**How to apply:** don't re-screen prior-saturation/flattening transforms. Scripts: that candidate's scratch/w/prof.py (uses replace_rvs_by_values, see [[factor-graded-multiples-finding]]).
