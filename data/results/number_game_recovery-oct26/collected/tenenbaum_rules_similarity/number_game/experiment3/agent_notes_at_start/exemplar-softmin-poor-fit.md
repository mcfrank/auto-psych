---
name: exemplar-softmin-poor-fit
description: exp2 iter_2 cand_0: Shepard per-exemplar similarity + soft-min combination smoke-fit loglik ~-2905 vs incumbent ~-2060
metadata:
  type: project
---
Exemplar accounts (per-exemplar Shepard generalization, combined by soft-min generalized mean, floor/ceiling mapping) fit far worse than joint Bayesian concept learning: smoke-fit loglik -2905 vs incumbent ELPD ~-2060. Code: experiment2/model_loop/iter_2/candidate_0/candidate.py (precomputes 100x100 single-exemplar similarity as a matmul; fast, ~7s/100 draws).
**Why:** brief forced an exemplar slot; the size principle across pooled examples seems essential.
**How to apply:** for future exemplar/prototype briefs, reuse that code skeleton; don't expect them to compete. See [[concept-space-saturated]].
