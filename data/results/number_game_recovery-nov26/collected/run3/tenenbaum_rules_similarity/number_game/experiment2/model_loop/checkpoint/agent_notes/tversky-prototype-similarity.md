---
name: tversky-prototype-similarity
description: exp2 iter_4 c2 similarity brief: Tversky ratio model to examples' shared-feature prototype, smoke ll -2746 (~-460 vs concept models)
metadata:
  type: project
---
Exp2 iter_4 candidate_2 (similarity brief): Tversky ratio similarity between target and the prototype (features shared by all examples: parity, multiples 3-12, squares/cubes/powers, primes, last digit, decade) plus exp-decay closeness to nearest example as one weighted property; Dirichlet salience over 7 dims, alpha/beta, sigmoid link. Passed check_candidate in ~2 s; smoke ll -2745.8, so expect it to lose ~450 nats to the concept learners (best LOO -2284), the same as other similarity models ([[exemplar-concept-similarity-pooling]], [[graded-neighbourhood-magnitude]]).
**Why:** the pure-similarity families keep landing hundreds of nats behind.
**How to apply:** for future similarity briefs, expect a large loss; the prototype/ratio layout gets admitted cheaply via prepare_observed per-kind counts.
