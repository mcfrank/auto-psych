---
name: working-memory-plus-ambiguity-doubt
description: exp3 iter_3 c5 - WM 3-example limit + all-sizes rule-ambiguity doubt graft; smoke -4660.0 (best smoke so far)
metadata:
  type: project
---
Grafted all-sizes rule-ambiguity doubt (rule_count over all examples, HalfNormal slopes, separate multi base rate) onto working_memory_three_example_limit. Smoke loglik -4660.0 vs incumbent parity_bounded_ambiguity_doubt smoke -4664.8. Smoke fit ~4 min per 200 iters (NSUB=5 matrix), so full fit is near the time limit.

**Why:** the two mechanisms act on different trials (4-example sets vs ambiguous sets), so they stack.
**How to apply:** next, adding parity-kept doubt ([[parity-bounded-ambiguity-doubt]]) on top is the obvious follow-up. See [[working-memory-three-example-limit]], [[rule-ambiguity-all-sizes]].
