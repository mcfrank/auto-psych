---
name: parity-doubt-three-example-memory
description: exp3 iter_3 c3 — working-memory 3-example limit grafted on parity_bounded_ambiguity_doubt (doubt+parity also from subset): smoke -4658.3 vs -4664.8
metadata:
  type: project
---
exp3 iter_3 candidate_3 (2026-11-02): `parity_doubt_three_example_memory`. It is the incumbent parity_bounded_ambiguity_doubt plus working_memory_three_example_limit. The full p (judge + rule-count doubt + parity-kept flat yes) is computed for the full set and for each 3-subset, then mixed by `drop` on n=4 sets. Check passed: smoke loglik -4658.3 (incumbent -4664.8), about 230 s for 200 iterations on 1 chain.

**Why:** the WM graft still adds about 6 nats on top of the all-sizes doubt, so the two mechanisms are not redundant.
**How to apply:** the WM graft on the incumbent is now taken. Related: [[working-memory-three-example-limit]] [[parity-bounded-ambiguity-doubt]]
