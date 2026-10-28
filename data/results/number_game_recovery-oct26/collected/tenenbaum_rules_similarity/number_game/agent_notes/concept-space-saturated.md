---
name: concept-space-saturated
description: Exp2 iter_1 probes: no added concept family or edit to the incumbent's 42 rules improves MAP log-lik by >1 nat
metadata:
  type: project
---
On 2026-10-19 (experiment2, iter_1 candidate_2), with prime_concept_excludes_one refit at MAP (log-lik about -2053), I added concept families with their own share and re-optimized everything (Powell). None gained more than 1.2 nats: repdigits, digit sums, multiples of 13-50, odd multiples, Fibonacci, triangular (+0.7), final-digit pairs, offset progressions, rule plus one exception, rule unions, near-miss (+-1) rules, unions of two decade blocks, composites, odd composites (+0.6). Edits to existing rules all hurt by 10-300 nats: squares/cubes/odd without 1, powers with 1, single digits dropped from the final-digit rules, multiples without k itself. Submitted odd_composite_concept anyway (a rule-with-exceptions kind).

**Why:** The incumbent's rule list is already tuned; with about 5 responses per cell, new concepts can't be told apart from noise.
**How to apply:** Don't spend another concept-space slot on additive families. Look at noise/response or interval-prior structure instead. NUTS posterior-mean p for the incumbent is in iter_1/candidate_2/scratch/p_inc.npy, and refit2.py there is a fast numpy refit harness. Related: [[prime-rule-includes-one]].
