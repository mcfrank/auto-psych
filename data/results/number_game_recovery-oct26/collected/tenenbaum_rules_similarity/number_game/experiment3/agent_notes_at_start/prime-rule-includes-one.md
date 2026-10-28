---
name: prime-rule-includes-one
description: All number-game models in the lineage listed 1 as a prime; removing it gained ~46 nats at MAP in experiment2
metadata:
  type: project
---
The shared `concept_sets()` rule list (copied in every model up to range_restricted_rule_concepts) put 1 in the "primes" rule. Experiment-2 data show people treat 1 as not prime (31,3,1 → "odd", not "prime"; after 67,31,17,61 they say no to 1). Removing 1 raised the incumbent's MAP log-lik from -2100.3 to -2054.1 (submitted as `prime_concept_excludes_one`, round 0 candidate_3_repair_1, 2026-10-19).

**Why:** Per-cell residuals of a MAP fit found this miss right away. Parameter-only tweaks (a separate region width) were rejected as near-duplicates because the conjunction share is only ~1-3%.
**How to apply:** Before proposing a new mechanism, run a MAP fit and rank the (set, target) cells by log loss; check the rule definitions for errors. See [[number-game-novelty-gate]].
